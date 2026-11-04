"""Unit tests for the discrete-time GPU thermal telemetry simulator.

Verifies:
1. Temperature evolution toward theoretical steady-state
2. Intake-heating anomaly response and resolution
3. Node power-budget enforcement across two workloads
4. Configurable demo time scaling (2.5x)
5. Deterministic repeatability across instances and resets
"""

import unittest
from datetime import timezone

from thermopilot.schemas import GPUTelemetry
from thermopilot.simulator.telemetry import SimulatorConfig, TelemetrySimulator


class TestTelemetrySimulator(unittest.TestCase):
    """Test suite covering the discrete-time telemetry simulator."""

    def setUp(self) -> None:
        self.config = SimulatorConfig(
            power_budget_w=400.0,
            baseline_ambient_c=22.0,
            anomaly_ambient_c=32.0,
            thermal_resistance=0.14,
            thermal_time_constant=15.0,
            time_scale=1.0,  # 1:1 scale for baseline physics tests
            base_dt=1.0,
            initial_temp_c=71.2,
            noise_std=0.0,  # Deterministic by default
            seed=42,
            default_workloads={"wl-alpha": 230.0, "wl-beta": 150.0},
        )
        self.sim = TelemetrySimulator(self.config)

    # -----------------------------------------------------------------------
    # 1. Temperature Evolution Tests
    # -----------------------------------------------------------------------

    def test_temperature_evolution_toward_steady_state(self) -> None:
        """Verify temperature monotonically approaches steady state: T_ss = T_amb + P * R_th."""
        # Total power = 230 + 150 = 380W; T_amb = 22°C; R_th = 0.14
        # T_ss = 22 + (380 * 0.14) = 75.2°C
        # Since initial_temp = 71.2°C, temperature should rise toward 75.2°C.
        initial_t = self.sim.core_temp_c
        self.assertAlmostEqual(initial_t, 71.2, places=2)

        prev_temp = initial_t
        for _ in range(30):
            telem = self.sim.step()
            self.assertIsInstance(telem, GPUTelemetry)
            self.assertGreater(telem.temperature_core_c, prev_temp)
            self.assertLess(telem.temperature_core_c, 75.25)
            prev_temp = telem.temperature_core_c

        # After 100 seconds (several tau=15s), should be within 0.1°C of steady state (75.2°C)
        for _ in range(70):
            telem = self.sim.step()
        self.assertAlmostEqual(telem.temperature_core_c, 75.2, delta=0.1)

    def test_temperature_decay_when_cooling(self) -> None:
        """Verify cooling physics when starting above steady state."""
        hot_config = self.config.model_copy(update={"initial_temp_c": 85.0})
        sim = TelemetrySimulator(hot_config)

        prev_temp = sim.core_temp_c
        for _ in range(20):
            telem = sim.step()
            self.assertLess(telem.temperature_core_c, prev_temp)
            prev_temp = telem.temperature_core_c

        # Verify decay heads toward 75.2°C
        self.assertLess(sim.core_temp_c, 80.0)

    def test_emitted_telemetry_schema_validity(self) -> None:
        """Verify that every emitted record strictly conforms to GPUTelemetry schema."""
        telem = self.sim.step()
        self.assertEqual(telem.gpu_id, 0)
        self.assertGreater(telem.temperature_junction_c, telem.temperature_core_c)
        self.assertGreaterEqual(telem.fan_speed_pct, 50.0)
        self.assertLessEqual(telem.fan_speed_pct, 100.0)
        self.assertEqual(telem.power_limit_w, 400.0)
        self.assertEqual(telem.timestamp.tzinfo, timezone.utc)

    # -----------------------------------------------------------------------
    # 2. Anomaly Response Tests
    # -----------------------------------------------------------------------

    def test_intake_heating_anomaly_trigger_and_resolve(self) -> None:
        """Verify ambient jump to 32°C induces runaway trajectory crossing 81.5°C."""
        # 1. Run baseline for 5 ticks
        for _ in range(5):
            self.sim.step()
        self.assertFalse(self.sim.is_anomaly_active)
        self.assertEqual(self.sim.ambient_temp_c, 22.0)

        # 2. Trigger anomaly: ambient jumps to 32°C
        # New steady state: 32 + (380 * 0.14) = 85.2°C
        self.sim.trigger_anomaly()
        self.assertTrue(self.sim.is_anomaly_active)
        self.assertEqual(self.sim.ambient_temp_c, 32.0)

        # 3. Step forward and observe core temp climbing past warning threshold (81.5°C)
        for _ in range(40):
            self.sim.step()

        self.assertGreater(self.sim.core_temp_c, 81.5)
        self.assertLessEqual(self.sim.core_temp_c, 85.2)

        # 4. Resolve anomaly: ambient restored to 22°C
        self.sim.resolve_anomaly()
        self.assertFalse(self.sim.is_anomaly_active)
        self.assertEqual(self.sim.ambient_temp_c, 22.0)

        # 5. Core temp begins cooling decay
        prev_t = self.sim.core_temp_c
        for _ in range(10):
            telem = self.sim.step()
            self.assertLess(telem.temperature_core_c, prev_t)
            prev_t = telem.temperature_core_c

    def test_scheduled_anomaly(self) -> None:
        """Verify deterministic scheduled anomaly start and end ticks."""
        self.sim.schedule_anomaly(start_tick=3, end_tick=8)

        # Ticks 0, 1, 2: normal
        for _ in range(3):
            self.sim.step()
        self.assertTrue(self.sim.is_anomaly_active)

        # Ticks 3, 4, 5, 6, 7: anomaly active
        for _ in range(5):
            self.sim.step()
        self.assertFalse(self.sim.is_anomaly_active)

    # -----------------------------------------------------------------------
    # 3. Power-Budget Enforcement Tests
    # -----------------------------------------------------------------------

    def test_two_workloads_initial_allocation(self) -> None:
        """Verify two workloads share the 400W power rail."""
        self.assertEqual(self.sim.get_workload_power("wl-alpha"), 230.0)
        self.assertEqual(self.sim.get_workload_power("wl-beta"), 150.0)
        self.assertEqual(self.sim.total_allocated_power_w, 380.0)
        self.assertEqual(self.sim.effective_power_draw_w, 380.0)

    def test_power_budget_clamping(self) -> None:
        """Verify that total power is clamped when requested power exceeds budget."""
        # Alpha wants 250W, Beta wants 200W -> sum is 450W (> 400W budget)
        self.sim.set_workload_power("wl-alpha", 250.0)
        allocated_beta = self.sim.set_workload_power("wl-beta", 200.0, clamp=True)

        # Beta should have been clamped to remaining 150W
        self.assertEqual(allocated_beta, 150.0)
        self.assertEqual(self.sim.get_workload_power("wl-beta"), 150.0)
        self.assertEqual(self.sim.effective_power_draw_w, 400.0)

    def test_power_budget_rejection_without_clamp(self) -> None:
        """Verify ValueError is raised if clamp=False and request exceeds budget."""
        self.sim.set_workload_power("wl-alpha", 250.0)
        with self.assertRaises(ValueError) as ctx:
            self.sim.set_workload_power("wl-beta", 200.0, clamp=False)
        self.assertIn("exceeds power budget", str(ctx.exception))

    def test_workload_removal_frees_power(self) -> None:
        """Verify pausing/removing a workload reduces power draw."""
        self.sim.remove_workload("wl-beta")
        self.assertEqual(self.sim.total_allocated_power_w, 230.0)
        self.assertEqual(self.sim.effective_power_draw_w, 230.0)

        # Beta can now be re-added up to remaining budget (170W)
        self.sim.set_workload_power("wl-beta", 170.0)
        self.assertEqual(self.sim.effective_power_draw_w, 400.0)

    # -----------------------------------------------------------------------
    # 4. Demo Time Scaling Tests
    # -----------------------------------------------------------------------

    def test_demo_time_scale_acceleration(self) -> None:
        """Verify that a 2.5x time scale accelerates simulated physics by 2.5x per step."""
        sim_1x = TelemetrySimulator(self.config.model_copy(update={"time_scale": 1.0}))
        sim_25x = TelemetrySimulator(self.config.model_copy(update={"time_scale": 2.5}))

        # Run both for 10 ticks
        for _ in range(10):
            sim_1x.step()
            sim_25x.step()

        # 10 ticks @ 1.0x = 10.0 simulated seconds
        self.assertEqual(sim_1x.sim_time_s, 10.0)
        self.assertEqual(sim_1x.tick_count, 10)

        # 10 ticks @ 2.5x = 25.0 simulated seconds
        self.assertEqual(sim_25x.sim_time_s, 25.0)
        self.assertEqual(sim_25x.tick_count, 10)

        # Since starting at 71.2°C and rising toward 75.2°C, 2.5x should have progressed further
        self.assertGreater(sim_25x.core_temp_c, sim_1x.core_temp_c)

    def test_simulation_decoupled_from_wall_clock(self) -> None:
        """Verify simulation advances strictly per step call, with zero wall-clock dependency."""
        import time

        t_start = time.perf_counter()
        for _ in range(50):
            self.sim.step()
        elapsed_wall = time.perf_counter() - t_start

        # 50 steps should execute in a few milliseconds (well under 0.5s wall-clock)
        self.assertLess(elapsed_wall, 0.5)
        self.assertEqual(self.sim.sim_time_s, 50.0)

    # -----------------------------------------------------------------------
    # 5. Deterministic Repeatability Tests
    # -----------------------------------------------------------------------

    def test_deterministic_repeatability_across_instances(self) -> None:
        """Verify two independent simulator instances produce identical telemetry sequences."""
        sim_a = TelemetrySimulator(self.config)
        sim_b = TelemetrySimulator(self.config)

        # Trigger anomaly on both at tick 5
        for tick in range(20):
            if tick == 5:
                sim_a.trigger_anomaly()
                sim_b.trigger_anomaly()

            telem_a = sim_a.step()
            telem_b = sim_b.step()

            self.assertEqual(telem_a.temperature_core_c, telem_b.temperature_core_c)
            self.assertEqual(telem_a.temperature_junction_c, telem_b.temperature_junction_c)
            self.assertEqual(telem_a.power_draw_w, telem_b.power_draw_w)
            self.assertEqual(telem_a.ambient_temp_c, telem_b.ambient_temp_c)

    def test_reset_reproducibility(self) -> None:
        """Verify reset() restores simulator to produce the exact initial sequence."""
        first_run_temps = []
        for _ in range(10):
            telem = self.sim.step()
            first_run_temps.append(telem.temperature_core_c)

        self.sim.reset()
        self.assertEqual(self.sim.sim_time_s, 0.0)
        self.assertEqual(self.sim.tick_count, 0)

        second_run_temps = []
        for _ in range(10):
            telem = self.sim.step()
            second_run_temps.append(telem.temperature_core_c)

        self.assertEqual(first_run_temps, second_run_temps)

    # -----------------------------------------------------------------------
    # 6. Workload Configuration & Validation Tests
    # -----------------------------------------------------------------------

    def test_initial_workload_exceeding_power_budget_rejected(self) -> None:
        """Verify SimulatorConfig and TelemetrySimulator reject initial allocations > power_budget_w."""
        # Config validation rejection
        with self.assertRaises(ValueError) as ctx:
            SimulatorConfig(
                power_budget_w=300.0,
                default_workloads={"wl-alpha": 200.0, "wl-beta": 150.0},
            )
        self.assertIn("exceed", str(ctx.exception).lower())

        # Direct instantiation validation rejection
        invalid_config = self.config.model_copy()
        object.__setattr__(invalid_config, "power_budget_w", 200.0)
        with self.assertRaises(ValueError) as ctx:
            TelemetrySimulator(invalid_config)
        self.assertIn("exceed", str(ctx.exception).lower())

    def test_initial_workload_negative_power_rejected(self) -> None:
        """Verify negative workload power allocations are rejected."""
        with self.assertRaises(ValueError) as ctx:
            SimulatorConfig(default_workloads={"wl-alpha": -20.0})
        self.assertIn("non-negative", str(ctx.exception).lower())

    def test_set_workload_power_negative_rejected(self) -> None:
        """Verify set_workload_power rejects negative power values."""
        with self.assertRaises(ValueError) as ctx:
            self.sim.set_workload_power("wl-alpha", -15.0)
        self.assertIn("non-negative", str(ctx.exception).lower())

    def test_workload_queries_and_removal_missing_id(self) -> None:
        """Verify querying or removing a non-existent workload behaves safely."""
        self.assertEqual(self.sim.get_workload_power("non-existent-workload"), 0.0)
        # Should complete without raising KeyError or exception
        self.sim.remove_workload("non-existent-workload")

    # -----------------------------------------------------------------------
    # 7. Numerical Integration Stability & Timestep Validation Tests
    # -----------------------------------------------------------------------

    def test_oversized_step_dt_destabilizes_raises_error(self) -> None:
        """Verify step() rejects oversized timesteps that destabilize forward Euler (effective_dt >= 2*tau)."""
        # tau = 15.0s, time_scale = 1.0 -> max stable dt < 30.0s
        with self.assertRaises(ValueError) as ctx:
            self.sim.step(dt=30.0)
        self.assertIn("destabilize", str(ctx.exception).lower())

        with self.assertRaises(ValueError) as ctx:
            self.sim.step(dt=50.0)
        self.assertIn("destabilize", str(ctx.exception).lower())

    def test_oversized_config_timestep_rejected(self) -> None:
        """Verify SimulatorConfig rejects base_dt * time_scale >= 2*tau."""
        with self.assertRaises(ValueError) as ctx:
            SimulatorConfig(
                base_dt=16.0,
                time_scale=2.0,
                thermal_time_constant=15.0,
            )
        self.assertIn("destabilize", str(ctx.exception).lower())

    def test_step_dt_zero_or_negative_rejected(self) -> None:
        """Verify step() rejects zero or negative timestep durations."""
        with self.assertRaises(ValueError) as ctx:
            self.sim.step(dt=0.0)
        self.assertIn("positive", str(ctx.exception).lower())

        with self.assertRaises(ValueError) as ctx:
            self.sim.step(dt=-2.0)
        self.assertIn("positive", str(ctx.exception).lower())

    # -----------------------------------------------------------------------
    # 8. Scheduled Anomaly Boundary Validation & Edge Cases
    # -----------------------------------------------------------------------

    def test_schedule_anomaly_negative_start_tick_rejected(self) -> None:
        """Verify schedule_anomaly rejects negative start_tick."""
        with self.assertRaises(ValueError) as ctx:
            self.sim.schedule_anomaly(start_tick=-1, end_tick=5)
        self.assertIn("non-negative", str(ctx.exception).lower())

    def test_schedule_anomaly_end_tick_not_greater_than_start_tick_rejected(self) -> None:
        """Verify schedule_anomaly rejects end_tick <= start_tick."""
        with self.assertRaises(ValueError) as ctx:
            self.sim.schedule_anomaly(start_tick=5, end_tick=5)
        self.assertIn("strictly greater", str(ctx.exception).lower())

        with self.assertRaises(ValueError) as ctx:
            self.sim.schedule_anomaly(start_tick=5, end_tick=3)
        self.assertIn("strictly greater", str(ctx.exception).lower())

    def test_schedule_anomaly_non_integer_type_rejected(self) -> None:
        """Verify schedule_anomaly rejects non-integer tick arguments."""
        with self.assertRaises(TypeError):
            self.sim.schedule_anomaly(start_tick=2.5, end_tick=5)  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            self.sim.schedule_anomaly(start_tick=2, end_tick=5.5)  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            self.sim.schedule_anomaly(start_tick=True, end_tick=5)  # type: ignore[arg-type]

    def test_scheduled_anomaly_open_ended(self) -> None:
        """Verify scheduled anomaly with end_tick=None remains active indefinitely after start_tick."""
        self.sim.schedule_anomaly(start_tick=2, end_tick=None)

        # Tick 0, 1: inactive
        self.sim.step()  # tick 0 -> 1
        self.assertFalse(self.sim.is_anomaly_active)
        self.sim.step()  # tick 1 -> 2
        self.assertTrue(self.sim.is_anomaly_active)

        # Remains active for subsequent ticks
        for _ in range(5):
            self.sim.step()
            self.assertTrue(self.sim.is_anomaly_active)

    def test_trigger_anomaly_with_custom_ambient(self) -> None:
        """Verify triggering an anomaly with an explicit ambient temperature override."""
        self.sim.trigger_anomaly(target_ambient_c=45.0)
        self.assertTrue(self.sim.is_anomaly_active)
        self.assertEqual(self.sim.ambient_temp_c, 45.0)

        telem = self.sim.step()
        self.assertEqual(telem.ambient_temp_c, 45.0)

    # -----------------------------------------------------------------------
    # 9. Steady-State Initialization & Physical Bounds
    # -----------------------------------------------------------------------

    def test_steady_state_initial_temp_when_none(self) -> None:
        """Verify that omitting initial_temp_c initializes temperature to theoretical steady state."""
        config = SimulatorConfig(
            power_budget_w=400.0,
            baseline_ambient_c=25.0,
            thermal_resistance=0.1,
            initial_temp_c=None,
            default_workloads={"wl-alpha": 200.0},
        )
        sim = TelemetrySimulator(config)
        # Expected T_ss = 25.0 + (200.0 * 0.1) = 45.0°C
        self.assertAlmostEqual(sim.core_temp_c, 45.0, places=2)

    def test_fan_speed_and_clock_bounds(self) -> None:
        """Verify fan speed clamps to 50% at low temp and 100% at high temp."""
        cold_sim = TelemetrySimulator(self.config.model_copy(update={"initial_temp_c": 50.0}))
        cold_telem = cold_sim.current_telemetry()
        self.assertEqual(cold_telem.fan_speed_pct, 50.0)

        hot_sim = TelemetrySimulator(self.config.model_copy(update={"initial_temp_c": 92.0}))
        hot_telem = hot_sim.current_telemetry()
        self.assertEqual(hot_telem.fan_speed_pct, 100.0)


if __name__ == "__main__":
    unittest.main()
