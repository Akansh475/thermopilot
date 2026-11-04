"""Unit tests for the Thermal Forecast Agent.

Verifies:
1. Insufficient telemetry sample handling (< min_samples)
2. Rising temperature detection and early incident trigger dispatch (before 85°C)
3. Stable temperature physics and nominal risk assessment
4. Falling temperature / cooling trajectory handling
5. Critical threshold crossings and breach state (>= 85°C)
6. Deterministic repeatability across instances and invocations
7. Rolling window eviction and reset behavior
8. Invalid inputs and timestamp edge cases
9. End-to-end integration with TelemetrySimulator
"""

import unittest
from datetime import datetime, timedelta, timezone

from thermopilot.agents.forecast import (
    ForecastAgent,
    ForecastConfig,
    ThermalRiskStatus,
    ThermalTrend,
)
from thermopilot.schemas import GPUTelemetry
from thermopilot.simulator.telemetry import SimulatorConfig, TelemetrySimulator


def make_telemetry(
    timestamp: datetime,
    core_temp_c: float,
    gpu_id: int = 0,
    power_draw_w: float = 380.0,
    ambient_temp_c: float = 22.0,
) -> GPUTelemetry:
    """Helper to construct valid GPUTelemetry records for tests."""
    return GPUTelemetry(
        timestamp=timestamp,
        gpu_id=gpu_id,
        temperature_core_c=core_temp_c,
        temperature_junction_c=core_temp_c + 8.0,
        power_draw_w=power_draw_w,
        power_limit_w=400.0,
        fan_speed_pct=70.0,
        clock_freq_mhz=1800,
        utilization_pct=95.0,
        ambient_temp_c=ambient_temp_c,
    )


class TestForecastAgent(unittest.TestCase):
    """Test suite for ForecastAgent perception and risk forecasting."""

    def setUp(self) -> None:
        self.config = ForecastConfig(
            critical_threshold_c=85.0,
            warning_threshold_c=81.5,
            incident_time_to_throttle_s=30.0,
            window_size=10,
            min_samples=3,
            trend_rate_threshold_c_per_s=0.02,
        )
        self.agent = ForecastAgent(self.config)
        self.base_time = datetime(2026, 10, 27, 10, 0, 0, tzinfo=timezone.utc)

    # -----------------------------------------------------------------------
    # 1. Insufficient Samples Tests
    # -----------------------------------------------------------------------

    def test_insufficient_samples_empty_history(self) -> None:
        """Verify empty history safely returns INSUFFICIENT_DATA without errors."""
        result = self.agent.forecast([])
        self.assertEqual(result.risk_status, ThermalRiskStatus.INSUFFICIENT_DATA)
        self.assertIsNone(result.time_to_threshold_s)
        self.assertIsNone(result.rate_c_per_s)
        self.assertIsNone(result.acceleration_c_per_s2)
        self.assertFalse(result.trigger_incident)
        self.assertEqual(result.sample_count, 0)
        self.assertIn("Insufficient", result.reason)

    def test_insufficient_samples_below_min_samples(self) -> None:
        """Verify fewer samples than min_samples returns INSUFFICIENT_DATA."""
        samples = [
            make_telemetry(self.base_time, 72.0),
            make_telemetry(self.base_time + timedelta(seconds=1), 73.0),
        ]
        result = self.agent.forecast(samples)
        self.assertEqual(result.risk_status, ThermalRiskStatus.INSUFFICIENT_DATA)
        self.assertEqual(result.sample_count, 2)
        self.assertFalse(result.trigger_incident)

        # Ingesting third sample meets min_samples=3 requirement
        sample_3 = make_telemetry(self.base_time + timedelta(seconds=2), 74.0)
        res_3 = self.agent.forecast(samples + [sample_3])
        self.assertNotEqual(res_3.risk_status, ThermalRiskStatus.INSUFFICIENT_DATA)
        self.assertEqual(res_3.sample_count, 3)

    # -----------------------------------------------------------------------
    # 2. Rising Temperature & Incident Trigger Tests
    # -----------------------------------------------------------------------

    def test_rising_temperature_and_early_incident_trigger(self) -> None:
        """Verify agent detects rising temperature and dispatches incident trigger before 85°C."""
        # Simulated climb: 80.0°C to 81.5°C over 10 step intervals (rate = 0.15°C/s)
        # 11 samples: i=0 (80.0°C) to i=10 (81.5°C)
        # Projected time to 85.0°C: (85.0 - 81.5) / 0.15 = 23.33s (<= 30s trigger window)
        samples = []
        for i in range(11):
            t = self.base_time + timedelta(seconds=i)
            temp = 80.0 + (i * 0.15)
            samples.append(make_telemetry(t, round(temp, 2)))

        result = self.agent.forecast(samples)

        self.assertEqual(result.trend, ThermalTrend.RISING)
        self.assertIsNotNone(result.rate_c_per_s)
        self.assertGreater(result.rate_c_per_s, 0.14)
        self.assertIsNotNone(result.time_to_threshold_s)
        self.assertAlmostEqual(result.time_to_threshold_s, 23.33, delta=0.5)

        # Reached warning temp (81.5°C) with TTT <= 30s -> CRITICAL_RISK incident trigger
        self.assertEqual(result.risk_status, ThermalRiskStatus.CRITICAL_RISK)
        self.assertTrue(result.trigger_incident)
        self.assertIn("Critical thermal runaway", result.reason)
        # Crucial requirement: triggered while still below 85°C
        self.assertLess(result.current_temp_c, 85.0)

    def test_moderate_rising_temperature_warning_only(self) -> None:
        """Verify slow temperature rise with TTT > 30s triggers WARNING, not immediate incident."""
        # Slow climb: 75.0°C rising at 0.05°C/s -> TTT = (85 - 75.25) / 0.05 = ~195s
        samples = [
            make_telemetry(self.base_time + timedelta(seconds=i), 75.0 + (i * 0.05))
            for i in range(6)
        ]
        result = self.agent.forecast(samples)
        self.assertEqual(result.trend, ThermalTrend.RISING)
        self.assertFalse(result.trigger_incident)
        self.assertIn(result.risk_status, (ThermalRiskStatus.NORMAL, ThermalRiskStatus.WARNING))

    # -----------------------------------------------------------------------
    # 3. Stable Temperature Tests
    # -----------------------------------------------------------------------

    def test_stable_temperature_nominal_assessment(self) -> None:
        """Verify stable temperature below warning threshold is evaluated as NORMAL."""
        samples = [
            make_telemetry(self.base_time + timedelta(seconds=i), 71.2)
            for i in range(8)
        ]
        result = self.agent.forecast(samples)

        self.assertEqual(result.trend, ThermalTrend.STABLE)
        self.assertAlmostEqual(result.rate_c_per_s, 0.0, places=3)
        self.assertAlmostEqual(result.acceleration_c_per_s2, 0.0, places=3)
        self.assertIsNone(result.time_to_threshold_s)
        self.assertEqual(result.risk_status, ThermalRiskStatus.NORMAL)
        self.assertFalse(result.trigger_incident)
        self.assertIn("nominal", result.reason.lower())

    # -----------------------------------------------------------------------
    # 4. Falling Temperature Tests
    # -----------------------------------------------------------------------

    def test_falling_temperature_cooling_trajectory(self) -> None:
        """Verify cooling physics results in FALLING trend with no threshold projection."""
        # Cooldown: 84.0°C down to 81.0°C at -0.3°C/s
        samples = [
            make_telemetry(self.base_time + timedelta(seconds=i), 84.0 - (i * 0.3))
            for i in range(10)
        ]
        result = self.agent.forecast(samples)

        self.assertEqual(result.trend, ThermalTrend.FALLING)
        self.assertLess(result.rate_c_per_s, -0.25)
        self.assertIsNone(result.time_to_threshold_s)
        self.assertFalse(result.trigger_incident)
        self.assertIn("cooling", result.reason.lower())

    # -----------------------------------------------------------------------
    # 5. Threshold Crossings & Breach Tests
    # -----------------------------------------------------------------------

    def test_critical_threshold_breached_at_or_above_85c(self) -> None:
        """Verify core temperature >= 85.0°C immediately triggers BREACHED state."""
        samples = [
            make_telemetry(self.base_time, 84.6),
            make_telemetry(self.base_time + timedelta(seconds=1), 84.9),
            make_telemetry(self.base_time + timedelta(seconds=2), 85.2),
        ]
        result = self.agent.forecast(samples)

        self.assertEqual(result.risk_status, ThermalRiskStatus.BREACHED)
        self.assertEqual(result.time_to_threshold_s, 0.0)
        self.assertTrue(result.trigger_incident)
        self.assertIn("breached", result.reason.lower())

    # -----------------------------------------------------------------------
    # 6. Deterministic Repeatability Tests
    # -----------------------------------------------------------------------

    def test_deterministic_forecast_reproducibility(self) -> None:
        """Verify identical telemetry sequences produce bit-exact ForecastResults."""
        samples = [
            make_telemetry(self.base_time + timedelta(seconds=i), 78.0 + (i * 0.2))
            for i in range(7)
        ]
        agent_a = ForecastAgent(self.config)
        agent_b = ForecastAgent(self.config)

        res_a1 = agent_a.forecast(samples)
        res_a2 = agent_a.forecast(samples)
        res_b1 = agent_b.forecast(samples)

        self.assertEqual(res_a1.model_dump(), res_a2.model_dump())
        self.assertEqual(res_a1.model_dump(), res_b1.model_dump())

    # -----------------------------------------------------------------------
    # 7. Rolling Buffer & Reset Tests
    # -----------------------------------------------------------------------

    def test_streaming_observe_and_rolling_window_eviction(self) -> None:
        """Verify observe() maintains rolling window size and reset() clears state."""
        # Feed 15 samples into an agent configured with window_size=10
        for i in range(15):
            t = self.base_time + timedelta(seconds=i)
            telem = make_telemetry(t, 70.0 + (i * 0.1))
            res = self.agent.observe(telem)

        self.assertEqual(self.agent.sample_count, 10)
        self.assertEqual(res.sample_count, 10)
        # Oldest sample in buffer should be i = 5 (15 - 10)
        self.assertEqual(
            self.agent.history[0].timestamp,
            self.base_time + timedelta(seconds=5),
        )

        # Verify reset()
        self.agent.reset()
        self.assertEqual(self.agent.sample_count, 0)
        self.assertEqual(len(self.agent.history), 0)

    # -----------------------------------------------------------------------
    # 8. Invalid Inputs & Timestamp Safety Tests
    # -----------------------------------------------------------------------

    def test_invalid_input_types_rejected(self) -> None:
        """Verify invalid arguments raise TypeError or ValueError."""
        with self.assertRaises(ValueError):
            self.agent.forecast(None)  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            self.agent.forecast("invalid_sequence")  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            self.agent.forecast([1, 2, 3])  # type: ignore[list-item]

        with self.assertRaises(TypeError):
            self.agent.observe("not_telemetry")  # type: ignore[arg-type]

    def test_out_of_order_timestamps_rejected(self) -> None:
        """Verify timestamps that regress in time raise ValueError."""
        t0 = self.base_time
        t1 = self.base_time - timedelta(seconds=1)  # regressed in time
        samples = [
            make_telemetry(t0, 75.0),
            make_telemetry(t1, 76.0),
        ]
        with self.assertRaises(ValueError) as ctx:
            self.agent.forecast(samples)
        self.assertIn("monotonically non-decreasing", str(ctx.exception))

        # Test regression in observe()
        self.agent.reset()
        self.agent.observe(make_telemetry(t0, 75.0))
        with self.assertRaises(ValueError) as ctx:
            self.agent.observe(make_telemetry(t1, 76.0))
        self.assertIn("cannot regress", str(ctx.exception))

    def test_identical_timestamps_handled_safely(self) -> None:
        """Verify identical timestamps (delta_t = 0) do not cause ZeroDivisionError."""
        samples = [
            make_telemetry(self.base_time, 75.0),
            make_telemetry(self.base_time, 76.0),
            make_telemetry(self.base_time, 77.0),
        ]
        result = self.agent.forecast(samples)
        self.assertEqual(result.rate_c_per_s, 0.0)
        self.assertEqual(result.acceleration_c_per_s2, 0.0)
        self.assertEqual(result.trend, ThermalTrend.STABLE)

    def test_config_threshold_validation(self) -> None:
        """Verify ForecastConfig rejects invalid threshold relationships."""
        with self.assertRaises(ValueError):
            # warning must be < critical
            ForecastConfig(warning_threshold_c=86.0, critical_threshold_c=85.0)

        with self.assertRaises(ValueError):
            # min_samples cannot exceed window_size
            ForecastConfig(min_samples=15, window_size=10)

    # -----------------------------------------------------------------------
    # 9. Simulator Integration Test
    # -----------------------------------------------------------------------

    def test_integration_with_simulator_thermal_anomaly(self) -> None:
        """Verify ForecastAgent intercepts intake heating anomaly and triggers incident before 85°C."""
        sim = TelemetrySimulator(
            SimulatorConfig(
                power_budget_w=400.0,
                baseline_ambient_c=22.0,
                anomaly_ambient_c=32.0,
                thermal_resistance=0.14,
                thermal_time_constant=15.0,
                time_scale=1.0,
                base_dt=1.0,
                initial_temp_c=74.0,
                noise_std=0.0,
                default_workloads={"wl-alpha": 230.0, "wl-beta": 150.0},
            )
        )

        # 1. Run 5 ticks under normal baseline
        for tick in range(5):
            telem = sim.step()
            res = self.agent.observe(telem)
            self.assertFalse(res.trigger_incident)
            if tick < 2:
                self.assertEqual(res.risk_status, ThermalRiskStatus.INSUFFICIENT_DATA)
            else:
                self.assertEqual(res.risk_status, ThermalRiskStatus.NORMAL)

        # 2. Trigger intake heating anomaly
        sim.trigger_anomaly()

        # 3. Step forward and observe predictive interception
        incident_triggered = False
        incident_temp = 0.0
        incident_ttt = 0.0

        for _ in range(40):
            telem = sim.step()
            res = self.agent.observe(telem)
            if res.trigger_incident:
                incident_triggered = True
                incident_temp = res.current_temp_c
                incident_ttt = res.time_to_threshold_s
                break

        # Must have triggered before 85°C hardware trip
        self.assertTrue(incident_triggered, "ForecastAgent failed to trigger incident")
        self.assertGreaterEqual(incident_temp, 81.5)
        self.assertLess(incident_temp, 85.0, f"Triggered too late: temp was {incident_temp}°C")
        self.assertGreater(incident_ttt, 0.0)
        self.assertLessEqual(incident_ttt, 30.0)


if __name__ == "__main__":
    unittest.main()
