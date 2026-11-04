"""Unit tests for the ThermoPilot Recovery Planner subsystem.

Verifies:
1. DeterministicFallbackPlanner generates three distinct feasible strategies.
2. Candidate origin tracking (PlanOrigin.LLM vs. PlanOrigin.FALLBACK).
3. Llama unconfigured or unavailable returns controlled failures without fabricating data.
4. Malformed or schema-invalid LLM output is safely rejected.
5. Strict separation of candidate generation from Safety Gate evaluation.
6. Safety Gate rejection of unsafe proposals (Plan D or thermal violations).
7. Preservation of P0 critical workloads and minimum power floors.
8. Simulation-only execution (no side effects on simulator or workloads).
"""

import json
import unittest
from datetime import datetime, timezone

from thermopilot.agents.forecast import ForecastResult, ThermalRiskStatus, ThermalTrend
from thermopilot.planner.fallback import DeterministicFallbackPlanner
from thermopilot.planner.interface import (
    CallableLlamaClient,
    LlamaClientConfig,
    LlamaConfigurationError,
    LlamaUnavailableError,
    MalformedPlanError,
)
from thermopilot.planner.llama import LlamaRecoveryPlanner
from thermopilot.planner.planner import RecoveryPlanner
from thermopilot.safety.gate import SafetyGate
from thermopilot.schemas import (
    ActionType,
    GPUTelemetry,
    NodeConstraints,
    PlanOrigin,
    StrategyType,
    WorkloadPriority,
    WorkloadSpec,
)
from thermopilot.simulator.telemetry import SimulatorConfig, TelemetrySimulator


def make_test_telemetry(core_temp_c: float = 81.5, power_draw_w: float = 380.0) -> GPUTelemetry:
    """Helper to create test telemetry snapshot."""
    return GPUTelemetry(
        timestamp=datetime(2026, 10, 27, 10, 0, 0, tzinfo=timezone.utc),
        gpu_id=0,
        temperature_core_c=core_temp_c,
        temperature_junction_c=core_temp_c + 8.0,
        power_draw_w=power_draw_w,
        power_limit_w=400.0,
        fan_speed_pct=92.0,
        clock_freq_mhz=1845,
        utilization_pct=98.0,
        ambient_temp_c=32.0,
    )


def make_test_forecast(core_temp_c: float = 81.5, t_ttt: float = 23.3) -> ForecastResult:
    """Helper to create forecast perception output."""
    return ForecastResult(
        timestamp=datetime(2026, 10, 27, 10, 0, 0, tzinfo=timezone.utc),
        gpu_id=0,
        current_temp_c=core_temp_c,
        critical_threshold_c=85.0,
        trend=ThermalTrend.RISING,
        rate_c_per_s=0.15,
        acceleration_c_per_s2=0.002,
        time_to_threshold_s=t_ttt,
        risk_status=ThermalRiskStatus.CRITICAL_RISK,
        trigger_incident=True,
        reason="Critical thermal runaway imminent",
        sample_count=10,
    )


def make_test_workloads() -> dict[str, WorkloadSpec]:
    """Helper returning standard Alpha (P0) and Beta (P2) workloads."""
    alpha = WorkloadSpec(
        workload_id="wl-alpha-inference",
        name="Financial Trading Batch",
        owner="Trading-Prod",
        priority=WorkloadPriority.P0_CRITICAL,
        current_power_w=250.0,
        min_power_w=200.0,
        max_power_w=250.0,
        can_preempt=False,
        can_power_cap=True,
        sla_deadline_s=90.0,
        sla_penalty_usd_per_s=5.50,
    )
    beta = WorkloadSpec(
        workload_id="wl-beta-training",
        name="Diffusion Fine-Tuning",
        owner="Research-Exploratory",
        priority=WorkloadPriority.P2_FLEXIBLE,
        current_power_w=150.0,
        min_power_w=150.0,
        max_power_w=200.0,
        can_preempt=True,
        can_power_cap=True,
        sla_deadline_s=3600.0,
        sla_penalty_usd_per_s=0.01,
    )
    return {"wl-alpha-inference": alpha, "wl-beta-training": beta}


def make_valid_llm_json_response() -> str:
    """Return JSON string simulating valid Llama response with Plans A, B, C."""
    plans = [
        {
            "plan_id": "PLAN-A",
            "name": "Conservative Workload Shed",
            "strategy_type": "CONSERVATIVE_SHED",
            "is_recommended": False,
            "projected_temp_max_c": 67.5,
            "cooling_time_to_safe_s": 14.0,
            "workload_actions": [
                {"workload_id": "wl-alpha-inference", "action": "POWER_CAP", "target_power_w": 220.0},
                {"workload_id": "wl-beta-training", "action": "PAUSE"},
            ],
            "estimated_cost_usd": 45.0,
            "sla_violations_expected": 0,
            "rationale": "Checkpoint-pause Beta; allocate safe power to Alpha.",
        },
        {
            "plan_id": "PLAN-B",
            "name": "Dynamic Workload Power-Cap Balancing",
            "strategy_type": "BALANCED_POWER_CAP",
            "is_recommended": True,
            "projected_temp_max_c": 74.0,
            "cooling_time_to_safe_s": 17.5,
            "workload_actions": [
                {"workload_id": "wl-alpha-inference", "action": "POWER_CAP", "target_power_w": 220.0},
                {"workload_id": "wl-beta-training", "action": "POWER_CAP", "target_power_w": 160.0},
            ],
            "estimated_cost_usd": 12.0,
            "sla_violations_expected": 0,
            "rationale": "Dynamic power caps balance thermal decay and SLA adherence.",
        },
        {
            "plan_id": "PLAN-C",
            "name": "Workload Migration / Offload",
            "strategy_type": "MIGRATION_OFFLOAD",
            "is_recommended": False,
            "projected_temp_max_c": 69.5,
            "cooling_time_to_safe_s": 12.0,
            "workload_actions": [
                {"workload_id": "wl-alpha-inference", "action": "POWER_CAP", "target_power_w": 250.0},
                {"workload_id": "wl-beta-training", "action": "MIGRATE"},
            ],
            "estimated_cost_usd": 28.0,
            "sla_violations_expected": 0,
            "rationale": "Migrate Beta to secondary capacity; Alpha retains 250W.",
        },
    ]
    return f"```json\n{json.dumps(plans, indent=2)}\n```"


class TestDeterministicFallbackPlanner(unittest.TestCase):
    """Test suite for the deterministic fallback plan generator."""

    def setUp(self) -> None:
        self.generator = DeterministicFallbackPlanner()
        self.telemetry = make_test_telemetry()
        self.workloads = make_test_workloads()
        self.forecast = make_test_forecast()

    def test_fallback_generates_three_distinct_strategies(self) -> None:
        """Verify fallback produces 3 distinct candidates: Plan A, Plan B, Plan C."""
        candidates = self.generator.generate_candidates(
            telemetry=self.telemetry,
            workloads=self.workloads,
            forecast=self.forecast,
        )
        self.assertEqual(len(candidates), 3)

        strategies = [c.strategy_type for c in candidates]
        self.assertIn(StrategyType.CONSERVATIVE_SHED, strategies)
        self.assertIn(StrategyType.BALANCED_POWER_CAP, strategies)
        self.assertIn(StrategyType.MIGRATION_OFFLOAD, strategies)

        # Plan IDs
        plan_ids = [c.plan_id for c in candidates]
        self.assertEqual(plan_ids, ["PLAN-A", "PLAN-B", "PLAN-C"])

    def test_fallback_candidate_origin_is_fallback(self) -> None:
        """Verify all fallback candidates are explicitly labeled with PlanOrigin.FALLBACK."""
        candidates = self.generator.generate_candidates(
            telemetry=self.telemetry,
            workloads=self.workloads,
        )
        for c in candidates:
            self.assertEqual(c.origin, PlanOrigin.FALLBACK)

    def test_fallback_recommended_plan_is_plan_b(self) -> None:
        """Verify Plan B is flagged as recommended."""
        candidates = self.generator.generate_candidates(
            telemetry=self.telemetry,
            workloads=self.workloads,
        )
        recommended = [c for c in candidates if c.is_recommended]
        self.assertEqual(len(recommended), 1)
        self.assertEqual(recommended[0].plan_id, "PLAN-B")

    def test_fallback_preserves_p0_and_power_floors(self) -> None:
        """Verify fallback generator strictly protects P0 and respects power floors."""
        candidates = self.generator.generate_candidates(
            telemetry=self.telemetry,
            workloads=self.workloads,
        )
        for plan in candidates:
            self.assertEqual(plan.sla_violations_expected, 0)
            self.assertLessEqual(plan.projected_temp_max_c, 80.0)
            for act in plan.workload_actions:
                if act.workload_id == "wl-alpha-inference":
                    # Alpha is never paused or migrated
                    self.assertNotEqual(act.action, ActionType.PAUSE)
                    self.assertNotEqual(act.action, ActionType.MIGRATE)
                    self.assertGreaterEqual(act.target_power_w, 200.0)
                elif act.workload_id == "wl-beta-training" and act.action == ActionType.POWER_CAP:
                    self.assertGreaterEqual(act.target_power_w, 150.0)

    def test_fallback_deterministic_repeatability(self) -> None:
        """Verify identical inputs yield bit-exact identical candidates."""
        res1 = self.generator.generate_candidates(self.telemetry, self.workloads)
        res2 = self.generator.generate_candidates(self.telemetry, self.workloads)
        self.assertEqual([c.model_dump() for c in res1], [c.model_dump() for c in res2])

    def test_fallback_with_unsafe_candidate(self) -> None:
        """Verify include_unsafe_candidate appends Plan D for safety testing."""
        candidates = self.generator.generate_candidates(
            telemetry=self.telemetry,
            workloads=self.workloads,
            include_unsafe_candidate=True,
        )
        self.assertEqual(len(candidates), 4)
        self.assertEqual(candidates[3].plan_id, "PLAN-D")
        self.assertEqual(candidates[3].strategy_type, StrategyType.UNSAFE_THROTTLE)


class TestLlamaRecoveryPlanner(unittest.TestCase):
    """Test suite for the Llama candidate generator and prompt/output validation."""

    def setUp(self) -> None:
        self.telemetry = make_test_telemetry()
        self.workloads = make_test_workloads()
        self.forecast = make_test_forecast()

    def test_llama_unconfigured_raises_configuration_error(self) -> None:
        """Verify unconfigured Llama generator raises LlamaConfigurationError."""
        planner = LlamaRecoveryPlanner(client=None, config=LlamaClientConfig())
        self.assertFalse(planner.is_configured())

        with self.assertRaises(LlamaConfigurationError) as ctx:
            planner.generate_candidates(self.telemetry, self.workloads)
        self.assertIn("not configured", str(ctx.exception).lower())

    def test_llama_client_failure_raises_unavailable_error(self) -> None:
        """Verify client exceptions are caught and wrapped in LlamaUnavailableError."""
        def failing_client(prompt: str, system_prompt: str | None) -> str:
            raise ConnectionError("Connection refused by endpoint http://llama:8000")

        client = CallableLlamaClient(failing_client)
        planner = LlamaRecoveryPlanner(client=client)

        with self.assertRaises(LlamaUnavailableError) as ctx:
            planner.generate_candidates(self.telemetry, self.workloads)
        self.assertIn("connection refused", str(ctx.exception).lower())

    def test_llama_empty_response_raises_malformed_error(self) -> None:
        """Verify empty LLM response raises MalformedPlanError."""
        client = CallableLlamaClient(lambda p, s: "   ")
        planner = LlamaRecoveryPlanner(client=client)

        with self.assertRaises(MalformedPlanError) as ctx:
            planner.generate_candidates(self.telemetry, self.workloads)
        self.assertIn("empty", str(ctx.exception).lower())

    def test_llama_malformed_json_raises_malformed_error(self) -> None:
        """Verify non-JSON response raises MalformedPlanError."""
        client = CallableLlamaClient(lambda p, s: "Here is your plan: [ {plan_id: PLAN-A truncated...")
        planner = LlamaRecoveryPlanner(client=client)

        with self.assertRaises(MalformedPlanError) as ctx:
            planner.generate_candidates(self.telemetry, self.workloads)
        self.assertIn("failed to parse", str(ctx.exception).lower())

    def test_llama_non_array_json_raises_malformed_error(self) -> None:
        """Verify JSON object instead of array raises MalformedPlanError."""
        client = CallableLlamaClient(lambda p, s: json.dumps({"plan_id": "PLAN-A"}))
        planner = LlamaRecoveryPlanner(client=client)

        with self.assertRaises(MalformedPlanError) as ctx:
            planner.generate_candidates(self.telemetry, self.workloads)
        self.assertIn("expected json array", str(ctx.exception).lower())

    def test_llama_schema_violation_raises_malformed_error(self) -> None:
        """Verify missing required fields raise MalformedPlanError."""
        # Missing workload_actions
        bad_plan = [
            {
                "plan_id": "PLAN-BAD",
                "name": "Bad Plan",
                "strategy_type": "BALANCED_POWER_CAP",
                "projected_temp_max_c": 72.0,
                "cooling_time_to_safe_s": 15.0,
                "estimated_cost_usd": 10.0,
                "rationale": "Missing workload actions",
            }
        ]
        client = CallableLlamaClient(lambda p, s: json.dumps(bad_plan))
        planner = LlamaRecoveryPlanner(client=client)

        with self.assertRaises(MalformedPlanError) as ctx:
            planner.generate_candidates(self.telemetry, self.workloads)
        self.assertIn("schema validation failed", str(ctx.exception).lower())

    def test_llama_valid_generation_stamps_origin_llm(self) -> None:
        """Verify valid Llama output is parsed into RecoveryPlanOptions tagged with PlanOrigin.LLM."""
        client = CallableLlamaClient(lambda p, s: make_valid_llm_json_response())
        planner = LlamaRecoveryPlanner(client=client)

        candidates = planner.generate_candidates(
            self.telemetry, self.workloads, forecast=self.forecast
        )
        self.assertEqual(len(candidates), 3)
        for plan in candidates:
            self.assertEqual(plan.origin, PlanOrigin.LLM)
        self.assertEqual(candidates[0].plan_id, "PLAN-A")
        self.assertEqual(candidates[1].plan_id, "PLAN-B")
        self.assertEqual(candidates[2].plan_id, "PLAN-C")


class TestRecoveryPlannerIntegration(unittest.TestCase):
    """End-to-end integration tests for RecoveryPlanner and deterministic SafetyGate."""

    def setUp(self) -> None:
        self.safety_gate = SafetyGate(NodeConstraints(max_temp_c=80.0, power_budget_w=400.0))
        self.telemetry = make_test_telemetry()
        self.workloads = make_test_workloads()
        self.forecast = make_test_forecast()

    def test_planner_with_valid_llama_certified_safe(self) -> None:
        """Verify end-to-end flow with valid Llama output passing SafetyGate."""
        client = CallableLlamaClient(lambda p, s: make_valid_llm_json_response())
        llama_planner = LlamaRecoveryPlanner(client=client)
        planner = RecoveryPlanner(safety_gate=self.safety_gate, llama_planner=llama_planner)

        result = planner.plan(
            telemetry=self.telemetry,
            workloads=self.workloads,
            forecast=self.forecast,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.generation_source, PlanOrigin.LLM)
        self.assertEqual(len(result.candidates), 3)

        # Every candidate evaluated by SafetyGate
        for candidate in result.candidates:
            self.assertEqual(candidate.origin, PlanOrigin.LLM)
            self.assertTrue(candidate.is_safe)
            self.assertTrue(candidate.safety_result.passed)
            self.assertEqual(len(candidate.safety_result.evaluations), 6)

        # Filter safe candidates
        self.assertEqual(len(result.safe_candidates), 3)
        self.assertIsNotNone(result.recommended_candidate)
        self.assertEqual(result.recommended_candidate.plan.plan_id, "PLAN-B")

    def test_planner_unconfigured_llama_fallback_disabled_controlled_failure(self) -> None:
        """Verify unconfigured Llama returns controlled failure when fallback is disabled."""
        unconfigured_llama = LlamaRecoveryPlanner(client=None, config=LlamaClientConfig())
        planner = RecoveryPlanner(safety_gate=self.safety_gate, llama_planner=unconfigured_llama)

        result = planner.plan(
            telemetry=self.telemetry,
            workloads=self.workloads,
            use_fallback_on_failure=False,
        )

        self.assertFalse(result.success)
        self.assertIsNone(result.generation_source)
        self.assertEqual(len(result.candidates), 0)
        self.assertIn("not configured", result.error_message.lower())

    def test_planner_unconfigured_llama_fallback_enabled(self) -> None:
        """Verify unconfigured Llama triggers controlled deterministic fallback with PlanOrigin.FALLBACK."""
        unconfigured_llama = LlamaRecoveryPlanner(client=None, config=LlamaClientConfig())
        planner = RecoveryPlanner(safety_gate=self.safety_gate, llama_planner=unconfigured_llama)

        result = planner.plan(
            telemetry=self.telemetry,
            workloads=self.workloads,
            use_fallback_on_failure=True,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.generation_source, PlanOrigin.FALLBACK)
        self.assertEqual(len(result.candidates), 3)
        for c in result.candidates:
            self.assertEqual(c.origin, PlanOrigin.FALLBACK)
            self.assertTrue(c.is_safe)

    def test_planner_llama_unavailable_fallback_enabled(self) -> None:
        """Verify LLM timeout / network outage falls back to deterministic generator."""
        def timeout_client(prompt: str, system_prompt: str | None) -> str:
            raise TimeoutError("Inference timed out after 2.5s")

        llama_planner = LlamaRecoveryPlanner(client=CallableLlamaClient(timeout_client))
        planner = RecoveryPlanner(safety_gate=self.safety_gate, llama_planner=llama_planner)

        result = planner.plan(
            telemetry=self.telemetry,
            workloads=self.workloads,
            use_fallback_on_failure=True,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.generation_source, PlanOrigin.FALLBACK)
        self.assertIn("timed out", result.error_message.lower())
        self.assertEqual(len(result.candidates), 3)

    def test_planner_llama_malformed_output_fallback_enabled(self) -> None:
        """Verify malformed LLM response is safely rejected and fallback is engaged."""
        bad_json_client = CallableLlamaClient(lambda p, s: "Error 500: Server Overloaded")
        llama_planner = LlamaRecoveryPlanner(client=bad_json_client)
        planner = RecoveryPlanner(safety_gate=self.safety_gate, llama_planner=llama_planner)

        result = planner.plan(
            telemetry=self.telemetry,
            workloads=self.workloads,
            use_fallback_on_failure=True,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.generation_source, PlanOrigin.FALLBACK)
        self.assertIn("failed to parse", result.error_message.lower())
        self.assertEqual(len(result.candidates), 3)

    def test_safety_gate_rejects_unsafe_candidate_from_llama(self) -> None:
        """Verify SafetyGate rejects unsafe proposal from Llama; LLM cannot override gate."""
        # Llama proposes Plans A, B, C AND Plan D (Blind 50% Throttle)
        plans = [
            {
                "plan_id": "PLAN-B",
                "name": "Dynamic Power Cap",
                "strategy_type": "BALANCED_POWER_CAP",
                "is_recommended": True,
                "projected_temp_max_c": 74.0,
                "cooling_time_to_safe_s": 17.5,
                "workload_actions": [
                    {"workload_id": "wl-alpha-inference", "action": "POWER_CAP", "target_power_w": 220.0},
                    {"workload_id": "wl-beta-training", "action": "POWER_CAP", "target_power_w": 160.0},
                ],
                "estimated_cost_usd": 12.0,
                "sla_violations_expected": 0,
                "rationale": "Balanced plan",
            },
            {
                "plan_id": "PLAN-D",
                "name": "Blind 50% Throttle",
                "strategy_type": "UNSAFE_THROTTLE",
                "is_recommended": False,
                "projected_temp_max_c": 65.0,
                "cooling_time_to_safe_s": 8.0,
                "workload_actions": [
                    {"workload_id": "wl-alpha-inference", "action": "POWER_CAP", "target_power_w": 125.0},
                    {"workload_id": "wl-beta-training", "action": "POWER_CAP", "target_power_w": 125.0},
                ],
                "estimated_cost_usd": 320.0,
                "sla_violations_expected": 1,
                "rationale": "Naive equal slash",
            },
        ]
        client = CallableLlamaClient(lambda p, s: json.dumps(plans))
        llama_planner = LlamaRecoveryPlanner(client=client)
        planner = RecoveryPlanner(safety_gate=self.safety_gate, llama_planner=llama_planner)

        result = planner.plan(self.telemetry, self.workloads)
        self.assertTrue(result.success)
        self.assertEqual(len(result.candidates), 2)

        # Plan B passed
        plan_b_cand = next(c for c in result.candidates if c.plan.plan_id == "PLAN-B")
        self.assertTrue(plan_b_cand.is_safe)
        self.assertTrue(plan_b_cand.safety_result.passed)

        # Plan D REJECTED by gate
        plan_d_cand = next(c for c in result.candidates if c.plan.plan_id == "PLAN-D")
        self.assertFalse(plan_d_cand.is_safe)
        self.assertFalse(plan_d_cand.safety_result.passed)
        self.assertGreater(len(plan_d_cand.safety_result.rejection_reasons), 0)

        # Safe candidates only includes Plan B
        self.assertEqual(len(result.safe_candidates), 1)
        self.assertEqual(result.safe_candidates[0].plan.plan_id, "PLAN-B")

    def test_safety_gate_rejects_temperature_ceiling_breach(self) -> None:
        """Verify SafetyGate rejects proposal exceeding 80°C temperature ceiling."""
        hot_plan = [
            {
                "plan_id": "PLAN-HOT",
                "name": "Hot Proposal",
                "strategy_type": "BALANCED_POWER_CAP",
                "is_recommended": False,
                "projected_temp_max_c": 82.5,  # Exceeds 80.0°C!
                "cooling_time_to_safe_s": 30.0,
                "workload_actions": [
                    {"workload_id": "wl-alpha-inference", "action": "POWER_CAP", "target_power_w": 220.0},
                    {"workload_id": "wl-beta-training", "action": "POWER_CAP", "target_power_w": 160.0},
                ],
                "estimated_cost_usd": 12.0,
                "sla_violations_expected": 0,
                "rationale": "Fails temperature ceiling",
            }
        ]
        client = CallableLlamaClient(lambda p, s: json.dumps(hot_plan))
        llama_planner = LlamaRecoveryPlanner(client=client)
        planner = RecoveryPlanner(safety_gate=self.safety_gate, llama_planner=llama_planner)

        result = planner.plan(self.telemetry, self.workloads)
        self.assertTrue(result.success)
        self.assertEqual(len(result.candidates), 1)
        self.assertFalse(result.candidates[0].is_safe)
        self.assertIn("RULE_TEMP_CEILING", [e.rule_name for e in result.candidates[0].safety_result.evaluations if not e.passed])

    def test_simulation_only_integrity_no_state_mutation(self) -> None:
        """Verify planning execution leaves simulator state completely untouched."""
        sim = TelemetrySimulator(
            SimulatorConfig(
                power_budget_w=400.0,
                baseline_ambient_c=22.0,
                default_workloads={"wl-alpha": 230.0, "wl-beta": 150.0},
            )
        )
        initial_temp = sim.core_temp_c
        initial_powers = sim.workload_powers

        # Run planner
        planner = RecoveryPlanner(safety_gate=self.safety_gate)
        planner.plan(self.telemetry, self.workloads)

        # Confirm simulator state has not changed
        self.assertEqual(sim.core_temp_c, initial_temp)
        self.assertEqual(sim.workload_powers, initial_powers)
        self.assertEqual(sim.tick_count, 0)


if __name__ == "__main__":
    unittest.main()
