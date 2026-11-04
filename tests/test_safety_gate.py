"""Unit tests for the Deterministic Safety Gate.

Verifies:
1. Rule 1 (RULE_TEMP_CEILING): Projected GPU temperature <= 80.0°C.
2. Rule 2 (RULE_NODE_POWER_BUDGET): Total node power <= 400.0W.
3. Rule 3 (RULE_NO_P0_PREEMPTION): P0_CRITICAL & non-preemptible jobs never preempted.
4. Rule 4 (RULE_P0_SLA_PRESERVATION): P0 SLA constraints preserved (violations = 0).
5. Rule 5 (RULE_MIN_POWER_FLOOR): Every workload remains at or above its minimum power floor.
6. Rule 6 (RULE_POWER_CAP_VALIDITY): POWER_CAP actions include valid target power and respect limits.
7. Architectural recovery plans: Plans A, B, C pass; Plan D is strictly rejected.
8. Multiple simultaneous violations: All violated rules and reasons are comprehensively captured.
9. Missing or invalid inputs: Gate fails safe (never defaults to approval).
10. Approval timeout governance: Timeout cannot authorize or execute workload mutations.
11. Deterministic reproducibility: Identical inputs produce bit-exact results.
"""

import unittest
from datetime import datetime, timezone

from thermopilot.safety.gate import SafetyGate
from thermopilot.schemas import (
    ActionType,
    ApprovalAction,
    ApprovalDecision,
    GPUTelemetry,
    NodeConstraints,
    RecoveryPlanOption,
    SafetyRuleId,
    StrategyType,
    WorkloadAction,
    WorkloadPriority,
    WorkloadSpec,
)


def make_test_telemetry(
    core_temp_c: float = 81.5,
    power_draw_w: float = 380.0,
    ambient_temp_c: float = 32.0,
) -> GPUTelemetry:
    """Helper to construct standard GPUTelemetry records for tests."""
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
        ambient_temp_c=ambient_temp_c,
    )


def make_standard_workloads() -> dict[str, WorkloadSpec]:
    """Helper returning standard Alpha (P0) and Beta (P2) workload specifications."""
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


class TestSafetyGate(unittest.TestCase):
    """Test suite for the deterministic SafetyGate."""

    def setUp(self) -> None:
        self.node_constraints = NodeConstraints(max_temp_c=80.0, power_budget_w=400.0)
        self.gate = SafetyGate(self.node_constraints)
        self.telemetry = make_test_telemetry()
        self.workloads = make_standard_workloads()

    # -----------------------------------------------------------------------
    # 1. Rule 1: RULE_TEMP_CEILING Tests
    # -----------------------------------------------------------------------

    def test_rule_temp_ceiling_passed_below_and_at_limit(self) -> None:
        """Verify projected temperature <= 80.0°C satisfies RULE_TEMP_CEILING."""
        plan = RecoveryPlanOption(
            plan_id="PLAN-TEMP-SAFE",
            name="Cooling Plan",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=74.0,
            cooling_time_to_safe_s=17.5,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=220.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=160.0),
            ],
            estimated_cost_usd=12.0,
            sla_violations_expected=0,
            rationale="Safe temp",
        )
        result = self.gate.evaluate(plan, self.telemetry, self.workloads)
        self.assertTrue(result.passed)
        eval_temp = next(e for e in result.evaluations if e.rule_name == SafetyRuleId.RULE_TEMP_CEILING.value)
        self.assertTrue(eval_temp.passed)
        self.assertEqual(eval_temp.observed_value, 74.0)
        self.assertEqual(eval_temp.limit_value, 80.0)

        # Boundary test at exactly 80.0°C
        plan_80 = plan.model_copy(update={"projected_temp_max_c": 80.0})
        res_80 = self.gate.evaluate(plan_80, self.telemetry, self.workloads)
        self.assertTrue(res_80.passed)

    def test_rule_temp_ceiling_failed_above_80c(self) -> None:
        """Verify projected temperature > 80.0°C is rejected with explicit reason."""
        plan = RecoveryPlanOption(
            plan_id="PLAN-HOT",
            name="Hot Plan",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=81.2,
            cooling_time_to_safe_s=35.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=220.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=160.0),
            ],
            estimated_cost_usd=12.0,
            sla_violations_expected=0,
            rationale="Too hot",
        )
        result = self.gate.evaluate(plan, self.telemetry, self.workloads)
        self.assertFalse(result.passed)
        eval_temp = next(e for e in result.evaluations if e.rule_name == SafetyRuleId.RULE_TEMP_CEILING.value)
        self.assertFalse(eval_temp.passed)
        self.assertIn("exceeds safety", result.rejection_reasons[0].lower())

    def test_rule_temp_ceiling_custom_node_constraints(self) -> None:
        """Verify safety gate honors custom node constraints (e.g. 75.0°C)."""
        strict_constraints = NodeConstraints(max_temp_c=75.0, power_budget_w=400.0)
        plan = RecoveryPlanOption(
            plan_id="PLAN-CUSTOM-TEMP",
            name="Custom Temp Plan",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=76.0,
            cooling_time_to_safe_s=20.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=220.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=160.0),
            ],
            estimated_cost_usd=12.0,
            sla_violations_expected=0,
            rationale="Exceeds custom 75C limit",
        )
        result = self.gate.evaluate(plan, self.telemetry, self.workloads, node_constraints=strict_constraints)
        self.assertFalse(result.passed)
        eval_temp = next(e for e in result.evaluations if e.rule_name == SafetyRuleId.RULE_TEMP_CEILING.value)
        self.assertFalse(eval_temp.passed)
        self.assertEqual(eval_temp.limit_value, 75.0)

    # -----------------------------------------------------------------------
    # 2. Rule 2: RULE_NODE_POWER_BUDGET Tests
    # -----------------------------------------------------------------------

    def test_rule_node_power_budget_passed_at_or_below_400w(self) -> None:
        """Verify total power <= 400.0W satisfies RULE_NODE_POWER_BUDGET."""
        # 250W + 150W = 400W
        plan = RecoveryPlanOption(
            plan_id="PLAN-POWER-OK",
            name="Power Budget OK",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=75.0,
            cooling_time_to_safe_s=15.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=240.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=160.0),
            ],
            estimated_cost_usd=10.0,
            sla_violations_expected=0,
            rationale="Exact 400W budget",
        )
        result = self.gate.evaluate(plan, self.telemetry, self.workloads)
        self.assertTrue(result.passed)
        eval_power = next(e for e in result.evaluations if e.rule_name == SafetyRuleId.RULE_NODE_POWER_BUDGET.value)
        self.assertTrue(eval_power.passed)
        self.assertEqual(eval_power.observed_value, 400.0)

    def test_rule_node_power_budget_failed_above_400w(self) -> None:
        """Verify total power > 400.0W is rejected with explicit rejection reason."""
        # 250W + 180W = 430W (> 400W budget)
        plan = RecoveryPlanOption(
            plan_id="PLAN-OVERPOWER",
            name="Power Exceeded",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=74.0,
            cooling_time_to_safe_s=15.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=250.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=180.0),
            ],
            estimated_cost_usd=10.0,
            sla_violations_expected=0,
            rationale="Over budget",
        )
        result = self.gate.evaluate(plan, self.telemetry, self.workloads)
        self.assertFalse(result.passed)
        eval_power = next(e for e in result.evaluations if e.rule_name == SafetyRuleId.RULE_NODE_POWER_BUDGET.value)
        self.assertFalse(eval_power.passed)
        self.assertEqual(eval_power.observed_value, 430.0)
        self.assertIn("exceeds budget", "; ".join(result.rejection_reasons).lower())

    def test_rule_node_power_budget_unmodified_workload_counted(self) -> None:
        """Verify workloads not mentioned in plan retain current power and count toward node budget."""
        # Beta is not mentioned; retains current_power_w = 150.0W
        # Alpha wants 260W (which exceeds budget: 260 + 150 = 410W)
        plan = RecoveryPlanOption(
            plan_id="PLAN-PARTIAL",
            name="Partial Action Plan",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=74.0,
            cooling_time_to_safe_s=15.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=260.0),
            ],
            estimated_cost_usd=10.0,
            sla_violations_expected=0,
            rationale="Alpha only",
        )
        result = self.gate.evaluate(plan, self.telemetry, self.workloads)
        self.assertFalse(result.passed)
        eval_power = next(e for e in result.evaluations if e.rule_name == SafetyRuleId.RULE_NODE_POWER_BUDGET.value)
        self.assertFalse(eval_power.passed)
        self.assertEqual(eval_power.observed_value, 410.0)

    # -----------------------------------------------------------------------
    # 3. Rule 3: RULE_NO_P0_PREEMPTION Tests
    # -----------------------------------------------------------------------

    def test_rule_no_p0_preemption_passed_when_p2_paused(self) -> None:
        """Verify pausing P2 workload (can_preempt=True) satisfies RULE_NO_P0_PREEMPTION."""
        plan = RecoveryPlanOption(
            plan_id="PLAN-PAUSE-BETA",
            name="Pause Beta",
            strategy_type=StrategyType.CONSERVATIVE_SHED,
            projected_temp_max_c=71.0,
            cooling_time_to_safe_s=14.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=220.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.PAUSE),
            ],
            estimated_cost_usd=45.0,
            sla_violations_expected=0,
            rationale="Pause flexible Beta",
        )
        result = self.gate.evaluate(plan, self.telemetry, self.workloads)
        self.assertTrue(result.passed)
        eval_preempt = next(e for e in result.evaluations if e.rule_name == SafetyRuleId.RULE_NO_P0_PREEMPTION.value)
        self.assertTrue(eval_preempt.passed)

    def test_rule_no_p0_preemption_failed_when_p0_paused(self) -> None:
        """Verify attempting to PAUSE P0_CRITICAL workload is rejected."""
        plan = RecoveryPlanOption(
            plan_id="PLAN-PAUSE-ALPHA",
            name="Illegal Pause Alpha",
            strategy_type=StrategyType.CONSERVATIVE_SHED,
            projected_temp_max_c=68.0,
            cooling_time_to_safe_s=10.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.PAUSE),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=160.0),
            ],
            estimated_cost_usd=500.0,
            sla_violations_expected=1,
            rationale="Pause Alpha",
        )
        result = self.gate.evaluate(plan, self.telemetry, self.workloads)
        self.assertFalse(result.passed)
        eval_preempt = next(e for e in result.evaluations if e.rule_name == SafetyRuleId.RULE_NO_P0_PREEMPTION.value)
        self.assertFalse(eval_preempt.passed)
        self.assertEqual(eval_preempt.observed_value, "PREEMPTED")
        self.assertIn("cannot be preempted", "; ".join(result.rejection_reasons).lower())

    def test_rule_no_p0_preemption_failed_when_p0_migrated(self) -> None:
        """Verify migrating non-preemptible P0 workload is rejected."""
        plan = RecoveryPlanOption(
            plan_id="PLAN-MIGRATE-ALPHA",
            name="Illegal Migrate Alpha",
            strategy_type=StrategyType.MIGRATION_OFFLOAD,
            projected_temp_max_c=70.0,
            cooling_time_to_safe_s=12.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.MIGRATE),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=160.0),
            ],
            estimated_cost_usd=500.0,
            sla_violations_expected=1,
            rationale="Migrate Alpha",
        )
        result = self.gate.evaluate(plan, self.telemetry, self.workloads)
        self.assertFalse(result.passed)
        eval_preempt = next(e for e in result.evaluations if e.rule_name == SafetyRuleId.RULE_NO_P0_PREEMPTION.value)
        self.assertFalse(eval_preempt.passed)

    # -----------------------------------------------------------------------
    # 4. Rule 4: RULE_P0_SLA_PRESERVATION Tests
    # -----------------------------------------------------------------------

    def test_rule_p0_sla_preservation_passed_zero_violations(self) -> None:
        """Verify 0 expected SLA violations satisfies RULE_P0_SLA_PRESERVATION."""
        plan = RecoveryPlanOption(
            plan_id="PLAN-SLA-SAFE",
            name="Safe SLA Plan",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=74.0,
            cooling_time_to_safe_s=17.5,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=220.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=160.0),
            ],
            estimated_cost_usd=12.0,
            sla_violations_expected=0,
            rationale="Zero SLA violations",
        )
        result = self.gate.evaluate(plan, self.telemetry, self.workloads)
        self.assertTrue(result.passed)
        eval_sla = next(e for e in result.evaluations if e.rule_name == SafetyRuleId.RULE_P0_SLA_PRESERVATION.value)
        self.assertTrue(eval_sla.passed)
        self.assertEqual(eval_sla.observed_value, 0)

    def test_rule_p0_sla_preservation_failed_positive_violations(self) -> None:
        """Verify plan expecting >= 1 SLA violations is rejected."""
        plan = RecoveryPlanOption(
            plan_id="PLAN-SLA-BREACH",
            name="SLA Breach Plan",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=74.0,
            cooling_time_to_safe_s=17.5,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=220.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=160.0),
            ],
            estimated_cost_usd=300.0,
            sla_violations_expected=2,
            rationale="Induces 2 violations",
        )
        result = self.gate.evaluate(plan, self.telemetry, self.workloads)
        self.assertFalse(result.passed)
        eval_sla = next(e for e in result.evaluations if e.rule_name == SafetyRuleId.RULE_P0_SLA_PRESERVATION.value)
        self.assertFalse(eval_sla.passed)
        self.assertEqual(eval_sla.observed_value, 2)
        self.assertIn("sla", "; ".join(result.rejection_reasons).lower())

    def test_rule_p0_sla_preservation_failed_when_p0_underpowered(self) -> None:
        """Verify reducing P0 power below min_power_w breaches SLA contract."""
        # Alpha min_power_w is 200W; capping at 180W violates SLA preservation
        plan = RecoveryPlanOption(
            plan_id="PLAN-UNDERPOWER-ALPHA",
            name="Underpower Alpha",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=72.0,
            cooling_time_to_safe_s=15.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=180.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=160.0),
            ],
            estimated_cost_usd=100.0,
            sla_violations_expected=0,
            rationale="Alpha below min floor",
        )
        result = self.gate.evaluate(plan, self.telemetry, self.workloads)
        self.assertFalse(result.passed)
        eval_sla = next(e for e in result.evaluations if e.rule_name == SafetyRuleId.RULE_P0_SLA_PRESERVATION.value)
        self.assertFalse(eval_sla.passed)

    # -----------------------------------------------------------------------
    # 5. Rule 5: RULE_MIN_POWER_FLOOR Tests
    # -----------------------------------------------------------------------

    def test_rule_min_power_floor_passed(self) -> None:
        """Verify workloads allocated >= min_power_w pass RULE_MIN_POWER_FLOOR."""
        plan = RecoveryPlanOption(
            plan_id="PLAN-FLOOR-OK",
            name="Floor OK",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=74.0,
            cooling_time_to_safe_s=17.5,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=200.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=150.0),
            ],
            estimated_cost_usd=12.0,
            sla_violations_expected=0,
            rationale="At exact power floors",
        )
        result = self.gate.evaluate(plan, self.telemetry, self.workloads)
        self.assertTrue(result.passed)
        eval_floor = next(e for e in result.evaluations if e.rule_name == SafetyRuleId.RULE_MIN_POWER_FLOOR.value)
        self.assertTrue(eval_floor.passed)

    def test_rule_min_power_floor_failed(self) -> None:
        """Verify power cap below min_power_w fails RULE_MIN_POWER_FLOOR."""
        # Alpha floor is 200W, Beta floor is 150W; capping Beta at 140W violates floor
        plan = RecoveryPlanOption(
            plan_id="PLAN-BETA-BELOW-FLOOR",
            name="Beta Below Floor",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=74.0,
            cooling_time_to_safe_s=17.5,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=220.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=140.0),
            ],
            estimated_cost_usd=12.0,
            sla_violations_expected=0,
            rationale="Beta at 140W",
        )
        result = self.gate.evaluate(plan, self.telemetry, self.workloads)
        self.assertFalse(result.passed)
        eval_floor = next(e for e in result.evaluations if e.rule_name == SafetyRuleId.RULE_MIN_POWER_FLOOR.value)
        self.assertFalse(eval_floor.passed)
        self.assertEqual(eval_floor.observed_value, 140.0)
        self.assertEqual(eval_floor.limit_value, 150.0)
        self.assertIn("minimum operational floor", "; ".join(result.rejection_reasons))

    # -----------------------------------------------------------------------
    # 6. Rule 6: RULE_POWER_CAP_VALIDITY Tests
    # -----------------------------------------------------------------------

    def test_rule_power_cap_validity_passed(self) -> None:
        """Verify valid POWER_CAP actions pass RULE_POWER_CAP_VALIDITY."""
        plan = RecoveryPlanOption(
            plan_id="PLAN-CAP-VALID",
            name="Valid Cap Plan",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=74.0,
            cooling_time_to_safe_s=17.5,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=220.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=160.0),
            ],
            estimated_cost_usd=12.0,
            sla_violations_expected=0,
            rationale="Valid caps",
        )
        result = self.gate.evaluate(plan, self.telemetry, self.workloads)
        self.assertTrue(result.passed)
        eval_val = next(e for e in result.evaluations if e.rule_name == SafetyRuleId.RULE_POWER_CAP_VALIDITY.value)
        self.assertTrue(eval_val.passed)

    def test_rule_power_cap_validity_failed_exceeds_workload_max(self) -> None:
        """Verify POWER_CAP exceeding workload max_power_w is rejected."""
        # Alpha max is 250W; capping at 255W violates limits
        plan = RecoveryPlanOption(
            plan_id="PLAN-EXCEED-MAX",
            name="Exceed Max Plan",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=74.0,
            cooling_time_to_safe_s=17.5,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=255.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=140.0),
            ],
            estimated_cost_usd=12.0,
            sla_violations_expected=0,
            rationale="Exceeds max",
        )
        result = self.gate.evaluate(plan, self.telemetry, self.workloads)
        self.assertFalse(result.passed)
        eval_val = next(e for e in result.evaluations if e.rule_name == SafetyRuleId.RULE_POWER_CAP_VALIDITY.value)
        self.assertFalse(eval_val.passed)
        self.assertIn("exceeds maximum", "; ".join(result.rejection_reasons).lower())

    def test_rule_power_cap_validity_failed_when_workload_cannot_power_cap(self) -> None:
        """Verify POWER_CAP action on workload with can_power_cap=False is rejected."""
        no_cap_workloads = dict(self.workloads)
        no_cap_workloads["wl-alpha-inference"] = self.workloads["wl-alpha-inference"].model_copy(
            update={"can_power_cap": False}
        )
        plan = RecoveryPlanOption(
            plan_id="PLAN-NO-CAP",
            name="No Cap Support",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=74.0,
            cooling_time_to_safe_s=17.5,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=220.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=160.0),
            ],
            estimated_cost_usd=12.0,
            sla_violations_expected=0,
            rationale="No cap support",
        )
        result = self.gate.evaluate(plan, self.telemetry, no_cap_workloads)
        self.assertFalse(result.passed)
        eval_val = next(e for e in result.evaluations if e.rule_name == SafetyRuleId.RULE_POWER_CAP_VALIDITY.value)
        self.assertFalse(eval_val.passed)
        self.assertIn("does not permit power capping", "; ".join(result.rejection_reasons))

    # -----------------------------------------------------------------------
    # 7. Architecture Recovery Plans Regression Tests (Plans A, B, C, D)
    # -----------------------------------------------------------------------

    def test_plan_a_conservative_shed_passes(self) -> None:
        """Verify Plan A (Pause Beta, Alpha 220W) satisfies all rules."""
        plan_a = RecoveryPlanOption(
            plan_id="PLAN-A",
            name="Conservative Workload Shed",
            strategy_type=StrategyType.CONSERVATIVE_SHED,
            is_recommended=False,
            projected_temp_max_c=67.5,
            cooling_time_to_safe_s=14.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=220.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.PAUSE),
            ],
            estimated_cost_usd=45.0,
            sla_violations_expected=0,
            rationale="Pause Beta to prioritize VIP Alpha.",
        )
        result = self.gate.evaluate(plan_a, self.telemetry, self.workloads)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.rejection_reasons), 0)
        self.assertEqual(len(result.evaluations), 6)
        self.assertTrue(all(e.passed for e in result.evaluations))

    def test_plan_b_balanced_power_cap_passes(self) -> None:
        """Verify Plan B (Alpha 220W, Beta 160W; total 380W) satisfies all rules."""
        plan_b = RecoveryPlanOption(
            plan_id="PLAN-B",
            name="Dynamic Workload Power-Cap Balancing",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            is_recommended=True,
            projected_temp_max_c=74.0,
            cooling_time_to_safe_s=17.5,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=220.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=160.0),
            ],
            estimated_cost_usd=12.0,
            sla_violations_expected=0,
            rationale="Balances cooling across jobs; preserves Alpha P0 deadline under 380W.",
        )
        result = self.gate.evaluate(plan_b, self.telemetry, self.workloads)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.rejection_reasons), 0)
        self.assertEqual(len(result.evaluations), 6)
        self.assertTrue(all(e.passed for e in result.evaluations))

    def test_plan_c_workload_migration_passes(self) -> None:
        """Verify Plan C (Migrate Beta, Alpha 250W; total 250W) satisfies all rules."""
        plan_c = RecoveryPlanOption(
            plan_id="PLAN-C",
            name="Workload Migration / Offload",
            strategy_type=StrategyType.MIGRATION_OFFLOAD,
            is_recommended=False,
            projected_temp_max_c=69.5,
            cooling_time_to_safe_s=12.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=250.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.MIGRATE),
            ],
            estimated_cost_usd=28.0,
            sla_violations_expected=0,
            rationale="Migrate Beta off-node to free thermal budget.",
        )
        result = self.gate.evaluate(plan_c, self.telemetry, self.workloads)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.rejection_reasons), 0)
        self.assertTrue(all(e.passed for e in result.evaluations))

    def test_plan_d_blind_50_percent_throttle_rejected(self) -> None:
        """Verify Plan D (Equal 125W throttle) is strictly rejected by Safety Gate."""
        # Plan D cuts Alpha to 125W (below 200W min floor) and Beta to 125W (below 150W min floor)
        plan_d = RecoveryPlanOption(
            plan_id="PLAN-D",
            name="Blind 50% Throttle",
            strategy_type=StrategyType.UNSAFE_THROTTLE,
            is_recommended=False,
            projected_temp_max_c=65.5,
            cooling_time_to_safe_s=8.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=125.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=125.0),
            ],
            estimated_cost_usd=320.0,
            sla_violations_expected=1,
            rationale="Naive throttle slashes both jobs equally, inducing severe SLA penalty.",
        )
        result = self.gate.evaluate(plan_d, self.telemetry, self.workloads)
        self.assertFalse(result.passed)

        # Confirm Rule 4 (P0 SLA preservation) and Rule 5 (Min Power Floor) failed
        eval_sla = next(e for e in result.evaluations if e.rule_name == SafetyRuleId.RULE_P0_SLA_PRESERVATION.value)
        eval_floor = next(e for e in result.evaluations if e.rule_name == SafetyRuleId.RULE_MIN_POWER_FLOOR.value)
        self.assertFalse(eval_sla.passed)
        self.assertFalse(eval_floor.passed)

        # Confirm explicit rejection reasons exist for both failures
        reasons_text = "; ".join(result.rejection_reasons).lower()
        self.assertIn("sla", reasons_text)
        self.assertIn("minimum operational floor", reasons_text)

    # -----------------------------------------------------------------------
    # 8. Multiple Simultaneous Violations
    # -----------------------------------------------------------------------

    def test_multiple_simultaneous_violations_captured(self) -> None:
        """Verify multiple simultaneous rule violations are all evaluated and captured."""
        # Violates:
        # 1. RULE_TEMP_CEILING: 84.5°C > 80.0°C
        # 2. RULE_NODE_POWER_BUDGET: 250W + 200W = 450W > 400W
        # 3. RULE_NO_P0_PREEMPTION: Alpha PAUSED
        # 4. RULE_P0_SLA_PRESERVATION: Alpha paused + expected SLA violations = 3
        # 5. RULE_MIN_POWER_FLOOR: Beta power = 100W < 150W floor
        # 6. RULE_POWER_CAP_VALIDITY: Beta power = 100W < min floor
        multi_fail_plan = RecoveryPlanOption(
            plan_id="PLAN-CATASTROPHIC",
            name="Catastrophic Plan",
            strategy_type=StrategyType.UNSAFE_THROTTLE,
            projected_temp_max_c=84.5,
            cooling_time_to_safe_s=40.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.PAUSE),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=100.0),
            ],
            estimated_cost_usd=900.0,
            sla_violations_expected=3,
            rationale="Violates almost every rule.",
        )
        result = self.gate.evaluate(multi_fail_plan, self.telemetry, self.workloads)
        self.assertFalse(result.passed)

        # Verify evaluations for all rules are present
        self.assertEqual(len(result.evaluations), 6)
        failed_rules = [e.rule_name for e in result.evaluations if not e.passed]
        self.assertIn(SafetyRuleId.RULE_TEMP_CEILING.value, failed_rules)
        self.assertIn(SafetyRuleId.RULE_NO_P0_PREEMPTION.value, failed_rules)
        self.assertIn(SafetyRuleId.RULE_P0_SLA_PRESERVATION.value, failed_rules)
        self.assertIn(SafetyRuleId.RULE_MIN_POWER_FLOOR.value, failed_rules)
        self.assertIn(SafetyRuleId.RULE_POWER_CAP_VALIDITY.value, failed_rules)
        self.assertGreater(len(result.rejection_reasons), 3)

    # -----------------------------------------------------------------------
    # 9. Missing or Invalid Inputs (Fail-Safe Defaults)
    # -----------------------------------------------------------------------

    def test_missing_or_invalid_plan_rejected(self) -> None:
        """Verify None or invalid plan types are rejected safely without exceptions."""
        res_none = self.gate.evaluate(None, self.telemetry, self.workloads)
        self.assertFalse(res_none.passed)
        self.assertIn("cannot be none", res_none.rejection_reasons[0].lower())

        res_str = self.gate.evaluate("invalid_plan_string", self.telemetry, self.workloads)
        self.assertFalse(res_str.passed)
        self.assertIn("invalid plan input", res_str.rejection_reasons[0].lower())

    def test_missing_or_invalid_telemetry_rejected(self) -> None:
        """Verify missing or invalid telemetry causes plan rejection."""
        plan = RecoveryPlanOption(
            plan_id="PLAN-SAFE",
            name="Plan",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=74.0,
            cooling_time_to_safe_s=15.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=220.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=160.0),
            ],
            estimated_cost_usd=12.0,
            sla_violations_expected=0,
            rationale="Test",
        )
        res_none_telem = self.gate.evaluate(plan, None, self.workloads)
        self.assertFalse(res_none_telem.passed)
        self.assertIn("telemetry", "; ".join(res_none_telem.rejection_reasons).lower())

        res_bad_telem = self.gate.evaluate(plan, "bad_telemetry", self.workloads)
        self.assertFalse(res_bad_telem.passed)
        self.assertIn("invalid telemetry", "; ".join(res_bad_telem.rejection_reasons).lower())

    def test_missing_or_empty_workloads_rejected(self) -> None:
        """Verify missing or empty workload specifications cause rejection."""
        plan = RecoveryPlanOption(
            plan_id="PLAN-SAFE",
            name="Plan",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=74.0,
            cooling_time_to_safe_s=15.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=220.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=160.0),
            ],
            estimated_cost_usd=12.0,
            sla_violations_expected=0,
            rationale="Test",
        )
        res_none_wl = self.gate.evaluate(plan, self.telemetry, None)
        self.assertFalse(res_none_wl.passed)

        res_empty_wl = self.gate.evaluate(plan, self.telemetry, [])
        self.assertFalse(res_empty_wl.passed)

        res_empty_dict = self.gate.evaluate(plan, self.telemetry, {})
        self.assertFalse(res_empty_dict.passed)

    def test_unknown_workload_id_in_action_rejected(self) -> None:
        """Verify actions referencing unknown workload IDs are rejected."""
        plan = RecoveryPlanOption(
            plan_id="PLAN-UNKNOWN-WL",
            name="Unknown Workload Plan",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=74.0,
            cooling_time_to_safe_s=15.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-nonexistent-job", action=ActionType.POWER_CAP, target_power_w=200.0),
            ],
            estimated_cost_usd=10.0,
            sla_violations_expected=0,
            rationale="Targets non-existent job",
        )
        result = self.gate.evaluate(plan, self.telemetry, self.workloads)
        self.assertFalse(result.passed)
        self.assertIn("unknown workload", "; ".join(result.rejection_reasons).lower())

    def test_conflicting_duplicate_actions_rejected(self) -> None:
        """Verify duplicate conflicting actions for the same workload are rejected."""
        plan = RecoveryPlanOption(
            plan_id="PLAN-DUPE-ACTION",
            name="Duplicate Actions Plan",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=74.0,
            cooling_time_to_safe_s=15.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=220.0),
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.PAUSE),
            ],
            estimated_cost_usd=50.0,
            sla_violations_expected=0,
            rationale="Duplicates on Alpha",
        )
        result = self.gate.evaluate(plan, self.telemetry, self.workloads)
        self.assertFalse(result.passed)
        self.assertIn("duplicate", "; ".join(result.rejection_reasons).lower())

    def test_invalid_node_constraints_rejected(self) -> None:
        """Verify invalid or non-positive node constraints are rejected."""
        bad_constraints = NodeConstraints(max_temp_c=80.0, power_budget_w=400.0)
        # Attempt negative temp override
        object.__setattr__(bad_constraints, "max_temp_c", -10.0)
        plan = RecoveryPlanOption(
            plan_id="PLAN-TEST",
            name="Plan",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=74.0,
            cooling_time_to_safe_s=15.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=220.0),
            ],
            estimated_cost_usd=10.0,
            sla_violations_expected=0,
            rationale="Test",
        )
        result = self.gate.evaluate(plan, self.telemetry, self.workloads, node_constraints=bad_constraints)
        self.assertFalse(result.passed)

    # -----------------------------------------------------------------------
    # 10. Approval Timeout Governance & Workload Mutation Safety Tests
    # -----------------------------------------------------------------------

    def test_approval_timeout_cannot_authorize_execution(self) -> None:
        """Verify TIMEOUT_ESCALATED strictly cannot authorize plan execution."""
        plan_b = RecoveryPlanOption(
            plan_id="PLAN-B",
            name="Plan B",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=74.0,
            cooling_time_to_safe_s=17.5,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=220.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=160.0),
            ],
            estimated_cost_usd=12.0,
            sla_violations_expected=0,
            rationale="Plan B",
        )
        safety_cert = self.gate.evaluate(plan_b, self.telemetry, self.workloads)
        self.assertTrue(safety_cert.passed)

        timeout_action = ApprovalAction(
            incident_id="INC-20261027-001",
            operator_id="operator-escalation",
            decision=ApprovalDecision.TIMEOUT_ESCALATED,
            selected_plan_id=None,
            notes="15 seconds elapsed without operator click.",
        )

        authorized = self.gate.authorize_execution(timeout_action, plan_b, safety_cert)
        self.assertFalse(authorized)

        auth_ok, reasons = self.gate.validate_authorization(timeout_action, plan_b, safety_cert)
        self.assertFalse(auth_ok)
        self.assertIn("timeout", "; ".join(reasons).lower())

    def test_approval_timeout_schema_rejects_selected_plan_id(self) -> None:
        """Verify ApprovalAction schema rejects any selected_plan_id for TIMEOUT_ESCALATED."""
        with self.assertRaises(ValueError) as ctx:
            ApprovalAction(
                incident_id="INC-20261027-001",
                operator_id="operator-escalation",
                decision=ApprovalDecision.TIMEOUT_ESCALATED,
                selected_plan_id="PLAN-B",  # Forbidden: timeout cannot select a plan!
            )
        self.assertIn("must be none", str(ctx.exception).lower())

    def test_approval_reject_decision_cannot_authorize_execution(self) -> None:
        """Verify operator REJECT decision cannot authorize plan execution."""
        plan_b = RecoveryPlanOption(
            plan_id="PLAN-B",
            name="Plan B",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=74.0,
            cooling_time_to_safe_s=17.5,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=220.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=160.0),
            ],
            estimated_cost_usd=12.0,
            sla_violations_expected=0,
            rationale="Plan B",
        )
        safety_cert = self.gate.evaluate(plan_b, self.telemetry, self.workloads)

        reject_action = ApprovalAction(
            incident_id="INC-20261027-001",
            operator_id="operator-oncall",
            decision=ApprovalDecision.REJECT,
            selected_plan_id=None,
            notes="Operator opted for manual chassis inspection.",
        )
        authorized = self.gate.authorize_execution(reject_action, plan_b, safety_cert)
        self.assertFalse(authorized)

    def test_approval_action_valid_authorization(self) -> None:
        """Verify valid APPROVE on certified plan succeeds in authorization."""
        plan_b = RecoveryPlanOption(
            plan_id="PLAN-B",
            name="Plan B",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=74.0,
            cooling_time_to_safe_s=17.5,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=220.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=160.0),
            ],
            estimated_cost_usd=12.0,
            sla_violations_expected=0,
            rationale="Plan B",
        )
        safety_cert = self.gate.evaluate(plan_b, self.telemetry, self.workloads)
        self.assertTrue(safety_cert.passed)

        approve_action = ApprovalAction(
            incident_id="INC-20261027-001",
            operator_id="operator-oncall",
            decision=ApprovalDecision.APPROVE,
            selected_plan_id="PLAN-B",
            notes="Approved recommended balanced plan.",
        )
        authorized = self.gate.authorize_execution(approve_action, plan_b, safety_cert)
        self.assertTrue(authorized)

    def test_approval_action_mismatched_plan_id_rejected(self) -> None:
        """Verify authorization fails if operator approved a different plan ID."""
        plan_b = RecoveryPlanOption(
            plan_id="PLAN-B",
            name="Plan B",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=74.0,
            cooling_time_to_safe_s=17.5,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=220.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=160.0),
            ],
            estimated_cost_usd=12.0,
            sla_violations_expected=0,
            rationale="Plan B",
        )
        safety_cert = self.gate.evaluate(plan_b, self.telemetry, self.workloads)

        approve_action = ApprovalAction(
            incident_id="INC-20261027-001",
            operator_id="operator-oncall",
            decision=ApprovalDecision.APPROVE,
            selected_plan_id="PLAN-A",  # Mismatch with PLAN-B!
        )
        authorized = self.gate.authorize_execution(approve_action, plan_b, safety_cert)
        self.assertFalse(authorized)

    def test_approval_action_unpassed_plan_rejected(self) -> None:
        """Verify unpassed safety gate certification cannot be authorized even if operator approved."""
        plan_d = RecoveryPlanOption(
            plan_id="PLAN-D",
            name="Blind Throttle",
            strategy_type=StrategyType.UNSAFE_THROTTLE,
            projected_temp_max_c=65.0,
            cooling_time_to_safe_s=8.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=125.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=125.0),
            ],
            estimated_cost_usd=320.0,
            sla_violations_expected=1,
            rationale="Unsafe plan",
        )
        safety_cert = self.gate.evaluate(plan_d, self.telemetry, self.workloads)
        self.assertFalse(safety_cert.passed)

        # Operator mistakenly clicks approve on Plan D
        approve_action = ApprovalAction(
            incident_id="INC-20261027-001",
            operator_id="operator-oncall",
            decision=ApprovalDecision.APPROVE,
            selected_plan_id="PLAN-D",
        )
        authorized = self.gate.authorize_execution(approve_action, plan_d, safety_cert)
        self.assertFalse(authorized)

    # -----------------------------------------------------------------------
    # 11. Deterministic Repeatability & Property Tests
    # -----------------------------------------------------------------------

    def test_deterministic_repeatability(self) -> None:
        """Verify identical inputs produce bit-exact SafetyGateResults across instances."""
        plan = RecoveryPlanOption(
            plan_id="PLAN-REPEATABLE",
            name="Repeatable Plan",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=74.0,
            cooling_time_to_safe_s=17.5,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=220.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=160.0),
            ],
            estimated_cost_usd=12.0,
            sla_violations_expected=0,
            rationale="Repeatable",
        )
        gate_a = SafetyGate(self.node_constraints)
        gate_b = SafetyGate(self.node_constraints)

        res_a = gate_a.evaluate(plan, self.telemetry, self.workloads)
        res_b = gate_b.evaluate(plan, self.telemetry, self.workloads)

        self.assertEqual(res_a.model_dump(), res_b.model_dump())
        self.assertEqual(res_a.rule_ids, res_b.rule_ids)


if __name__ == "__main__":
    unittest.main()
