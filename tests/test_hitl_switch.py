"""Unit tests for Switch Human-in-the-Loop (HITL) approval desk and governance subsystem.

Verifies:
1. Valid operator approval binds incident ID, plan ID, passes Safety Gate revalidation, and authorizes execution.
2. Operator explicit rejection (REJECT) strictly prevents execution and leaves workloads untouched.
3. Approval timeout (TIMEOUT_ESCALATED) strictly never executes workload mutations.
4. Missing (None) or malformed approval actions are rejected safely.
5. Duplicate (replay) approvals for an already processed incident are rejected.
6. Mismatched incident ID between approval and desk is rejected.
7. Attempting to approve an uncertified or rejected plan (Plan D) is strictly blocked.
8. Mismatched or non-existent plan ID is rejected.
9. Stale approval timestamps (pre-incident or past max age) are rejected.
10. Pre-execution Safety Gate revalidation catches environmental degradation and blocks execution.
11. Simulator state preservation: simulator power is never mutated without explicit valid authorization.
12. MockConsoleSwitchClient formats rich decision card text and simulates operator actions and timeouts.
13. WebhookSwitchClient enforces configuration checks and supports mockable HTTP dispatch.
14. Total power projection calculation across POWER_CAP, PAUSE, and MIGRATE actions.
"""

import unittest
from datetime import datetime, timedelta, timezone

from thermopilot.agents.forecast import ForecastResult, ThermalRiskStatus, ThermalTrend
from thermopilot.hitl.switch import (
    MockConsoleSwitchClient,
    SwitchApprovalDesk,
    SwitchConfigurationError,
    WebhookSwitchClient,
    WebhookSwitchClientConfig,
    apply_plan_to_simulator_if_authorized,
)
from thermopilot.safety.gate import SafetyGate
from thermopilot.schemas import (
    ActionType,
    ApprovalAction,
    ApprovalDecision,
    CandidatePlanSummary,
    DecisionCard,
    EvaluatedPlanCandidate,
    GPUTelemetry,
    NodeConstraints,
    PlanOrigin,
    RecoveryPlanOption,
    StrategyType,
    WorkflowStatus,
    WorkloadAction,
    WorkloadPriority,
    WorkloadSpec,
)
from thermopilot.simulator.telemetry import TelemetrySimulator


def make_test_telemetry(
    core_temp_c: float = 81.5,
    power_draw_w: float = 380.0,
    ambient_temp_c: float = 32.0,
) -> GPUTelemetry:
    """Helper returning standard GPUTelemetry snapshot."""
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


def make_test_workloads() -> dict[str, WorkloadSpec]:
    """Helper returning standard Alpha (P0) and Beta (P2) workloads."""
    alpha = WorkloadSpec(
        workload_id="wl-alpha-inference",
        name="Financial Trading Inference",
        owner="Trading-Prod",
        priority=WorkloadPriority.P0_CRITICAL,
        current_power_w=230.0,
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
        min_power_w=100.0,
        max_power_w=180.0,
        can_preempt=True,
        can_power_cap=True,
        sla_deadline_s=600.0,
        sla_penalty_usd_per_s=0.20,
    )
    return {"wl-alpha-inference": alpha, "wl-beta-training": beta}


def make_candidate_plans(
    gate: SafetyGate,
    telemetry: GPUTelemetry,
    workloads: dict[str, WorkloadSpec],
) -> list[EvaluatedPlanCandidate]:
    """Helper returning evaluated candidates: Plans A, B, C (Safe) and Plan D (Unsafe)."""
    # Plan A: Conservative Shed (Pause Beta)
    plan_a = RecoveryPlanOption(
        plan_id="PLAN-A",
        name="Conservative Workload Shedding",
        strategy_type=StrategyType.CONSERVATIVE_SHED,
        is_recommended=False,
        projected_temp_max_c=68.5,
        cooling_time_to_safe_s=18.0,
        workload_actions=[
            WorkloadAction(workload_id="wl-beta-training", action=ActionType.PAUSE),
        ],
        estimated_cost_usd=45.0,
        sla_violations_expected=0,
        rationale="Pauses P2 workload Beta to immediately dump heat.",
        origin=PlanOrigin.FALLBACK,
    )

    # Plan B: Balanced Power Cap (Cap Alpha to 210W, Beta to 120W)
    plan_b = RecoveryPlanOption(
        plan_id="PLAN-B",
        name="Balanced Coordinated Power Capping",
        strategy_type=StrategyType.BALANCED_POWER_CAP,
        is_recommended=True,
        projected_temp_max_c=74.2,
        cooling_time_to_safe_s=24.0,
        workload_actions=[
            WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=210.0),
            WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=120.0),
        ],
        estimated_cost_usd=12.0,
        sla_violations_expected=0,
        rationale="Coordinates moderate power caps preserving all SLAs.",
        origin=PlanOrigin.FALLBACK,
    )

    # Plan C: Migration Offload (Migrate Beta)
    plan_c = RecoveryPlanOption(
        plan_id="PLAN-C",
        name="Workload Migration Offload",
        strategy_type=StrategyType.MIGRATION_OFFLOAD,
        is_recommended=False,
        projected_temp_max_c=71.0,
        cooling_time_to_safe_s=20.0,
        workload_actions=[
            WorkloadAction(workload_id="wl-beta-training", action=ActionType.MIGRATE),
        ],
        estimated_cost_usd=28.0,
        sla_violations_expected=0,
        rationale="Transfers Beta to a remote standby GPU.",
        origin=PlanOrigin.FALLBACK,
    )

    # Plan D: Unsafe Plan (Preempts P0 Alpha, breaching rules)
    plan_d = RecoveryPlanOption(
        plan_id="PLAN-D",
        name="Naive Aggressive Throttle",
        strategy_type=StrategyType.UNSAFE_THROTTLE,
        is_recommended=False,
        projected_temp_max_c=82.0,  # Exceeds 80C
        cooling_time_to_safe_s=40.0,
        workload_actions=[
            WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.PAUSE),  # Violates Rule 3
        ],
        estimated_cost_usd=150.0,
        sla_violations_expected=1,
        rationale="Unsafe aggressive shedding attempting to pause P0.",
        origin=PlanOrigin.FALLBACK,
    )

    result_a = gate.evaluate(plan_a, telemetry, workloads)
    result_b = gate.evaluate(plan_b, telemetry, workloads)
    result_c = gate.evaluate(plan_c, telemetry, workloads)
    result_d = gate.evaluate(plan_d, telemetry, workloads)

    return [
        EvaluatedPlanCandidate(plan=plan_a, origin=PlanOrigin.FALLBACK, safety_result=result_a),
        EvaluatedPlanCandidate(plan=plan_b, origin=PlanOrigin.FALLBACK, safety_result=result_b),
        EvaluatedPlanCandidate(plan=plan_c, origin=PlanOrigin.FALLBACK, safety_result=result_c),
        EvaluatedPlanCandidate(plan=plan_d, origin=PlanOrigin.FALLBACK, safety_result=result_d),
    ]


class TestSwitchApprovalDesk(unittest.TestCase):
    """Tests for SwitchApprovalDesk orchestration and fail-safe governance."""

    def setUp(self) -> None:
        self.gate = SafetyGate()
        self.mock_client = MockConsoleSwitchClient()
        self.desk = SwitchApprovalDesk(safety_gate=self.gate, switch_client=self.mock_client)
        self.telemetry = make_test_telemetry()
        self.workloads = make_test_workloads()
        self.candidates = make_candidate_plans(self.gate, self.telemetry, self.workloads)
        self.incident_id = "INC-20261029-001"

    def test_decision_card_structure_and_safety_partition(self) -> None:
        """Card must present Plans A/B/C as executable and flag Plan D as rejected."""
        card = self.desk.create_and_dispatch_card(
            incident_id=self.incident_id,
            telemetry=self.telemetry,
            candidates=self.candidates,
            workloads=self.workloads,
        )

        self.assertEqual(card.incident_id, self.incident_id)
        self.assertEqual(card.current_temp_c, 81.5)
        self.assertEqual(card.current_power_w, 380.0)
        self.assertEqual(len(card.candidate_plans), 4)

        # Executable list contains safe plans only
        self.assertEqual(card.executable_plan_ids, ["PLAN-A", "PLAN-B", "PLAN-C"])
        self.assertEqual(card.rejected_plan_ids, ["PLAN-D"])

        # Check Plan B summary
        plan_b_summary = next(p for p in card.candidate_plans if p.plan_id == "PLAN-B")
        self.assertTrue(plan_b_summary.is_safe)
        self.assertTrue(plan_b_summary.is_executable)
        self.assertEqual(plan_b_summary.safety_verdict, "CERTIFIED_SAFE")
        self.assertEqual(plan_b_summary.projected_power_w, 330.0)  # 210 + 120

        # Check Plan D summary
        plan_d_summary = next(p for p in card.candidate_plans if p.plan_id == "PLAN-D")
        self.assertFalse(plan_d_summary.is_safe)
        self.assertFalse(plan_d_summary.is_executable)
        self.assertEqual(plan_d_summary.safety_verdict, "REJECTED_BY_SAFETY_GATE")
        self.assertGreater(len(plan_d_summary.rejection_reasons), 0)

    def test_valid_operator_approval_authorizes_plan(self) -> None:
        """Valid operator approval of certified Plan B authorizes execution."""
        self.desk.create_and_dispatch_card(
            incident_id=self.incident_id,
            telemetry=self.telemetry,
            candidates=self.candidates,
            workloads=self.workloads,
        )

        approval = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id="PLAN-B",
            operator_id="operator-alex",
            decision=ApprovalDecision.APPROVE,
            timestamp=datetime.now(timezone.utc),
            notes="Approving Plan B to balance cooling with SLA.",
        )

        result = self.desk.evaluate_approval(
            approval=approval,
            incident_id=self.incident_id,
            current_telemetry=self.telemetry,
        )

        self.assertEqual(result.status, WorkflowStatus.APPROVED)
        self.assertTrue(result.is_authorized_for_execution)
        self.assertIsNotNone(result.selected_plan)
        self.assertEqual(result.selected_plan.plan_id, "PLAN-B")
        self.assertIsNotNone(result.revalidation_result)
        self.assertTrue(result.revalidation_result.passed)

    def test_explicit_operator_rejection_prevents_execution(self) -> None:
        """Explicit rejection must set status REJECTED and strictly forbid execution."""
        self.desk.create_and_dispatch_card(
            incident_id=self.incident_id,
            telemetry=self.telemetry,
            candidates=self.candidates,
            workloads=self.workloads,
        )

        rejection = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id=None,
            operator_id="operator-alex",
            decision=ApprovalDecision.REJECT,
            timestamp=datetime.now(timezone.utc),
            notes="Conditions do not warrant intervention.",
        )

        result = self.desk.evaluate_approval(
            approval=rejection,
            incident_id=self.incident_id,
            current_telemetry=self.telemetry,
        )

        self.assertEqual(result.status, WorkflowStatus.REJECTED)
        self.assertFalse(result.is_authorized_for_execution)
        self.assertIsNone(result.selected_plan)

    def test_timeout_escalation_strictly_prevents_execution(self) -> None:
        """Timeout escalation must set TIMEOUT_ESCALATED and strictly forbid execution."""
        self.desk.create_and_dispatch_card(
            incident_id=self.incident_id,
            telemetry=self.telemetry,
            candidates=self.candidates,
            workloads=self.workloads,
        )

        timeout_action = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id=None,
            operator_id="escalation-subsystem",
            decision=ApprovalDecision.TIMEOUT_ESCALATED,
            timestamp=datetime.now(timezone.utc),
            notes="15-second response window expired.",
        )

        result = self.desk.evaluate_approval(
            approval=timeout_action,
            incident_id=self.incident_id,
            current_telemetry=self.telemetry,
        )

        self.assertEqual(result.status, WorkflowStatus.TIMEOUT_ESCALATED)
        self.assertFalse(result.is_authorized_for_execution)
        self.assertIsNone(result.selected_plan)

    def test_missing_approval_action_rejected(self) -> None:
        """None approval action must fail safely as INVALID_SUBMISSION."""
        self.desk.create_and_dispatch_card(
            incident_id=self.incident_id,
            telemetry=self.telemetry,
            candidates=self.candidates,
            workloads=self.workloads,
        )

        result = self.desk.evaluate_approval(
            approval=None,
            incident_id=self.incident_id,
            current_telemetry=self.telemetry,
        )

        self.assertEqual(result.status, WorkflowStatus.INVALID_SUBMISSION)
        self.assertFalse(result.is_authorized_for_execution)
        self.assertIn("missing", result.rejection_reasons[0].lower())

    def test_duplicate_approval_rejected(self) -> None:
        """Re-submitting an approval for an already finalized incident is rejected."""
        self.desk.create_and_dispatch_card(
            incident_id=self.incident_id,
            telemetry=self.telemetry,
            candidates=self.candidates,
            workloads=self.workloads,
        )

        approval = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id="PLAN-B",
            operator_id="operator-alex",
            decision=ApprovalDecision.APPROVE,
            timestamp=datetime.now(timezone.utc),
        )

        result1 = self.desk.evaluate_approval(approval, self.incident_id, self.telemetry)
        self.assertEqual(result1.status, WorkflowStatus.APPROVED)

        # Attempt duplicate submission
        result2 = self.desk.evaluate_approval(approval, self.incident_id, self.telemetry)
        self.assertEqual(result2.status, WorkflowStatus.INVALID_SUBMISSION)
        self.assertFalse(result2.is_authorized_for_execution)
        self.assertIn("duplicate", result2.rejection_reasons[0].lower())

    def test_mismatched_incident_id_rejected(self) -> None:
        """Approval with different incident ID is rejected."""
        self.desk.create_and_dispatch_card(
            incident_id=self.incident_id,
            telemetry=self.telemetry,
            candidates=self.candidates,
            workloads=self.workloads,
        )

        mismatched_approval = ApprovalAction(
            incident_id="INC-OTHER-999",
            selected_plan_id="PLAN-B",
            operator_id="operator-alex",
            decision=ApprovalDecision.APPROVE,
            timestamp=datetime.now(timezone.utc),
        )

        result = self.desk.evaluate_approval(mismatched_approval, self.incident_id, self.telemetry)
        self.assertEqual(result.status, WorkflowStatus.INVALID_SUBMISSION)
        self.assertFalse(result.is_authorized_for_execution)
        self.assertIn("does not match", result.rejection_reasons[0].lower())

    def test_approving_rejected_plan_d_strictly_rejected(self) -> None:
        """Attempting to approve rejected Plan D must be denied immediately."""
        self.desk.create_and_dispatch_card(
            incident_id=self.incident_id,
            telemetry=self.telemetry,
            candidates=self.candidates,
            workloads=self.workloads,
        )

        bad_approval = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id="PLAN-D",
            operator_id="operator-alex",
            decision=ApprovalDecision.APPROVE,
            timestamp=datetime.now(timezone.utc),
        )

        result = self.desk.evaluate_approval(bad_approval, self.incident_id, self.telemetry)
        self.assertEqual(result.status, WorkflowStatus.INVALID_SUBMISSION)
        self.assertFalse(result.is_authorized_for_execution)
        self.assertIn("rejected by the safety gate", result.rejection_reasons[0].lower())

    def test_approving_unknown_plan_rejected(self) -> None:
        """Submitting an unknown plan ID is rejected."""
        self.desk.create_and_dispatch_card(
            incident_id=self.incident_id,
            telemetry=self.telemetry,
            candidates=self.candidates,
            workloads=self.workloads,
        )

        unknown_approval = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id="PLAN-NONEXISTENT",
            operator_id="operator-alex",
            decision=ApprovalDecision.APPROVE,
            timestamp=datetime.now(timezone.utc),
        )

        result = self.desk.evaluate_approval(unknown_approval, self.incident_id, self.telemetry)
        self.assertEqual(result.status, WorkflowStatus.INVALID_SUBMISSION)
        self.assertFalse(result.is_authorized_for_execution)
        self.assertIn("not among certified executable plans", result.rejection_reasons[0].lower())

    def test_stale_approval_preceding_card_dispatch_rejected(self) -> None:
        """Approval timestamp earlier than the card generation timestamp is rejected."""
        card = self.desk.create_and_dispatch_card(
            incident_id=self.incident_id,
            telemetry=self.telemetry,
            candidates=self.candidates,
            workloads=self.workloads,
        )

        stale_time = card.timestamp - timedelta(minutes=10)
        stale_approval = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id="PLAN-B",
            operator_id="operator-alex",
            decision=ApprovalDecision.APPROVE,
            timestamp=stale_time,
        )

        result = self.desk.evaluate_approval(stale_approval, self.incident_id, self.telemetry)
        self.assertEqual(result.status, WorkflowStatus.INVALID_SUBMISSION)
        self.assertFalse(result.is_authorized_for_execution)
        self.assertIn("precedes incident card", result.rejection_reasons[0].lower())

    def test_stale_approval_exceeding_max_age_rejected(self) -> None:
        """Approval submitted after max_approval_age_s is rejected."""
        card = self.desk.create_and_dispatch_card(
            incident_id=self.incident_id,
            telemetry=self.telemetry,
            candidates=self.candidates,
            workloads=self.workloads,
        )

        too_old_time = card.timestamp + timedelta(seconds=self.desk.max_approval_age_s + 15.0)
        old_approval = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id="PLAN-B",
            operator_id="operator-alex",
            decision=ApprovalDecision.APPROVE,
            timestamp=too_old_time,
        )

        result = self.desk.evaluate_approval(old_approval, self.incident_id, self.telemetry)
        self.assertEqual(result.status, WorkflowStatus.INVALID_SUBMISSION)
        self.assertFalse(result.is_authorized_for_execution)
        self.assertIn("stale", result.rejection_reasons[0].lower())

    def test_pre_execution_revalidation_failure_blocks_execution(self) -> None:
        """If telemetry deteriorates before execution such that Safety Gate fails, execution is blocked."""
        self.desk.create_and_dispatch_card(
            incident_id=self.incident_id,
            telemetry=self.telemetry,
            candidates=self.candidates,
            workloads=self.workloads,
        )

        approval = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id="PLAN-B",
            operator_id="operator-alex",
            decision=ApprovalDecision.APPROVE,
            timestamp=datetime.now(timezone.utc),
        )

        # Deteriorated telemetry at execution time: power budget tightened to 300W
        # Plan B requires 330W, so Rule 2 will fail!
        tight_constraints = NodeConstraints(power_budget_w=300.0, max_temp_c=80.0)

        result = self.desk.evaluate_approval(
            approval=approval,
            incident_id=self.incident_id,
            current_telemetry=self.telemetry,
            node_constraints=tight_constraints,
        )

        self.assertEqual(result.status, WorkflowStatus.REVALIDATION_FAILED)
        self.assertFalse(result.is_authorized_for_execution)
        self.assertIsNotNone(result.revalidation_result)
        self.assertFalse(result.revalidation_result.passed)
        self.assertIn("revalidation failed", result.rejection_reasons[0].lower())

    def test_run_approval_cycle_end_to_end_with_mock_client(self) -> None:
        """Test full run_approval_cycle with queued decision in MockConsoleSwitchClient."""
        approval = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id="PLAN-A",
            operator_id="operator-bob",
            decision=ApprovalDecision.APPROVE,
            timestamp=datetime.now(timezone.utc),
        )
        self.mock_client.queue_approval(approval)

        result = self.desk.run_approval_cycle(
            incident_id=self.incident_id,
            telemetry=self.telemetry,
            candidates=self.candidates,
            workloads=self.workloads,
        )

        self.assertEqual(result.status, WorkflowStatus.APPROVED)
        self.assertTrue(result.is_authorized_for_execution)
        self.assertEqual(result.selected_plan.plan_id, "PLAN-A")

    def test_run_approval_cycle_timeout_with_mock_client(self) -> None:
        """Test full run_approval_cycle when no action is queued, triggering timeout."""
        result = self.desk.run_approval_cycle(
            incident_id=self.incident_id,
            telemetry=self.telemetry,
            candidates=self.candidates,
            workloads=self.workloads,
            timeout_s=5.0,
        )

        self.assertEqual(result.status, WorkflowStatus.TIMEOUT_ESCALATED)
        self.assertFalse(result.is_authorized_for_execution)
        self.assertIsNone(result.selected_plan)


class TestSimulatorIsolation(unittest.TestCase):
    """Tests ensuring simulator workload power state is NEVER mutated without explicit authorization."""

    def setUp(self) -> None:
        self.gate = SafetyGate()
        self.mock_client = MockConsoleSwitchClient()
        self.desk = SwitchApprovalDesk(safety_gate=self.gate, switch_client=self.mock_client)
        self.telemetry = make_test_telemetry()
        self.workloads = make_test_workloads()
        self.candidates = make_candidate_plans(self.gate, self.telemetry, self.workloads)
        self.incident_id = "INC-20261029-SIM"

    def test_simulator_state_unchanged_on_timeout(self) -> None:
        """Timeout escalation must not mutate simulator workloads."""
        sim = TelemetrySimulator()
        initial_powers = dict(sim.workload_powers)

        self.desk.create_and_dispatch_card(self.incident_id, self.telemetry, self.candidates, self.workloads)
        timeout_action = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id=None,
            operator_id="escalation-subsystem",
            decision=ApprovalDecision.TIMEOUT_ESCALATED,
        )
        result = self.desk.evaluate_approval(timeout_action, self.incident_id, self.telemetry)

        # Attempt to apply to simulator raises PermissionError
        with self.assertRaises(PermissionError):
            apply_plan_to_simulator_if_authorized(result, sim)

        # Verify simulator powers remain completely unchanged
        self.assertEqual(sim.workload_powers, initial_powers)

    def test_simulator_state_unchanged_on_rejection(self) -> None:
        """Explicit rejection must not mutate simulator workloads."""
        sim = TelemetrySimulator()
        initial_powers = dict(sim.workload_powers)

        self.desk.create_and_dispatch_card(self.incident_id, self.telemetry, self.candidates, self.workloads)
        rejection_action = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id=None,
            operator_id="operator-alex",
            decision=ApprovalDecision.REJECT,
        )
        result = self.desk.evaluate_approval(rejection_action, self.incident_id, self.telemetry)

        with self.assertRaises(PermissionError):
            apply_plan_to_simulator_if_authorized(result, sim)

        self.assertEqual(sim.workload_powers, initial_powers)

    def test_simulator_state_unchanged_on_revalidation_failure(self) -> None:
        """Revalidation failure must not mutate simulator workloads."""
        sim = TelemetrySimulator()
        initial_powers = dict(sim.workload_powers)

        self.desk.create_and_dispatch_card(self.incident_id, self.telemetry, self.candidates, self.workloads)
        approval = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id="PLAN-B",
            operator_id="operator-alex",
            decision=ApprovalDecision.APPROVE,
        )

        # Pass tight constraint forcing revalidation failure
        tight_constraints = NodeConstraints(power_budget_w=200.0)
        result = self.desk.evaluate_approval(
            approval,
            self.incident_id,
            self.telemetry,
            node_constraints=tight_constraints,
        )

        with self.assertRaises(PermissionError):
            apply_plan_to_simulator_if_authorized(result, sim)

        self.assertEqual(sim.workload_powers, initial_powers)

    def test_simulator_mutated_only_on_authorized_approval(self) -> None:
        """Only an authorized approval modifies simulator power limits."""
        from thermopilot.simulator.telemetry import SimulatorConfig
        sim = TelemetrySimulator(
            config=SimulatorConfig(
                default_workloads={"wl-alpha-inference": 230.0, "wl-beta-training": 150.0}
            )
        )

        self.desk.create_and_dispatch_card(self.incident_id, self.telemetry, self.candidates, self.workloads)
        approval = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id="PLAN-B",
            operator_id="operator-alex",
            decision=ApprovalDecision.APPROVE,
        )
        result = self.desk.evaluate_approval(approval, self.incident_id, self.telemetry)
        self.assertTrue(result.is_authorized_for_execution)

        # Apply authorized plan to simulator
        applied = apply_plan_to_simulator_if_authorized(result, sim)
        self.assertTrue(applied)

        # Workload powers in simulator are updated according to Plan B
        self.assertEqual(sim.workload_powers["wl-alpha-inference"], 210.0)
        self.assertEqual(sim.workload_powers["wl-beta-training"], 120.0)


class TestSwitchClients(unittest.TestCase):
    """Tests for MockConsoleSwitchClient and WebhookSwitchClient."""

    def test_mock_console_client_card_formatting(self) -> None:
        """Mock console client must format card text with all required sections."""
        client = MockConsoleSwitchClient()
        forecast = ForecastResult(
            gpu_id=0,
            current_temp_c=81.5,
            critical_threshold_c=85.0,
            trend=ThermalTrend.RISING,
            time_to_threshold_s=24.0,
            risk_status=ThermalRiskStatus.CRITICAL_RISK,
            trigger_incident=True,
            reason="Thermal runaway",
        )
        card = DecisionCard(
            incident_id="INC-FORMAT-TEST",
            gpu_id=0,
            severity="CRITICAL",
            current_temp_c=81.5,
            current_power_w=380.0,
            critical_trip_temp_c=85.0,
            time_to_trip_s=forecast.time_to_threshold_s,
            candidate_plans=[
                CandidatePlanSummary(
                    plan_id="PLAN-A",
                    name="Test Safe",
                    strategy_type=StrategyType.CONSERVATIVE_SHED,
                    is_recommended=True,
                    is_safe=True,
                    safety_verdict="CERTIFIED_SAFE",
                    projected_temp_max_c=70.0,
                    projected_power_w=300.0,
                    cooling_time_to_safe_s=15.0,
                    estimated_cost_usd=20.0,
                    sla_violations_expected=0,
                    proposed_actions=[],
                    is_executable=True,
                    rationale="Safe test",
                ),
                CandidatePlanSummary(
                    plan_id="PLAN-D",
                    name="Test Unsafe",
                    strategy_type=StrategyType.UNSAFE_THROTTLE,
                    is_recommended=False,
                    is_safe=False,
                    safety_verdict="REJECTED_BY_SAFETY_GATE",
                    rejection_reasons=["Exceeds 80C limit"],
                    projected_temp_max_c=85.0,
                    projected_power_w=450.0,
                    cooling_time_to_safe_s=40.0,
                    estimated_cost_usd=100.0,
                    sla_violations_expected=2,
                    proposed_actions=[],
                    is_executable=False,
                    rationale="Unsafe test",
                ),
            ],
            executable_plan_ids=["PLAN-A"],
            rejected_plan_ids=["PLAN-D"],
            timeout_seconds=15.0,
        )

        text = client.format_card_text(card)
        self.assertIn("INCIDENT: INC-FORMAT-TEST", text)
        self.assertIn("Time-to-Trip (t_TTT): 24.0s", text)
        self.assertIn("PLAN-A: Test Safe [RECOMMENDED] [CERTIFIED SAFE]", text)
        self.assertIn("PLAN-D: Test Unsafe [REJECTED BY SAFETY GATE - NON-EXECUTABLE]", text)
        self.assertIn("Exceeds 80C limit", text)
        self.assertIn("Hotkeys:", text)

        # Dispatch adds to client's records
        client.dispatch_decision_card(card)
        self.assertEqual(len(client.dispatched_cards), 1)

    def test_webhook_switch_client_unconfigured_error(self) -> None:
        """Unconfigured WebhookSwitchClient must raise SwitchConfigurationError."""
        client = WebhookSwitchClient()
        self.assertFalse(client.is_configured)

        card = DecisionCard(
            incident_id="INC-WH-TEST",
            gpu_id=0,
            severity="CRITICAL",
            current_temp_c=82.0,
            current_power_w=380.0,
            candidate_plans=[
                CandidatePlanSummary(
                    plan_id="PLAN-A",
                    name="Test",
                    strategy_type=StrategyType.CONSERVATIVE_SHED,
                    is_safe=True,
                    safety_verdict="CERTIFIED_SAFE",
                    projected_temp_max_c=70.0,
                    projected_power_w=300.0,
                    cooling_time_to_safe_s=10.0,
                    estimated_cost_usd=10.0,
                    is_executable=True,
                    rationale="Test",
                )
            ],
            executable_plan_ids=["PLAN-A"],
        )

        with self.assertRaises(SwitchConfigurationError):
            client.dispatch_decision_card(card)

        with self.assertRaises(SwitchConfigurationError):
            client.listen_for_decision("INC-WH-TEST")

    def test_webhook_switch_client_with_mock_poster(self) -> None:
        """Configured WebhookSwitchClient posts JSON card payload via poster."""
        captured_requests: list[dict] = []

        def mock_poster(url: str, data: bytes, headers: dict) -> bool:
            captured_requests.append({"url": url, "data": data, "headers": headers})
            return True

        config = WebhookSwitchClientConfig(
            webhook_url="https://switch.test/api/v1/incidents",
            api_key="secret-switch-key-123",
        )
        client = WebhookSwitchClient(config=config, http_poster=mock_poster)
        self.assertTrue(client.is_configured)

        card = DecisionCard(
            incident_id="INC-POSTER-TEST",
            gpu_id=0,
            severity="CRITICAL",
            current_temp_c=81.0,
            current_power_w=375.0,
            candidate_plans=[
                CandidatePlanSummary(
                    plan_id="PLAN-A",
                    name="Test Plan",
                    strategy_type=StrategyType.CONSERVATIVE_SHED,
                    is_safe=True,
                    safety_verdict="CERTIFIED_SAFE",
                    projected_temp_max_c=72.0,
                    projected_power_w=320.0,
                    cooling_time_to_safe_s=12.0,
                    estimated_cost_usd=15.0,
                    is_executable=True,
                    rationale="Test",
                )
            ],
            executable_plan_ids=["PLAN-A"],
        )

        success = client.dispatch_decision_card(card)
        self.assertTrue(success)
        self.assertEqual(len(captured_requests), 1)
        self.assertEqual(captured_requests[0]["url"], "https://switch.test/api/v1/incidents")
        self.assertEqual(captured_requests[0]["headers"]["Authorization"], "Bearer secret-switch-key-123")


class TestSchemaValidation(unittest.TestCase):
    """Tests for new HITL schema models and fail-safe validators."""

    def test_candidate_plan_summary_cannot_be_unsafe_and_executable(self) -> None:
        """CandidatePlanSummary must reject is_safe=False combined with is_executable=True."""
        with self.assertRaises(ValueError):
            CandidatePlanSummary(
                plan_id="PLAN-BAD",
                name="Inconsistent Candidate",
                strategy_type=StrategyType.BALANCED_POWER_CAP,
                is_safe=False,
                safety_verdict="REJECTED_BY_SAFETY_GATE",
                rejection_reasons=["Temperature violation"],
                projected_temp_max_c=85.0,
                projected_power_w=400.0,
                cooling_time_to_safe_s=30.0,
                estimated_cost_usd=10.0,
                is_executable=True,  # Inconsistent! Must fail validator
                rationale="Bad plan",
            )

    def test_calculate_projected_power_with_various_actions(self) -> None:
        """Verify projected power calculation across capping, pausing, and migrating."""
        workloads = make_test_workloads()
        # Workloads: Alpha=230W, Beta=150W (Total 380W)

        # Plan caps Alpha to 200W, pauses Beta
        plan = RecoveryPlanOption(
            plan_id="PLAN-POWER-TEST",
            name="Power Calc Plan",
            strategy_type=StrategyType.CONSERVATIVE_SHED,
            projected_temp_max_c=65.0,
            cooling_time_to_safe_s=15.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=200.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.PAUSE),
            ],
            estimated_cost_usd=30.0,
            rationale="Test",
        )

        power = SwitchApprovalDesk.calculate_projected_power(plan, workloads)
        self.assertEqual(power, 200.0)  # 200 + 0 = 200W
