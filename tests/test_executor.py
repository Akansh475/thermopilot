"""Unit tests for the Simulation Executor subsystem.

Verifies:
1. Successful execution of Plan A (POWER_CAP Alpha, PAUSE Beta).
2. Successful execution of Plan B (dynamic power capping across both workloads).
3. Successful execution of Plan C (POWER_CAP Alpha, MIGRATE Beta).
4. Individual execution of each supported action (POWER_CAP, PAUSE, MIGRATE, RESUME).
5. Rejection of unsupported or invalid actions.
6. Rejection of unapproved decisions (TIMEOUT_ESCALATED, REJECT, None).
7. Rejection of stale approvals (preceding incident timestamp or exceeding max age).
8. Rejection of mismatched plan IDs and incident IDs.
9. Rejection of uncertified or failed SafetyGate evaluations.
10. Pre-execution SafetyGate revalidation failure blocks execution.
11. Node electrical power budget (400W) enforcement.
12. Workload power floor enforcement.
13. P0 SLA protection enforcement (non-preemption, zero penalties).
14. Atomic partial failure rollback (simulator restored to pre-execution state).
15. STRICT GUARANTEE: Unauthorized execution leaves simulator state 100% unchanged.
16. Execution via ApprovalWorkflowResult.
17. raise_on_failure parameter raising ExecutionDeniedError / PermissionError.
18. Isolation guarantee: strictly requires TelemetrySimulator, rejects real hardware targets.
"""

import unittest
from datetime import datetime, timedelta, timezone

from thermopilot.executor.executor import (
    ExecutionDeniedError,
    ExecutionStatus,
    SimulationExecutor,
)
from thermopilot.safety.gate import SafetyGate
from thermopilot.schemas import (
    ActionType,
    ApprovalAction,
    ApprovalDecision,
    ApprovalWorkflowResult,
    GPUTelemetry,
    NodeConstraints,
    PlanOrigin,
    RecoveryPlanOption,
    SafetyGateResult,
    StrategyType,
    WorkflowStatus,
    WorkloadAction,
    WorkloadPriority,
    WorkloadSpec,
)
from thermopilot.simulator.telemetry import SimulatorConfig, TelemetrySimulator


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
    return {alpha.workload_id: alpha, beta.workload_id: beta}


def make_test_simulator() -> TelemetrySimulator:
    """Helper returning TelemetrySimulator matching standard test workloads."""
    config = SimulatorConfig(
        power_budget_w=400.0,
        baseline_ambient_c=22.0,
        anomaly_ambient_c=32.0,
        initial_temp_c=81.5,
        noise_std=0.0,
        default_workloads={"wl-alpha-inference": 230.0, "wl-beta-training": 150.0},
    )
    sim = TelemetrySimulator(config)
    sim.trigger_anomaly(32.0)
    return sim


class TestSimulationExecutor(unittest.TestCase):
    """Test suite verifying SimulationExecutor functionality and safety boundaries."""

    def setUp(self) -> None:
        self.incident_id = "INC-20261027-001"
        self.workloads = make_test_workloads()
        self.telemetry = make_test_telemetry()
        self.safety_gate = SafetyGate()
        self.executor = SimulationExecutor(safety_gate=self.safety_gate, max_approval_age_s=60.0)

        # Plan A: Conservative Workload Shed (Pause Beta, Cap Alpha 220W)
        self.plan_a = RecoveryPlanOption(
            plan_id="PLAN-A",
            name="Conservative Workload Shed",
            strategy_type=StrategyType.CONSERVATIVE_SHED,
            is_recommended=False,
            projected_temp_max_c=68.0,
            cooling_time_to_safe_s=14.0,
            workload_actions=[
                WorkloadAction(
                    workload_id="wl-alpha-inference",
                    action=ActionType.POWER_CAP,
                    target_power_w=220.0,
                ),
                WorkloadAction(
                    workload_id="wl-beta-training",
                    action=ActionType.PAUSE,
                ),
            ],
            estimated_cost_usd=45.0,
            sla_violations_expected=0,
            rationale="Pause flexible training job Beta; allocate safe power to VIP Alpha.",
            origin=PlanOrigin.FALLBACK,
        )

        # Plan B: Balanced Dynamic Cap (Alpha 210W, Beta 120W)
        self.plan_b = RecoveryPlanOption(
            plan_id="PLAN-B",
            name="Dynamic Workload Power-Cap Balancing",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            is_recommended=True,
            projected_temp_max_c=74.0,
            cooling_time_to_safe_s=17.5,
            workload_actions=[
                WorkloadAction(
                    workload_id="wl-alpha-inference",
                    action=ActionType.POWER_CAP,
                    target_power_w=210.0,
                ),
                WorkloadAction(
                    workload_id="wl-beta-training",
                    action=ActionType.POWER_CAP,
                    target_power_w=120.0,
                ),
            ],
            estimated_cost_usd=12.0,
            sla_violations_expected=0,
            rationale="Dynamic power-cap balancing across both jobs within 330W budget.",
            origin=PlanOrigin.FALLBACK,
        )

        # Plan C: Workload Migration (Alpha 240W, Migrate Beta)
        self.plan_c = RecoveryPlanOption(
            plan_id="PLAN-C",
            name="Workload Migration / Offload",
            strategy_type=StrategyType.MIGRATION_OFFLOAD,
            is_recommended=False,
            projected_temp_max_c=70.0,
            cooling_time_to_safe_s=12.0,
            workload_actions=[
                WorkloadAction(
                    workload_id="wl-alpha-inference",
                    action=ActionType.POWER_CAP,
                    target_power_w=240.0,
                ),
                WorkloadAction(
                    workload_id="wl-beta-training",
                    action=ActionType.MIGRATE,
                ),
            ],
            estimated_cost_usd=28.0,
            sla_violations_expected=0,
            rationale="Export Beta checkpoint and offload to secondary node.",
            origin=PlanOrigin.FALLBACK,
        )

        # Safety Gate certification for Plan B
        self.safety_result_b = self.safety_gate.evaluate(
            plan=self.plan_b,
            telemetry=self.telemetry,
            workloads=self.workloads,
        )
        self.assertTrue(self.safety_result_b.passed)

        # Valid Approval Action for Plan B
        self.approval_b = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id="PLAN-B",
            operator_id="operator-alex",
            decision=ApprovalDecision.APPROVE,
            timestamp=self.telemetry.timestamp + timedelta(seconds=5),
            notes="Approved recommended balanced dynamic power cap.",
        )

    # -----------------------------------------------------------------------
    # 1. Successful Plan Execution Tests
    # -----------------------------------------------------------------------

    def test_successful_execution_plan_b_balanced_cap(self) -> None:
        """Verify successful execution of Plan B: updates workload powers and returns SUCCESS."""
        sim = make_test_simulator()
        initial_powers = dict(sim.workload_powers)
        self.assertEqual(initial_powers["wl-alpha-inference"], 230.0)
        self.assertEqual(initial_powers["wl-beta-training"], 150.0)

        result = self.executor.execute(
            plan=self.plan_b,
            approval=self.approval_b,
            safety_result=self.safety_result_b,
            simulator=sim,
            workloads=self.workloads,
            incident_id=self.incident_id,
            reference_time=self.telemetry.timestamp,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.status, ExecutionStatus.SUCCESS)
        self.assertEqual(len(result.applied_actions), 2)
        self.assertTrue(result.is_simulated)
        self.assertIsNotNone(result.revalidation_result)
        self.assertTrue(result.revalidation_result.passed)

        # Workload allocations updated in simulator
        self.assertEqual(sim.workload_powers["wl-alpha-inference"], 210.0)
        self.assertEqual(sim.workload_powers["wl-beta-training"], 120.0)
        self.assertEqual(sim.effective_power_draw_w, 330.0)

    def test_successful_execution_plan_a_pause(self) -> None:
        """Verify successful execution of Plan A: caps Alpha and removes Beta (PAUSE)."""
        sim = make_test_simulator()
        safety_res_a = self.safety_gate.evaluate(self.plan_a, self.telemetry, self.workloads)
        self.assertTrue(safety_res_a.passed)

        approval_a = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id="PLAN-A",
            operator_id="operator-alex",
            decision=ApprovalDecision.APPROVE,
            timestamp=self.telemetry.timestamp + timedelta(seconds=5),
        )

        result = self.executor.execute(
            plan=self.plan_a,
            approval=approval_a,
            safety_result=safety_res_a,
            simulator=sim,
            workloads=self.workloads,
            incident_id=self.incident_id,
            reference_time=self.telemetry.timestamp,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.status, ExecutionStatus.SUCCESS)
        self.assertEqual(sim.get_workload_power("wl-alpha-inference"), 220.0)
        self.assertEqual(sim.get_workload_power("wl-beta-training"), 0.0)
        self.assertNotIn("wl-beta-training", sim.workload_powers)
        self.assertEqual(sim.effective_power_draw_w, 220.0)

    def test_successful_execution_plan_c_migrate(self) -> None:
        """Verify successful execution of Plan C: caps Alpha and removes Beta (MIGRATE)."""
        sim = make_test_simulator()
        safety_res_c = self.safety_gate.evaluate(self.plan_c, self.telemetry, self.workloads)
        self.assertTrue(safety_res_c.passed)

        approval_c = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id="PLAN-C",
            operator_id="operator-alex",
            decision=ApprovalDecision.APPROVE,
            timestamp=self.telemetry.timestamp + timedelta(seconds=8),
        )

        result = self.executor.execute(
            plan=self.plan_c,
            approval=approval_c,
            safety_result=safety_res_c,
            simulator=sim,
            workloads=self.workloads,
            incident_id=self.incident_id,
            reference_time=self.telemetry.timestamp,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.status, ExecutionStatus.SUCCESS)
        self.assertEqual(sim.get_workload_power("wl-alpha-inference"), 240.0)
        self.assertEqual(sim.get_workload_power("wl-beta-training"), 0.0)
        self.assertNotIn("wl-beta-training", sim.workload_powers)

    def test_each_supported_action_individually(self) -> None:
        """Verify explicit support for each discrete action: POWER_CAP, PAUSE, MIGRATE, RESUME."""
        sim = make_test_simulator()

        # 1. POWER_CAP
        plan_cap = RecoveryPlanOption(
            plan_id="PLAN-CAP",
            name="Power Cap Action",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=74.0,
            cooling_time_to_safe_s=15.0,
            workload_actions=[WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=205.0)],
            estimated_cost_usd=5.0,
            rationale="Test POWER_CAP action",
        )
        safe_res = self.safety_gate.evaluate(plan_cap, self.telemetry, self.workloads)
        app = ApprovalAction(incident_id=self.incident_id, selected_plan_id="PLAN-CAP", operator_id="op", decision=ApprovalDecision.APPROVE)
        res = self.executor.execute(plan_cap, app, safe_res, sim, self.workloads, reference_time=app.timestamp)
        self.assertTrue(res.success)
        self.assertEqual(sim.get_workload_power("wl-alpha-inference"), 205.0)

        # 2. PAUSE
        plan_pause = RecoveryPlanOption(
            plan_id="PLAN-PAUSE",
            name="Pause Action",
            strategy_type=StrategyType.CONSERVATIVE_SHED,
            projected_temp_max_c=70.0,
            cooling_time_to_safe_s=10.0,
            workload_actions=[WorkloadAction(workload_id="wl-beta-training", action=ActionType.PAUSE)],
            estimated_cost_usd=10.0,
            rationale="Test PAUSE action",
        )
        safe_res = self.safety_gate.evaluate(plan_pause, self.telemetry, self.workloads)
        app = ApprovalAction(incident_id=self.incident_id, selected_plan_id="PLAN-PAUSE", operator_id="op", decision=ApprovalDecision.APPROVE)
        res = self.executor.execute(plan_pause, app, safe_res, sim, self.workloads, reference_time=app.timestamp)
        self.assertTrue(res.success)
        self.assertEqual(sim.get_workload_power("wl-beta-training"), 0.0)

        # 3. RESUME
        plan_resume = RecoveryPlanOption(
            plan_id="PLAN-RESUME",
            name="Resume Action",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=74.0,
            cooling_time_to_safe_s=15.0,
            workload_actions=[WorkloadAction(workload_id="wl-beta-training", action=ActionType.RESUME, target_power_w=140.0)],
            estimated_cost_usd=0.0,
            rationale="Test RESUME action",
        )
        safe_res = self.safety_gate.evaluate(plan_resume, self.telemetry, self.workloads)
        app = ApprovalAction(incident_id=self.incident_id, selected_plan_id="PLAN-RESUME", operator_id="op", decision=ApprovalDecision.APPROVE)
        res = self.executor.execute(plan_resume, app, safe_res, sim, self.workloads, reference_time=app.timestamp)
        self.assertTrue(res.success)
        self.assertEqual(sim.get_workload_power("wl-beta-training"), 140.0)

        # 4. MIGRATE
        plan_migrate = RecoveryPlanOption(
            plan_id="PLAN-MIGRATE",
            name="Migrate Action",
            strategy_type=StrategyType.MIGRATION_OFFLOAD,
            projected_temp_max_c=70.0,
            cooling_time_to_safe_s=10.0,
            workload_actions=[WorkloadAction(workload_id="wl-beta-training", action=ActionType.MIGRATE)],
            estimated_cost_usd=15.0,
            rationale="Test MIGRATE action",
        )
        safe_res = self.safety_gate.evaluate(plan_migrate, self.telemetry, self.workloads)
        app = ApprovalAction(incident_id=self.incident_id, selected_plan_id="PLAN-MIGRATE", operator_id="op", decision=ApprovalDecision.APPROVE)
        res = self.executor.execute(plan_migrate, app, safe_res, sim, self.workloads, reference_time=app.timestamp)
        self.assertTrue(res.success)
        self.assertEqual(sim.get_workload_power("wl-beta-training"), 0.0)

    # -----------------------------------------------------------------------
    # 2. Authorization & Governance Rejection Tests
    # -----------------------------------------------------------------------

    def test_timeout_escalated_decision_rejected(self) -> None:
        """Verify TIMEOUT_ESCALATED decision is strictly rejected without mutating workloads."""
        sim = make_test_simulator()
        initial_powers = dict(sim.workload_powers)

        timeout_approval = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id=None,
            operator_id="escalation-subsystem",
            decision=ApprovalDecision.TIMEOUT_ESCALATED,
        )

        result = self.executor.execute(
            plan=self.plan_b,
            approval=timeout_approval,
            safety_result=self.safety_result_b,
            simulator=sim,
            workloads=self.workloads,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status, ExecutionStatus.REJECTED_UNAUTHORIZED)
        self.assertIn("timed out", result.rejection_reasons[0].lower())
        self.assertEqual(sim.workload_powers, initial_powers)

    def test_operator_reject_decision_rejected(self) -> None:
        """Verify explicit operator REJECT decision is strictly blocked."""
        sim = make_test_simulator()
        initial_powers = dict(sim.workload_powers)

        rejection_approval = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id=None,
            operator_id="operator-alex",
            decision=ApprovalDecision.REJECT,
        )

        result = self.executor.execute(
            plan=self.plan_b,
            approval=rejection_approval,
            safety_result=self.safety_result_b,
            simulator=sim,
            workloads=self.workloads,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status, ExecutionStatus.REJECTED_UNAUTHORIZED)
        self.assertIn("rejected", result.rejection_reasons[0].lower())
        self.assertEqual(sim.workload_powers, initial_powers)

    def test_stale_approval_preceding_incident_rejected(self) -> None:
        """Verify approval timestamp preceding reference telemetry is rejected as stale."""
        sim = make_test_simulator()
        initial_powers = dict(sim.workload_powers)

        stale_time = self.telemetry.timestamp - timedelta(minutes=5)
        stale_approval = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id="PLAN-B",
            operator_id="operator-alex",
            decision=ApprovalDecision.APPROVE,
            timestamp=stale_time,
        )

        result = self.executor.execute(
            plan=self.plan_b,
            approval=stale_approval,
            safety_result=self.safety_result_b,
            simulator=sim,
            workloads=self.workloads,
            reference_time=self.telemetry.timestamp,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status, ExecutionStatus.REJECTED_STALE_APPROVAL)
        self.assertIn("precedes reference telemetry", result.rejection_reasons[0].lower())
        self.assertEqual(sim.workload_powers, initial_powers)

    def test_stale_approval_exceeding_max_age_rejected(self) -> None:
        """Verify approval submitted past max_approval_age_s is rejected."""
        sim = make_test_simulator()
        initial_powers = dict(sim.workload_powers)

        too_old_time = self.telemetry.timestamp + timedelta(seconds=120)
        old_approval = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id="PLAN-B",
            operator_id="operator-alex",
            decision=ApprovalDecision.APPROVE,
            timestamp=too_old_time,
        )

        result = self.executor.execute(
            plan=self.plan_b,
            approval=old_approval,
            safety_result=self.safety_result_b,
            simulator=sim,
            workloads=self.workloads,
            reference_time=self.telemetry.timestamp,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status, ExecutionStatus.REJECTED_STALE_APPROVAL)
        self.assertIn("stale", result.rejection_reasons[0].lower())
        self.assertEqual(sim.workload_powers, initial_powers)

    def test_mismatched_plan_id_rejected(self) -> None:
        """Verify approval targeting a different plan ID than submitted plan is rejected."""
        sim = make_test_simulator()
        initial_powers = dict(sim.workload_powers)

        mismatched_approval = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id="PLAN-A",  # Approving A, but executing B
            operator_id="operator-alex",
            decision=ApprovalDecision.APPROVE,
            timestamp=self.telemetry.timestamp + timedelta(seconds=5),
        )

        result = self.executor.execute(
            plan=self.plan_b,
            approval=mismatched_approval,
            safety_result=self.safety_result_b,
            simulator=sim,
            workloads=self.workloads,
            reference_time=self.telemetry.timestamp,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status, ExecutionStatus.REJECTED_UNAUTHORIZED)
        self.assertIn("does not match plan id", result.rejection_reasons[0].lower())
        self.assertEqual(sim.workload_powers, initial_powers)

    def test_mismatched_incident_id_rejected(self) -> None:
        """Verify approval for a different incident ID is rejected."""
        sim = make_test_simulator()
        initial_powers = dict(sim.workload_powers)

        wrong_incident_approval = ApprovalAction(
            incident_id="INC-OTHER-999",
            selected_plan_id="PLAN-B",
            operator_id="operator-alex",
            decision=ApprovalDecision.APPROVE,
            timestamp=self.telemetry.timestamp + timedelta(seconds=5),
        )

        result = self.executor.execute(
            plan=self.plan_b,
            approval=wrong_incident_approval,
            safety_result=self.safety_result_b,
            simulator=sim,
            workloads=self.workloads,
            incident_id=self.incident_id,
            reference_time=self.telemetry.timestamp,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status, ExecutionStatus.REJECTED_UNAUTHORIZED)
        self.assertIn("does not match target incident id", result.rejection_reasons[0].lower())
        self.assertEqual(sim.workload_powers, initial_powers)

    def test_uncertified_or_failed_safety_result_rejected(self) -> None:
        """Verify that an uncertified plan (SafetyGate passed=False) cannot be executed."""
        sim = make_test_simulator()
        initial_powers = dict(sim.workload_powers)

        failed_safety_res = SafetyGateResult(
            plan_id="PLAN-B",
            passed=False,
            evaluations=[],
            rejection_reasons=["Node temperature limit violated."],
        )

        result = self.executor.execute(
            plan=self.plan_b,
            approval=self.approval_b,
            safety_result=failed_safety_res,
            simulator=sim,
            workloads=self.workloads,
            reference_time=self.telemetry.timestamp,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status, ExecutionStatus.REJECTED_SAFETY_VIOLATION)
        self.assertIn("failed safety gate certification", result.rejection_reasons[0].lower())
        self.assertEqual(sim.workload_powers, initial_powers)

    # -----------------------------------------------------------------------
    # 3. Safety Gate Revalidation & Constraint Enforcement Tests
    # -----------------------------------------------------------------------

    def test_pre_execution_safety_revalidation_failure_blocks_execution(self) -> None:
        """Verify that if environment degrades, revalidation fails immediately and blocks execution."""
        sim = make_test_simulator()
        initial_powers = dict(sim.workload_powers)

        # Force revalidation to fail by setting tight power budget constraint (200W)
        tight_constraints = NodeConstraints(power_budget_w=200.0)

        result = self.executor.execute(
            plan=self.plan_b,
            approval=self.approval_b,
            safety_result=self.safety_result_b,
            simulator=sim,
            workloads=self.workloads,
            node_constraints=tight_constraints,
            reference_time=self.telemetry.timestamp,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status, ExecutionStatus.REJECTED_SAFETY_VIOLATION)
        self.assertIn("revalidation failed", result.rejection_reasons[0].lower())
        self.assertEqual(sim.workload_powers, initial_powers)

    def test_power_budget_violation_rejected(self) -> None:
        """Verify plan that allocates total power exceeding 400W is rejected."""
        sim = make_test_simulator()
        initial_powers = dict(sim.workload_powers)

        # Plan allocating 240W + 180W = 420W (> 400W)
        excess_plan = RecoveryPlanOption(
            plan_id="PLAN-EXCESS",
            name="Excess Power Plan",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=78.0,
            cooling_time_to_safe_s=15.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=240.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=180.0),
            ],
            estimated_cost_usd=5.0,
            rationale="Allocates 420W total",
        )
        safe_res = self.safety_gate.evaluate(excess_plan, self.telemetry, self.workloads)
        # SafetyGate already marks it rejected
        self.assertFalse(safe_res.passed)

        app = ApprovalAction(incident_id=self.incident_id, selected_plan_id="PLAN-EXCESS", operator_id="op", decision=ApprovalDecision.APPROVE)
        result = self.executor.execute(excess_plan, app, safe_res, sim, self.workloads, reference_time=app.timestamp)

        self.assertFalse(result.success)
        self.assertEqual(sim.workload_powers, initial_powers)

    def test_p0_protection_pause_rejected(self) -> None:
        """Verify plan attempting to PAUSE P0 workload is strictly rejected."""
        sim = make_test_simulator()
        initial_powers = dict(sim.workload_powers)

        pause_p0_plan = RecoveryPlanOption(
            plan_id="PLAN-PAUSE-P0",
            name="Pause P0 Plan",
            strategy_type=StrategyType.CONSERVATIVE_SHED,
            projected_temp_max_c=65.0,
            cooling_time_to_safe_s=10.0,
            workload_actions=[WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.PAUSE)],
            estimated_cost_usd=500.0,
            rationale="Attempts to pause P0",
        )
        safe_res = self.safety_gate.evaluate(pause_p0_plan, self.telemetry, self.workloads)
        self.assertFalse(safe_res.passed)

        app = ApprovalAction(incident_id=self.incident_id, selected_plan_id="PLAN-PAUSE-P0", operator_id="op", decision=ApprovalDecision.APPROVE)
        result = self.executor.execute(pause_p0_plan, app, safe_res, sim, self.workloads, reference_time=app.timestamp)

        self.assertFalse(result.success)
        self.assertEqual(sim.workload_powers, initial_powers)

    def test_p0_protection_migrate_rejected(self) -> None:
        """Verify plan attempting to MIGRATE P0 non-preemptible workload is rejected."""
        sim = make_test_simulator()
        initial_powers = dict(sim.workload_powers)

        migrate_p0_plan = RecoveryPlanOption(
            plan_id="PLAN-MIGRATE-P0",
            name="Migrate P0 Plan",
            strategy_type=StrategyType.MIGRATION_OFFLOAD,
            projected_temp_max_c=65.0,
            cooling_time_to_safe_s=10.0,
            workload_actions=[WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.MIGRATE)],
            estimated_cost_usd=500.0,
            rationale="Attempts to migrate P0",
        )
        safe_res = self.safety_gate.evaluate(migrate_p0_plan, self.telemetry, self.workloads)
        self.assertFalse(safe_res.passed)

        app = ApprovalAction(incident_id=self.incident_id, selected_plan_id="PLAN-MIGRATE-P0", operator_id="op", decision=ApprovalDecision.APPROVE)
        result = self.executor.execute(migrate_p0_plan, app, safe_res, sim, self.workloads, reference_time=app.timestamp)

        self.assertFalse(result.success)
        self.assertEqual(sim.workload_powers, initial_powers)

    def test_workload_power_floor_violation_rejected(self) -> None:
        """Verify plan setting power below min_power_w is rejected."""
        sim = make_test_simulator()
        initial_powers = dict(sim.workload_powers)

        # Alpha min_power_w is 200W; attempt to cap to 150W
        floor_plan = RecoveryPlanOption(
            plan_id="PLAN-FLOOR-VIOLATION",
            name="Sub Floor Plan",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=72.0,
            cooling_time_to_safe_s=15.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=150.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=120.0),
            ],
            estimated_cost_usd=100.0,
            rationale="Cuts Alpha to 150W below floor",
        )
        safe_res = self.safety_gate.evaluate(floor_plan, self.telemetry, self.workloads)
        self.assertFalse(safe_res.passed)

        app = ApprovalAction(incident_id=self.incident_id, selected_plan_id="PLAN-FLOOR-VIOLATION", operator_id="op", decision=ApprovalDecision.APPROVE)
        result = self.executor.execute(floor_plan, app, safe_res, sim, self.workloads, reference_time=app.timestamp)

        self.assertFalse(result.success)
        self.assertEqual(sim.workload_powers, initial_powers)

    # -----------------------------------------------------------------------
    # 4. Partial Failure Rollback & Unauthorized Invariance Tests
    # -----------------------------------------------------------------------

    def test_partial_failure_rollback(self) -> None:
        """Verify atomic execution: if an action fails, all previous actions are rolled back."""
        sim = make_test_simulator()
        initial_powers = dict(sim.workload_powers)

        # Plan targets a valid workload, then an invalid/unknown workload
        partial_plan = RecoveryPlanOption(
            plan_id="PLAN-PARTIAL",
            name="Partial Failure Plan",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=74.0,
            cooling_time_to_safe_s=15.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=210.0),
                WorkloadAction(workload_id="wl-unknown-nonexistent", action=ActionType.PAUSE),
            ],
            estimated_cost_usd=0.0,
            rationale="Second action targets unknown workload",
        )
        safe_res = self.safety_gate.evaluate(partial_plan, self.telemetry, self.workloads)
        app = ApprovalAction(incident_id=self.incident_id, selected_plan_id="PLAN-PARTIAL", operator_id="op", decision=ApprovalDecision.APPROVE)

        result = self.executor.execute(partial_plan, app, safe_res, sim, self.workloads, reference_time=app.timestamp)

        self.assertFalse(result.success)
        # Crucial check: Alpha was NOT left at 210W; it remained at initial 230W
        self.assertEqual(sim.workload_powers, initial_powers)

    def test_unauthorized_execution_leaves_simulator_state_unchanged(self) -> None:
        """Prove that unauthorized execution attempts leave simulator state 100% invariant."""
        sim = make_test_simulator()

        # Record exact state
        initial_powers = dict(sim.workload_powers)
        initial_temp = sim.core_temp_c
        initial_sim_time = sim.sim_time_s
        initial_ticks = sim.tick_count

        # Attempt multiple unauthorized executions:
        # 1. Timeout
        timeout_action = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id=None,
            operator_id="esc",
            decision=ApprovalDecision.TIMEOUT_ESCALATED,
        )
        self.executor.execute(self.plan_b, timeout_action, self.safety_result_b, sim, self.workloads)

        # 2. Rejection
        reject_action = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id=None,
            operator_id="op",
            decision=ApprovalDecision.REJECT,
        )
        self.executor.execute(self.plan_b, reject_action, self.safety_result_b, sim, self.workloads)

        # 3. Mismatched plan
        mismatch_action = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id="PLAN-WRONG",
            operator_id="op",
            decision=ApprovalDecision.APPROVE,
        )
        self.executor.execute(self.plan_b, mismatch_action, self.safety_result_b, sim, self.workloads)

        # Assert zero change to simulator
        self.assertEqual(sim.workload_powers, initial_powers)
        self.assertEqual(sim.core_temp_c, initial_temp)
        self.assertEqual(sim.sim_time_s, initial_sim_time)
        self.assertEqual(sim.tick_count, initial_ticks)

    # -----------------------------------------------------------------------
    # 5. Workflow Result & Error Handling Tests
    # -----------------------------------------------------------------------

    def test_execute_from_approval_workflow_result_authorized(self) -> None:
        """Verify executing directly from an authorized ApprovalWorkflowResult."""
        sim = make_test_simulator()
        wf_result = ApprovalWorkflowResult(
            incident_id=self.incident_id,
            status=WorkflowStatus.APPROVED,
            decision=ApprovalDecision.APPROVE,
            selected_plan=self.plan_b,
            is_authorized_for_execution=True,
            revalidation_result=self.safety_result_b,
            action=self.approval_b,
        )

        res = self.executor.execute_workflow_result(wf_result, sim, self.workloads)
        self.assertTrue(res.success)
        self.assertEqual(sim.workload_powers["wl-alpha-inference"], 210.0)
        self.assertEqual(sim.workload_powers["wl-beta-training"], 120.0)

    def test_execute_from_approval_workflow_result_unauthorized_rejected(self) -> None:
        """Verify un-authorized ApprovalWorkflowResult is rejected safely."""
        sim = make_test_simulator()
        initial_powers = dict(sim.workload_powers)

        wf_result = ApprovalWorkflowResult(
            incident_id=self.incident_id,
            status=WorkflowStatus.TIMEOUT_ESCALATED,
            decision=ApprovalDecision.TIMEOUT_ESCALATED,
            selected_plan=None,
            is_authorized_for_execution=False,
            rejection_reasons=["Approval timed out."],
        )

        res = self.executor.execute_workflow_result(wf_result, sim, self.workloads)
        self.assertFalse(res.success)
        self.assertEqual(res.status, ExecutionStatus.REJECTED_UNAUTHORIZED)
        self.assertEqual(sim.workload_powers, initial_powers)

    def test_raise_on_failure_flag_raises_exception(self) -> None:
        """Verify raise_on_failure=True raises ExecutionDeniedError (inheriting from PermissionError)."""
        sim = make_test_simulator()
        timeout_action = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id=None,
            operator_id="esc",
            decision=ApprovalDecision.TIMEOUT_ESCALATED,
        )

        with self.assertRaises(ExecutionDeniedError) as ctx:
            self.executor.execute(
                self.plan_b,
                timeout_action,
                self.safety_result_b,
                sim,
                self.workloads,
                raise_on_failure=True,
            )

        self.assertIsInstance(ctx.exception, PermissionError)
        self.assertIn("denied", str(ctx.exception).lower())

    def test_isolation_never_controls_real_hardware(self) -> None:
        """Verify executor strictly refuses to operate on non-TelemetrySimulator targets."""
        with self.assertRaises(TypeError) as ctx:
            self.executor.execute(
                self.plan_b,
                self.approval_b,
                self.safety_result_b,
                simulator=object(),  # Non-simulator object
                workloads=self.workloads,
            )

        self.assertIn("telemetrysimulator", str(ctx.exception).lower())
        self.assertIn("hardware", str(ctx.exception).lower())

    def test_execute_and_verify_convenience_method(self) -> None:
        """Verify execute_and_verify returns execution result and verification report."""
        sim = make_test_simulator()
        safety_res = self.safety_gate.evaluate(self.plan_a, self.telemetry, self.workloads)
        approval = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id="PLAN-A",
            operator_id="operator-alex",
            decision=ApprovalDecision.APPROVE,
            timestamp=self.telemetry.timestamp + timedelta(seconds=5),
        )

        exec_res, report = self.executor.execute_and_verify(
            plan=self.plan_a,
            approval=approval,
            safety_result=safety_res,
            simulator=sim,
            workloads=self.workloads,
            incident_id=self.incident_id,
            reference_time=self.telemetry.timestamp,
            max_ticks=20,
        )

        self.assertTrue(exec_res.success)
        self.assertIsNotNone(report)
        self.assertTrue(report.recovery_successful)
        self.assertTrue(report.is_simulated)

        # When unauthorized, returns (exec_res, None)
        timeout_approval = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id=None,
            operator_id="esc",
            decision=ApprovalDecision.TIMEOUT_ESCALATED,
        )
        fail_exec, fail_report = self.executor.execute_and_verify(
            plan=self.plan_a,
            approval=timeout_approval,
            safety_result=safety_res,
            simulator=sim,
            workloads=self.workloads,
        )
        self.assertFalse(fail_exec.success)
        self.assertIsNone(fail_report)


if __name__ == "__main__":
    unittest.main()
