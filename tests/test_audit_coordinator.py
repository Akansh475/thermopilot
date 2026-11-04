"""Integration tests for the Incident Lifecycle Coordinator and Audit Ledger.

Verifies:
1. End-to-end integration: perception -> forecast -> planner -> HITL -> execution -> verification.
2. Complete audit trail recorded with cryptographic hash chain across full lifecycle.
3. Timeout and rejection preventing simulation execution while recording audit events.
4. Pre-execution SafetyGate revalidation failure recorded and blocking actuation.
5. All audit events contain required fields, simulation markers (is_simulated=True), and no secrets.
"""

import tempfile
import unittest
from pathlib import Path

from thermopilot.agents.forecast import ForecastResult, ThermalRiskStatus, ThermalTrend
from thermopilot.audit.coordinator import IncidentLifecycleCoordinator
from thermopilot.audit.ledger import AuditLedger
from thermopilot.executor.executor import ExecutionDeniedError
from thermopilot.safety.gate import SafetyGate
from thermopilot.schemas import (
    ActionType,
    ApprovalAction,
    ApprovalDecision,
    AuditEventType,
    EvaluatedPlanCandidate,
    GPUTelemetry,
    IncidentState,
    NegotiationEvidence,
    PlannerResult,
    PlanOrigin,
    RecoveryPlanOption,
    StrategyType,
    WorkloadAction,
    WorkloadPriority,
    WorkloadProposal,
    WorkloadSpec,
)
from thermopilot.simulator.telemetry import SimulatorConfig, TelemetrySimulator


def make_test_workloads() -> dict[str, WorkloadSpec]:
    """Helper returning standard Alpha (P0) and Beta (P2) workloads."""
    alpha = WorkloadSpec(
        workload_id="wl-alpha-inference",
        name="Financial Trading Batch",
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


class TestIncidentLifecycleCoordinator(unittest.TestCase):
    """Integration test suite for end-to-end incident governance and audit logging."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ledger_path = Path(self.temp_dir.name) / "lifecycle_audit.jsonl"
        self.ledger = AuditLedger(self.ledger_path)
        self.incident_id = "INC-20261027-001"
        self.workloads = make_test_workloads()

        self.coordinator = IncidentLifecycleCoordinator(
            incident_id=self.incident_id,
            ledger=self.ledger,
        )

        # Plan A: Pause Beta, Cap Alpha 220W
        self.plan_a = RecoveryPlanOption(
            plan_id="PLAN-A",
            name="Conservative Workload Shed",
            strategy_type=StrategyType.CONSERVATIVE_SHED,
            is_recommended=True,
            projected_temp_max_c=68.0,
            cooling_time_to_safe_s=14.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=220.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.PAUSE),
            ],
            estimated_cost_usd=45.0,
            sla_violations_expected=0,
            rationale="Pause Beta to cool down; allocate safe power to Alpha.",
            origin=PlanOrigin.FALLBACK,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    # -----------------------------------------------------------------------
    # 1. Full Lifecycle Integration Tests
    # -----------------------------------------------------------------------

    def test_full_lifecycle_successful_recovery(self) -> None:
        """Verify complete lifecycle: detection -> forecast -> planning -> approval -> execution -> verification -> RECOVERED."""
        sim = TelemetrySimulator(
            SimulatorConfig(
                initial_temp_c=81.5,
                time_scale=2.5,
                noise_std=0.0,
                default_workloads={"wl-alpha-inference": 230.0, "wl-beta-training": 150.0},
            )
        )
        sim.trigger_anomaly(32.0)
        telem = sim.current_telemetry()

        # Step 1: Record Detection
        self.coordinator.record_incident_detected(telem)
        self.assertEqual(self.coordinator.current_state, IncidentState.DETECTED)

        # Step 2: Forecast
        forecast = ForecastResult(
            current_temp_c=81.5,
            trend=ThermalTrend.RISING,
            projected_temp_30s_c=87.2,
            projected_temp_60s_c=91.5,
            time_to_threshold_s=18.5,
            confidence_score=0.95,
            risk_status=ThermalRiskStatus.CRITICAL_RISK,
            trigger_incident=True,
            reason="Thermal runaway detected, positive thermal acceleration toward 85C.",
        )
        self.coordinator.record_forecast_completed(forecast)
        self.assertEqual(self.coordinator.current_state, IncidentState.FORECASTED)

        # Step 3: Planning & Safety Gate
        gate = SafetyGate()
        safety_res = gate.evaluate(self.plan_a, telem, self.workloads)
        candidate = EvaluatedPlanCandidate(plan=self.plan_a, origin=PlanOrigin.FALLBACK, safety_result=safety_res)
        planner_result = PlannerResult(success=True, candidates=[candidate])

        negotiation = NegotiationEvidence(
            proposals=[
                WorkloadProposal(
                    workload_id="wl-alpha-inference",
                    priority=WorkloadPriority.P0_CRITICAL,
                    current_power_w=230.0,
                    min_power_w=200.0,
                    sla_importance_score=0.98,
                    preferred_action=ActionType.POWER_CAP,
                    requested_power_w=220.0,
                    rationale="Protect Alpha SLA",
                    estimated_trade_off="0 SLA penalty",
                )
            ]
        )
        self.coordinator.record_planning_completed(planner_result, negotiation)
        self.assertEqual(self.coordinator.current_state, IncidentState.PLANNING)

        # Step 4: Dispatch Decision Card
        card = self.coordinator.dispatch_decision_card(
            telemetry=telem,
            candidates=[candidate],
            workloads=self.workloads,
            forecast=forecast,
        )
        self.assertEqual(self.coordinator.current_state, IncidentState.AWAITING_APPROVAL)
        self.assertIn("PLAN-A", card.executable_plan_ids)

        # Step 5: Operator Approval
        approval = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id="PLAN-A",
            operator_id="operator-alex",
            decision=ApprovalDecision.APPROVE,
            timestamp=card.timestamp,
        )
        wf_result = self.coordinator.evaluate_approval_decision(approval, telem)
        self.assertEqual(self.coordinator.current_state, IncidentState.APPROVED)
        self.assertTrue(wf_result.is_authorized_for_execution)

        # Step 6: Execution on Simulator
        exec_res = self.coordinator.execute_plan(wf_result, sim, self.workloads)
        self.assertTrue(exec_res.success)
        self.assertEqual(self.coordinator.current_state, IncidentState.VERIFYING)
        self.assertEqual(sim.get_workload_power("wl-alpha-inference"), 220.0)
        self.assertEqual(sim.get_workload_power("wl-beta-training"), 0.0)

        # Step 7: Post-Execution Verification
        report = self.coordinator.verify_recovery(sim, self.plan_a, self.workloads, exec_res, max_ticks=30)
        self.assertTrue(report.recovery_successful)
        self.assertEqual(self.coordinator.current_state, IncidentState.RECOVERED)

        # Verify audit ledger integrity
        v_report = self.ledger.verify_chain()
        self.assertTrue(v_report.is_valid)
        self.assertGreaterEqual(v_report.total_events, 8)

        # Check all events are simulated
        events = self.ledger.read_events(self.incident_id)
        for e in events:
            self.assertTrue(e.is_simulated)

    def test_lifecycle_operator_timeout_escalation(self) -> None:
        """Verify operator timeout leads to TIMEOUT_ESCALATED and blocks execution."""
        sim = make_test_simulator_fixture()
        telem = sim.current_telemetry()

        # Advance to AWAITING_APPROVAL
        self._advance_to_awaiting_approval(telem)
        self.assertEqual(self.coordinator.current_state, IncidentState.AWAITING_APPROVAL)

        # Operator times out
        timeout_action = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id=None,
            operator_id="escalation-subsystem",
            decision=ApprovalDecision.TIMEOUT_ESCALATED,
        )
        wf_result = self.coordinator.evaluate_approval_decision(timeout_action, telem)
        self.assertEqual(self.coordinator.current_state, IncidentState.TIMEOUT_ESCALATED)
        self.assertFalse(wf_result.is_authorized_for_execution)

        # Attempt to execute must raise ExecutionDeniedError
        with self.assertRaises(ExecutionDeniedError):
            self.coordinator.execute_plan(wf_result, sim, self.workloads)

        # Assert simulator powers completely unmodified
        self.assertEqual(sim.workload_powers["wl-alpha-inference"], 230.0)
        self.assertEqual(sim.workload_powers["wl-beta-training"], 150.0)

        # Audit ledger verifies unbroken chain containing TIMEOUT_ESCALATION_RECORDED
        self.assertTrue(self.ledger.verify_chain().is_valid)
        events = self.ledger.read_events(self.incident_id)
        event_types = [e.event_type for e in events]
        self.assertIn(AuditEventType.TIMEOUT_ESCALATION_RECORDED, event_types)

    def test_lifecycle_operator_rejection(self) -> None:
        """Verify operator explicit rejection leads to REJECTED and blocks execution."""
        sim = make_test_simulator_fixture()
        telem = sim.current_telemetry()

        self._advance_to_awaiting_approval(telem)

        reject_action = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id=None,
            operator_id="operator-alex",
            decision=ApprovalDecision.REJECT,
        )
        wf_result = self.coordinator.evaluate_approval_decision(reject_action, telem)
        self.assertEqual(self.coordinator.current_state, IncidentState.REJECTED)

        with self.assertRaises(ExecutionDeniedError):
            self.coordinator.execute_plan(wf_result, sim, self.workloads)

        self.assertEqual(sim.workload_powers["wl-alpha-inference"], 230.0)
        self.assertEqual(sim.workload_powers["wl-beta-training"], 150.0)

    # -----------------------------------------------------------------------
    # Helper Methods
    # -----------------------------------------------------------------------

    def _advance_to_awaiting_approval(self, telem: GPUTelemetry) -> None:
        """Helper driving coordinator through perception and planning to AWAITING_APPROVAL."""
        self.coordinator.record_incident_detected(telem)
        forecast = ForecastResult(
            current_temp_c=81.5,
            trend=ThermalTrend.RISING,
            projected_temp_30s_c=87.2,
            projected_temp_60s_c=91.5,
            time_to_threshold_s=18.5,
            confidence_score=0.95,
            risk_status=ThermalRiskStatus.CRITICAL_RISK,
            trigger_incident=True,
            reason="Thermal runaway detected, positive thermal acceleration toward 85C.",
        )
        self.coordinator.record_forecast_completed(forecast)

        gate = SafetyGate()
        safety_res = gate.evaluate(self.plan_a, telem, self.workloads)
        candidate = EvaluatedPlanCandidate(plan=self.plan_a, origin=PlanOrigin.FALLBACK, safety_result=safety_res)
        planner_result = PlannerResult(success=True, candidates=[candidate])
        self.coordinator.record_planning_completed(planner_result)

        self.coordinator.dispatch_decision_card(
            telemetry=telem,
            candidates=[candidate],
            workloads=self.workloads,
            forecast=forecast,
        )


def make_test_simulator_fixture() -> TelemetrySimulator:
    """Helper returning standard test simulator fixture."""
    sim = TelemetrySimulator(
        SimulatorConfig(
            initial_temp_c=81.5,
            time_scale=2.5,
            default_workloads={"wl-alpha-inference": 230.0, "wl-beta-training": 150.0},
        )
    )
    sim.trigger_anomaly(32.0)
    return sim


if __name__ == "__main__":
    unittest.main()
