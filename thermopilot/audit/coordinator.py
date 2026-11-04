"""Incident Lifecycle Coordinator for ThermoPilot Compute Desk.

Orchestrates the complete thermal governance lifecycle by seamlessly connecting:
- IncidentStateMachine (deterministic lifecycle states & transitions)
- AuditLedger (append-only cryptographic hash chain event ledger)
- SwitchApprovalDesk (decision card dispatch & approval validation)
- SimulationExecutor (fail-safe simulator plan actuation)
- VerificationAgent (post-execution thermal decay verification)

Strictly simulation-only: never connects to real GPUs or cloud infrastructure.
"""

from collections.abc import Mapping, Sequence

from thermopilot.agents.forecast import ForecastResult
from thermopilot.audit.ledger import AuditLedger
from thermopilot.audit.state_machine import IncidentStateMachine
from thermopilot.executor.executor import ExecutionDeniedError, SimulationExecutor
from thermopilot.executor.verifier import VerificationAgent
from thermopilot.hitl.switch import SwitchApprovalDesk
from thermopilot.schemas import (
    ApprovalAction,
    ApprovalWorkflowResult,
    AuditEventType,
    DecisionCard,
    EvaluatedPlanCandidate,
    ExecutionResult,
    GPUTelemetry,
    IncidentState,
    NegotiationEvidence,
    NodeConstraints,
    PlannerResult,
    RecoveryPlanOption,
    VerificationReport,
    WorkflowStatus,
    WorkloadSpec,
)
from thermopilot.simulator.telemetry import TelemetrySimulator


class IncidentLifecycleCoordinator:
    """Coordinates perception, planning, governance, actuation, and audit logging for an incident."""

    def __init__(
        self,
        incident_id: str,
        ledger: AuditLedger,
        state_machine: IncidentStateMachine | None = None,
        desk: SwitchApprovalDesk | None = None,
        executor: SimulationExecutor | None = None,
        verifier: VerificationAgent | None = None,
    ) -> None:
        """Initialize IncidentLifecycleCoordinator.

        Args:
            incident_id: Active thermal incident ID.
            ledger: Cryptographic AuditLedger instance.
            state_machine: State machine instance (created if None).
            desk: Switch approval desk (created if None).
            executor: Simulation executor (created if None).
            verifier: Verification agent (created if None).
        """
        self.incident_id = incident_id
        self.ledger = ledger
        self.state_machine = state_machine or IncidentStateMachine(
            incident_id=incident_id,
            initial_state=IncidentState.DETECTED,
            ledger=ledger,
        )
        self.desk = desk or SwitchApprovalDesk()
        self.executor = executor or SimulationExecutor()
        self.verifier = verifier or VerificationAgent()

    @property
    def current_state(self) -> IncidentState:
        """Current incident state in state machine."""
        return self.state_machine.current_state

    # -----------------------------------------------------------------------
    # 1. Perception & Forecasting
    # -----------------------------------------------------------------------

    def record_incident_detected(self, telemetry: GPUTelemetry) -> None:
        """Record initial incident trigger in audit ledger."""
        self.ledger.append(
            incident_id=self.incident_id,
            event_type=AuditEventType.INCIDENT_TRIGGERED,
            payload={
                "core_temp_c": telemetry.temperature_core_c,
                "power_draw_w": telemetry.power_draw_w,
                "ambient_temp_c": telemetry.ambient_temp_c,
                "gpu_id": telemetry.gpu_id,
                "timestamp": telemetry.timestamp.isoformat(),
            },
            is_simulated=True,
        )

    def record_forecast_completed(self, forecast: ForecastResult) -> None:
        """Transition from DETECTED to FORECASTED and log forecast trajectory."""
        self.state_machine.transition(
            to_state=IncidentState.FORECASTED,
            reason=f"Thermal forecast computed: t_TTT={forecast.time_to_threshold_s}s, trend={forecast.trend.value}.",
            payload={
                "time_to_threshold_s": forecast.time_to_threshold_s,
                "trend": forecast.trend.value,
                "risk_status": forecast.risk_status.value,
                "trigger_incident": forecast.trigger_incident,
            },
        )
        self.ledger.append(
            incident_id=self.incident_id,
            event_type=AuditEventType.FORECAST_COMPLETED,
            payload={
                "time_to_threshold_s": forecast.time_to_threshold_s,
                "trend": forecast.trend.value,
                "risk_status": forecast.risk_status.value,
                "rate_c_per_s": forecast.rate_c_per_s,
                "reason": forecast.reason,
            },
            is_simulated=True,
        )

    # -----------------------------------------------------------------------
    # 2. Planning & Safety Evaluation
    # -----------------------------------------------------------------------

    def record_planning_completed(
        self,
        planner_result: PlannerResult,
        negotiation: NegotiationEvidence | None = None,
    ) -> None:
        """Transition from FORECASTED to PLANNING and log proposals and candidates."""
        self.state_machine.transition(
            to_state=IncidentState.PLANNING,
            reason=f"Planning complete: {len(planner_result.candidates)} candidates evaluated.",
        )

        # Log workload proposals if present
        if negotiation is not None:
            self.ledger.append(
                incident_id=self.incident_id,
                event_type=AuditEventType.WORKLOAD_BIDS_COLLECTED,
                payload={
                    "proposals": [p.model_dump(mode="json") for p in negotiation.proposals]
                },
                is_simulated=True,
            )

        # Log planner candidate generation
        self.ledger.append(
            incident_id=self.incident_id,
            event_type=AuditEventType.PLANNER_PLANS_GENERATED,
            payload={
                "success": planner_result.success,
                "plan_ids": [c.plan.plan_id for c in planner_result.candidates],
                "safe_plan_ids": [c.plan.plan_id for c in planner_result.safe_candidates],
            },
            is_simulated=True,
        )

        # Log SafetyGate evaluations for each candidate
        for c in planner_result.candidates:
            self.ledger.append(
                incident_id=self.incident_id,
                event_type=AuditEventType.SAFETY_GATE_EVALUATED,
                payload={
                    "plan_id": c.plan.plan_id,
                    "passed": c.safety_result.passed,
                    "rejection_reasons": c.safety_result.rejection_reasons,
                    "evaluations": [e.model_dump(mode="json") for e in c.safety_result.evaluations],
                },
                is_simulated=True,
            )

    # -----------------------------------------------------------------------
    # 3. Decision Desk & Governance
    # -----------------------------------------------------------------------

    def dispatch_decision_card(
        self,
        telemetry: GPUTelemetry,
        candidates: Sequence[EvaluatedPlanCandidate],
        workloads: Sequence[WorkloadSpec] | Mapping[str, WorkloadSpec],
        forecast: ForecastResult | None = None,
        node_constraints: NodeConstraints | None = None,
        timeout_s: float | None = None,
    ) -> DecisionCard:
        """Transition to AWAITING_APPROVAL and dispatch decision card to Switch desk."""
        card = self.desk.create_and_dispatch_card(
            incident_id=self.incident_id,
            telemetry=telemetry,
            candidates=candidates,
            workloads=workloads,
            forecast=forecast,
            node_constraints=node_constraints,
            timeout_s=timeout_s,
        )

        self.state_machine.transition(
            to_state=IncidentState.AWAITING_APPROVAL,
            reason="Decision card dispatched to on-call operator via Switch.",
            payload={
                "executable_plans": card.executable_plan_ids,
                "rejected_plans": card.rejected_plan_ids,
                "timeout_seconds": card.timeout_seconds,
            },
        )

        self.ledger.append(
            incident_id=self.incident_id,
            event_type=AuditEventType.DECISION_CARD_DISPATCHED,
            payload={
                "incident_id": card.incident_id,
                "executable_plans": card.executable_plan_ids,
                "rejected_plans": card.rejected_plan_ids,
                "severity": card.severity,
            },
            is_simulated=True,
        )
        return card

    def evaluate_approval_decision(
        self,
        approval: ApprovalAction,
        current_telemetry: GPUTelemetry,
        node_constraints: NodeConstraints | None = None,
    ) -> ApprovalWorkflowResult:
        """Process operator decision, revalidate safety, and transition state accordingly."""
        wf_result = self.desk.evaluate_approval(
            approval=approval,
            incident_id=self.incident_id,
            current_telemetry=current_telemetry,
            node_constraints=node_constraints,
        )

        if wf_result.status == WorkflowStatus.APPROVED:
            # Transition to APPROVED
            self.state_machine.transition(
                to_state=IncidentState.APPROVED,
                plan_id=approval.selected_plan_id,
                reason=f"Operator approved plan '{approval.selected_plan_id}' with pre-execution safety certification.",
                payload={
                    "operator_id": approval.operator_id,
                    "selected_plan_id": approval.selected_plan_id,
                    "decision": approval.decision.value,
                },
                is_authorized=True,
                revalidation_passed=True,
            )
            # Log revalidation report
            self.ledger.append(
                incident_id=self.incident_id,
                event_type=AuditEventType.PRE_EXECUTION_REVALIDATED,
                payload={
                    "plan_id": approval.selected_plan_id,
                    "passed": True,
                    "revalidation": (
                        wf_result.revalidation_result.model_dump(mode="json")
                        if wf_result.revalidation_result
                        else {}
                    ),
                },
                is_simulated=True,
            )

        elif wf_result.status == WorkflowStatus.TIMEOUT_ESCALATED:
            self.state_machine.transition(
                to_state=IncidentState.TIMEOUT_ESCALATED,
                reason="Operator approval window timed out without response. Workload mutations blocked.",
                payload={"operator_id": approval.operator_id},
            )

        elif wf_result.status == WorkflowStatus.REJECTED:
            self.state_machine.transition(
                to_state=IncidentState.REJECTED,
                reason="Operator explicitly rejected all recovery plans.",
                payload={"operator_id": approval.operator_id},
            )

        elif wf_result.status == WorkflowStatus.REVALIDATION_FAILED:
            # Pre-execution revalidation failed
            self.state_machine.transition(
                to_state=IncidentState.APPROVED,
                plan_id=approval.selected_plan_id,
                reason="Operator approved plan.",
            )
            self.state_machine.transition(
                to_state=IncidentState.REVALIDATION_FAILED,
                plan_id=approval.selected_plan_id,
                reason=f"Pre-execution revalidation failed: {'; '.join(wf_result.rejection_reasons)}.",
            )
            self.ledger.append(
                incident_id=self.incident_id,
                event_type=AuditEventType.PRE_EXECUTION_REVALIDATED,
                payload={
                    "plan_id": approval.selected_plan_id,
                    "passed": False,
                    "rejection_reasons": wf_result.rejection_reasons,
                },
                is_simulated=True,
            )

        return wf_result

    # -----------------------------------------------------------------------
    # 4. Execution & Actuation
    # -----------------------------------------------------------------------

    def execute_plan(
        self,
        workflow_result: ApprovalWorkflowResult,
        simulator: TelemetrySimulator,
        workloads: Sequence[WorkloadSpec] | Mapping[str, WorkloadSpec],
        node_constraints: NodeConstraints | None = None,
    ) -> ExecutionResult:
        """Transition to EXECUTING and apply approved plan to simulator."""
        if not workflow_result.is_authorized_for_execution or workflow_result.selected_plan is None:
            self.ledger.append(
                incident_id=self.incident_id,
                event_type=AuditEventType.EXECUTION_DENIED,
                payload={
                    "status": workflow_result.status.value,
                    "rejection_reasons": workflow_result.rejection_reasons,
                },
                is_simulated=True,
            )
            raise ExecutionDeniedError(
                f"Execution denied: approval status is {workflow_result.status.value}."
            )

        selected_plan = workflow_result.selected_plan

        # Transition state machine to EXECUTING (audit written BEFORE simulator action)
        self.state_machine.transition(
            to_state=IncidentState.EXECUTING,
            plan_id=selected_plan.plan_id,
            reason=f"Applying plan '{selected_plan.plan_id}' actions to simulation environment.",
            is_authorized=True,
            revalidation_passed=True,
        )

        # Apply plan to simulator
        exec_res = self.executor.execute_workflow_result(
            workflow_result=workflow_result,
            simulator=simulator,
            workloads=workloads,
            node_constraints=node_constraints,
        )

        if exec_res.success:
            self.ledger.append(
                incident_id=self.incident_id,
                event_type=AuditEventType.PLAN_EXECUTED,
                payload={
                    "execution_id": exec_res.execution_id,
                    "plan_id": selected_plan.plan_id,
                    "applied_actions": [a.model_dump(mode="json") for a in exec_res.applied_actions],
                    "total_node_power_w": exec_res.details.get("total_node_power_w"),
                },
                is_simulated=True,
            )
            # Advance to VERIFYING
            self.state_machine.transition(
                to_state=IncidentState.VERIFYING,
                plan_id=selected_plan.plan_id,
                reason="Plan executed successfully; beginning post-execution thermal verification.",
            )
        else:
            self.ledger.append(
                incident_id=self.incident_id,
                event_type=AuditEventType.EXECUTION_DENIED,
                payload={
                    "plan_id": selected_plan.plan_id,
                    "rejection_reasons": exec_res.rejection_reasons,
                },
                is_simulated=True,
            )
            self.state_machine.transition(
                to_state=IncidentState.RECOVERY_FAILED,
                plan_id=selected_plan.plan_id,
                reason=f"Simulation actuation failed: {'; '.join(exec_res.rejection_reasons)}.",
            )

        return exec_res

    # -----------------------------------------------------------------------
    # 5. Verification & Final Outcome
    # -----------------------------------------------------------------------

    def verify_recovery(
        self,
        simulator: TelemetrySimulator,
        plan: RecoveryPlanOption,
        workloads: Sequence[WorkloadSpec] | Mapping[str, WorkloadSpec],
        execution_result: ExecutionResult | None = None,
        max_ticks: int = 30,
        node_constraints: NodeConstraints | None = None,
    ) -> VerificationReport:
        """Run post-execution verification and transition to RECOVERED or RECOVERY_FAILED."""
        report = self.verifier.verify(
            simulator=simulator,
            plan=plan,
            workloads=workloads,
            execution_result=execution_result,
            incident_id=self.incident_id,
            max_ticks=max_ticks,
            node_constraints=node_constraints,
        )

        self.ledger.append(
            incident_id=self.incident_id,
            event_type=AuditEventType.VERIFICATION_COMPLETED,
            payload={
                "verification_id": report.verification_id,
                "recovery_successful": report.recovery_successful,
                "initial_temp_c": report.initial_temp_c,
                "final_temp_c": report.final_temp_c,
                "observed_cooling_rate_c_per_s": report.observed_cooling_rate_c_per_s,
                "reached_target": report.reached_target,
                "ticks_to_target": report.ticks_to_target,
                "failure_reasons": report.failure_reasons,
            },
            is_simulated=True,
        )

        if report.recovery_successful:
            self.state_machine.transition(
                to_state=IncidentState.RECOVERED,
                plan_id=plan.plan_id,
                reason="Thermal envelope successfully restored (<= 75°C) with verified SLA compliance.",
            )
        else:
            self.state_machine.transition(
                to_state=IncidentState.RECOVERY_FAILED,
                plan_id=plan.plan_id,
                reason=f"Recovery failed: {'; '.join(report.failure_reasons)}.",
            )

        return report
