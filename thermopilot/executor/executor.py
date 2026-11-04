"""Simulation Executor for ThermoPilot Compute Desk.

Applies human-approved, safety-certified recovery plans to the discrete-time
telemetry simulator. Operates under strict fail-safe governance:

GOVERNANCE & SAFETY GUARANTEES:
1. Accepts only explicitly approved (ApprovalDecision.APPROVE), certified plans.
2. Rejects timeouts (TIMEOUT_ESCALATED), operator rejections (REJECT), stale
   approvals, mismatched incidents/plans, and uncertified candidates.
3. Revalidates safety immediately before execution using SafetyGate.
4. Enforces 400W node power budget, 80°C projected steady-state temperature ceiling,
   minimum workload power floors, and P0 workload non-preemption SLA protections.
5. Atomic execution: if any validation or action fails, the simulator state is
   completely unmodified or restored to its exact pre-execution state.
6. ISOLATION: Strictly interacts with in-memory TelemetrySimulator instances.
   Never executes commands on real GPUs, production workloads, or cloud infrastructure.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, ClassVar

from thermopilot.safety.gate import SafetyGate
from thermopilot.schemas import (
    ActionType,
    ApprovalAction,
    ApprovalDecision,
    ApprovalWorkflowResult,
    ExecutionResult,
    ExecutionStatus,
    GPUTelemetry,
    NodeConstraints,
    RecoveryPlanOption,
    SafetyGateResult,
    WorkloadAction,
    WorkloadPriority,
    WorkloadSpec,
)
from thermopilot.simulator.telemetry import TelemetrySimulator

# ---------------------------------------------------------------------------
# 1. Custom Exceptions
# ---------------------------------------------------------------------------

class ExecutorError(Exception):
    """Base exception for all executor subsystem errors."""


class ExecutionDeniedError(ExecutorError, PermissionError):
    """Raised when execution is rejected due to unauthorized, rejected, stale, or unsafe plans."""


class ExecutionValidationError(ExecutorError, ValueError):
    """Raised when plan, workload, or action specifications are malformed or invalid."""


class ExecutionSafetyError(ExecutorError):
    """Raised when pre-execution safety revalidation fails."""


# ---------------------------------------------------------------------------
# 2. Simulation Executor
# ---------------------------------------------------------------------------

class SimulationExecutor:
    """Deterministic, safety-gated actuator applying recovery plans to the simulator."""

    SUPPORTED_ACTIONS: ClassVar[set[ActionType]] = {
        ActionType.POWER_CAP,
        ActionType.PAUSE,
        ActionType.MIGRATE,
        ActionType.RESUME,
    }

    def __init__(
        self,
        safety_gate: SafetyGate | None = None,
        node_constraints: NodeConstraints | None = None,
        max_approval_age_s: float = 60.0,
    ) -> None:
        """Initialize SimulationExecutor.

        Args:
            safety_gate: Deterministic SafetyGate for pre-execution revalidation.
            node_constraints: Node power and thermal physical boundary constraints.
            max_approval_age_s: Maximum allowable approval age before rejection as stale.
        """
        self.safety_gate = safety_gate or SafetyGate()
        self.node_constraints = node_constraints or NodeConstraints()
        self.max_approval_age_s = max_approval_age_s

    @staticmethod
    def _normalize_workloads(
        workloads: Sequence[WorkloadSpec] | Mapping[str, WorkloadSpec] | Any,
    ) -> dict[str, WorkloadSpec]:
        """Normalize workload specifications into a mapping keyed by workload_id."""
        workload_map: dict[str, WorkloadSpec] = {}
        if isinstance(workloads, Mapping):
            for wid, item in workloads.items():
                if isinstance(item, WorkloadSpec):
                    workload_map[item.workload_id] = item
                else:
                    raise ExecutionValidationError(
                        f"Workload mapping contains invalid item for '{wid}': expected WorkloadSpec, got {type(item).__name__}."
                    )
        elif isinstance(workloads, (list, tuple, set)):
            for item in workloads:
                if isinstance(item, WorkloadSpec):
                    workload_map[item.workload_id] = item
                else:
                    raise ExecutionValidationError(
                        f"Workload list contains invalid item: expected WorkloadSpec, got {type(item).__name__}."
                    )
        else:
            raise ExecutionValidationError(
                f"Invalid workloads input: expected Sequence or Mapping of WorkloadSpec, got {type(workloads).__name__}."
            )

        if not workload_map:
            raise ExecutionValidationError("Workload specifications mapping cannot be empty.")
        return workload_map

    def execute(
        self,
        plan: RecoveryPlanOption,
        approval: ApprovalAction,
        safety_result: SafetyGateResult,
        simulator: TelemetrySimulator,
        workloads: Sequence[WorkloadSpec] | Mapping[str, WorkloadSpec],
        incident_id: str | None = None,
        node_constraints: NodeConstraints | None = None,
        reference_time: datetime | None = None,
        raise_on_failure: bool = False,
    ) -> ExecutionResult:
        """Validate authorization, revalidate safety, and apply plan actions to the simulator.

        Strictly guarantees:
        - If any check fails, the simulator state is 100% unchanged.
        - Only explicitly approved (APPROVE) plans pass.
        - Pre-execution revalidation is performed immediately before actuator updates.
        - Total power budget and temperature limits are enforced.

        Args:
            plan: Candidate RecoveryPlanOption to apply.
            approval: Operator ApprovalAction authorizing execution.
            safety_result: SafetyGate certification result for the plan.
            simulator: TelemetrySimulator instance to update.
            workloads: Active workload specifications.
            incident_id: Optional incident ID check. If provided, must match approval.
            node_constraints: Optional node constraint overrides.
            reference_time: Optional reference timestamp for staleness checking.
            raise_on_failure: If True, raises ExecutionDeniedError on rejection.

        Returns:
            ExecutionResult detailing success status, applied actions, and telemetry.

        Raises:
            ExecutionDeniedError: If raise_on_failure is True and execution is rejected.
            TypeError: If simulator is not a TelemetrySimulator instance.
        """
        # Strict Isolation Check: never control real hardware
        if not isinstance(simulator, TelemetrySimulator):
            raise TypeError(
                f"SimulationExecutor strictly operates ONLY on TelemetrySimulator instances. "
                f"Received invalid executor target of type '{type(simulator).__name__}'. "
                f"Direct hardware actuation is strictly prohibited."
            )

        active_constraints = node_constraints or self.node_constraints
        target_incident_id = incident_id or (approval.incident_id if approval else "UNKNOWN-INCIDENT")
        plan_id = plan.plan_id if plan else "UNKNOWN-PLAN"

        # Snapshot current simulator state before any evaluation or action
        initial_telemetry: GPUTelemetry = simulator.current_telemetry()
        initial_workload_powers: dict[str, float] = dict(simulator.workload_powers)

        def _fail(
            status: ExecutionStatus,
            reasons: list[str],
            reval: SafetyGateResult | None = None,
        ) -> ExecutionResult:
            # Ensure simulator workload powers remain untouched or rolled back
            self._rollback_simulator(simulator, initial_workload_powers)
            result = ExecutionResult(
                incident_id=target_incident_id,
                plan_id=plan_id,
                success=False,
                status=status,
                applied_actions=[],
                initial_telemetry=initial_telemetry,
                post_execution_telemetry=simulator.current_telemetry(),
                revalidation_result=reval,
                rejection_reasons=reasons,
                is_simulated=True,
                details={"rejection_count": len(reasons)},
            )
            if raise_on_failure:
                raise ExecutionDeniedError(
                    f"Plan execution denied ({status.value}): {'; '.join(reasons)}"
                )
            return result

        # -------------------------------------------------------------------
        # 1. Validate Input Types
        # -------------------------------------------------------------------
        if plan is None or not isinstance(plan, RecoveryPlanOption):
            return _fail(
                ExecutionStatus.REJECTED_INVALID_PLAN,
                [f"Invalid plan input: expected RecoveryPlanOption, got {type(plan).__name__}."],
            )

        if approval is None or not isinstance(approval, ApprovalAction):
            return _fail(
                ExecutionStatus.REJECTED_UNAUTHORIZED,
                [f"Invalid approval input: expected ApprovalAction, got {type(approval).__name__}."],
            )

        if safety_result is None or not isinstance(safety_result, SafetyGateResult):
            return _fail(
                ExecutionStatus.REJECTED_SAFETY_VIOLATION,
                [f"Invalid safety result input: expected SafetyGateResult, got {type(safety_result).__name__}."],
            )

        try:
            workload_map = self._normalize_workloads(workloads)
        except ExecutionValidationError as exc:
            return _fail(ExecutionStatus.REJECTED_INVALID_PLAN, [str(exc)])

        # -------------------------------------------------------------------
        # 2. Verify Approval Authorization & Binding
        # -------------------------------------------------------------------
        # Check explicit decision
        if approval.decision == ApprovalDecision.TIMEOUT_ESCALATED:
            return _fail(
                ExecutionStatus.REJECTED_UNAUTHORIZED,
                ["Approval timed out (TIMEOUT_ESCALATED). Workload mutations strictly blocked."],
            )

        if approval.decision == ApprovalDecision.REJECT:
            return _fail(
                ExecutionStatus.REJECTED_UNAUTHORIZED,
                ["Plan execution was explicitly rejected by the operator."],
            )

        if approval.decision != ApprovalDecision.APPROVE:
            return _fail(
                ExecutionStatus.REJECTED_UNAUTHORIZED,
                [f"Approval decision is not APPROVE (decision: {approval.decision.value})."],
            )

        # Check incident ID match
        if incident_id is not None and approval.incident_id != incident_id:
            return _fail(
                ExecutionStatus.REJECTED_UNAUTHORIZED,
                [f"Approval incident ID '{approval.incident_id}' does not match target incident ID '{incident_id}'."],
            )

        # Check plan ID match
        if approval.selected_plan_id != plan.plan_id:
            return _fail(
                ExecutionStatus.REJECTED_UNAUTHORIZED,
                [f"Approval selected plan ID '{approval.selected_plan_id}' does not match plan ID '{plan.plan_id}'."],
            )

        # Check prior safety gate result certification
        if not safety_result.passed:
            return _fail(
                ExecutionStatus.REJECTED_SAFETY_VIOLATION,
                [f"Plan '{plan.plan_id}' failed Safety Gate certification: {'; '.join(safety_result.rejection_reasons)}."],
                reval=safety_result,
            )

        if safety_result.plan_id != plan.plan_id:
            return _fail(
                ExecutionStatus.REJECTED_SAFETY_VIOLATION,
                [f"SafetyGateResult plan ID '{safety_result.plan_id}' does not match plan '{plan.plan_id}'."],
                reval=safety_result,
            )

        # -------------------------------------------------------------------
        # 3. Check Approval Freshness (Staleness Protection)
        # -------------------------------------------------------------------
        ref_dt = reference_time or initial_telemetry.timestamp
        time_diff = (approval.timestamp - ref_dt).total_seconds()
        if time_diff < -1.0:
            return _fail(
                ExecutionStatus.REJECTED_STALE_APPROVAL,
                [
                    (
                        f"Approval timestamp ({approval.timestamp.isoformat()}) precedes reference telemetry timestamp "
                        f"({ref_dt.isoformat()}). Stale approval rejected."
                    )
                ],
            )
        if time_diff > self.max_approval_age_s:
            return _fail(
                ExecutionStatus.REJECTED_STALE_APPROVAL,
                [
                    (
                        f"Approval is stale: submitted {time_diff:.1f}s after incident reference time "
                        f"(maximum allowable age: {self.max_approval_age_s:.1f}s)."
                    )
                ],
            )

        # -------------------------------------------------------------------
        # 4. Immediate Pre-Execution Safety Revalidation
        # -------------------------------------------------------------------
        reval_result = self.safety_gate.evaluate(
            plan=plan,
            telemetry=initial_telemetry,
            workloads=workload_map,
            node_constraints=active_constraints,
        )

        if not reval_result.passed:
            return _fail(
                ExecutionStatus.REJECTED_SAFETY_VIOLATION,
                [
                    f"Pre-execution Safety Gate revalidation failed: {'; '.join(reval_result.rejection_reasons)}"
                ],
                reval=reval_result,
            )

        # -------------------------------------------------------------------
        # 5. Validate Action Payloads & Guardrails
        # -------------------------------------------------------------------
        action_rejections: list[str] = []
        if not plan.workload_actions:
            action_rejections.append("Plan contains no workload actions to execute.")

        for act in plan.workload_actions:
            if not isinstance(act, WorkloadAction):
                action_rejections.append(f"Invalid action item type: {type(act).__name__}.")
                continue

            if act.action not in self.SUPPORTED_ACTIONS:
                action_rejections.append(
                    f"Action type '{act.action.value}' is not supported by SimulationExecutor."
                )
                continue

            spec = workload_map.get(act.workload_id)
            if spec is None:
                action_rejections.append(
                    f"Action targets unknown workload '{act.workload_id}' not present in active workloads."
                )
                continue

            # Check POWER_CAP action constraints
            if act.action == ActionType.POWER_CAP:
                if not spec.can_power_cap:
                    action_rejections.append(
                        f"Workload '{spec.workload_id}' does not allow power capping (can_power_cap=False)."
                    )
                if act.target_power_w is None or act.target_power_w <= 0.0:
                    action_rejections.append(
                        f"Workload '{spec.workload_id}' target_power_w must be positive, got {act.target_power_w}."
                    )
                else:
                    if act.target_power_w < spec.min_power_w:
                        action_rejections.append(
                            f"Workload '{spec.workload_id}' target power ({act.target_power_w:.1f}W) violates "
                            f"minimum operational floor ({spec.min_power_w:.1f}W)."
                        )
                    if act.target_power_w > spec.max_power_w:
                        action_rejections.append(
                            f"Workload '{spec.workload_id}' target power ({act.target_power_w:.1f}W) exceeds "
                            f"maximum capacity ({spec.max_power_w:.1f}W)."
                        )

            # Check PAUSE / MIGRATE constraints (P0 protection)
            elif act.action in (ActionType.PAUSE, ActionType.MIGRATE):
                if spec.priority == WorkloadPriority.P0_CRITICAL or not spec.can_preempt:
                    action_rejections.append(
                        f"P0_CRITICAL / non-preemptible workload '{spec.workload_id}' cannot be preempted "
                        f"(action: {act.action.value})."
                    )

        # Enforce projected steady-state temperature ceiling (<= 80.0°C)
        if plan.projected_temp_max_c > active_constraints.max_temp_c:
            action_rejections.append(
                f"Plan projected maximum temperature ({plan.projected_temp_max_c:.1f}°C) exceeds "
                f"safety ceiling ({active_constraints.max_temp_c:.1f}°C)."
            )

        # Enforce projected total node power budget (<= 400.0W)
        projected_powers: dict[str, float] = {
            wid: simulator.get_workload_power(wid) for wid in workload_map
        }
        for act in plan.workload_actions:
            if act.action == ActionType.POWER_CAP and act.target_power_w is not None:
                projected_powers[act.workload_id] = act.target_power_w
            elif act.action in (ActionType.PAUSE, ActionType.MIGRATE):
                projected_powers[act.workload_id] = 0.0
            elif act.action == ActionType.RESUME:
                spec = workload_map.get(act.workload_id)
                projected_powers[act.workload_id] = (
                    act.target_power_w if act.target_power_w is not None else (spec.current_power_w if spec else 0.0)
                )

        total_projected_power = sum(projected_powers.values())
        if total_projected_power > active_constraints.power_budget_w:
            action_rejections.append(
                f"Total projected node power ({total_projected_power:.1f}W) exceeds "
                f"power budget ({active_constraints.power_budget_w:.1f}W)."
            )

        if action_rejections:
            return _fail(
                ExecutionStatus.REJECTED_INVALID_PLAN,
                action_rejections,
                reval=reval_result,
            )

        # -------------------------------------------------------------------
        # 6. Apply Actions Atomically to Simulator
        # -------------------------------------------------------------------
        applied_actions: list[WorkloadAction] = []
        try:
            for act in plan.workload_actions:
                if act.action == ActionType.POWER_CAP and act.target_power_w is not None:
                    simulator.set_workload_power(act.workload_id, act.target_power_w, clamp=False)
                    applied_actions.append(act)
                elif act.action in (ActionType.PAUSE, ActionType.MIGRATE):
                    simulator.remove_workload(act.workload_id)
                    applied_actions.append(act)
                elif act.action == ActionType.RESUME:
                    target_p = act.target_power_w or workload_map[act.workload_id].current_power_w
                    simulator.set_workload_power(act.workload_id, target_p, clamp=False)
                    applied_actions.append(act)
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            # Partial failure rollback: restore initial state completely
            self._rollback_simulator(simulator, initial_workload_powers)
            return _fail(
                ExecutionStatus.FAILED_EXECUTION,
                [f"Actuation failure during plan execution: {exc}. Simulator state rolled back."],
                reval=reval_result,
            )

        # Record post-execution telemetry snapshot
        post_execution_telemetry: GPUTelemetry = simulator.current_telemetry()

        return ExecutionResult(
            incident_id=target_incident_id,
            plan_id=plan.plan_id,
            success=True,
            status=ExecutionStatus.SUCCESS,
            timestamp=datetime.now(timezone.utc),
            applied_actions=applied_actions,
            initial_telemetry=initial_telemetry,
            post_execution_telemetry=post_execution_telemetry,
            revalidation_result=reval_result,
            rejection_reasons=[],
            is_simulated=True,
            details={
                "workload_powers": dict(simulator.workload_powers),
                "total_node_power_w": simulator.effective_power_draw_w,
                "strategy_type": plan.strategy_type.value,
            },
        )

    def execute_workflow_result(
        self,
        workflow_result: ApprovalWorkflowResult,
        simulator: TelemetrySimulator,
        workloads: Sequence[WorkloadSpec] | Mapping[str, WorkloadSpec],
        node_constraints: NodeConstraints | None = None,
        reference_time: datetime | None = None,
        raise_on_failure: bool = False,
    ) -> ExecutionResult:
        """Execute recovery plan directly from an ApprovalWorkflowResult."""
        if workflow_result is None or not isinstance(workflow_result, ApprovalWorkflowResult):
            reasons = [f"Invalid workflow result input: expected ApprovalWorkflowResult, got {type(workflow_result).__name__}."]
            if raise_on_failure:
                raise ExecutionDeniedError(reasons[0])
            return ExecutionResult(
                incident_id="UNKNOWN",
                plan_id="UNKNOWN",
                success=False,
                status=ExecutionStatus.REJECTED_UNAUTHORIZED,
                rejection_reasons=reasons,
                is_simulated=True,
            )

        if not workflow_result.is_authorized_for_execution or workflow_result.selected_plan is None:
            reasons = list(workflow_result.rejection_reasons) or [
                f"Approval workflow status is '{workflow_result.status.value}'. Execution not authorized."
            ]
            if raise_on_failure:
                raise ExecutionDeniedError(f"Workflow result not authorized: {'; '.join(reasons)}")
            return ExecutionResult(
                incident_id=workflow_result.incident_id,
                plan_id=getattr(workflow_result.selected_plan, "plan_id", "UNKNOWN"),
                success=False,
                status=ExecutionStatus.REJECTED_UNAUTHORIZED,
                rejection_reasons=reasons,
                is_simulated=True,
            )

        if workflow_result.action is None:
            reasons = ["ApprovalWorkflowResult missing required operator action."]
            if raise_on_failure:
                raise ExecutionDeniedError(reasons[0])
            return ExecutionResult(
                incident_id=workflow_result.incident_id,
                plan_id=workflow_result.selected_plan.plan_id,
                success=False,
                status=ExecutionStatus.REJECTED_UNAUTHORIZED,
                rejection_reasons=reasons,
                is_simulated=True,
            )

        # Delegate to primary execute method
        safety_res = workflow_result.revalidation_result or SafetyGateResult(
            plan_id=workflow_result.selected_plan.plan_id,
            passed=True,
            evaluations=[],
            rejection_reasons=[],
        )

        return self.execute(
            plan=workflow_result.selected_plan,
            approval=workflow_result.action,
            safety_result=safety_res,
            simulator=simulator,
            workloads=workloads,
            incident_id=workflow_result.incident_id,
            node_constraints=node_constraints,
            reference_time=reference_time or (workflow_result.action.timestamp if workflow_result.action else None),
            raise_on_failure=raise_on_failure,
        )

    def execute_and_verify(
        self,
        plan: RecoveryPlanOption,
        approval: ApprovalAction,
        safety_result: SafetyGateResult,
        simulator: TelemetrySimulator,
        workloads: Sequence[WorkloadSpec] | Mapping[str, WorkloadSpec],
        incident_id: str | None = None,
        node_constraints: NodeConstraints | None = None,
        reference_time: datetime | None = None,
        verifier: Any | None = None,
        max_ticks: int = 30,
        stop_on_target: bool = False,
        raise_on_failure: bool = False,
    ) -> tuple[ExecutionResult, Any | None]:
        """Execute plan on simulator and immediately run post-execution verification.

        If execution fails or is unauthorized, returns (exec_result, None) without advancing.
        """
        exec_result = self.execute(
            plan=plan,
            approval=approval,
            safety_result=safety_result,
            simulator=simulator,
            workloads=workloads,
            incident_id=incident_id,
            node_constraints=node_constraints,
            reference_time=reference_time or (approval.timestamp if approval else None),
            raise_on_failure=raise_on_failure,
        )

        if exec_result.success:
            from thermopilot.executor.verifier import VerificationAgent

            active_verifier = verifier or VerificationAgent()
            report = active_verifier.verify(
                simulator=simulator,
                plan=plan,
                workloads=workloads,
                execution_result=exec_result,
                incident_id=incident_id,
                max_ticks=max_ticks,
                node_constraints=node_constraints,
                stop_on_target=stop_on_target,
            )
            return exec_result, report

        return exec_result, None

    @staticmethod
    def _rollback_simulator(
        simulator: TelemetrySimulator,
        initial_powers: dict[str, float],
    ) -> None:
        """Rollback simulator workloads to the exact pre-execution state."""
        current_wids = list(simulator.workload_powers.keys())
        for wid in current_wids:
            if wid not in initial_powers:
                simulator.remove_workload(wid)

        for wid, p in initial_powers.items():
            simulator.set_workload_power(wid, p, clamp=False)
