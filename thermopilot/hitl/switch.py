"""Switch Human-in-the-Loop (HITL) approval desk and governance subsystem.

Implements the operator decision interface, candidate card presentation,
fail-safe timeout escalation, approval validation, pre-execution Safety Gate
revalidation, and simulator isolation.
"""

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from thermopilot.agents.forecast import ForecastResult
from thermopilot.safety.gate import SafetyGate
from thermopilot.schemas import (
    ActionType,
    ApprovalAction,
    ApprovalDecision,
    ApprovalWorkflowResult,
    CandidatePlanSummary,
    DecisionCard,
    EvaluatedPlanCandidate,
    GPUTelemetry,
    NodeConstraints,
    RecoveryPlanOption,
    WorkflowStatus,
    WorkloadSpec,
)
from thermopilot.simulator.telemetry import TelemetrySimulator

# ---------------------------------------------------------------------------
# 1. Exceptions
# ---------------------------------------------------------------------------

class SwitchError(Exception):
    """Base exception for all Switch HITL governance and integration errors."""


class SwitchConfigurationError(SwitchError):
    """Raised when Switch client configuration (endpoint, credentials) is missing or unverified."""


class SwitchTimeoutError(SwitchError):
    """Raised when operator approval window expires without explicit authorization."""


class InvalidApprovalError(SwitchError):
    """Raised when an operator submission is malformed, stale, duplicate, or targets an uncertified plan."""


# ---------------------------------------------------------------------------
# 2. Client Interfaces & Implementations
# ---------------------------------------------------------------------------

class SwitchClient(ABC):
    """Abstract interface contract for Switch Human-in-the-Loop clients."""

    @abstractmethod
    def dispatch_decision_card(self, card: DecisionCard) -> bool:
        """Present or dispatch the interactive decision card to the operator interface.

        Args:
            card: Validated DecisionCard containing certified and rejected candidates.

        Returns:
            True if the card was successfully dispatched or displayed.
        """

    @abstractmethod
    def listen_for_decision(self, incident_id: str, timeout_s: float = 15.0) -> ApprovalAction:
        """Wait for human operator sign-off or return a safe timeout escalation.

        Args:
            incident_id: Identifier of the active incident.
            timeout_s: Maximum seconds to wait before automatic timeout escalation.

        Returns:
            Validated ApprovalAction from the operator or TIMEOUT_ESCALATED.
        """


class MockConsoleSwitchClient(SwitchClient):
    """Local development and CLI fallback client for testing and demo interaction.

    Renders human-readable formatted decision tables and accepts pre-queued
    actions, simulated timeouts, or fallback decisions.
    """

    def __init__(self, render_to_console: bool = False) -> None:
        """Initialize mock console client.

        Args:
            render_to_console: Whether to print formatted card text to standard output.
        """
        self.render_to_console = render_to_console
        self.dispatched_cards: list[DecisionCard] = []
        self._queued_actions: dict[str, ApprovalAction] = {}
        self._simulate_timeout_incident_ids: set[str] = set()

    def queue_approval(self, action: ApprovalAction) -> None:
        """Queue an approval action for a specific incident."""
        self._queued_actions[action.incident_id] = action

    def simulate_timeout(self, incident_id: str) -> None:
        """Designate an incident to trigger a timeout escalation upon listening."""
        self._simulate_timeout_incident_ids.add(incident_id)

    def clear(self) -> None:
        """Reset all queued actions, timeouts, and dispatched cards."""
        self.dispatched_cards.clear()
        self._queued_actions.clear()
        self._simulate_timeout_incident_ids.clear()

    @staticmethod
    def format_card_text(card: DecisionCard) -> str:
        """Format a DecisionCard into structured console text with visual markers."""
        lines: list[str] = [
            "================================================================================",
            f" [SWITCH DECISION DESK] INCIDENT: {card.incident_id} | SEVERITY: {card.severity}",
            "================================================================================",
            f" Telemetry Snapshot: Core Temp: {card.current_temp_c:.1f}°C | Power: {card.current_power_w:.1f}W",
            f" Hardware Safety: Trip Limit: {card.critical_trip_temp_c:.1f}°C | "
            + (f"Time-to-Trip (t_TTT): {card.time_to_trip_s:.1f}s" if card.time_to_trip_s is not None else "Trend: Nominal"),
            f" Governance Timeout: {card.timeout_seconds:.1f}s (Strict policy: Timeout never mutates workloads)",
            "--------------------------------------------------------------------------------",
            " CANDIDATE RECOVERY STRATEGIES:",
            "--------------------------------------------------------------------------------",
        ]

        for plan in card.candidate_plans:
            rec_tag = " [RECOMMENDED]" if plan.is_recommended else ""
            if plan.is_safe:
                status_header = f"  >>> {plan.plan_id}: {plan.name}{rec_tag} [CERTIFIED SAFE]"
                lines.append(status_header)
                lines.append(
                    f"      Strategy: {plan.strategy_type.value} | Projected Temp: {plan.projected_temp_max_c:.1f}°C | "
                    f"Power: {plan.projected_power_w:.1f}W | Cost: ${plan.estimated_cost_usd:.2f}"
                )
                lines.append(f"      Cooldown: {plan.cooling_time_to_safe_s:.1f}s | Expected SLA Breaches: {plan.sla_violations_expected}")
                actions_str = ", ".join(
                    f"{a.workload_id}->{a.action.value}" + (f"({a.target_power_w}W)" if a.target_power_w is not None else "")
                    for a in plan.proposed_actions
                )
                lines.append(f"      Actions: {actions_str}")
                lines.append(f"      Rationale: {plan.rationale}")
                lines.append(f"      [Action Available: Approve {plan.plan_id}]")
            else:
                status_header = f"  XXX {plan.plan_id}: {plan.name} [REJECTED BY SAFETY GATE - NON-EXECUTABLE]"
                lines.append(status_header)
                lines.append(f"      Violations: {'; '.join(plan.rejection_reasons)}")
                lines.append("      [Execution Button: DISABLED - SAFETY VIOLATION]")
            lines.append("  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")

        lines.extend([
            " Operator Controls:",
            f"   Executable Options: {', '.join(card.executable_plan_ids) if card.executable_plan_ids else 'None'}",
            f"   Blocked Options:    {', '.join(card.rejected_plan_ids) if card.rejected_plan_ids else 'None'}",
            "   Hotkeys: [1] Approve Plan A  [2] Approve Plan B  [3] Approve Plan C  [R] Reject  [E] Escalate",
            "================================================================================",
        ])
        return "\n".join(lines)

    def dispatch_decision_card(self, card: DecisionCard) -> bool:
        """Store dispatched card and optionally print to console."""
        self.dispatched_cards.append(card)
        if self.render_to_console:
            print(self.format_card_text(card))  # pragma: no cover
        return True

    def listen_for_decision(self, incident_id: str, timeout_s: float = 15.0) -> ApprovalAction:
        """Return queued operator decision or default fail-safe timeout escalation."""
        if incident_id in self._simulate_timeout_incident_ids:
            return ApprovalAction(
                incident_id=incident_id,
                selected_plan_id=None,
                operator_id="escalation-subsystem",
                decision=ApprovalDecision.TIMEOUT_ESCALATED,
                timestamp=datetime.now(timezone.utc),
                notes=f"Simulated operator response timeout after {timeout_s:.1f}s window.",
            )

        if incident_id in self._queued_actions:
            return self._queued_actions.pop(incident_id)

        # Default fail-safe behavior: unattended cards trigger timeout escalation
        return ApprovalAction(
            incident_id=incident_id,
            selected_plan_id=None,
            operator_id="escalation-subsystem",
            decision=ApprovalDecision.TIMEOUT_ESCALATED,
            timestamp=datetime.now(timezone.utc),
            notes=f"No operator response received within {timeout_s:.1f}s window. Automatic timeout escalation.",
        )


class WebhookSwitchClientConfig:
    """Configuration settings for connecting to a remote Switch webhook endpoint."""

    def __init__(
        self,
        webhook_url: str | None = None,
        api_key: str | None = None,
        callback_url: str | None = None,
        timeout_s: float = 15.0,
    ) -> None:
        self.webhook_url = webhook_url
        self.api_key = api_key
        self.callback_url = callback_url
        self.timeout_s = timeout_s


class WebhookSwitchClient(SwitchClient):
    """Adapter for remote Switch webhook integration.

    When live credentials and endpoints are unverified, fails controlled and provides
    clear diagnostic errors rather than inventing endpoints or credentials.
    """

    def __init__(
        self,
        config: WebhookSwitchClientConfig | None = None,
        http_poster: Any | None = None,
    ) -> None:
        """Initialize WebhookSwitchClient.

        Args:
            config: Optional endpoint configuration.
            http_poster: Optional mockable HTTP poster callable for testing: (url, data, headers) -> response.
        """
        self.config = config or WebhookSwitchClientConfig()
        self._http_poster = http_poster

    @property
    def is_configured(self) -> bool:
        """Whether the client possesses a non-empty destination webhook URL."""
        return bool(self.config.webhook_url and self.config.webhook_url.strip())

    def dispatch_decision_card(self, card: DecisionCard) -> bool:
        """Post DecisionCard payload to configured webhook URL.

        Raises:
            SwitchConfigurationError: If webhook_url is missing or unverified.
        """
        if not self.is_configured:
            raise SwitchConfigurationError(
                "WebhookSwitchClient is unconfigured: live Switch endpoint and credentials "
                "are not provided in the environment or project configuration."
            )

        payload = card.model_dump(mode="json")
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ThermoPilot-SwitchClient/0.1.0",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        if self._http_poster is not None:
            return bool(self._http_poster(self.config.webhook_url, data, headers))

        req = urllib.request.Request(
            url=str(self.config.webhook_url),
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as response:
                return response.status in (200, 201, 202)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise SwitchError(f"Failed to dispatch card to Switch webhook: {exc}") from exc

    def listen_for_decision(self, incident_id: str, timeout_s: float = 15.0) -> ApprovalAction:
        """Listen for decision callback from Switch.

        Raises:
            SwitchConfigurationError: If webhook_url is missing or unverified.
        """
        if not self.is_configured:
            raise SwitchConfigurationError(
                "WebhookSwitchClient is unconfigured: cannot listen for live Switch callbacks."
            )

        # In production, this awaits an incoming webhook or polls the resolution gateway.
        # As documented in docs/architecture.md, Switch callback paths are unverified.
        # Fail safe: return timeout escalation without workload mutation.
        return ApprovalAction(
            incident_id=incident_id,
            selected_plan_id=None,
            operator_id="escalation-subsystem",
            decision=ApprovalDecision.TIMEOUT_ESCALATED,
            timestamp=datetime.now(timezone.utc),
            notes=f"Live Switch webhook listener timed out after {timeout_s:.1f}s.",
        )


# ---------------------------------------------------------------------------
# 3. Approval Workflow Orchestrator
# ---------------------------------------------------------------------------

class SwitchApprovalDesk:
    """Orchestrator for Human-in-the-Loop decision governance via Switch.

    Responsibilities:
    1. Builds decision cards displaying candidate plans with physical and SLA trade-offs.
    2. Strips execution availability from rejected plans (e.g. Plan D).
    3. Dispatches cards via the configured Switch client.
    4. Binds approvals to incident ID, plan ID, and certified gate evaluations.
    5. Rejects stale, mismatched, uncertified, or duplicate approvals.
    6. Enforces fail-safe timeout policy: timeout NEVER mutates workloads.
    7. Revalidates selected plan against the deterministic Safety Gate immediately before execution.
    """

    def __init__(
        self,
        safety_gate: SafetyGate | None = None,
        switch_client: SwitchClient | None = None,
        default_timeout_s: float = 15.0,
        max_approval_age_s: float = 60.0,
    ) -> None:
        """Initialize SwitchApprovalDesk.

        Args:
            safety_gate: Deterministic SafetyGate instance for verification and revalidation.
            switch_client: SwitchClient implementation (defaults to MockConsoleSwitchClient).
            default_timeout_s: Default governance window in seconds before timeout escalation.
            max_approval_age_s: Maximum allowable age in seconds of an operator approval timestamp.
        """
        self.safety_gate = safety_gate or SafetyGate()
        self.switch_client = switch_client or MockConsoleSwitchClient()
        self.default_timeout_s = default_timeout_s
        self.max_approval_age_s = max_approval_age_s

        # In-memory governance state
        self._processed_incidents: set[str] = set()
        self._dispatched_cards: dict[str, DecisionCard] = {}
        self._incident_candidates: dict[str, list[EvaluatedPlanCandidate]] = {}
        self._incident_workloads: dict[str, list[WorkloadSpec]] = {}
        self._incident_constraints: dict[str, NodeConstraints] = {}

    @staticmethod
    def calculate_projected_power(
        plan: RecoveryPlanOption,
        workloads: Sequence[WorkloadSpec] | Mapping[str, WorkloadSpec],
    ) -> float:
        """Calculate total node electrical power draw in Watts under a proposed recovery plan."""
        workload_map: dict[str, WorkloadSpec] = (
            dict(workloads) if isinstance(workloads, Mapping) else {w.workload_id: w for w in workloads}
        )
        allocated_powers: dict[str, float] = {w_id: w.current_power_w for w_id, w in workload_map.items()}

        for action in plan.workload_actions:
            if action.workload_id in allocated_powers:
                if action.action == ActionType.POWER_CAP and action.target_power_w is not None:
                    allocated_powers[action.workload_id] = action.target_power_w
                elif action.action in (ActionType.PAUSE, ActionType.MIGRATE):
                    allocated_powers[action.workload_id] = 0.0

        return sum(allocated_powers.values())

    def build_decision_card(
        self,
        incident_id: str,
        telemetry: GPUTelemetry,
        candidates: Sequence[EvaluatedPlanCandidate],
        workloads: Sequence[WorkloadSpec] | Mapping[str, WorkloadSpec],
        forecast: ForecastResult | None = None,
        timeout_s: float | None = None,
    ) -> DecisionCard:
        """Construct a validated DecisionCard summarizing candidate plans and safety verdicts."""
        summaries: list[CandidatePlanSummary] = []
        executable_plan_ids: list[str] = []
        rejected_plan_ids: list[str] = []

        for candidate in candidates:
            plan = candidate.plan
            is_safe = candidate.is_safe
            projected_power = self.calculate_projected_power(plan, workloads)

            summary = CandidatePlanSummary(
                plan_id=plan.plan_id,
                name=plan.name,
                strategy_type=plan.strategy_type,
                is_recommended=plan.is_recommended,
                is_safe=is_safe,
                safety_verdict="CERTIFIED_SAFE" if is_safe else "REJECTED_BY_SAFETY_GATE",
                rejection_reasons=list(candidate.safety_result.rejection_reasons) if not is_safe else [],
                projected_temp_max_c=plan.projected_temp_max_c,
                projected_power_w=projected_power,
                cooling_time_to_safe_s=plan.cooling_time_to_safe_s,
                estimated_cost_usd=plan.estimated_cost_usd,
                sla_violations_expected=plan.sla_violations_expected,
                proposed_actions=list(plan.workload_actions),
                is_executable=is_safe,
                rationale=plan.rationale,
            )
            summaries.append(summary)

            if is_safe:
                executable_plan_ids.append(plan.plan_id)
            else:
                rejected_plan_ids.append(plan.plan_id)

        severity = "CRITICAL" if ((forecast and forecast.trigger_incident) or telemetry.temperature_core_c >= 80.0) else "WARNING"
        time_to_trip = forecast.time_to_threshold_s if forecast is not None else None

        return DecisionCard(
            incident_id=incident_id,
            timestamp=datetime.now(timezone.utc),
            gpu_id=telemetry.gpu_id,
            severity=severity,
            current_temp_c=telemetry.temperature_core_c,
            current_power_w=telemetry.power_draw_w,
            critical_trip_temp_c=85.0,
            time_to_trip_s=time_to_trip,
            candidate_plans=summaries,
            executable_plan_ids=executable_plan_ids,
            rejected_plan_ids=rejected_plan_ids,
            timeout_seconds=timeout_s or self.default_timeout_s,
        )

    def create_and_dispatch_card(
        self,
        incident_id: str,
        telemetry: GPUTelemetry,
        candidates: Sequence[EvaluatedPlanCandidate],
        workloads: Sequence[WorkloadSpec] | Mapping[str, WorkloadSpec],
        forecast: ForecastResult | None = None,
        node_constraints: NodeConstraints | None = None,
        timeout_s: float | None = None,
    ) -> DecisionCard:
        """Build and dispatch the decision card to the Switch interface."""
        if not incident_id or not incident_id.strip():
            raise ValueError("incident_id must be a non-empty string.")
        if not candidates:
            raise ValueError("candidates list cannot be empty.")

        card = self.build_decision_card(
            incident_id=incident_id,
            telemetry=telemetry,
            candidates=candidates,
            workloads=workloads,
            forecast=forecast,
            timeout_s=timeout_s,
        )

        # Store incident governance context
        workload_list = list(workloads.values()) if isinstance(workloads, Mapping) else list(workloads)
        self._dispatched_cards[incident_id] = card
        self._incident_candidates[incident_id] = list(candidates)
        self._incident_workloads[incident_id] = workload_list
        self._incident_constraints[incident_id] = node_constraints or NodeConstraints()

        self.switch_client.dispatch_decision_card(card)
        return card

    def evaluate_approval(
        self,
        approval: ApprovalAction | None,
        incident_id: str,
        current_telemetry: GPUTelemetry,
        node_constraints: NodeConstraints | None = None,
    ) -> ApprovalWorkflowResult:
        """Evaluate an operator approval submission and perform pre-execution Safety Gate revalidation.

        Enforces:
        - Missing or None approval is rejected.
        - Duplicate approvals for an already-finalized incident are rejected.
        - Mismatched incident ID is rejected.
        - Stale approval timestamps are rejected.
        - Submissions approving uncertified or rejected plans are rejected.
        - TIMEOUT_ESCALATED and REJECT decisions NEVER authorize execution.
        - Pre-execution Safety Gate revalidation: evaluates the selected plan against current telemetry
          immediately before authorization. If revalidation fails, execution is strictly blocked.
        """
        # 1. Reject missing submission
        if approval is None:
            return ApprovalWorkflowResult(
                incident_id=incident_id,
                status=WorkflowStatus.INVALID_SUBMISSION,
                is_authorized_for_execution=False,
                rejection_reasons=["Approval action is missing (None)."],
            )

        # 2. Reject duplicate / replay
        if incident_id in self._processed_incidents:
            return ApprovalWorkflowResult(
                incident_id=incident_id,
                status=WorkflowStatus.INVALID_SUBMISSION,
                is_authorized_for_execution=False,
                rejection_reasons=[
                    f"Incident '{incident_id}' has already been processed. Duplicate approval rejected."
                ],
                action=approval,
            )

        # 3. Reject mismatched incident ID
        if approval.incident_id != incident_id:
            return ApprovalWorkflowResult(
                incident_id=incident_id,
                status=WorkflowStatus.INVALID_SUBMISSION,
                is_authorized_for_execution=False,
                rejection_reasons=[
                    f"Approval incident ID '{approval.incident_id}' does not match active incident '{incident_id}'."
                ],
                action=approval,
            )

        # 4. Check if decision card exists
        card = self._dispatched_cards.get(incident_id)
        if card is None:
            return ApprovalWorkflowResult(
                incident_id=incident_id,
                status=WorkflowStatus.INVALID_SUBMISSION,
                is_authorized_for_execution=False,
                rejection_reasons=[f"No active decision card found for incident '{incident_id}'."],
                action=approval,
            )

        # 5. Check for stale approval
        time_diff = (approval.timestamp - card.timestamp).total_seconds()
        if time_diff < -1.0:
            return ApprovalWorkflowResult(
                incident_id=incident_id,
                status=WorkflowStatus.INVALID_SUBMISSION,
                is_authorized_for_execution=False,
                rejection_reasons=[
                    f"Approval timestamp ({approval.timestamp.isoformat()}) precedes incident card dispatch timestamp ({card.timestamp.isoformat()}). Stale/invalid approval rejected."
                ],
                action=approval,
            )
        if time_diff > self.max_approval_age_s:
            return ApprovalWorkflowResult(
                incident_id=incident_id,
                status=WorkflowStatus.INVALID_SUBMISSION,
                is_authorized_for_execution=False,
                rejection_reasons=[
                    f"Approval is stale: submitted {time_diff:.1f}s after incident card dispatch (maximum allowable: {self.max_approval_age_s:.1f}s)."
                ],
                action=approval,
            )

        # 6. Handle TIMEOUT_ESCALATED
        if approval.decision == ApprovalDecision.TIMEOUT_ESCALATED:
            self._processed_incidents.add(incident_id)
            return ApprovalWorkflowResult(
                incident_id=incident_id,
                status=WorkflowStatus.TIMEOUT_ESCALATED,
                decision=ApprovalDecision.TIMEOUT_ESCALATED,
                selected_plan=None,
                is_authorized_for_execution=False,
                rejection_reasons=[
                    "Approval window timed out (TIMEOUT_ESCALATED). Workload mutations strictly blocked."
                ],
                action=approval,
            )

        # 7. Handle REJECT
        if approval.decision == ApprovalDecision.REJECT:
            self._processed_incidents.add(incident_id)
            return ApprovalWorkflowResult(
                incident_id=incident_id,
                status=WorkflowStatus.REJECTED,
                decision=ApprovalDecision.REJECT,
                selected_plan=None,
                is_authorized_for_execution=False,
                rejection_reasons=["Operator explicitly rejected recovery plans. No action authorized."],
                action=approval,
            )

        # 8. Handle APPROVE
        if approval.decision == ApprovalDecision.APPROVE:
            selected_plan_id = approval.selected_plan_id

            # Check if operator attempted to approve a rejected / non-executable plan
            if selected_plan_id in card.rejected_plan_ids:
                return ApprovalWorkflowResult(
                    incident_id=incident_id,
                    status=WorkflowStatus.INVALID_SUBMISSION,
                    decision=approval.decision,
                    selected_plan=None,
                    is_authorized_for_execution=False,
                    rejection_reasons=[
                        f"Selected plan '{selected_plan_id}' was rejected by the Safety Gate and is strictly NOT executable."
                    ],
                    action=approval,
                )

            # Check if selected plan is in executable plans
            if selected_plan_id not in card.executable_plan_ids:
                return ApprovalWorkflowResult(
                    incident_id=incident_id,
                    status=WorkflowStatus.INVALID_SUBMISSION,
                    decision=approval.decision,
                    selected_plan=None,
                    is_authorized_for_execution=False,
                    rejection_reasons=[
                        f"Selected plan '{selected_plan_id}' is not among certified executable plans for incident '{incident_id}'."
                    ],
                    action=approval,
                )

            # Find matching candidate
            candidates = self._incident_candidates.get(incident_id, [])
            matching_candidate = next((c for c in candidates if c.plan.plan_id == selected_plan_id), None)
            if matching_candidate is None:
                return ApprovalWorkflowResult(
                    incident_id=incident_id,
                    status=WorkflowStatus.INVALID_SUBMISSION,
                    decision=approval.decision,
                    selected_plan=None,
                    is_authorized_for_execution=False,
                    rejection_reasons=[f"Plan option '{selected_plan_id}' could not be located in candidate registry."],
                    action=approval,
                )

            selected_plan = matching_candidate.plan

            # ---------------------------------------------------------------
            # 9. PRE-EXECUTION REVALIDATION (Mandatory Requirement)
            # ---------------------------------------------------------------
            active_constraints = node_constraints or self._incident_constraints.get(incident_id, NodeConstraints())
            active_workloads = self._incident_workloads.get(incident_id, [])

            reval_result = self.safety_gate.evaluate(
                plan=selected_plan,
                telemetry=current_telemetry,
                workloads=active_workloads,
                node_constraints=active_constraints,
            )

            if not reval_result.passed:
                self._processed_incidents.add(incident_id)
                return ApprovalWorkflowResult(
                    incident_id=incident_id,
                    status=WorkflowStatus.REVALIDATION_FAILED,
                    decision=approval.decision,
                    selected_plan=selected_plan,
                    is_authorized_for_execution=False,
                    revalidation_result=reval_result,
                    rejection_reasons=[
                        f"Pre-execution Safety Gate revalidation failed: {'; '.join(reval_result.rejection_reasons)}"
                    ],
                    action=approval,
                )

            # Validate SafetyGate authorization
            is_auth, auth_reasons = self.safety_gate.validate_authorization(
                approval=approval,
                plan=selected_plan,
                safety_result=reval_result,
            )
            if not is_auth:
                return ApprovalWorkflowResult(
                    incident_id=incident_id,
                    status=WorkflowStatus.INVALID_SUBMISSION,
                    decision=approval.decision,
                    selected_plan=selected_plan,
                    is_authorized_for_execution=False,
                    revalidation_result=reval_result,
                    rejection_reasons=auth_reasons,
                    action=approval,
                )

            # Execution authorized
            self._processed_incidents.add(incident_id)
            return ApprovalWorkflowResult(
                incident_id=incident_id,
                status=WorkflowStatus.APPROVED,
                decision=approval.decision,
                selected_plan=selected_plan,
                is_authorized_for_execution=True,
                revalidation_result=reval_result,
                action=approval,
            )

        return ApprovalWorkflowResult(
            incident_id=incident_id,
            status=WorkflowStatus.INVALID_SUBMISSION,
            is_authorized_for_execution=False,
            rejection_reasons=[f"Unhandled approval decision: {approval.decision}."],
            action=approval,
        )

    def run_approval_cycle(
        self,
        incident_id: str,
        telemetry: GPUTelemetry,
        candidates: Sequence[EvaluatedPlanCandidate],
        workloads: Sequence[WorkloadSpec] | Mapping[str, WorkloadSpec],
        forecast: ForecastResult | None = None,
        node_constraints: NodeConstraints | None = None,
        timeout_s: float | None = None,
        current_telemetry_at_approval: GPUTelemetry | None = None,
    ) -> ApprovalWorkflowResult:
        """Execute full end-to-end approval cycle: dispatch card, listen, and evaluate authorization."""
        self.create_and_dispatch_card(
            incident_id=incident_id,
            telemetry=telemetry,
            candidates=candidates,
            workloads=workloads,
            forecast=forecast,
            node_constraints=node_constraints,
            timeout_s=timeout_s,
        )

        effective_timeout = timeout_s or self.default_timeout_s
        action = self.switch_client.listen_for_decision(incident_id=incident_id, timeout_s=effective_timeout)
        eval_telemetry = current_telemetry_at_approval or telemetry

        return self.evaluate_approval(
            approval=action,
            incident_id=incident_id,
            current_telemetry=eval_telemetry,
            node_constraints=node_constraints,
        )


# ---------------------------------------------------------------------------
# 4. Fail-Safe Simulator Isolation Helper
# ---------------------------------------------------------------------------

def apply_plan_to_simulator_if_authorized(
    result: ApprovalWorkflowResult,
    simulator: TelemetrySimulator,
) -> bool:
    """Safely apply an approved plan to the simulator IF AND ONLY IF execution is authorized.

    Strictly refuses to touch or mutate simulator state if authorization is False.

    Args:
        result: Outcome from SwitchApprovalDesk.
        simulator: TelemetrySimulator instance.

    Returns:
        True if plan actions were successfully simulated.

    Raises:
        PermissionError: If execution is unauthorized, timed out, rejected, or invalid.
    """
    if not result.is_authorized_for_execution or result.selected_plan is None:
        raise PermissionError(
            f"Execution denied: workflow status is {result.status.value}. "
            f"Reasons: {'; '.join(result.rejection_reasons or ['Not authorized.'])}"
        )

    # Apply discrete actions to simulation model
    for action in result.selected_plan.workload_actions:
        if action.action == ActionType.POWER_CAP and action.target_power_w is not None:
            simulator.set_workload_power(action.workload_id, action.target_power_w)
        elif action.action in (ActionType.PAUSE, ActionType.MIGRATE):
            simulator.remove_workload(action.workload_id)

    return True
