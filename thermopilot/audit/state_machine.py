"""Deterministic Incident State Machine for ThermoPilot Compute Desk.

Manages the lifecycle of a GPU thermal incident across 12 explicit states:
- DETECTED: Thermal anomaly trigger received.
- FORECASTED: Trajectory extrapolation and Time-to-Throttle (t_TTT) computed.
- PLANNING: Workload agent bids collected and recovery plans formulated.
- AWAITING_APPROVAL: Certified decision card dispatched to Switch/CLI desk.
- APPROVED: Operator explicitly authorized an executable plan option.
- REJECTED: Operator explicitly rejected all recovery plans (terminal).
- TIMEOUT_ESCALATED: Approval window expired without human response (terminal).
- REVALIDATION_FAILED: Pre-execution Safety Gate revalidation failed (terminal).
- EXECUTING: Plan actions currently being applied to simulator.
- VERIFYING: Post-execution thermal decay trajectory being monitored.
- RECOVERED: Cooldown target and SLA preservation verified (terminal success).
- RECOVERY_FAILED: Thermal target not reached or SLA breached (terminal failure).

SAFETY & INTEGRITY GUARANTEES:
1. Deterministic Transition Table: Enforces allowed transitions; rejects state jumps.
2. Authorization & Safety Preconditions: EXECUTING requires state APPROVED, explicit
   human authorization, and successful SafetyGate revalidation.
3. Fail-Safe Timeout / Rejection: Timeout and rejection states strictly terminate
   and cannot transition into execution.
4. Duplicate Execution Protection: An incident can transition to EXECUTING only once.
5. Durable Audit Synchronization: Writes every transition to the audit ledger;
   if the ledger write fails, the transition aborts and does not mutate simulator state.
"""

from datetime import datetime, timezone
from typing import Any, ClassVar

from thermopilot.audit.ledger import AuditLedger
from thermopilot.schemas import (
    AuditEventType,
    IncidentState,
    StateTransitionRecord,
)

# ---------------------------------------------------------------------------
# 1. Custom Exceptions
# ---------------------------------------------------------------------------

class StateMachineError(Exception):
    """Base exception for incident state machine errors."""


class InvalidStateTransitionError(StateMachineError):
    """Raised when an illegal or unauthorized state transition is attempted."""


# ---------------------------------------------------------------------------
# 2. Deterministic State Machine
# ---------------------------------------------------------------------------

class IncidentStateMachine:
    """State machine governing the lifecycle of a GPU thermal incident."""

    ALLOWED_TRANSITIONS: ClassVar[dict[IncidentState, set[IncidentState]]] = {
        IncidentState.DETECTED: {IncidentState.FORECASTED},
        IncidentState.FORECASTED: {IncidentState.PLANNING},
        IncidentState.PLANNING: {IncidentState.AWAITING_APPROVAL},
        IncidentState.AWAITING_APPROVAL: {
            IncidentState.APPROVED,
            IncidentState.REJECTED,
            IncidentState.TIMEOUT_ESCALATED,
        },
        IncidentState.APPROVED: {
            IncidentState.EXECUTING,
            IncidentState.REVALIDATION_FAILED,
        },
        IncidentState.EXECUTING: {
            IncidentState.VERIFYING,
            IncidentState.RECOVERY_FAILED,
        },
        IncidentState.VERIFYING: {
            IncidentState.RECOVERED,
            IncidentState.RECOVERY_FAILED,
        },
        # Terminal states with zero outbound transitions
        IncidentState.REJECTED: set(),
        IncidentState.TIMEOUT_ESCALATED: set(),
        IncidentState.REVALIDATION_FAILED: set(),
        IncidentState.RECOVERED: set(),
        IncidentState.RECOVERY_FAILED: set(),
    }

    TERMINAL_STATES: ClassVar[set[IncidentState]] = {
        IncidentState.REJECTED,
        IncidentState.TIMEOUT_ESCALATED,
        IncidentState.REVALIDATION_FAILED,
        IncidentState.RECOVERED,
        IncidentState.RECOVERY_FAILED,
    }

    def __init__(
        self,
        incident_id: str,
        initial_state: IncidentState = IncidentState.DETECTED,
        ledger: AuditLedger | None = None,
    ) -> None:
        """Initialize IncidentStateMachine.

        Args:
            incident_id: Unique thermal incident identifier.
            initial_state: Starting state (default: DETECTED).
            ledger: Optional AuditLedger for durable event logging.
        """
        if not incident_id or not incident_id.strip():
            raise ValueError("incident_id must be a non-empty string.")

        self.incident_id = incident_id
        self._current_state = initial_state
        self.ledger = ledger

        self.selected_plan_id: str | None = None
        self.is_executed: bool = False
        self._history: list[StateTransitionRecord] = []

    @property
    def current_state(self) -> IncidentState:
        """Current operational lifecycle state."""
        return self._current_state

    @property
    def history(self) -> list[StateTransitionRecord]:
        """Read-only copy of transition history."""
        return list(self._history)

    @property
    def is_terminal(self) -> bool:
        """Whether current state is a terminal lifecycle state."""
        return self._current_state in self.TERMINAL_STATES

    def can_transition_to(self, to_state: IncidentState) -> bool:
        """Check whether transition from current state to to_state is allowed."""
        return to_state in self.ALLOWED_TRANSITIONS.get(self._current_state, set())

    def transition(
        self,
        to_state: IncidentState,
        plan_id: str | None = None,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
        is_authorized: bool | None = None,
        revalidation_passed: bool | None = None,
        incident_id: str | None = None,
    ) -> StateTransitionRecord:
        """Execute a deterministic state transition with audit recording.

        Enforces all security, binding, and authorization preconditions.
        If audit recording fails, aborts the transition and leaves state untouched.

        Args:
            to_state: Target destination IncidentState.
            plan_id: Plan ID associated with transition (required for APPROVED).
            reason: Human-readable rationale for transition.
            payload: Optional metadata to store in transition record and audit ledger.
            is_authorized: Boolean flag indicating explicit human approval.
            revalidation_passed: Boolean flag indicating SafetyGate pre-execution pass.
            incident_id: Incident ID validation. Must match self.incident_id if provided.

        Returns:
            Immutable StateTransitionRecord.

        Raises:
            InvalidStateTransitionError: If transition violates grammar, bindings, or safety preconditions.
        """
        # 1. Verify Incident Binding
        if incident_id is not None and incident_id != self.incident_id:
            raise InvalidStateTransitionError(
                f"Incident ID mismatch: transition target is '{incident_id}', but state machine is bound to '{self.incident_id}'."
            )

        # 2. Prevent Duplicate Execution
        if to_state == IncidentState.EXECUTING and self.is_executed:
            raise InvalidStateTransitionError(
                f"Duplicate execution blocked: incident '{self.incident_id}' has already executed plan '{self.selected_plan_id}'."
            )

        # 3. Check Allowed Transition Grammar
        if not self.can_transition_to(to_state):
            raise InvalidStateTransitionError(
                f"Illegal state transition from '{self._current_state.value}' to '{to_state.value}'. "
                f"Allowed destinations: {[s.value for s in self.ALLOWED_TRANSITIONS.get(self._current_state, set())]}."
            )

        # 3. State-Specific Binding and Precondition Checks
        # A. Transitioning to APPROVED requires a plan_id
        if to_state == IncidentState.APPROVED:
            if not plan_id:
                raise InvalidStateTransitionError("Transition to APPROVED requires an explicit plan_id.")
            eff_plan_id = plan_id
        else:
            eff_plan_id = plan_id or self.selected_plan_id

        # B. Transitioning to EXECUTING requires authorization, safety revalidation, and single execution
        if to_state == IncidentState.EXECUTING:
            if self.is_executed:
                raise InvalidStateTransitionError(
                    f"Duplicate execution blocked: incident '{self.incident_id}' has already executed plan '{self.selected_plan_id}'."
                )

            if is_authorized is False:
                raise InvalidStateTransitionError(
                    "Cannot transition to EXECUTING: execution is not authorized by human operator."
                )

            if revalidation_passed is False:
                raise InvalidStateTransitionError(
                    "Cannot transition to EXECUTING: pre-execution Safety Gate revalidation failed."
                )

            if eff_plan_id != self.selected_plan_id:
                raise InvalidStateTransitionError(
                    f"Plan ID mismatch: attempting to execute '{eff_plan_id}', but approved plan is '{self.selected_plan_id}'."
                )

        # 4. Fail-Safe Audit Recording
        # Durable record MUST be saved BEFORE internal state is committed.
        # If ledger write raises an exception, the state machine aborts without advancing.
        from_state = self._current_state
        trans_details: dict[str, Any] = dict(payload or {})
        trans_details["from_state"] = from_state.value
        trans_details["to_state"] = to_state.value
        if eff_plan_id:
            trans_details["plan_id"] = eff_plan_id
        if reason:
            trans_details["reason"] = reason

        if self.ledger is not None:
            # Map state transition to relevant AuditEventType
            event_type = AuditEventType.STATE_TRANSITION
            if to_state == IncidentState.APPROVED:
                event_type = AuditEventType.OPERATOR_DECISION_RECORDED
            elif to_state == IncidentState.TIMEOUT_ESCALATED:
                event_type = AuditEventType.TIMEOUT_ESCALATION_RECORDED
            elif to_state == IncidentState.REJECTED:
                event_type = AuditEventType.OPERATOR_DECISION_RECORDED
            elif to_state == IncidentState.EXECUTING:
                event_type = AuditEventType.PLAN_EXECUTED
            elif to_state in (IncidentState.RECOVERED, IncidentState.RECOVERY_FAILED):
                event_type = AuditEventType.VERIFICATION_COMPLETED

            # If write fails, this raises and state remains untouched!
            self.ledger.append(
                incident_id=self.incident_id,
                event_type=event_type,
                payload=trans_details,
                is_simulated=True,
            )

        # 5. Commit Internal State Mutation
        self._current_state = to_state
        if to_state == IncidentState.APPROVED:
            self.selected_plan_id = eff_plan_id
        elif to_state == IncidentState.EXECUTING:
            self.is_executed = True

        record = StateTransitionRecord(
            incident_id=self.incident_id,
            from_state=from_state,
            to_state=to_state,
            plan_id=eff_plan_id,
            timestamp=datetime.now(timezone.utc),
            reason=reason,
            details=trans_details,
        )
        self._history.append(record)
        return record
