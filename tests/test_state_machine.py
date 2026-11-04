"""Unit tests for the Deterministic Incident State Machine.

Verifies:
1. Valid state transitions across complete incident lifecycles (recovery and failures).
2. Rejection of illegal state transitions (jumping states, reversing direction).
3. Terminal state permanence: zero outbound transitions from REJECTED, TIMEOUT_ESCALATED, etc.
4. Strict guarantee: Timeout and rejection never reach EXECUTING.
5. Incident ID mismatch rejection.
6. Plan ID binding and mismatch rejection.
7. Authorization and SafetyGate revalidation preconditions for EXECUTING.
8. Duplicate execution prevention (EXECUTING can only occur once per incident).
9. Fail-safe audit synchronization: ledger write failure aborts transition and preserves state.
"""

import tempfile
import unittest
from pathlib import Path

from thermopilot.audit.ledger import AuditLedger
from thermopilot.audit.state_machine import (
    IncidentStateMachine,
    InvalidStateTransitionError,
)
from thermopilot.schemas import IncidentState


class TestIncidentStateMachine(unittest.TestCase):
    """Test suite for the deterministic incident state machine."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ledger_path = Path(self.temp_dir.name) / "state_audit.jsonl"
        self.ledger = AuditLedger(self.ledger_path)
        self.incident_id = "INC-20261027-001"
        self.sm = IncidentStateMachine(
            incident_id=self.incident_id,
            initial_state=IncidentState.DETECTED,
            ledger=self.ledger,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    # -----------------------------------------------------------------------
    # 1. Valid Lifecycle Transitions
    # -----------------------------------------------------------------------

    def test_full_recovery_lifecycle(self) -> None:
        """Verify complete linear transition path from DETECTED to RECOVERED."""
        self.assertEqual(self.sm.current_state, IncidentState.DETECTED)

        # 1. DETECTED -> FORECASTED
        r1 = self.sm.transition(IncidentState.FORECASTED, reason="Forecast calculated")
        self.assertEqual(self.sm.current_state, IncidentState.FORECASTED)
        self.assertEqual(r1.from_state, IncidentState.DETECTED)
        self.assertEqual(r1.to_state, IncidentState.FORECASTED)

        # 2. FORECASTED -> PLANNING
        self.sm.transition(IncidentState.PLANNING, reason="Planner evaluating candidates")
        self.assertEqual(self.sm.current_state, IncidentState.PLANNING)

        # 3. PLANNING -> AWAITING_APPROVAL
        self.sm.transition(IncidentState.AWAITING_APPROVAL, reason="Card dispatched")
        self.assertEqual(self.sm.current_state, IncidentState.AWAITING_APPROVAL)

        # 4. AWAITING_APPROVAL -> APPROVED
        self.sm.transition(IncidentState.APPROVED, plan_id="PLAN-B", reason="Operator approved Plan B")
        self.assertEqual(self.sm.current_state, IncidentState.APPROVED)
        self.assertEqual(self.sm.selected_plan_id, "PLAN-B")

        # 5. APPROVED -> EXECUTING
        self.sm.transition(
            IncidentState.EXECUTING,
            plan_id="PLAN-B",
            is_authorized=True,
            revalidation_passed=True,
            reason="Executing on simulator",
        )
        self.assertEqual(self.sm.current_state, IncidentState.EXECUTING)
        self.assertTrue(self.sm.is_executed)

        # 6. EXECUTING -> VERIFYING
        self.sm.transition(IncidentState.VERIFYING, reason="Monitoring thermal decay")
        self.assertEqual(self.sm.current_state, IncidentState.VERIFYING)

        # 7. VERIFYING -> RECOVERED
        self.sm.transition(IncidentState.RECOVERED, reason="Cooldown target <= 75C verified")
        self.assertEqual(self.sm.current_state, IncidentState.RECOVERED)
        self.assertTrue(self.sm.is_terminal)

        # Verify history length
        self.assertEqual(len(self.sm.history), 7)

        # Verify all events recorded in ledger
        events = self.ledger.read_events(self.incident_id)
        self.assertEqual(len(events), 7)
        self.assertTrue(self.ledger.verify_chain().is_valid)

    def test_rejection_lifecycle_terminates_safely(self) -> None:
        """Verify AWAITING_APPROVAL -> REJECTED reaches terminal state with no outbound paths."""
        self.sm.transition(IncidentState.FORECASTED)
        self.sm.transition(IncidentState.PLANNING)
        self.sm.transition(IncidentState.AWAITING_APPROVAL)
        self.sm.transition(IncidentState.REJECTED, reason="Operator rejected plans")

        self.assertEqual(self.sm.current_state, IncidentState.REJECTED)
        self.assertTrue(self.sm.is_terminal)
        self.assertFalse(self.sm.can_transition_to(IncidentState.EXECUTING))

    def test_timeout_lifecycle_terminates_safely(self) -> None:
        """Verify AWAITING_APPROVAL -> TIMEOUT_ESCALATED terminates safely."""
        self.sm.transition(IncidentState.FORECASTED)
        self.sm.transition(IncidentState.PLANNING)
        self.sm.transition(IncidentState.AWAITING_APPROVAL)
        self.sm.transition(IncidentState.TIMEOUT_ESCALATED, reason="15s approval window elapsed")

        self.assertEqual(self.sm.current_state, IncidentState.TIMEOUT_ESCALATED)
        self.assertTrue(self.sm.is_terminal)
        self.assertFalse(self.sm.can_transition_to(IncidentState.EXECUTING))

    def test_revalidation_failure_terminates_safely(self) -> None:
        """Verify APPROVED -> REVALIDATION_FAILED terminates safely without executing."""
        self.sm.transition(IncidentState.FORECASTED)
        self.sm.transition(IncidentState.PLANNING)
        self.sm.transition(IncidentState.AWAITING_APPROVAL)
        self.sm.transition(IncidentState.APPROVED, plan_id="PLAN-B")
        self.sm.transition(IncidentState.REVALIDATION_FAILED, reason="Telemetry degraded beyond 80C ceiling")

        self.assertEqual(self.sm.current_state, IncidentState.REVALIDATION_FAILED)
        self.assertTrue(self.sm.is_terminal)
        self.assertFalse(self.sm.can_transition_to(IncidentState.EXECUTING))

    # -----------------------------------------------------------------------
    # 2. Strict Rejections & Constraint Enforcements
    # -----------------------------------------------------------------------

    def test_timeout_and_rejection_never_reach_executing(self) -> None:
        """Prove that neither TIMEOUT_ESCALATED nor REJECTED can ever transition to EXECUTING."""
        # 1. From TIMEOUT_ESCALATED
        sm_timeout = IncidentStateMachine("INC-TIMEOUT", IncidentState.TIMEOUT_ESCALATED)
        with self.assertRaises(InvalidStateTransitionError):
            sm_timeout.transition(IncidentState.EXECUTING)

        # 2. From REJECTED
        sm_reject = IncidentStateMachine("INC-REJECT", IncidentState.REJECTED)
        with self.assertRaises(InvalidStateTransitionError):
            sm_reject.transition(IncidentState.EXECUTING)

        # 3. From REVALIDATION_FAILED
        sm_reval = IncidentStateMachine("INC-REVAL", IncidentState.REVALIDATION_FAILED)
        with self.assertRaises(InvalidStateTransitionError):
            sm_reval.transition(IncidentState.EXECUTING)

    def test_state_jumping_rejected(self) -> None:
        """Verify illegal state transitions (jumping intermediate steps) are rejected."""
        # DETECTED cannot jump directly to EXECUTING
        with self.assertRaises(InvalidStateTransitionError):
            self.sm.transition(IncidentState.EXECUTING)

        # DETECTED cannot jump to APPROVED
        with self.assertRaises(InvalidStateTransitionError):
            self.sm.transition(IncidentState.APPROVED, plan_id="PLAN-B")

        # PLANNING cannot jump to RECOVERED
        self.sm.transition(IncidentState.FORECASTED)
        self.sm.transition(IncidentState.PLANNING)
        with self.assertRaises(InvalidStateTransitionError):
            self.sm.transition(IncidentState.RECOVERED)

    def test_reverse_transitions_rejected(self) -> None:
        """Verify state machine cannot transition backward in lifecycle."""
        self.sm.transition(IncidentState.FORECASTED)
        self.sm.transition(IncidentState.PLANNING)

        with self.assertRaises(InvalidStateTransitionError):
            self.sm.transition(IncidentState.DETECTED)

    def test_incident_id_mismatch_rejected(self) -> None:
        """Verify transition with mismatched incident_id is rejected."""
        with self.assertRaises(InvalidStateTransitionError) as ctx:
            self.sm.transition(IncidentState.FORECASTED, incident_id="INC-DIFFERENT-999")

        self.assertIn("mismatch", str(ctx.exception).lower())

    def test_plan_id_required_for_approved_state(self) -> None:
        """Verify transition to APPROVED fails if plan_id is missing."""
        self.sm.transition(IncidentState.FORECASTED)
        self.sm.transition(IncidentState.PLANNING)
        self.sm.transition(IncidentState.AWAITING_APPROVAL)

        with self.assertRaises(InvalidStateTransitionError) as ctx:
            self.sm.transition(IncidentState.APPROVED, plan_id=None)

        self.assertIn("requires an explicit plan_id", str(ctx.exception).lower())

    def test_plan_id_mismatch_on_execution_rejected(self) -> None:
        """Verify executing a different plan than approved raises InvalidStateTransitionError."""
        self.sm.transition(IncidentState.FORECASTED)
        self.sm.transition(IncidentState.PLANNING)
        self.sm.transition(IncidentState.AWAITING_APPROVAL)
        self.sm.transition(IncidentState.APPROVED, plan_id="PLAN-B")

        # Attempt to execute PLAN-A instead of approved PLAN-B
        with self.assertRaises(InvalidStateTransitionError) as ctx:
            self.sm.transition(
                IncidentState.EXECUTING,
                plan_id="PLAN-A",
                is_authorized=True,
                revalidation_passed=True,
            )

        self.assertIn("plan id mismatch", str(ctx.exception).lower())

    def test_unauthorized_or_revalidation_failed_cannot_execute(self) -> None:
        """Verify is_authorized=False or revalidation_passed=False blocks transition to EXECUTING."""
        self.sm.transition(IncidentState.FORECASTED)
        self.sm.transition(IncidentState.PLANNING)
        self.sm.transition(IncidentState.AWAITING_APPROVAL)
        self.sm.transition(IncidentState.APPROVED, plan_id="PLAN-B")

        # 1. is_authorized=False
        with self.assertRaises(InvalidStateTransitionError):
            self.sm.transition(
                IncidentState.EXECUTING,
                plan_id="PLAN-B",
                is_authorized=False,
                revalidation_passed=True,
            )

        # 2. revalidation_passed=False
        with self.assertRaises(InvalidStateTransitionError):
            self.sm.transition(
                IncidentState.EXECUTING,
                plan_id="PLAN-B",
                is_authorized=True,
                revalidation_passed=False,
            )

    def test_duplicate_execution_blocked(self) -> None:
        """Verify an incident cannot transition to EXECUTING multiple times."""
        self.sm.transition(IncidentState.FORECASTED)
        self.sm.transition(IncidentState.PLANNING)
        self.sm.transition(IncidentState.AWAITING_APPROVAL)
        self.sm.transition(IncidentState.APPROVED, plan_id="PLAN-B")
        self.sm.transition(
            IncidentState.EXECUTING,
            plan_id="PLAN-B",
            is_authorized=True,
            revalidation_passed=True,
        )
        self.assertTrue(self.sm.is_executed)

        # Attempt second execution
        with self.assertRaises(InvalidStateTransitionError) as ctx:
            self.sm.transition(
                IncidentState.EXECUTING,
                plan_id="PLAN-B",
                is_authorized=True,
                revalidation_passed=True,
            )

        self.assertIn("duplicate execution", str(ctx.exception).lower())

    # -----------------------------------------------------------------------
    # 3. Fail-Safe Audit Durability
    # -----------------------------------------------------------------------

    def test_ledger_write_failure_aborts_transition_and_preserves_state(self) -> None:
        """Prove that if audit ledger write fails, state machine transition aborts and state is unmodified."""
        class FailingLedger:
            def append(self, *args, **kwargs):
                raise OSError("Simulated disk write failure / full partition")

        failing_sm = IncidentStateMachine(
            incident_id=self.incident_id,
            initial_state=IncidentState.APPROVED,
            ledger=FailingLedger(),  # type: ignore[arg-type]
        )
        failing_sm.selected_plan_id = "PLAN-B"

        # Attempt to transition to EXECUTING with failing ledger
        with self.assertRaises(OSError) as ctx:
            failing_sm.transition(
                IncidentState.EXECUTING,
                plan_id="PLAN-B",
                is_authorized=True,
                revalidation_passed=True,
            )

        self.assertIn("simulated disk write failure", str(ctx.exception).lower())

        # CRUCIAL ASSERTION: state was NOT committed to EXECUTING; remained APPROVED
        self.assertEqual(failing_sm.current_state, IncidentState.APPROVED)
        self.assertFalse(failing_sm.is_executed)


if __name__ == "__main__":
    unittest.main()
