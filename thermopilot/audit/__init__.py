"""Audit Ledger & Incident State Machine Subsystem for ThermoPilot Compute Desk.

Exposes the immutable, append-only cryptographic JSONL audit ledger, the
deterministic incident state machine, and the end-to-end incident lifecycle coordinator.
"""

from thermopilot.audit.coordinator import IncidentLifecycleCoordinator
from thermopilot.audit.ledger import (
    GENESIS_HASH,
    AuditLedger,
    AuditLedgerError,
    LedgerCorruptionError,
    LedgerTamperError,
    compute_event_hash,
    sanitize_payload,
)
from thermopilot.audit.state_machine import (
    IncidentStateMachine,
    InvalidStateTransitionError,
    StateMachineError,
)
from thermopilot.schemas import (
    AuditEvent,
    AuditEventType,
    IncidentState,
    LedgerVerificationReport,
    StateTransitionRecord,
)

__all__ = [
    "GENESIS_HASH",
    "AuditEvent",
    "AuditEventType",
    "AuditLedger",
    "AuditLedgerError",
    "IncidentLifecycleCoordinator",
    "IncidentState",
    "IncidentStateMachine",
    "InvalidStateTransitionError",
    "LedgerCorruptionError",
    "LedgerTamperError",
    "LedgerVerificationReport",
    "StateMachineError",
    "StateTransitionRecord",
    "compute_event_hash",
    "sanitize_payload",
]
