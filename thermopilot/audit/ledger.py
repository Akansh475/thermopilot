"""Append-Only Cryptographic Audit Ledger for ThermoPilot Compute Desk.

Provides an immutable, tamper-evident JSONL audit event store with a SHA-256
cryptographic hash chain. Guarantees:

1. Append-Only Interface: The application API only supports appending verified events.
2. Cryptographic Hash Chain: Each event includes the SHA-256 hash of the preceding
   event and its own canonical hash. Any modification, insertion, or deletion is detected.
3. Thread-Safe Concurrency: Serializes concurrent writes using a threading lock and
   enforces persistent disk syncing (`os.fsync`).
4. Secret Sanitization: Automatically sanitizes sensitive keys and secret values
   (API keys, tokens, credentials) from event payloads before hashing and writing.
5. Corruption Protection: Validates existing files on open; rejects appending to a
   corrupted or tampered ledger.
6. Simulation Marker: Clearly marks all simulated infrastructure events with
   `is_simulated = True`.

SECURITY NOTICE & LIMITATION:
A cryptographic hash chain provides tamper-evident integrity: any modification,
insertion, deletion, or reordering of historical records invalidates subsequent hashes.
However, like any local cryptographic chain, a hash chain stored on a local filesystem
detects unauthorized modification after the fact, but does NOT independently prevent
a malicious actor with root/filesystem write access from recalculating hashes and
rewriting the entire file from genesis. In production environments, tamper-resistance
requires anchoring hashes to append-only WORM storage (Write Once Read Many), remote
cloud SIEMs, or signed witness timestamps.
"""

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thermopilot.schemas import (
    AuditEvent,
    AuditEventType,
    LedgerVerificationReport,
)

GENESIS_HASH: str = "0" * 64

SECRET_KEY_SUBSTRINGS: set[str] = {
    "token",
    "secret",
    "api_key",
    "password",
    "auth",
    "credential",
    "bearer",
    "private_key",
}


# ---------------------------------------------------------------------------
# 1. Custom Exceptions
# ---------------------------------------------------------------------------

class AuditLedgerError(Exception):
    """Base exception for audit ledger errors."""


class LedgerCorruptionError(AuditLedgerError):
    """Raised when ledger file contains malformed JSON or corrupted chain on disk."""


class LedgerTamperError(AuditLedgerError):
    """Raised when tampering or invalid cryptographic hashes are detected in the ledger."""


# ---------------------------------------------------------------------------
# 2. Secret Sanitization
# ---------------------------------------------------------------------------

def _sanitize_value(key: str, val: Any) -> Any:
    """Recursively scrub secrets from payload values."""
    key_lower = key.lower()
    if any(sub in key_lower for sub in SECRET_KEY_SUBSTRINGS):
        return "[REDACTED_SECRET]"
    if isinstance(val, str):
        val_lower = val.lower()
        if val_lower.startswith("bearer ") or val.startswith(("ghp_", "sk-", "xoxb-")):
            return "[REDACTED_SECRET]"
        return val
    if isinstance(val, dict):
        return {k: _sanitize_value(k, v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [_sanitize_value("", item) for item in val]
    return val


def sanitize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Scrub tokens, API keys, credentials, and passwords from an audit event payload."""
    if not isinstance(payload, dict):
        return {"data": _sanitize_value("data", payload)}
    return {k: _sanitize_value(k, v) for k, v in payload.items()}


# ---------------------------------------------------------------------------
# 3. Canonical Cryptographic Hashing
# ---------------------------------------------------------------------------

def compute_event_hash(
    event_id: str,
    incident_id: str,
    event_type: str,
    timestamp: str | datetime,
    sequence_num: int,
    prev_hash: str | None,
    is_simulated: bool,
    payload: dict[str, Any],
) -> str:
    """Compute deterministic SHA-256 hash over canonical representation of an event.

    Excludes the event's own 'hash' field while including 'prev_hash' to bind the chain.
    """
    ts_str = timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)
    canonical_dict = {
        "event_id": str(event_id),
        "event_type": str(event_type),
        "incident_id": str(incident_id),
        "is_simulated": bool(is_simulated),
        "payload": payload,
        "prev_hash": prev_hash or GENESIS_HASH,
        "sequence_num": int(sequence_num),
        "timestamp": ts_str,
    }
    canonical_bytes = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


# ---------------------------------------------------------------------------
# 4. Append-Only Audit Ledger
# ---------------------------------------------------------------------------

class AuditLedger:
    """Append-only, thread-safe JSONL audit ledger with cryptographic hash chain."""

    def __init__(self, ledger_path: str | Path, auto_verify_on_open: bool = True) -> None:
        """Initialize AuditLedger.

        Args:
            ledger_path: File path to the JSONL audit ledger file.
            auto_verify_on_open: Whether to verify existing ledger chain on initialization.

        Raises:
            LedgerCorruptionError: If existing file is corrupted, malformed, or tampered.
        """
        self.ledger_path = Path(ledger_path)
        self._lock = threading.Lock()
        self._last_hash: str | None = None
        self._last_sequence_num: int = -1

        # Ensure parent directory exists
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

        # Inspect or initialize existing ledger
        if self.ledger_path.exists() and self.ledger_path.stat().st_size > 0:
            if auto_verify_on_open:
                report = self.verify_chain()
                if not report.is_valid:
                    raise LedgerCorruptionError(
                        f"Existing audit ledger at '{self.ledger_path}' is corrupted: {'; '.join(report.errors)}"
                    )
            # Seed last hash and sequence from existing file
            events = self.read_events()
            if events:
                self._last_hash = events[-1].hash
                self._last_sequence_num = events[-1].sequence_num

    @property
    def event_count(self) -> int:
        """Total number of audit events currently recorded."""
        with self._lock:
            return self._last_sequence_num + 1

    @property
    def last_hash(self) -> str | None:
        """The SHA-256 hash of the most recent audit event in the chain."""
        with self._lock:
            return self._last_hash

    def __len__(self) -> int:
        return self.event_count

    def get_last_event(self) -> AuditEvent | None:
        """Retrieve the most recent audit event in the chain."""
        with self._lock:
            events = self.read_events()
            return events[-1] if events else None

    def append(
        self,
        incident_id: str,
        event_type: AuditEventType | str,
        payload: dict[str, Any],
        is_simulated: bool = True,
        event_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> AuditEvent:
        """Append a new lifecycle event to the ledger with hash chaining and disk sync.

        Args:
            incident_id: Thermal incident identifier.
            event_type: Categorized AuditEventType or event string.
            payload: Structured dictionary of event attributes (will be scrubbed for secrets).
            is_simulated: Marker confirming simulated vs. real infrastructure event.
            event_id: Optional UUID override (generated if None).
            timestamp: Optional timestamp override (UTC now if None).

        Returns:
            The created, hashed, and persistently written AuditEvent.

        Raises:
            AuditLedgerError: If write or disk synchronization fails.
        """
        with self._lock:
            # 1. Determine sequence number and previous hash
            next_seq = self._last_sequence_num + 1
            prev_hash = self._last_hash or GENESIS_HASH

            # 2. Scrub secrets from payload
            clean_payload = sanitize_payload(payload)

            # 3. Canonical event timestamp and type
            event_ts = timestamp or datetime.now(timezone.utc)
            type_val = event_type.value if isinstance(event_type, AuditEventType) else str(event_type)

            # 4. Generate unique event ID if omitted
            import uuid
            eff_event_id = event_id or str(uuid.uuid4())

            # 5. Compute cryptographic SHA-256 hash
            event_hash = compute_event_hash(
                event_id=eff_event_id,
                incident_id=incident_id,
                event_type=type_val,
                timestamp=event_ts,
                sequence_num=next_seq,
                prev_hash=prev_hash,
                is_simulated=is_simulated,
                payload=clean_payload,
            )

            # 6. Construct validated AuditEvent
            # Normalize event_type to enum if possible
            try:
                enum_type = AuditEventType(type_val)
            except ValueError:
                enum_type = AuditEventType.STATE_TRANSITION  # fallback for custom states

            event = AuditEvent(
                event_id=eff_event_id,
                incident_id=incident_id,
                event_type=enum_type,
                timestamp=event_ts,
                sequence_num=next_seq,
                prev_hash=prev_hash,
                hash=event_hash,
                is_simulated=is_simulated,
                payload=clean_payload,
            )

            # 7. Write to JSONL file and fsync for durability
            line_dict = {
                "event_id": event.event_id,
                "incident_id": event.incident_id,
                "event_type": type_val,
                "timestamp": event.timestamp.isoformat(),
                "sequence_num": event.sequence_num,
                "prev_hash": event.prev_hash,
                "hash": event.hash,
                "is_simulated": event.is_simulated,
                "payload": event.payload,
            }
            jsonl_line = json.dumps(line_dict, separators=(",", ":"), default=str) + "\n"

            try:
                with open(self.ledger_path, "a", encoding="utf-8") as f:
                    f.write(jsonl_line)
                    f.flush()
                    os.fsync(f.fileno())
            except OSError as exc:
                raise AuditLedgerError(f"Failed to durably write audit event to disk: {exc}") from exc

            # 8. Update in-memory state
            self._last_hash = event.hash
            self._last_sequence_num = next_seq

            return event

    def append_event(self, event: AuditEvent) -> AuditEvent:
        """Append an existing AuditEvent instance through the hash chain."""
        return self.append(
            incident_id=event.incident_id,
            event_type=event.event_type,
            payload=event.payload,
            is_simulated=event.is_simulated,
            event_id=event.event_id,
            timestamp=event.timestamp,
        )

    def verify_chain(self) -> LedgerVerificationReport:
        """Verify the cryptographic hash chain of the entire JSONL ledger.

        Checks:
        - Line-by-line JSON validity and completeness.
        - Strict sequence number continuity (0, 1, 2, ...).
        - Genesis event prev_hash binding (GENESIS_HASH).
        - Chain linkage: prev_hash == previous event's hash.
        - Cryptographic integrity: recomputed hash matches recorded hash.

        Returns:
            LedgerVerificationReport detailing validity, event count, and error reasons.
        """
        if not self.ledger_path.exists():
            return LedgerVerificationReport(is_valid=True, total_events=0, errors=[])

        errors: list[str] = []
        expected_seq = 0
        expected_prev_hash = GENESIS_HASH
        corrupted_index: int | None = None

        with open(self.ledger_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        if not lines:
            return LedgerVerificationReport(is_valid=True, total_events=0, errors=[])

        for idx, line in enumerate(lines):
            stripped = line.strip()
            # Interrupted / partial record check
            if not stripped:
                continue

            # Check if line ends with proper newline (detect interrupted write at EOF)
            if idx == len(lines) - 1 and not line.endswith("\n"):
                errors.append(f"Line {idx}: Interrupted or partial final record (missing newline).")
                corrupted_index = idx if corrupted_index is None else corrupted_index

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {idx}: Malformed JSON record: {exc}.")
                if corrupted_index is None:
                    corrupted_index = idx
                continue

            if not isinstance(record, dict):
                errors.append(f"Line {idx}: Record is not a valid JSON object.")
                if corrupted_index is None:
                    corrupted_index = idx
                continue

            # Check required fields
            req_fields = {"event_id", "incident_id", "event_type", "timestamp", "sequence_num", "prev_hash", "hash", "payload"}
            missing = req_fields - set(record.keys())
            if missing:
                errors.append(f"Line {idx}: Missing required audit fields: {sorted(missing)}.")
                if corrupted_index is None:
                    corrupted_index = idx
                continue

            # 1. Verify sequence number
            actual_seq = record.get("sequence_num")
            if actual_seq != expected_seq:
                errors.append(
                    f"Line {idx} (event '{record.get('event_id')}'): Sequence number discontinuity. "
                    f"Expected {expected_seq}, got {actual_seq}."
                )
                if corrupted_index is None:
                    corrupted_index = idx

            # 2. Verify previous hash link
            actual_prev_hash = record.get("prev_hash")
            if actual_prev_hash != expected_prev_hash:
                errors.append(
                    f"Line {idx} (event '{record.get('event_id')}'): Broken hash chain link. "
                    f"Expected prev_hash '{expected_prev_hash}', got '{actual_prev_hash}'."
                )
                if corrupted_index is None:
                    corrupted_index = idx

            # 3. Recompute and verify cryptographic hash
            expected_hash = compute_event_hash(
                event_id=record["event_id"],
                incident_id=record["incident_id"],
                event_type=record["event_type"],
                timestamp=record["timestamp"],
                sequence_num=actual_seq,
                prev_hash=actual_prev_hash,
                is_simulated=record.get("is_simulated", True),
                payload=record["payload"],
            )

            actual_hash = record.get("hash")
            if actual_hash != expected_hash:
                errors.append(
                    f"Line {idx} (event '{record.get('event_id')}'): Tampering detected! "
                    f"Recorded hash '{actual_hash}' does not match recomputed hash '{expected_hash}'."
                )
                if corrupted_index is None:
                    corrupted_index = idx

            expected_prev_hash = actual_hash
            expected_seq = actual_seq + 1

        is_valid = len(errors) == 0
        return LedgerVerificationReport(
            is_valid=is_valid,
            total_events=len(lines),
            corrupted_event_index=corrupted_index,
            errors=errors,
        )

    def read_events(self, incident_id: str | None = None) -> list[AuditEvent]:
        """Read all validated audit events from the ledger, optionally filtered by incident ID."""
        if not self.ledger_path.exists():
            return []

        events: list[AuditEvent] = []
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                    # Convert event_type string to enum
                    t_str = data.get("event_type", "STATE_TRANSITION")
                    try:
                        enum_t = AuditEventType(t_str)
                    except ValueError:
                        enum_t = AuditEventType.STATE_TRANSITION

                    event = AuditEvent(
                        event_id=data["event_id"],
                        incident_id=data["incident_id"],
                        event_type=enum_t,
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        sequence_num=data["sequence_num"],
                        prev_hash=data["prev_hash"],
                        hash=data["hash"],
                        is_simulated=data.get("is_simulated", True),
                        payload=data.get("payload", {}),
                    )
                    if incident_id is None or event.incident_id == incident_id:
                        events.append(event)
                except (json.JSONDecodeError, KeyError):
                    continue

        return events
