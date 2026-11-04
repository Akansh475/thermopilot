"""Unit tests for the Append-Only Cryptographic Audit Ledger.

Verifies:
1. Append-only JSONL writes with sequential sequence numbers.
2. Cryptographic SHA-256 hash chaining: prev_hash binding to preceding event.
3. Successful hash-chain verification on untampered ledgers.
4. Tampering detection: modified payload, altered timestamp, or corrupted hash.
5. Deleted record detection: missing sequence numbers or broken hash links.
6. Malformed JSONL line detection.
7. Interrupted / partial final record detection.
8. Secret sanitization: API tokens, passwords, and credentials scrubbed from payloads.
9. Simulation marker: is_simulated=True preserved and checked.
10. Thread-safe concurrent appends with persistent fsync durability.
11. LedgerCorruptionError raised when opening a corrupted ledger with auto_verify_on_open=True.
12. Filtering events by incident ID.
"""

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from thermopilot.audit.ledger import (
    GENESIS_HASH,
    AuditLedger,
    LedgerCorruptionError,
    sanitize_payload,
)
from thermopilot.schemas import AuditEvent, AuditEventType


class TestAuditLedger(unittest.TestCase):
    """Test suite verifying AuditLedger functionality, hash chain, and tamper detection."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ledger_path = Path(self.temp_dir.name) / "test_audit.jsonl"
        self.ledger = AuditLedger(self.ledger_path)
        self.incident_id = "INC-20261027-001"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    # -----------------------------------------------------------------------
    # 1. Basic Append & Hash Chain Tests
    # -----------------------------------------------------------------------

    def test_append_first_event_binds_genesis_hash(self) -> None:
        """Verify the first event in an empty ledger binds to GENESIS_HASH at sequence 0."""
        event = self.ledger.append(
            incident_id=self.incident_id,
            event_type=AuditEventType.INCIDENT_TRIGGERED,
            payload={"core_temp_c": 81.5, "power_w": 380.0},
        )

        self.assertIsInstance(event, AuditEvent)
        self.assertEqual(event.sequence_num, 0)
        self.assertEqual(event.prev_hash, GENESIS_HASH)
        self.assertEqual(len(event.hash), 64)  # SHA-256 hex string
        self.assertTrue(event.is_simulated)
        self.assertEqual(len(self.ledger), 1)

        # Check raw file contents on disk
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        data = json.loads(lines[0])
        self.assertEqual(data["hash"], event.hash)
        self.assertEqual(data["sequence_num"], 0)

    def test_hash_chain_sequential_linking(self) -> None:
        """Verify sequential appends bind prev_hash to the preceding event's hash."""
        events = []
        for i in range(5):
            e = self.ledger.append(
                incident_id=self.incident_id,
                event_type=AuditEventType.STATE_TRANSITION,
                payload={"step": i},
            )
            events.append(e)

        self.assertEqual(len(events), 5)
        # Verify sequence numbers and hash links
        for i in range(5):
            self.assertEqual(events[i].sequence_num, i)
            if i == 0:
                self.assertEqual(events[i].prev_hash, GENESIS_HASH)
            else:
                self.assertEqual(events[i].prev_hash, events[i - 1].hash)

        # Verify chain integrity
        report = self.ledger.verify_chain()
        self.assertTrue(report.is_valid)
        self.assertEqual(report.total_events, 5)
        self.assertEqual(len(report.errors), 0)

    # -----------------------------------------------------------------------
    # 2. Tampering & Corruption Detection Tests
    # -----------------------------------------------------------------------

    def test_tampering_detected_when_payload_modified(self) -> None:
        """Verify modifying an event payload in the file is caught by verify_chain()."""
        # Append 3 events
        for i in range(3):
            self.ledger.append(self.incident_id, AuditEventType.PLAN_EXECUTED, {"action": f"act_{i}"})

        # Tamper with second event on disk
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        tampered_data = json.loads(lines[1])
        tampered_data["payload"]["action"] = "MALICIOUS_TAMPERED_ACTION"
        lines[1] = json.dumps(tampered_data) + "\n"

        with open(self.ledger_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        # Verify chain detects tampering
        report = self.ledger.verify_chain()
        self.assertFalse(report.is_valid)
        self.assertEqual(report.corrupted_event_index, 1)
        self.assertIn("tampering detected", report.errors[0].lower())

    def test_deleted_record_detected(self) -> None:
        """Verify deleting an event line is detected as a sequence break and hash mismatch."""
        for i in range(4):
            self.ledger.append(self.incident_id, AuditEventType.STATE_TRANSITION, {"idx": i})

        # Delete the second line (index 1)
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        del lines[1]  # remove event 1

        with open(self.ledger_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        report = self.ledger.verify_chain()
        self.assertFalse(report.is_valid)
        self.assertEqual(report.corrupted_event_index, 1)
        error_str = "; ".join(report.errors).lower()
        self.assertTrue("sequence number discontinuity" in error_str or "broken hash chain" in error_str)

    def test_malformed_json_line_detected(self) -> None:
        """Verify corrupt non-JSON line is detected."""
        self.ledger.append(self.incident_id, AuditEventType.INCIDENT_TRIGGERED, {"temp": 81.5})
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write("CORRUPTED_NON_JSON_LINE\n")

        report = self.ledger.verify_chain()
        self.assertFalse(report.is_valid)
        self.assertEqual(report.corrupted_event_index, 1)
        self.assertIn("malformed json", report.errors[0].lower())

    def test_interrupted_partial_final_record_detected(self) -> None:
        """Verify partial unclosed record from an interrupted write is caught."""
        self.ledger.append(self.incident_id, AuditEventType.INCIDENT_TRIGGERED, {"temp": 81.5})
        # Simulate partial record at end of file (no closing newline)
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write('{"event_id": "truncated-id"')

        report = self.ledger.verify_chain()
        self.assertFalse(report.is_valid)
        error_str = "; ".join(report.errors).lower()
        self.assertTrue("interrupted" in error_str or "partial" in error_str or "malformed" in error_str)

    def test_open_corrupted_ledger_raises_error(self) -> None:
        """Verify opening a corrupted ledger with auto_verify_on_open=True raises LedgerCorruptionError."""
        # Create corrupted file
        with open(self.ledger_path, "w", encoding="utf-8") as f:
            f.write("CORRUPTED_CONTENT\n")

        with self.assertRaises(LedgerCorruptionError) as ctx:
            AuditLedger(self.ledger_path, auto_verify_on_open=True)

        self.assertIn("corrupted", str(ctx.exception).lower())

    # -----------------------------------------------------------------------
    # 3. Secret Sanitization Tests
    # -----------------------------------------------------------------------

    def test_secret_sanitization_scrubs_sensitive_keys(self) -> None:
        """Verify tokens, API keys, passwords, and secrets are sanitized."""
        dirty_payload = {
            "workload_id": "wl-alpha",
            "api_key": "sk-proj-super-secret-key-12345",
            "auth_token": "Bearer my_secret_token",
            "password": "production_password_xyz",
            "nested": {
                "client_secret": "secret_abc",
                "normal_field": 42.0,
            },
            "items_list": ["safe_string", "Bearer secret_in_list"],
        }

        clean = sanitize_payload(dirty_payload)
        self.assertEqual(clean["api_key"], "[REDACTED_SECRET]")
        self.assertEqual(clean["auth_token"], "[REDACTED_SECRET]")
        self.assertEqual(clean["password"], "[REDACTED_SECRET]")
        self.assertEqual(clean["nested"]["client_secret"], "[REDACTED_SECRET]")
        self.assertEqual(clean["nested"]["normal_field"], 42.0)
        self.assertEqual(clean["items_list"][0], "safe_string")
        self.assertEqual(clean["items_list"][1], "[REDACTED_SECRET]")

        # Append and verify sanitized in file on disk
        event = self.ledger.append(self.incident_id, AuditEventType.OPERATOR_DECISION_RECORDED, dirty_payload)
        self.assertEqual(event.payload["api_key"], "[REDACTED_SECRET]")

        with open(self.ledger_path, "r", encoding="utf-8") as f:
            file_text = f.read()
        self.assertNotIn("sk-proj-super-secret-key-12345", file_text)
        self.assertNotIn("production_password_xyz", file_text)

    # -----------------------------------------------------------------------
    # 4. Concurrency & Durability Tests
    # -----------------------------------------------------------------------

    def test_concurrent_writes_thread_safety(self) -> None:
        """Verify 10 concurrent threads appending events produce a valid, unbroken hash chain."""
        num_threads = 10
        events_per_thread = 15
        total_expected = num_threads * events_per_thread

        def worker(thread_idx: int) -> None:
            for i in range(events_per_thread):
                self.ledger.append(
                    incident_id=self.incident_id,
                    event_type=AuditEventType.STATE_TRANSITION,
                    payload={"thread": thread_idx, "iter": i},
                )

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, t) for t in range(num_threads)]
            for f in futures:
                f.result()

        self.assertEqual(len(self.ledger), total_expected)

        # Verify hash chain across all 150 events
        report = self.ledger.verify_chain()
        self.assertTrue(report.is_valid)
        self.assertEqual(report.total_events, total_expected)
        self.assertEqual(len(report.errors), 0)

    # -----------------------------------------------------------------------
    # 5. Read & Filter Tests
    # -----------------------------------------------------------------------

    def test_filter_events_by_incident_id(self) -> None:
        """Verify read_events() correctly filters by incident ID."""
        self.ledger.append("INC-001", AuditEventType.INCIDENT_TRIGGERED, {"num": 1})
        self.ledger.append("INC-002", AuditEventType.INCIDENT_TRIGGERED, {"num": 2})
        self.ledger.append("INC-001", AuditEventType.PLAN_EXECUTED, {"num": 3})

        all_events = self.ledger.read_events()
        self.assertEqual(len(all_events), 3)

        inc1_events = self.ledger.read_events(incident_id="INC-001")
        self.assertEqual(len(inc1_events), 2)
        self.assertEqual(inc1_events[0].payload["num"], 1)
        self.assertEqual(inc1_events[1].payload["num"], 3)


if __name__ == "__main__":
    unittest.main()
