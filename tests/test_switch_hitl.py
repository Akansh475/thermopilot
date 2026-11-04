"""Tests for Switch HITL governance, command parsing, and incident/plan binding."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from thermopilot.audit.ledger import AuditLedger
from thermopilot.schemas import IncidentState, NodeConstraints
from thermopilot.switch_adapter.client import SwitchHttpClient
from thermopilot.switch_adapter.config import SwitchAdapterConfig
from thermopilot.switch_adapter.messages import (
    ExecutionResultPayload,
    VerificationReportPayload,
    WorkloadBidItem,
    WorkloadBidPayload,
    parse_switch_message,
)
from thermopilot.switch_adapter.planner_adapter import PlannerSwitchAdapter


class TestSwitchHitlBinding(unittest.IsolatedAsyncioTestCase):
    """Test suite for Switch HITL command parsing and incident/plan binding."""

    def setUp(self) -> None:
        self.room_id = "test-room-hitl"
        self.config = SwitchAdapterConfig(
            agent_id="test-planner-id",
            api_key="test-planner-key",
            client_name="planner-agent",
            allowed_rooms=frozenset({self.room_id}),
            authorized_operators=frozenset({"operator-alice", "oncall-bob"}),
            approval_timeout_s=30.0,
        )
        self.mock_http = AsyncMock(spec=SwitchHttpClient)
        self.adapter = PlannerSwitchAdapter(self.config, http_client=self.mock_http)

    def test_parse_governance_command_variants(self) -> None:
        """Verify command parsing for explicit, shorthand, and uppercase/lowercase tokens."""
        # 1. Explicit command: approve <incident_id> <plan_id>
        cmd, inc_id, p_id = self.adapter.parse_governance_command("approve inc-9001 PLAN-B")
        self.assertEqual(cmd, "approve")
        self.assertEqual(inc_id, "inc-9001")
        self.assertEqual(p_id, "PLAN-B")

        # 2. Case-insensitive plan normalization
        cmd, inc_id, p_id = self.adapter.parse_governance_command("approve inc-9001 plan-b")
        self.assertEqual(p_id, "PLAN-B")

        # 3. Single-letter plan shorthand
        cmd, inc_id, p_id = self.adapter.parse_governance_command("approve inc-9001 b")
        self.assertEqual(p_id, "PLAN-B")

        # 4. Explicit reject
        cmd, inc_id, p_id = self.adapter.parse_governance_command("reject inc-9001")
        self.assertEqual(cmd, "reject")
        self.assertEqual(inc_id, "inc-9001")
        self.assertIsNone(p_id)

        # 5. Non-governance commands
        cmd, inc_id, p_id = self.adapter.parse_governance_command("status")
        self.assertEqual(cmd, "")

    def test_parse_governance_command_single_incident_inference(self) -> None:
        """Verify incident inference when exactly one incident is active."""
        self.adapter.active_incidents["inc-active-42"] = {"room_id": self.room_id}

        cmd, inc_id, p_id = self.adapter.parse_governance_command("approve PLAN-B")
        self.assertEqual(cmd, "approve")
        self.assertEqual(inc_id, "inc-active-42")
        self.assertEqual(p_id, "PLAN-B")

        cmd, inc_id, _ = self.adapter.parse_governance_command("reject")
        self.assertEqual(cmd, "reject")
        self.assertEqual(inc_id, "inc-active-42")

    async def test_binding_rejection_when_no_active_incident(self) -> None:
        """Reject approval when no incident is currently active."""
        event = {
            "type": "message",
            "room_id": self.room_id,
            "payload": {
                "sender_name": "operator-alice",
                "addressed": True,
                "body": "@planner-agent approve inc-9999 PLAN-B",
            },
        }
        reply = await self.adapter.handle_message_event(event)
        self.assertIsNotNone(reply)
        self.assertIn("Execution Denied: Unknown Incident", reply)

    async def test_binding_rejection_for_mismatched_incident(self) -> None:
        """Reject approval when incident ID does not match active incident."""
        # Seed an active incident via plan_and_certify
        bids = WorkloadBidPayload(
            incident_id="inc-correct-123",
            proposals=[
                WorkloadBidItem(
                    workload_id="wl-alpha",
                    name="Alpha",
                    priority="P0_CRITICAL",
                    current_power_w=250.0,
                    min_power_w=200.0,
                    requested_power_w=220.0,
                    can_preempt=False,
                    sla_importance_score=0.9,
                    rationale="Critical",
                    estimated_trade_off="None",
                )
            ],
        )
        self.adapter.plan_and_certify(bids, room_id=self.room_id)

        # Operator attempts to approve mismatched incident ID
        event = {
            "type": "message",
            "room_id": self.room_id,
            "payload": {
                "sender_name": "operator-alice",
                "addressed": True,
                "body": "@planner-agent approve inc-wrong-999 PLAN-B",
            },
        }
        reply = await self.adapter.handle_message_event(event)
        self.assertIsNotNone(reply)
        self.assertIn("Execution Denied: Unknown Incident", reply)

    async def test_binding_rejection_for_missing_plan_id(self) -> None:
        """Prompt operator for plan ID when omitted in multi-candidate situation."""
        bids = WorkloadBidPayload(
            incident_id="inc-multi-plan",
            proposals=[
                WorkloadBidItem(
                    workload_id="wl-alpha",
                    name="Alpha",
                    priority="P0_CRITICAL",
                    current_power_w=250.0,
                    min_power_w=200.0,
                    requested_power_w=220.0,
                    can_preempt=False,
                    sla_importance_score=0.9,
                    rationale="Critical",
                    estimated_trade_off="None",
                )
            ],
        )
        self.adapter.plan_and_certify(bids, room_id=self.room_id)

        event = {
            "type": "message",
            "room_id": self.room_id,
            "payload": {
                "sender_name": "operator-alice",
                "addressed": True,
                "body": "@planner-agent approve inc-multi-plan",
            },
        }
        reply = await self.adapter.handle_message_event(event)
        self.assertIsNotNone(reply)
        self.assertIn("Plan ID Required", reply)


class TestSwitchOperatorAuthorization(unittest.IsolatedAsyncioTestCase):
    """Test suite for operator authorization and security boundaries."""

    def setUp(self) -> None:
        self.room_id = "test-room-auth"
        self.config = SwitchAdapterConfig(
            agent_id="test-planner-id",
            api_key="test-planner-key",
            client_name="planner-agent",
            allowed_rooms=frozenset({self.room_id}),
            authorized_operators=frozenset({"operator-alice", "oncall-bob"}),
            approval_timeout_s=30.0,
        )
        self.mock_http = AsyncMock(spec=SwitchHttpClient)
        self.adapter = PlannerSwitchAdapter(self.config, http_client=self.mock_http)

        # Seed an active incident
        bids = WorkloadBidPayload(
            incident_id="inc-auth-test",
            proposals=[
                WorkloadBidItem(
                    workload_id="wl-alpha-inference",
                    name="Financial Trading Batch",
                    priority="P0_CRITICAL",
                    current_power_w=250.0,
                    min_power_w=200.0,
                    requested_power_w=220.0,
                    can_preempt=False,
                    sla_importance_score=0.98,
                    rationale="VIP non-preemptible",
                    estimated_trade_off="None",
                ),
                WorkloadBidItem(
                    workload_id="wl-beta-training",
                    name="Diffusion Fine-Tuning",
                    priority="P2_FLEXIBLE",
                    current_power_w=150.0,
                    min_power_w=150.0,
                    requested_power_w=160.0,
                    can_preempt=True,
                    sla_importance_score=0.15,
                    rationale="Elastic batch training",
                    estimated_trade_off="Can cap or pause",
                ),
            ],
        )
        self.adapter.plan_and_certify(bids, room_id=self.room_id)

    async def test_autonomous_agents_strictly_blocked_from_approving(self) -> None:
        """Autonomous agents must never be permitted to approve recovery plans."""
        for agent_sender in ("forecast-agent", "workload-agent", "eval-agent", "custom-adapter"):
            event = {
                "type": "message",
                "room_id": self.room_id,
                "payload": {
                    "sender_name": agent_sender,
                    "addressed": True,
                    "body": "@planner-agent approve inc-auth-test PLAN-B",
                },
            }
            reply = await self.adapter.handle_message_event(event)
            self.assertIsNotNone(reply)
            self.assertIn("Autonomous agents cannot authorize execution", reply)
            self.assertIn("Execution Denied", reply)

    async def test_unauthorized_user_blocked_from_approving(self) -> None:
        """Arbitrary chat users not in authorized_operators must be rejected."""
        event = {
            "type": "message",
            "room_id": self.room_id,
            "payload": {
                "sender_name": "mallory-attacker",
                "addressed": True,
                "body": "@planner-agent approve inc-auth-test PLAN-B",
            },
        }
        reply = await self.adapter.handle_message_event(event)
        self.assertIsNotNone(reply)
        self.assertIn("Unauthorized Operator", reply)
        self.assertIn("Execution Denied", reply)

    async def test_authorized_human_operator_allowed(self) -> None:
        """Authorized operator approvals are accepted through the security check."""
        event = {
            "type": "message",
            "room_id": self.room_id,
            "payload": {
                "sender_name": "operator-alice",
                "addressed": True,
                "body": "@planner-agent approve inc-auth-test PLAN-B",
            },
        }
        reply = await self.adapter.handle_message_event(event)
        self.assertIsNotNone(reply)
        combined = "\n".join(reply) if isinstance(reply, list) else str(reply)
        self.assertIn("Plan `PLAN-B` Approved and Pre-Execution Certified by SafetyGate", combined)
        self.assertIn("operator-alice", combined)
        self.assertIn("execution_result", combined)

    async def test_non_approval_mutation_commands_rejected_for_everyone(self) -> None:
        """Raw execution or mitigation commands in chat are blocked even for authorized operators."""
        for cmd in ("execute PLAN-B", "apply changes", "kill wl-alpha", "throttle now", "mitigate"):
            event = {
                "type": "message",
                "room_id": self.room_id,
                "payload": {
                    "sender_name": "operator-alice",
                    "addressed": True,
                    "body": f"@planner-agent {cmd}",
                },
            }
            reply = await self.adapter.handle_message_event(event)
            self.assertIsNotNone(reply)
            self.assertIn("Recovery plans cannot be executed from chat", reply)
            self.assertIn("Execution Denied", reply)


class TestSwitchTimeoutAndDuplicate(unittest.IsolatedAsyncioTestCase):
    """Test suite for approval timeout escalation and duplicate protection."""

    def setUp(self) -> None:
        self.room_id = "test-room-timeout"
        self.config = SwitchAdapterConfig(
            agent_id="test-planner-id",
            api_key="test-planner-key",
            client_name="planner-agent",
            allowed_rooms=frozenset({self.room_id}),
            authorized_operators=frozenset({"operator-alice"}),
            approval_timeout_s=30.0,
        )
        self.mock_http = AsyncMock(spec=SwitchHttpClient)
        self.adapter = PlannerSwitchAdapter(self.config, http_client=self.mock_http)

    def _seed_incident(self, incident_id: str) -> None:
        bids = WorkloadBidPayload(
            incident_id=incident_id,
            proposals=[
                WorkloadBidItem(
                    workload_id="wl-alpha-inference",
                    name="Financial Trading Batch",
                    priority="P0_CRITICAL",
                    current_power_w=250.0,
                    min_power_w=200.0,
                    requested_power_w=220.0,
                    can_preempt=False,
                    sla_importance_score=0.98,
                    rationale="VIP non-preemptible",
                    estimated_trade_off="None",
                ),
                WorkloadBidItem(
                    workload_id="wl-beta-training",
                    name="Diffusion Fine-Tuning",
                    priority="P2_FLEXIBLE",
                    current_power_w=150.0,
                    min_power_w=150.0,
                    requested_power_w=160.0,
                    can_preempt=True,
                    sla_importance_score=0.15,
                    rationale="Elastic batch training",
                    estimated_trade_off="Can cap or pause",
                ),
            ],
        )
        self.adapter.plan_and_certify(bids, room_id=self.room_id)

    async def test_approval_timeout_escalation(self) -> None:
        """When approval is submitted after governance window, escalate to timeout."""
        self._seed_incident("inc-timeout-1")
        # Simulate elapsed time beyond 30.0s window
        self.adapter.active_incidents["inc-timeout-1"]["timestamp"] = time.time() - 35.0

        event = {
            "type": "message",
            "room_id": self.room_id,
            "payload": {
                "sender_name": "operator-alice",
                "addressed": True,
                "body": "@planner-agent approve inc-timeout-1 PLAN-B",
            },
        }
        reply = await self.adapter.handle_message_event(event)
        self.assertIsNotNone(reply)
        self.assertIn("TIMEOUT_ESCALATED", reply)
        self.assertIn("Workload mutations are strictly blocked", reply)
        self.assertIn("inc-timeout-1", self.adapter.finalized_incidents)
        self.assertNotIn("inc-timeout-1", self.adapter.active_incidents)
        sm = self.adapter.state_machines["inc-timeout-1"]
        self.assertEqual(sm.current_state, IncidentState.TIMEOUT_ESCALATED)

    async def test_duplicate_approval_rejected(self) -> None:
        """Subsequent approval attempt for finalized incident is rejected."""
        self._seed_incident("inc-dup-1")

        event = {
            "type": "message",
            "room_id": self.room_id,
            "payload": {
                "sender_name": "operator-alice",
                "addressed": True,
                "body": "@planner-agent approve inc-dup-1 PLAN-B",
            },
        }
        # First approval
        reply1 = await self.adapter.handle_message_event(event)
        combined1 = "\n".join(reply1) if isinstance(reply1, list) else str(reply1)
        self.assertIn("Approved", combined1)
        self.assertIn("execution_result", combined1)

        # Second approval attempt (duplicate replay)
        reply2 = await self.adapter.handle_message_event(event)
        self.assertIn("Duplicate Governance Submission", reply2)
        self.assertIn("Execution Denied", reply2)

    async def test_duplicate_rejection_rejected(self) -> None:
        """Subsequent rejection attempt for finalized incident is rejected."""
        self._seed_incident("inc-dup-2")

        event = {
            "type": "message",
            "room_id": self.room_id,
            "payload": {
                "sender_name": "operator-alice",
                "addressed": True,
                "body": "@planner-agent reject inc-dup-2",
            },
        }
        reply1 = await self.adapter.handle_message_event(event)
        self.assertIn("Rejected by Operator", reply1)
        sm = self.adapter.state_machines["inc-dup-2"]
        self.assertEqual(sm.current_state, IncidentState.REJECTED)

        # Second rejection attempt
        reply2 = await self.adapter.handle_message_event(event)
        self.assertIn("Duplicate Governance Submission", reply2)
        self.assertIn("Execution Denied", reply2)


class TestSwitchPreExecutionSafetyGate(unittest.IsolatedAsyncioTestCase):
    """Test suite verifying SafetyGate pre-execution revalidation and unsafe plan blocking."""

    def setUp(self) -> None:
        self.room_id = "test-room-safety"
        self.config = SwitchAdapterConfig(
            agent_id="test-planner-id",
            api_key="test-planner-key",
            client_name="planner-agent",
            allowed_rooms=frozenset({self.room_id}),
            authorized_operators=frozenset({"operator-alice"}),
            approval_timeout_s=30.0,
        )
        self.mock_http = AsyncMock(spec=SwitchHttpClient)
        self.adapter = PlannerSwitchAdapter(self.config, http_client=self.mock_http)

        # Seed an incident with candidate plans including unsafe Plan D
        bids = WorkloadBidPayload(
            incident_id="inc-safety-gate",
            proposals=[
                WorkloadBidItem(
                    workload_id="wl-alpha-inference",
                    name="Financial Trading Batch",
                    priority="P0_CRITICAL",
                    current_power_w=250.0,
                    min_power_w=200.0,
                    requested_power_w=220.0,
                    can_preempt=False,
                    sla_importance_score=0.98,
                    rationale="VIP non-preemptible",
                    estimated_trade_off="None",
                ),
                WorkloadBidItem(
                    workload_id="wl-beta-training",
                    name="Diffusion Fine-Tuning",
                    priority="P2_FLEXIBLE",
                    current_power_w=150.0,
                    min_power_w=150.0,
                    requested_power_w=160.0,
                    can_preempt=True,
                    sla_importance_score=0.15,
                    rationale="Elastic batch training",
                    estimated_trade_off="Can cap or pause",
                ),
            ],
        )
        self.rec, self.saf, self.neg, self.card = self.adapter.plan_and_certify(
            bids, room_id=self.room_id
        )

    def test_unsafe_plan_d_marked_rejected_in_decision_card(self) -> None:
        """Verify Plan D is categorized as rejected and non-executable in the decision card."""
        self.assertIn("PLAN-D", self.card.rejected_plan_ids)
        self.assertNotIn("PLAN-D", self.card.executable_plan_ids)
        self.assertIn("PLAN-B", self.card.executable_plan_ids)

    async def test_attempted_approval_of_unsafe_plan_d_strictly_blocked(self) -> None:
        """Even if operator attempts to approve Plan D, SafetyGate strictly blocks execution."""
        event = {
            "type": "message",
            "room_id": self.room_id,
            "payload": {
                "sender_name": "operator-alice",
                "addressed": True,
                "body": "@planner-agent approve inc-safety-gate PLAN-D",
            },
        }
        reply = await self.adapter.handle_message_event(event)
        self.assertIsNotNone(reply)
        self.assertIn("Execution Denied by SafetyGate", reply)
        self.assertIn("Selected plan 'PLAN-D' was rejected by the Safety Gate", reply)
        self.assertIn("inc-safety-gate", self.adapter.finalized_incidents)
        self.assertNotIn("inc-safety-gate", self.adapter.active_incidents)

    async def test_pre_execution_revalidation_failure_on_tighter_constraints(self) -> None:
        """If node constraints tighten prior to approval, revalidation blocks execution."""
        # Tighten power budget below Plan B's projected power (~350W)
        self.adapter.desk._incident_constraints["inc-safety-gate"] = NodeConstraints(
            power_budget_w=200.0,
            max_temp_c=70.0,
        )

        event = {
            "type": "message",
            "room_id": self.room_id,
            "payload": {
                "sender_name": "operator-alice",
                "addressed": True,
                "body": "@planner-agent approve inc-safety-gate PLAN-B",
            },
        }
        reply = await self.adapter.handle_message_event(event)
        self.assertIsNotNone(reply)
        self.assertIn("Execution Denied by SafetyGate", reply)
        self.assertIn("Pre-execution Safety Gate revalidation failed", reply)


class TestSwitchSimulationExecutor(unittest.IsolatedAsyncioTestCase):
    """Test suite verifying SimulationExecutor integration on plan approval."""

    def setUp(self) -> None:
        self.room_id = "test-room-executor"
        self.config = SwitchAdapterConfig(
            agent_id="test-planner-id",
            api_key="test-planner-key",
            client_name="planner-agent",
            allowed_rooms=frozenset({self.room_id}),
            authorized_operators=frozenset({"operator-alice"}),
            approval_timeout_s=30.0,
        )
        self.mock_http = AsyncMock(spec=SwitchHttpClient)
        self.adapter = PlannerSwitchAdapter(self.config, http_client=self.mock_http)

        bids = WorkloadBidPayload(
            incident_id="inc-exec-test",
            proposals=[
                WorkloadBidItem(
                    workload_id="wl-alpha-inference",
                    name="Financial Trading Batch",
                    priority="P0_CRITICAL",
                    current_power_w=250.0,
                    min_power_w=200.0,
                    requested_power_w=220.0,
                    can_preempt=False,
                    sla_importance_score=0.98,
                    rationale="VIP non-preemptible",
                    estimated_trade_off="None",
                ),
                WorkloadBidItem(
                    workload_id="wl-beta-training",
                    name="Diffusion Fine-Tuning",
                    priority="P2_FLEXIBLE",
                    current_power_w=150.0,
                    min_power_w=150.0,
                    requested_power_w=160.0,
                    can_preempt=True,
                    sla_importance_score=0.15,
                    rationale="Elastic batch training",
                    estimated_trade_off="Can cap or pause",
                ),
            ],
        )
        self.adapter.plan_and_certify(bids, room_id=self.room_id)

    async def test_simulation_executor_applied_and_dispatched_on_approval(self) -> None:
        """Approving certified plan triggers SimulationExecutor and emits execution card."""
        event = {
            "type": "message",
            "room_id": self.room_id,
            "payload": {
                "sender_name": "operator-alice",
                "addressed": True,
                "body": "@planner-agent approve inc-exec-test PLAN-B",
            },
        }
        res = await self.adapter.handle_message_event(event)
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 3)

        # First message: operator authorization confirmation
        self.assertIn("Approved and Pre-Execution Certified", res[0])

        # Second message: ExecutionResultPayload
        msg_exec = parse_switch_message(res[1])
        self.assertIsInstance(msg_exec, ExecutionResultPayload)
        self.assertEqual(msg_exec.incident_id, "inc-exec-test")
        self.assertEqual(msg_exec.plan_id, "PLAN-B")
        self.assertEqual(msg_exec.operator_id, "operator-alice")
        self.assertTrue(msg_exec.revalidation_passed)
        self.assertTrue(msg_exec.is_simulated)
        self.assertGreater(msg_exec.node_power_w, 0.0)

        # Third message: VerificationReportPayload
        msg_verif = parse_switch_message(res[2])
        self.assertIsInstance(msg_verif, VerificationReportPayload)
        self.assertEqual(msg_verif.incident_id, "inc-exec-test")
        self.assertEqual(msg_verif.plan_id, "PLAN-B")
        self.assertTrue(msg_verif.recovery_successful)
        self.assertTrue(msg_verif.reached_target)
        self.assertLessEqual(msg_verif.final_temp_c, 75.0)

        # State machine transition to RECOVERED
        sm = self.adapter.state_machines["inc-exec-test"]
        self.assertEqual(sm.current_state, IncidentState.RECOVERED)


class TestSwitchAuditLedger(unittest.IsolatedAsyncioTestCase):
    """Test suite verifying cryptographic hash chain audit logging across HITL workflow."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ledger_path = Path(self.temp_dir.name) / "test_audit.jsonl"
        self.ledger = AuditLedger(ledger_path=self.ledger_path)
        self.room_id = "test-room-ledger"
        self.config = SwitchAdapterConfig(
            agent_id="test-planner-id",
            api_key="test-planner-key",
            client_name="planner-agent",
            allowed_rooms=frozenset({self.room_id}),
            authorized_operators=frozenset({"operator-alice"}),
            approval_timeout_s=30.0,
        )
        self.mock_http = AsyncMock(spec=SwitchHttpClient)
        self.adapter = PlannerSwitchAdapter(
            self.config,
            ledger=self.ledger,
            http_client=self.mock_http,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    async def test_full_lifecycle_audit_hash_chain_integrity(self) -> None:
        """Every HITL phase writes to append-only ledger and validates cryptographic chain."""
        incident_id = "inc-ledger-chain-1"
        bids = WorkloadBidPayload(
            incident_id=incident_id,
            proposals=[
                WorkloadBidItem(
                    workload_id="wl-alpha-inference",
                    name="Financial Trading Batch",
                    priority="P0_CRITICAL",
                    current_power_w=250.0,
                    min_power_w=200.0,
                    requested_power_w=220.0,
                    can_preempt=False,
                    sla_importance_score=0.98,
                    rationale="VIP non-preemptible",
                    estimated_trade_off="None",
                ),
                WorkloadBidItem(
                    workload_id="wl-beta-training",
                    name="Diffusion Fine-Tuning",
                    priority="P2_FLEXIBLE",
                    current_power_w=150.0,
                    min_power_w=150.0,
                    requested_power_w=160.0,
                    can_preempt=True,
                    sla_importance_score=0.15,
                    rationale="Elastic batch training",
                    estimated_trade_off="Can cap or pause",
                ),
            ],
        )

        # 1. Perception, Candidate Formulation & Card Dispatch
        self.adapter.plan_and_certify(bids, room_id=self.room_id)

        # 2. Operator Approval & Actuation
        event = {
            "type": "message",
            "room_id": self.room_id,
            "payload": {
                "sender_name": "operator-alice",
                "addressed": True,
                "body": f"@planner-agent approve {incident_id} PLAN-B",
            },
        }
        res = await self.adapter.handle_message_event(event)
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 3)

        # 3. Check Verification Report includes valid cryptographic hash
        msg_verif = parse_switch_message(res[2])
        self.assertIsInstance(msg_verif, VerificationReportPayload)
        self.assertIsNotNone(msg_verif.audit_hash)
        self.assertEqual(msg_verif.audit_hash, self.ledger.last_hash)

        # 4. Validate Cryptographic Hash Chain
        chain_report = self.ledger.verify_chain()
        self.assertTrue(chain_report.is_valid, f"Chain corrupted: {chain_report.errors}")
        self.assertGreaterEqual(chain_report.total_events, 8)

        # 5. Verify all events have simulated flag set to True
        events = self.ledger.read_events(incident_id=incident_id)
        for ev in events:
            self.assertTrue(ev.is_simulated)


class TestSwitchDemonstrationScenarios(unittest.IsolatedAsyncioTestCase):
    """End-to-end verification of the four primary ThermoPilot demonstration scenarios.

    Scenario A: Successful Recovery Lifecycle
        Simulated incident -> multi-agent negotiation -> Plan B certified -> operator approve
        -> simulation execute -> verification passed (RECOVERED).

    Scenario B: Unsafe Plan Blocked by SafetyGate
        Operator attempts to approve unsafe candidate (Plan D) -> SafetyGate blocks
        -> execution blocked -> incident marked REVALIDATION_FAILED.

    Scenario C: Human Rejection
        Plan B certified -> operator issues explicit rejection -> execution blocked
        -> incident marked REJECTED.

    Scenario D: Approval Timeout Escalation
        Plan B certified -> operator fails to respond within governance timeout
        -> submission rejected -> incident marked TIMEOUT_ESCALATED.
    """

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ledger_path = Path(self.temp_dir.name) / "scenario_ledger.jsonl"
        self.ledger = AuditLedger(ledger_path=self.ledger_path)
        self.room_id = "test-demo-room"
        self.config = SwitchAdapterConfig(
            agent_id="test-planner-id",
            api_key="test-planner-key",
            client_name="planner-agent",
            allowed_rooms=frozenset({self.room_id}),
            authorized_operators=frozenset({"operator-alice", "lead-engineer"}),
            approval_timeout_s=10.0,
        )
        self.mock_http = AsyncMock(spec=SwitchHttpClient)
        self.adapter = PlannerSwitchAdapter(
            self.config,
            ledger=self.ledger,
            http_client=self.mock_http,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_standard_bids(self, incident_id: str) -> WorkloadBidPayload:
        return WorkloadBidPayload(
            incident_id=incident_id,
            proposals=[
                WorkloadBidItem(
                    workload_id="wl-alpha-inference",
                    name="Financial Trading Inference",
                    priority="P0_CRITICAL",
                    current_power_w=250.0,
                    min_power_w=200.0,
                    requested_power_w=220.0,
                    can_preempt=False,
                    sla_importance_score=0.99,
                    rationale="Ultra-low latency inference",
                    estimated_trade_off="Power cap to 220W acceptable",
                ),
                WorkloadBidItem(
                    workload_id="wl-beta-training",
                    name="Diffusion Model Pre-Training",
                    priority="P2_FLEXIBLE",
                    current_power_w=150.0,
                    min_power_w=150.0,
                    requested_power_w=160.0,
                    can_preempt=True,
                    sla_importance_score=0.15,
                    rationale="Batch offline workload",
                    estimated_trade_off="Can throttle or pause",
                ),
            ],
        )

    async def test_scenario_a_successful_recovery(self) -> None:
        """Scenario A: Simulated incident -> Plan B certified -> Operator approve -> Executed -> Recovered."""
        incident_id = "inc-demo-scenario-a"
        bids = self._create_standard_bids(incident_id)

        # 1. Perception, formulation, and decision card dispatch
        _, safety_res, _, card = self.adapter.plan_and_certify(bids, room_id=self.room_id)
        self.assertTrue(any(ev.passed for ev in safety_res.evaluations))
        self.assertIn("PLAN-B", card.executable_plan_ids)
        self.assertIn("PLAN-D", card.rejected_plan_ids)

        sm = self.adapter.state_machines[incident_id]
        self.assertEqual(sm.current_state, IncidentState.AWAITING_APPROVAL)

        # 2. Operator submits approval for certified Plan B
        event = {
            "type": "message",
            "room_id": self.room_id,
            "payload": {
                "sender_name": "operator-alice",
                "addressed": True,
                "body": f"@planner-agent approve {incident_id} PLAN-B",
            },
        }
        res = await self.adapter.handle_message_event(event)
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 3)

        # 3. Verify Execution Result
        exec_msg = parse_switch_message(res[1])
        self.assertIsInstance(exec_msg, ExecutionResultPayload)
        self.assertTrue(exec_msg.success)
        self.assertTrue(exec_msg.revalidation_passed)
        self.assertTrue(exec_msg.is_simulated)

        # 4. Verify Post-Execution Verification Report
        verif_msg = parse_switch_message(res[2])
        self.assertIsInstance(verif_msg, VerificationReportPayload)
        self.assertTrue(verif_msg.recovery_successful)
        self.assertTrue(verif_msg.reached_target)
        self.assertGreater(verif_msg.cooling_rate_c_per_s, 0.0)
        self.assertEqual(verif_msg.audit_hash, self.ledger.last_hash)

        # 5. Verify State Machine is in RECOVERED state
        self.assertEqual(sm.current_state, IncidentState.RECOVERED)
        self.assertTrue(sm.is_terminal)

        # 6. Verify Audit Ledger integrity
        report = self.ledger.verify_chain()
        self.assertTrue(report.is_valid)
        self.assertGreaterEqual(report.total_events, 8)

    async def test_scenario_b_unsafe_plan_blocked_by_safetygate(self) -> None:
        """Scenario B: Unsafe Plan D -> SafetyGate blocks -> Pre-execution revalidation fails -> REVALIDATION_FAILED."""
        incident_id = "inc-demo-scenario-b"
        bids = self._create_standard_bids(incident_id)

        self.adapter.plan_and_certify(bids, room_id=self.room_id)
        sm = self.adapter.state_machines[incident_id]

        # Operator attempts to approve uncertified/unsafe Plan D
        event = {
            "type": "message",
            "room_id": self.room_id,
            "payload": {
                "sender_name": "operator-alice",
                "addressed": True,
                "body": f"@planner-agent approve {incident_id} PLAN-D",
            },
        }
        reply = await self.adapter.handle_message_event(event)
        self.assertIsInstance(reply, str)
        self.assertIn("Execution Denied by SafetyGate", reply)
        self.assertIn("PLAN-D", reply)

        # Confirm state transitioned to terminal REVALIDATION_FAILED
        self.assertEqual(sm.current_state, IncidentState.REVALIDATION_FAILED)
        self.assertTrue(sm.is_terminal)

        # Verify incident cannot be executed afterwards
        self.assertIn(incident_id, self.adapter.finalized_incidents)

    async def test_scenario_c_explicit_human_rejection(self) -> None:
        """Scenario C: Plan B certified -> Operator rejects -> No execution -> REJECTED."""
        incident_id = "inc-demo-scenario-c"
        bids = self._create_standard_bids(incident_id)

        self.adapter.plan_and_certify(bids, room_id=self.room_id)
        sm = self.adapter.state_machines[incident_id]

        # Operator rejects
        event = {
            "type": "message",
            "room_id": self.room_id,
            "payload": {
                "sender_name": "lead-engineer",
                "addressed": True,
                "body": f"@planner-agent reject {incident_id}",
            },
        }
        reply = await self.adapter.handle_message_event(event)
        self.assertIsInstance(reply, str)
        self.assertIn("Rejected by Operator", reply)

        # Confirm state transitioned to terminal REJECTED
        self.assertEqual(sm.current_state, IncidentState.REJECTED)
        self.assertTrue(sm.is_terminal)
        self.assertIn(incident_id, self.adapter.finalized_incidents)

    async def test_scenario_d_approval_timeout_escalation(self) -> None:
        """Scenario D: Governance window expires -> Approval rejected -> TIMEOUT_ESCALATED."""
        incident_id = "inc-demo-scenario-d"
        bids = self._create_standard_bids(incident_id)

        self.adapter.plan_and_certify(bids, room_id=self.room_id)
        sm = self.adapter.state_machines[incident_id]

        # Artificially age the incident timestamp beyond 10s timeout
        self.adapter.active_incidents[incident_id]["timestamp"] = time.time() - 25.0

        # Operator tries to approve after timeout
        event = {
            "type": "message",
            "room_id": self.room_id,
            "payload": {
                "sender_name": "operator-alice",
                "addressed": True,
                "body": f"@planner-agent approve {incident_id} PLAN-B",
            },
        }
        reply = await self.adapter.handle_message_event(event)
        self.assertIsInstance(reply, str)
        self.assertIn("TIMEOUT_ESCALATED", reply)
        self.assertIn("Approval Window Expired", reply)

        # Confirm state transitioned to terminal TIMEOUT_ESCALATED
        self.assertEqual(sm.current_state, IncidentState.TIMEOUT_ESCALATED)
        self.assertTrue(sm.is_terminal)
        self.assertIn(incident_id, self.adapter.finalized_incidents)






