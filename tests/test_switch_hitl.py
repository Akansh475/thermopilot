"""Tests for Switch HITL governance, command parsing, and incident/plan binding."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock

from thermopilot.switch_adapter.client import SwitchHttpClient
from thermopilot.switch_adapter.config import SwitchAdapterConfig
from thermopilot.switch_adapter.messages import (
    WorkloadBidItem,
    WorkloadBidPayload,
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
