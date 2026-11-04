"""Tests for ThermoPilot Multi-Agent Switch Negotiation (Milestone 2).

Verifies:
- Independent agent connections & credential isolation (forecast, workload, planner).
- Structured message schemas & round-trip markdown/JSON parsing.
- Inter-agent message routing (incident -> alert -> bids -> recovery -> safety -> complete).
- Deterministic SafetyGate evaluation & explicit rejection of unsafe plans (Plan D).
- Mutation command rejection across all agents.
- Loopback suppression and duplicate event idempotency.
- MultiAgentSwitchRunner startup and shutdown lifecycle.
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from thermopilot.planner.planner import RecoveryPlanner
from thermopilot.safety.gate import SafetyGate
from thermopilot.schemas import NodeConstraints
from thermopilot.switch_adapter.adapter import ForecastSwitchAdapter
from thermopilot.switch_adapter.client import (
    SwitchHttpClient,
    SwitchWebSocketClient,
)
from thermopilot.switch_adapter.config import SwitchAdapterConfig
from thermopilot.switch_adapter.messages import (
    DecisionCardItem,
    DecisionCardPayload,
    ExecutionResultPayload,
    NegotiationCompletePayload,
    RecoveryCandidateItem,
    RecoveryProposalPayload,
    SafetyEvaluationItem,
    SafetyResultPayload,
    ThermalAlertPayload,
    VerificationReportPayload,
    WorkloadActionItem,
    WorkloadBidItem,
    WorkloadBidPayload,
    format_discord_message,
    parse_switch_message,
)
from thermopilot.switch_adapter.planner_adapter import PlannerSwitchAdapter
from thermopilot.switch_adapter.runner import MultiAgentSwitchRunner
from thermopilot.switch_adapter.workload_adapter import WorkloadSwitchAdapter


class TestSwitchMultiAgent(unittest.IsolatedAsyncioTestCase):
    """Test suite for three-agent Switch communication and governance."""

    def setUp(self) -> None:
        self.room_id = "test-room-discord-1234"

        self.cfg_forecast = SwitchAdapterConfig(
            bridge_url="http://localhost:8000",
            agent_id="11111111-1111-1111-1111-111111111111",
            api_key="secret-forecast-key",
            connection_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            authorized_room_ids=frozenset([self.room_id]),
            client_name="thermopilot-forecast-adapter",
        )
        self.cfg_workload = SwitchAdapterConfig(
            bridge_url="http://localhost:8000",
            agent_id="22222222-2222-2222-2222-222222222222",
            api_key="secret-workload-key",
            connection_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            authorized_room_ids=frozenset([self.room_id]),
            client_name="thermopilot-workload-adapter",
        )
        self.cfg_planner = SwitchAdapterConfig(
            bridge_url="http://localhost:8000",
            agent_id="33333333-3333-3333-3333-333333333333",
            api_key="secret-planner-key",
            connection_id="cccccccc-cccc-cccc-cccc-cccccccccccc",
            authorized_room_ids=frozenset([self.room_id]),
            client_name="thermopilot-planner-adapter",
        )

    # -----------------------------------------------------------------------
    # 1. Independent Agent Connections & Credential Isolation
    # -----------------------------------------------------------------------

    def test_credential_and_connection_isolation(self) -> None:
        """Verify each agent has independent credentials, connection IDs, and names."""
        self.assertNotEqual(self.cfg_forecast.agent_id, self.cfg_workload.agent_id)
        self.assertNotEqual(self.cfg_workload.agent_id, self.cfg_planner.agent_id)
        self.assertNotEqual(self.cfg_forecast.api_key, self.cfg_workload.api_key)
        self.assertNotEqual(
            self.cfg_forecast.connection_id, self.cfg_workload.connection_id
        )
        self.assertNotEqual(
            self.cfg_workload.connection_id, self.cfg_planner.connection_id
        )
        self.assertEqual(self.cfg_forecast.client_name, "thermopilot-forecast-adapter")
        self.assertEqual(self.cfg_workload.client_name, "thermopilot-workload-adapter")
        self.assertEqual(self.cfg_planner.client_name, "thermopilot-planner-adapter")

    def test_independent_cursor_and_connection_state(self) -> None:
        """Verify modifying one adapter's cursor or generation does not affect others."""
        mock_http = MagicMock(spec=SwitchHttpClient)
        mock_ws = MagicMock(spec=SwitchWebSocketClient)

        adapter_f = ForecastSwitchAdapter(
            self.cfg_forecast, http_client=mock_http, ws_client=mock_ws
        )
        adapter_w = WorkloadSwitchAdapter(
            self.cfg_workload, http_client=mock_http, ws_client=mock_ws
        )
        adapter_p = PlannerSwitchAdapter(
            self.cfg_planner, http_client=mock_http, ws_client=mock_ws
        )

        adapter_f.current_cursor = 100
        adapter_w.current_cursor = 200
        adapter_p.current_cursor = 300

        self.assertEqual(adapter_f.current_cursor, 100)
        self.assertEqual(adapter_w.current_cursor, 200)
        self.assertEqual(adapter_p.current_cursor, 300)

    # -----------------------------------------------------------------------
    # 2. Structured Message Schemas & Parsing
    # -----------------------------------------------------------------------

    def test_structured_messages_serialization_roundtrip(self) -> None:
        """Test formatting and parsing roundtrip for all 5 message types."""
        # A. Thermal Alert
        alert = ThermalAlertPayload(
            incident_id="inc-test-1",
            gpu_id=0,
            current_temp_c=83.2,
            critical_threshold_c=85.0,
            time_to_threshold_s=21.0,
            rate_c_per_s=0.35,
            risk_status="CRITICAL_RISK",
            reason="Rapid temperature increase",
        )
        txt = format_discord_message(alert)
        self.assertIn("🚨 **[ThermoPilot Perception Alert]**", txt)
        self.assertIn("```thermopilot", txt)
        parsed_alert = parse_switch_message(txt)
        self.assertIsInstance(parsed_alert, ThermalAlertPayload)
        self.assertEqual(parsed_alert.incident_id, "inc-test-1")
        self.assertEqual(parsed_alert.current_temp_c, 83.2)

        # B. Workload Bid
        bid = WorkloadBidPayload(
            incident_id="inc-test-1",
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
                    rationale="VIP workload non-preemptible",
                    estimated_trade_off="30W electrical concession",
                )
            ],
        )
        txt = format_discord_message(bid)
        self.assertIn("💼 **[ThermoPilot Workload Bids]**", txt)
        parsed_bid = parse_switch_message(txt)
        self.assertIsInstance(parsed_bid, WorkloadBidPayload)
        self.assertEqual(parsed_bid.incident_id, "inc-test-1")
        self.assertEqual(len(parsed_bid.proposals), 1)

        # C. Recovery Proposal
        proposal = RecoveryProposalPayload(
            incident_id="inc-test-1",
            candidates=[
                RecoveryCandidateItem(
                    plan_id="PLAN-B",
                    name="Balanced Dynamic Power Cap",
                    strategy_type="BALANCED_POWER_CAP",
                    is_recommended=True,
                    projected_temp_max_c=76.4,
                    cooling_time_to_safe_s=12.0,
                    actions=[
                        WorkloadActionItem(
                            workload_id="wl-alpha",
                            action="POWER_CAP",
                            target_power_w=220.0,
                        )
                    ],
                    estimated_cost_usd=0.0,
                    sla_violations_expected=0,
                    rationale="Balanced trade-off",
                    origin="FALLBACK",
                )
            ],
        )
        txt = format_discord_message(proposal)
        self.assertIn("📋 **[ThermoPilot Recovery Candidates]**", txt)
        parsed_prop = parse_switch_message(txt)
        self.assertIsInstance(parsed_prop, RecoveryProposalPayload)

        # D. Safety Result
        safety = SafetyResultPayload(
            incident_id="inc-test-1",
            evaluations=[
                SafetyEvaluationItem(plan_id="PLAN-B", passed=True),
                SafetyEvaluationItem(
                    plan_id="PLAN-D",
                    passed=False,
                    rejection_reasons=["Violates RULE_MIN_POWER_FLOOR"],
                ),
            ],
        )
        txt = format_discord_message(safety)
        self.assertIn("🛡️ **[ThermoPilot SafetyGate Audit]**", txt)
        self.assertIn("BLOCKED BY SAFETYGATE", txt)
        parsed_safety = parse_switch_message(txt)
        self.assertIsInstance(parsed_safety, SafetyResultPayload)
        self.assertFalse(parsed_safety.evaluations[1].passed)

        # E. Negotiation Complete
        complete = NegotiationCompletePayload(
            incident_id="inc-test-1",
            recommended_plan_id="PLAN-B",
            plan_name="Balanced Dynamic Power Cap",
            projected_temp_max_c=76.4,
            cooling_time_s=12.0,
            actions_summary="Alpha 220W, Beta 160W",
            sla_status="0 SLA breaches expected",
        )
        txt = format_discord_message(complete)
        self.assertIn("🏁 **[ThermoPilot Multi-Agent Consensus]**", txt)
        parsed_comp = parse_switch_message(txt)
        self.assertIsInstance(parsed_comp, NegotiationCompletePayload)
        self.assertEqual(parsed_comp.recommended_plan_id, "PLAN-B")

        # F. Decision Card
        card = DecisionCardPayload(
            incident_id="inc-test-1",
            current_temp_c=82.5,
            current_power_w=380.0,
            candidate_plans=[
                DecisionCardItem(
                    plan_id="PLAN-B",
                    name="Balanced Dynamic Power Cap",
                    strategy_type="BALANCED_POWER_CAP",
                    is_recommended=True,
                    is_safe=True,
                    safety_verdict="CERTIFIED_SAFE",
                    projected_temp_max_c=76.4,
                    projected_power_w=380.0,
                    cooling_time_to_safe_s=12.0,
                    estimated_cost_usd=12.0,
                    sla_violations_expected=0,
                    actions_summary="Alpha 220W, Beta 160W",
                    is_executable=True,
                ),
                DecisionCardItem(
                    plan_id="PLAN-D",
                    name="Blind 50% Throttle",
                    strategy_type="UNSAFE_THROTTLE",
                    is_recommended=False,
                    is_safe=False,
                    safety_verdict="REJECTED_BY_SAFETY_GATE",
                    rejection_reasons=["Violates RULE_MIN_POWER_FLOOR"],
                    projected_temp_max_c=70.0,
                    projected_power_w=200.0,
                    cooling_time_to_safe_s=8.0,
                    estimated_cost_usd=500.0,
                    sla_violations_expected=1,
                    actions_summary="Alpha 125W, Beta 75W",
                    is_executable=False,
                ),
            ],
            executable_plan_ids=["PLAN-B"],
            rejected_plan_ids=["PLAN-D"],
            approval_command="@planner-agent approve inc-test-1 PLAN-B",
            rejection_command="@planner-agent reject inc-test-1",
        )
        txt = format_discord_message(card)
        self.assertIn("⚖️ **[ThermoPilot HITL Governance Desk]**", txt)
        self.assertIn("@planner-agent approve", txt)
        parsed_card = parse_switch_message(txt)
        self.assertIsInstance(parsed_card, DecisionCardPayload)
        self.assertEqual(parsed_card.incident_id, "inc-test-1")
        self.assertEqual(parsed_card.executable_plan_ids, ["PLAN-B"])
        self.assertEqual(parsed_card.rejected_plan_ids, ["PLAN-D"])

        # G. Execution Result
        exec_res = ExecutionResultPayload(
            incident_id="inc-test-1",
            execution_id="exec-12345",
            plan_id="PLAN-B",
            operator_id="operator-alice",
            success=True,
            applied_actions=[
                {
                    "workload_id": "wl-alpha",
                    "action": "POWER_CAP",
                    "target_power_w": 220.0,
                }
            ],
            node_power_w=380.0,
            revalidation_passed=True,
        )
        txt = format_discord_message(exec_res)
        self.assertIn("⚡ **[ThermoPilot Actuation Report]**", txt)
        parsed_exec = parse_switch_message(txt)
        self.assertIsInstance(parsed_exec, ExecutionResultPayload)
        self.assertTrue(parsed_exec.success)

        # H. Verification Report
        verif = VerificationReportPayload(
            incident_id="inc-test-1",
            verification_id="verif-12345",
            plan_id="PLAN-B",
            recovery_successful=True,
            initial_temp_c=82.5,
            final_temp_c=74.2,
            cooling_rate_c_per_s=-0.415,
            reached_target=True,
            ticks_to_target=14,
            audit_hash="a1b2c3d4e5f607182930",
        )
        txt = format_discord_message(verif)
        self.assertIn("🎯 **[ThermoPilot Verification Report]**", txt)
        parsed_verif = parse_switch_message(txt)
        self.assertIsInstance(parsed_verif, VerificationReportPayload)
        self.assertTrue(parsed_verif.recovery_successful)

    def test_parse_invalid_or_non_json_messages(self) -> None:
        """Verify parser returns None for unstructured chat text."""
        self.assertIsNone(parse_switch_message(""))
        self.assertIsNone(
            parse_switch_message("Hello @ForecastAgent what is the temperature?")
        )
        self.assertIsNone(parse_switch_message("```json\n{'not': 'valid json'}\n```"))
        self.assertIsNone(
            parse_switch_message('```thermopilot\n{"type": "unknown_type"}\n```')
        )

    # -----------------------------------------------------------------------
    # 3. Inter-Agent Negotiation Message Routing
    # -----------------------------------------------------------------------

    async def test_full_negotiation_workflow(self) -> None:
        """Test complete 3-agent pipeline: incident -> alert -> bids -> plan & safety."""
        mock_http_f = AsyncMock(spec=SwitchHttpClient)
        mock_http_w = AsyncMock(spec=SwitchHttpClient)
        mock_http_p = AsyncMock(spec=SwitchHttpClient)

        adapter_f = ForecastSwitchAdapter(self.cfg_forecast, http_client=mock_http_f)
        adapter_w = WorkloadSwitchAdapter(self.cfg_workload, http_client=mock_http_w)
        adapter_p = PlannerSwitchAdapter(self.cfg_planner, http_client=mock_http_p)

        # Step 1: Operator triggers simulated incident via ForecastAgent
        trigger_event = {
            "type": "message",
            "room_id": self.room_id,
            "payload": {
                "sender_name": "operator-alice",
                "addressed": True,
                "body": "@ForecastAgent simulate-incident",
            },
        }
        alert_content = await adapter_f.handle_message_event(trigger_event)
        self.assertIsNotNone(alert_content)
        mock_http_f.send_message.assert_awaited_once()
        sent_alert_text = mock_http_f.send_message.call_args[1]["content"]

        # Parse alert emitted by forecast-agent
        alert_msg = parse_switch_message(sent_alert_text)
        self.assertIsInstance(alert_msg, ThermalAlertPayload)
        self.assertEqual(alert_msg.risk_status, "CRITICAL_RISK")

        # Step 2: WorkloadAgent receives alert and generates bids
        workload_event = {
            "type": "message",
            "room_id": self.room_id,
            "payload": {
                "sender_name": "forecast-agent",
                "addressed": False,
                "body": sent_alert_text,
            },
        }
        bids_content = await adapter_w.handle_message_event(workload_event)
        self.assertIsNotNone(bids_content)
        mock_http_w.send_message.assert_awaited_once()
        sent_bids_text = mock_http_w.send_message.call_args[1]["content"]

        # Parse bids emitted by workload-agent
        bids_msg = parse_switch_message(sent_bids_text)
        self.assertIsInstance(bids_msg, WorkloadBidPayload)
        self.assertEqual(bids_msg.incident_id, alert_msg.incident_id)
        # Verify Alpha is ranked first due to P0 priority
        self.assertEqual(bids_msg.proposals[0].priority, "P0_CRITICAL")
        self.assertFalse(bids_msg.proposals[0].can_preempt)

        # Step 3: PlannerAgent receives bids, evaluates SafetyGate, and emits recovery cards
        planner_event = {
            "type": "message",
            "room_id": self.room_id,
            "payload": {
                "sender_name": "workload-agent",
                "addressed": False,
                "body": sent_bids_text,
            },
        }
        planner_content = await adapter_p.handle_message_event(planner_event)
        self.assertIsNotNone(planner_content)
        self.assertIsInstance(planner_content, list)
        self.assertEqual(len(planner_content), 4)

        # Verify 4 messages dispatched by planner: proposal, safety, complete, decision card
        self.assertEqual(mock_http_p.send_message.await_count, 4)

        msg_proposal = parse_switch_message(planner_content[0])
        msg_safety = parse_switch_message(planner_content[1])
        msg_complete = parse_switch_message(planner_content[2])
        msg_card = parse_switch_message(planner_content[3])

        self.assertIsInstance(msg_proposal, RecoveryProposalPayload)
        self.assertIsInstance(msg_safety, SafetyResultPayload)
        self.assertIsInstance(msg_complete, NegotiationCompletePayload)
        self.assertIsInstance(msg_card, DecisionCardPayload)
        self.assertEqual(msg_card.incident_id, alert_msg.incident_id)
        self.assertIn("PLAN-B", msg_card.executable_plan_ids)
        self.assertIn("PLAN-D", msg_card.rejected_plan_ids)

        # Step 4: Verify SafetyGate certification & unsafe plan rejection
        evaluations = {e.plan_id: e for e in msg_safety.evaluations}
        self.assertTrue(evaluations["PLAN-B"].passed)
        self.assertTrue(evaluations["PLAN-A"].passed)
        # Plan D must be rejected by SafetyGate!
        self.assertIn("PLAN-D", evaluations)
        self.assertFalse(evaluations["PLAN-D"].passed)
        self.assertGreater(len(evaluations["PLAN-D"].rejection_reasons), 0)

        # Verify consensus complete card recommends safe Plan B
        self.assertEqual(msg_complete.recommended_plan_id, "PLAN-B")

    # -----------------------------------------------------------------------
    # 4. Safety Boundary Enforcement & Rejection
    # -----------------------------------------------------------------------

    async def test_mutation_rejection_across_all_agents(self) -> None:
        """Verify neither workload-agent nor planner-agent allow chat actuation."""
        mock_http_w = AsyncMock(spec=SwitchHttpClient)
        mock_http_p = AsyncMock(spec=SwitchHttpClient)

        adapter_w = WorkloadSwitchAdapter(self.cfg_workload, http_client=mock_http_w)
        adapter_p = PlannerSwitchAdapter(self.cfg_planner, http_client=mock_http_p)

        for mutation_cmd in [
            "approve plan B",
            "execute PLAN-A",
            "apply",
            "kill wl-beta",
            "throttle now",
            "mitigate",
        ]:
            # Workload agent rejection
            res_w = await adapter_w.handle_message_event(
                {
                    "type": "message",
                    "room_id": self.room_id,
                    "payload": {
                        "sender_name": "operator",
                        "addressed": True,
                        "body": f"@WorkloadAgent {mutation_cmd}",
                    },
                }
            )
            self.assertIn("Execution Denied", res_w)

            # Planner agent rejection
            res_p = await adapter_p.handle_message_event(
                {
                    "type": "message",
                    "room_id": self.room_id,
                    "payload": {
                        "sender_name": "operator",
                        "addressed": True,
                        "body": f"@PlannerAgent {mutation_cmd}",
                    },
                }
            )
            self.assertIn("Execution Denied", res_p)

    def test_safety_gate_rejects_unsafe_candidate(self) -> None:
        """Directly verify that SafetyGate rejects candidate Plan D."""
        planner = RecoveryPlanner(safety_gate=SafetyGate(NodeConstraints()))
        bids = WorkloadBidPayload(
            incident_id="inc-test-gate",
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

        adapter_p = PlannerSwitchAdapter(self.cfg_planner, planner=planner)
        _rec_prop, saf_res, _neg_comp, dec_card = adapter_p.plan_and_certify(bids)
        self.assertIn("PLAN-D", dec_card.rejected_plan_ids)
        self.assertIn("PLAN-B", dec_card.executable_plan_ids)

        eval_d = next(e for e in saf_res.evaluations if e.plan_id == "PLAN-D")
        self.assertFalse(eval_d.passed)
        reasons_str = "; ".join(eval_d.rejection_reasons)
        # Unsafe Plan D cuts Alpha to 125W which violates its 200W min floor
        self.assertTrue(
            "minimum operational power floor" in reasons_str
            or "RULE_MIN_POWER_FLOOR" in reasons_str
            or "SLA" in reasons_str
        )

    # -----------------------------------------------------------------------
    # 5. Duplicate Handling, Loopback & Idempotency
    # -----------------------------------------------------------------------

    async def test_loopback_and_duplicate_suppression(self) -> None:
        """Verify agents ignore self messages and do not process duplicate incident IDs."""
        mock_http_w = AsyncMock(spec=SwitchHttpClient)
        adapter_w = WorkloadSwitchAdapter(self.cfg_workload, http_client=mock_http_w)

        # 1. Self-message loopback test
        self_event = {
            "type": "message",
            "room_id": self.room_id,
            "payload": {
                "sender_name": "workload-agent",
                "addressed": False,
                "body": "Hello world",
            },
        }
        res = await adapter_w.handle_message_event(self_event)
        self.assertIsNone(res)

        # 2. Duplicate incident alert test
        alert = ThermalAlertPayload(
            incident_id="inc-unique-999",
            gpu_id=0,
            current_temp_c=82.5,
            critical_threshold_c=85.0,
            risk_status="CRITICAL_RISK",
            reason="Thermal anomaly",
        )
        msg_text = format_discord_message(alert)
        event = {
            "type": "message",
            "room_id": self.room_id,
            "payload": {
                "sender_name": "forecast-agent",
                "addressed": False,
                "body": msg_text,
            },
        }

        # First delivery: processed
        res1 = await adapter_w.handle_message_event(event)
        self.assertIsNotNone(res1)
        self.assertEqual(mock_http_w.send_message.await_count, 1)

        # Duplicate delivery: ignored
        res2 = await adapter_w.handle_message_event(event)
        self.assertIsNone(res2)
        # Call count unchanged!
        self.assertEqual(mock_http_w.send_message.await_count, 1)

    # -----------------------------------------------------------------------
    # 6. MultiAgentSwitchRunner Lifecycle
    # -----------------------------------------------------------------------

    async def test_runner_start_and_stop_lifecycle(self) -> None:
        """Verify MultiAgentSwitchRunner starts and stops all 3 agents cleanly."""
        mock_http = AsyncMock(spec=SwitchHttpClient)
        mock_ws_f = MagicMock(spec=SwitchWebSocketClient)
        mock_ws_w = MagicMock(spec=SwitchWebSocketClient)
        mock_ws_p = MagicMock(spec=SwitchWebSocketClient)

        # Mock event stream that yields ping then sleeps
        async def fake_stream():
            yield {
                "event": "connection_state",
                "data": {"rooms": [self.room_id], "generation": 1},
            }
            yield {"event": "ping", "data": {}}
            while True:
                await asyncio.sleep(0.1)

        mock_ws_f.connect_events = MagicMock(return_value=fake_stream())
        mock_ws_w.connect_events = MagicMock(return_value=fake_stream())
        mock_ws_p.connect_events = MagicMock(return_value=fake_stream())

        mock_ws_f.close = AsyncMock()
        mock_ws_w.close = AsyncMock()
        mock_ws_p.close = AsyncMock()

        adapter_f = ForecastSwitchAdapter(
            self.cfg_forecast, http_client=mock_http, ws_client=mock_ws_f
        )
        adapter_w = WorkloadSwitchAdapter(
            self.cfg_workload, http_client=mock_http, ws_client=mock_ws_w
        )
        adapter_p = PlannerSwitchAdapter(
            self.cfg_planner, http_client=mock_http, ws_client=mock_ws_p
        )

        runner = MultiAgentSwitchRunner(
            forecast_adapter=adapter_f,
            workload_adapter=adapter_w,
            planner_adapter=adapter_p,
        )

        run_task = asyncio.create_task(runner.start())
        await asyncio.sleep(0.05)
        self.assertTrue(runner.is_running)

        await runner.stop()
        await asyncio.sleep(0.05)
        self.assertFalse(runner.is_running)

        try:
            await asyncio.wait_for(run_task, timeout=0.5)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

        mock_ws_f.close.assert_awaited()
        mock_ws_w.close.assert_awaited()
        mock_ws_p.close.assert_awaited()
