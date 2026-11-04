"""Unit and integration test suite for ThermoPilot Compute Desk UI, CLI, and Web Dashboard.

Tests:
1. ComputeDesk terminal rendering (ANSI formatting, telemetry, workloads, plans, safety gate, verification, audit).
2. DemoRunner full 8-stage lifecycle execution.
3. Simulator reset repeatability and state isolation.
4. Plan D SafetyGate block enforcement (preventing approval and execution).
5. Operator explicit rejection and timeout escalation leaving simulator unmutated.
6. Web dashboard HTTP server, HTML template, and REST API endpoints (GET /api/state, POST /api/action, POST /api/reset, POST /api/step).
7. End-to-end cryptographic audit hash chain integrity.
"""

import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from thermopilot.agents.forecast import ForecastResult, ThermalRiskStatus, ThermalTrend
from thermopilot.audit.ledger import AuditLedger
from thermopilot.schemas import (
    ActionType,
    ApprovalDecision,
    EvaluatedPlanCandidate,
    IncidentState,
    NegotiationEvidence,
    PlanOrigin,
    RecoveryPlanOption,
    SafetyGateResult,
    StrategyType,
    VerificationReport,
    WorkflowStatus,
    WorkloadAction,
    WorkloadPriority,
    WorkloadProposal,
)
from thermopilot.ui.demo import DemoRunner, create_demo_simulator, create_demo_workloads
from thermopilot.ui.desk import ComputeDesk
from thermopilot.ui.web import launch_web_dashboard


class TestComputeDeskRendering(unittest.TestCase):
    """Test suite for terminal ANSI and plain-text control desk formatting."""

    def setUp(self) -> None:
        self.desk_color = ComputeDesk(use_color=True)
        self.desk_plain = ComputeDesk(use_color=False)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ledger_path = Path(self.temp_dir.name) / "test_desk_audit.jsonl"
        self.ledger = AuditLedger(self.ledger_path)
        self.workloads = create_demo_workloads()
        self.sim = create_demo_simulator()
        self.telem = self.sim.current_telemetry()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_render_header(self) -> None:
        """Verify header banner contains title, incident ID, and simulation notice."""
        out = self.desk_plain.render_header(incident_id="INC-TEST-001", state=IncidentState.DETECTED)
        self.assertIn("THERMOPILOT COMPUTE DESK", out)
        self.assertIn("INC-TEST-001", out)
        self.assertIn("[DETECTED]", out)
        self.assertIn("SIMULATED ENVIRONMENT", out)

    def test_render_telemetry_panel(self) -> None:
        """Verify telemetry panel contains temperature, power bar, and time-to-trip."""
        forecast = ForecastResult(
            timestamp=self.telem.timestamp,
            gpu_id=0,
            current_temp_c=81.5,
            critical_threshold_c=85.0,
            trend=ThermalTrend.RISING,
            rate_c_per_s=0.15,
            acceleration_c_per_s2=0.002,
            time_to_threshold_s=23.3,
            confidence_score=0.98,
            risk_status=ThermalRiskStatus.CRITICAL_RISK,
            trigger_incident=True,
            reason="Thermal runaway",
        )
        out = self.desk_plain.render_telemetry_panel(self.telem, forecast=forecast)
        self.assertIn("81.5°C", out)
        self.assertIn("380.0W / 400.0W", out)
        self.assertIn("23.3s to 85.0°C Trip", out)
        self.assertIn("32.0°C", out)
        self.assertIn("ANOMALY ACTIVE", out)

    def test_render_workloads_panel(self) -> None:
        """Verify workload tenant panel displays priorities, SLA deadlines, and proposals."""
        negotiation = NegotiationEvidence(
            proposals=[
                WorkloadProposal(
                    workload_id="wl-alpha-inference",
                    priority=WorkloadPriority.P0_CRITICAL,
                    current_power_w=230.0,
                    min_power_w=200.0,
                    sla_importance_score=0.99,
                    requested_power_w=220.0,
                    preferred_action=ActionType.POWER_CAP,
                    can_preempt=False,
                    can_power_cap=True,
                    rationale="VIP trading contract requires min 200W",
                    estimated_trade_off="Negligible throughput impact ($0 penalty)",
                )
            ]
        )
        out = self.desk_plain.render_workloads_panel(self.workloads, negotiation=negotiation)
        self.assertIn("Financial Trading Batch", out)
        self.assertIn("P0_CRITICAL", out)
        self.assertIn("230W", out)
        self.assertIn("200W", out)
        self.assertIn("0.99", out)
        self.assertIn("Proposal: POWER_CAP -> 220W", out)

    def test_render_plans_panel_with_blocked_plan_d(self) -> None:
        """Verify candidate table highlights certified plans and shows explicit SafetyGate rejection for Plan D."""
        plan_a = RecoveryPlanOption(
            plan_id="PLAN-A",
            name="Conservative Workload Shed",
            strategy_type=StrategyType.CONSERVATIVE_SHED,
            is_recommended=True,
            projected_temp_max_c=68.0,
            cooling_time_to_safe_s=14.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=220.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.PAUSE),
            ],
            estimated_cost_usd=45.0,
            sla_violations_expected=0,
            rationale="Pause Beta to cool down",
            origin=PlanOrigin.FALLBACK,
        )
        plan_d = RecoveryPlanOption(
            plan_id="PLAN-D",
            name="Blind 50% Throttle",
            strategy_type=StrategyType.UNSAFE_THROTTLE,
            is_recommended=False,
            projected_temp_max_c=65.0,
            cooling_time_to_safe_s=8.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=125.0),
            ],
            estimated_cost_usd=320.0,
            sla_violations_expected=1,
            rationale="Naive cuts",
            origin=PlanOrigin.FALLBACK,
        )

        cand_a = EvaluatedPlanCandidate(
            plan=plan_a,
            origin=PlanOrigin.FALLBACK,
            safety_result=SafetyGateResult(plan_id="PLAN-A", passed=True, evaluations=[]),
        )
        cand_d = EvaluatedPlanCandidate(
            plan=plan_d,
            origin=PlanOrigin.FALLBACK,
            safety_result=SafetyGateResult(
                plan_id="PLAN-D",
                passed=False,
                rejection_reasons=["P0 SLA contract breach: Expected SLA violations (1) > 0"],
                evaluations=[],
            ),
        )

        out = self.desk_plain.render_plans_panel([cand_a, cand_d])
        self.assertIn("PLAN-A", out)
        self.assertIn("[PASS - CERTIFIED]", out)
        self.assertIn("PLAN-D", out)
        self.assertIn("[BLOCKED BY GATE]", out)
        self.assertIn("P0 SLA contract breach", out)

    def test_render_verification_panel(self) -> None:
        """Verify verification report panel renders cooling rate, Delta T, and SLA pass."""
        report = VerificationReport(
            incident_id="INC-001",
            plan_id="PLAN-A",
            recovery_successful=True,
            initial_temp_c=81.5,
            final_temp_c=63.0,
            min_temp_c=63.0,
            max_temp_c=81.5,
            temp_delta_c=-18.5,
            reached_target=True,
            ticks_to_target=3,
            sim_time_to_target_s=7.5,
            observed_cooling_rate_c_per_s=1.05,
            target_cooling_rate_c_per_s=0.5,
            meets_cooling_rate_target=True,
            ticks_monitored=20,
            simulated_duration_s=50.0,
            summary="Recovery successful: 75C safe envelope reached in 3 ticks.",
            checks=[],
            failure_reasons=[],
        )
        out = self.desk_plain.render_verification_panel(report=report)
        self.assertIn("[RECOVERY SUCCESSFUL - 75°C TARGET ACHIEVED]", out)
        self.assertIn("81.5°C -> 63.0°C", out)
        self.assertIn("1.05°C/s simulated", out)
        self.assertIn("PASS (0 Violations)", out)

    def test_render_audit_panel_and_full_screen(self) -> None:
        """Verify audit trail rendering and complete screen generation."""
        from thermopilot.schemas import AuditEventType
        self.ledger.append(incident_id="INC-001", event_type=AuditEventType.INCIDENT_TRIGGERED, payload={"k": "v"}, is_simulated=True)
        out = self.desk_plain.render_audit_panel(self.ledger)
        self.assertIn("CHAIN VERIFIED", out)
        self.assertIn("INCIDENT_TRIGGERED", out)

        # Full screen rendering without exception
        full = self.desk_plain.render_screen(
            incident_id="INC-001",
            state=IncidentState.DETECTED,
            telemetry=self.telem,
            workloads=self.workloads,
            candidates=[],
            ledger=self.ledger,
        )
        self.assertIsInstance(full, str)
        self.assertGreater(len(full), 200)


class TestDemoRunner(unittest.TestCase):
    """Test suite for DemoRunner end-to-end lifecycle and safety constraints."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ledger_path = Path(self.temp_dir.name) / "test_demo_audit.jsonl"
        self.runner = DemoRunner(incident_id="INC-TEST-001", ledger_path=self.ledger_path, use_color=False)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_demo_reset_repeatability(self) -> None:
        """Verify reset() restores clean state with 81.5°C initial temp, 380W power, and empty ledger."""
        self.runner.step_1_detect()
        self.runner.step_2_forecast()
        self.assertGreaterEqual(len(self.runner.ledger.read_events()), 2)

        # Reset
        self.runner.reset()
        self.assertEqual(self.runner.state_machine.current_state, IncidentState.DETECTED)
        self.assertEqual(self.runner.simulator.effective_power_draw_w, 380.0)
        self.assertAlmostEqual(self.runner.simulator.core_temp_c, 81.5, places=1)
        self.assertEqual(len(self.runner.ledger.read_events()), 0)

    def test_full_auto_demo_execution(self) -> None:
        """Verify run_auto_demo executes all 8 stages, achieving verified recovery and intact hash chain."""
        res = self.runner.run_auto_demo(plan_id="PLAN-A", delay_s=0.0, print_output=False)
        self.assertEqual(res["final_state"], "RECOVERED")
        self.assertTrue(res["recovery_successful"])
        self.assertLessEqual(res["final_temp_c"], 75.0)
        self.assertGreaterEqual(res["observed_cooling_rate_c_per_s"], 0.50)
        self.assertTrue(res["audit_chain_valid"])
        self.assertGreaterEqual(res["audit_events_count"], 10)

        # Verify state machine state
        self.assertEqual(self.runner.state_machine.current_state, IncidentState.RECOVERED)

    def test_plan_d_safety_gate_blocking(self) -> None:
        """Verify Plan D is rejected by SafetyGate and cannot be approved or executed."""
        self.runner.step_1_detect()
        self.runner.step_2_forecast()
        self.runner.step_3_plan()
        card = self.runner.step_4_dispatch_card()

        # Plan D must be in rejected_plan_ids
        self.assertIn("PLAN-D", card.rejected_plan_ids)
        self.assertNotIn("PLAN-D", card.executable_plan_ids)

        # Attempting to approve Plan D must be denied
        with self.assertRaises(ValueError) as ctx:
            self.runner.step_5_operator_decision(ApprovalDecision.APPROVE, plan_id="PLAN-D")
        self.assertIn("BLOCKED by SafetyGate", str(ctx.exception))

        # Simulator power must remain unchanged at 380W
        self.assertEqual(self.runner.simulator.effective_power_draw_w, 380.0)
        self.assertEqual(self.runner.state_machine.current_state, IncidentState.AWAITING_APPROVAL)

    def test_operator_explicit_rejection(self) -> None:
        """Verify operator explicit rejection transitions to REJECTED and leaves simulator unmutated."""
        self.runner.step_1_detect()
        self.runner.step_2_forecast()
        self.runner.step_3_plan()
        self.runner.step_4_dispatch_card()

        wf_res = self.runner.step_5_operator_decision(ApprovalDecision.REJECT, reason="Operator manual veto")
        self.assertEqual(wf_res.status, WorkflowStatus.REJECTED)
        self.assertEqual(self.runner.state_machine.current_state, IncidentState.REJECTED)

        # Simulator power and workloads must be completely unmutated
        self.assertEqual(self.runner.simulator.effective_power_draw_w, 380.0)
        self.assertEqual(self.runner.simulator.get_workload_power("wl-alpha-inference"), 230.0)
        self.assertEqual(self.runner.simulator.get_workload_power("wl-beta-training"), 150.0)

    def test_operator_timeout_escalation(self) -> None:
        """Verify timeout escalation transitions to TIMEOUT_ESCALATED and protects workloads."""
        self.runner.step_1_detect()
        self.runner.step_2_forecast()
        self.runner.step_3_plan()
        self.runner.step_4_dispatch_card()

        wf_res = self.runner.step_5_operator_decision(
            ApprovalDecision.TIMEOUT_ESCALATED, reason="Operator approval window expired"
        )
        self.assertEqual(wf_res.status, WorkflowStatus.TIMEOUT_ESCALATED)
        self.assertEqual(self.runner.state_machine.current_state, IncidentState.TIMEOUT_ESCALATED)

        # Workloads protected from mutation
        self.assertEqual(self.runner.simulator.effective_power_draw_w, 380.0)

    def test_get_state_dict_serialization(self) -> None:
        """Verify get_state_dict produces valid JSON-serializable structure with all required fields."""
        self.runner.step_1_detect()
        self.runner.step_2_forecast()
        self.runner.step_3_plan()
        self.runner.step_4_dispatch_card()

        state_dict = self.runner.get_state_dict()
        self.assertEqual(state_dict["incident_id"], "INC-TEST-001")
        self.assertEqual(state_dict["state"], "AWAITING_APPROVAL")
        self.assertTrue(state_dict["is_simulated"])
        self.assertIn("telemetry", state_dict)
        self.assertIn("workloads", state_dict)
        self.assertIn("candidates", state_dict)
        self.assertIn("audit_ledger", state_dict)
        self.assertTrue(state_dict["audit_ledger"]["chain_valid"])

        # JSON encode test
        encoded = json.dumps(state_dict)
        self.assertIsInstance(encoded, str)
        self.assertGreater(len(encoded), 100)

    def test_get_state_dict_reflects_paused_workload_power(self) -> None:
        """Verify get_state_dict reflects 0W power for paused or migrated workloads."""
        self.runner.step_1_detect()
        self.runner.step_2_forecast()
        self.runner.step_3_plan()
        self.runner.step_4_dispatch_card()
        self.runner.step_5_operator_decision(ApprovalDecision.APPROVE, plan_id="PLAN-A")
        self.runner.step_6_execute()

        state_dict = self.runner.get_state_dict()
        w_map = {w["workload_id"]: w["current_power_w"] for w in state_dict["workloads"]}
        self.assertEqual(w_map["wl-alpha-inference"], 220.0)
        self.assertEqual(w_map["wl-beta-training"], 0.0)
        self.assertEqual(w_map["wl-gamma-analytics"], 0.0)


class TestWebDashboardAPI(unittest.TestCase):
    """Test suite for local Web Control Room Dashboard server and REST APIs."""

    server = None
    port = 8899

    @classmethod
    def setUpClass(cls) -> None:
        """Start local web dashboard on dedicated port."""
        cls.temp_dir = tempfile.TemporaryDirectory()
        ledger_path = Path(cls.temp_dir.name) / "web_test_audit.jsonl"
        runner = DemoRunner(incident_id="INC-WEB-001", ledger_path=ledger_path, use_color=False)
        cls.server = launch_web_dashboard(
            port=cls.port,
            open_browser=False,
            run_in_background=True,
            runner=runner,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Shutdown local HTTP server."""
        if cls.server:
            cls.server.shutdown()
        cls.temp_dir.cleanup()

    def test_get_index_html(self) -> None:
        """Verify GET / returns 200 with HTML control room dashboard."""
        url = f"http://127.0.0.1:{self.port}/"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/html", resp.headers.get("Content-Type", ""))
            content = resp.read().decode("utf-8")
            self.assertIn("ThermoPilot Compute Desk", content)
            self.assertIn("SIMULATED TELEMETRY ONLY", content)

    def test_get_api_state(self) -> None:
        """Verify GET /api/state returns valid JSON with telemetry and workload data."""
        url = f"http://127.0.0.1:{self.port}/api/state"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["incident_id"], "INC-WEB-001")
            self.assertTrue(data["is_simulated"])
            self.assertIn("telemetry", data)
            self.assertIn("workloads", data)

    def test_post_api_action_and_step(self) -> None:
        """Verify POST /api/action approves plan and POST /api/step advances simulator."""
        action_url = f"http://127.0.0.1:{self.port}/api/action"
        req_body = json.dumps({"action": "APPROVE", "plan_id": "PLAN-A"}).encode("utf-8")
        req = urllib.request.Request(
            action_url, data=req_body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("status", data)
            self.assertEqual(data["status"], "RECOVERED")

        # Step simulator
        step_url = f"http://127.0.0.1:{self.port}/api/step"
        step_body = json.dumps({"ticks": 5}).encode("utf-8")
        req_step = urllib.request.Request(
            step_url, data=step_body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req_step) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("telemetry", data)

    def test_post_api_reset(self) -> None:
        """Verify POST /api/reset restores initial incident conditions."""
        reset_url = f"http://127.0.0.1:{self.port}/api/reset"
        req = urllib.request.Request(reset_url, data=b"{}", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["state"], "AWAITING_APPROVAL")
            self.assertEqual(data["telemetry"]["power_draw_w"], 380.0)

    def test_post_api_action_plan_d_blocked(self) -> None:
        """Verify POST /api/action rejects Plan D with HTTP 400 and blocks approval."""
        # Ensure fresh state
        reset_url = f"http://127.0.0.1:{self.port}/api/reset"
        req_reset = urllib.request.Request(reset_url, data=b"{}", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req_reset) as resp:
            self.assertEqual(resp.status, 200)

        action_url = f"http://127.0.0.1:{self.port}/api/action"
        req_body = json.dumps({"action": "APPROVE", "plan_id": "PLAN-D"}).encode("utf-8")
        req = urllib.request.Request(action_url, data=req_body, headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)
        err_data = json.loads(ctx.exception.read().decode("utf-8"))
        self.assertIn("error", err_data)
        self.assertIn("BLOCKED by SafetyGate", err_data["error"])

    def test_post_api_action_reject(self) -> None:
        """Verify POST /api/action with REJECT transitions to REJECTED and protects workloads."""
        reset_url = f"http://127.0.0.1:{self.port}/api/reset"
        req_reset = urllib.request.Request(reset_url, data=b"{}", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req_reset) as resp:
            self.assertEqual(resp.status, 200)

        action_url = f"http://127.0.0.1:{self.port}/api/action"
        req_body = json.dumps({"action": "REJECT"}).encode("utf-8")
        req = urllib.request.Request(action_url, data=req_body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "REJECTED")

        # Telemetry power draw must remain unchanged
        state_url = f"http://127.0.0.1:{self.port}/api/state"
        with urllib.request.urlopen(state_url) as resp:
            state_data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(state_data["state"], "REJECTED")
            self.assertEqual(state_data["telemetry"]["power_draw_w"], 380.0)

    def test_post_api_action_timeout(self) -> None:
        """Verify POST /api/action with TIMEOUT transitions to TIMEOUT_ESCALATED without workload mutations."""
        reset_url = f"http://127.0.0.1:{self.port}/api/reset"
        req_reset = urllib.request.Request(reset_url, data=b"{}", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req_reset) as resp:
            self.assertEqual(resp.status, 200)

        action_url = f"http://127.0.0.1:{self.port}/api/action"
        req_body = json.dumps({"action": "TIMEOUT"}).encode("utf-8")
        req = urllib.request.Request(action_url, data=req_body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "TIMEOUT_ESCALATED")

        state_url = f"http://127.0.0.1:{self.port}/api/state"
        with urllib.request.urlopen(state_url) as resp:
            state_data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(state_data["state"], "TIMEOUT_ESCALATED")
            self.assertEqual(state_data["telemetry"]["power_draw_w"], 380.0)


if __name__ == "__main__":
    unittest.main()
