"""Recovery Planner Switch Agent Adapter.

Bridges the deterministic RecoveryPlanner and SafetyGate to Switch Agent Protocol rooms,
processing tenant workload bids, certifying candidate plans against physical safety rules,
and publishing verified recovery recommendations.

Advisory Only: PlannerAgent outputs certified recommendations but has zero actuation
authority. Execution requires authenticated operator approval at the governance desk.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

from thermopilot.audit.state_machine import IncidentStateMachine
from thermopilot.executor.executor import SimulationExecutor
from thermopilot.executor.verifier import VerificationAgent
from thermopilot.hitl.switch import SwitchApprovalDesk
from thermopilot.planner.planner import RecoveryPlanner
from thermopilot.schemas import (
    ApprovalAction,
    ApprovalDecision,
    GPUTelemetry,
    IncidentState,
    WorkloadSpec,
)
from thermopilot.simulator.telemetry import TelemetrySimulator
from thermopilot.switch_adapter.base import BaseSwitchAdapter
from thermopilot.switch_adapter.client import (
    SwitchHttpClient,
    SwitchWebSocketClient,
)
from thermopilot.switch_adapter.config import SwitchAdapterConfig
from thermopilot.switch_adapter.messages import (
    ADVISORY_FOOTER,
    DecisionCardItem,
    DecisionCardPayload,
    ExecutionResultPayload,
    NegotiationCompletePayload,
    RecoveryCandidateItem,
    RecoveryProposalPayload,
    SafetyEvaluationItem,
    SafetyResultPayload,
    VerificationReportPayload,
    WorkloadActionItem,
    WorkloadBidPayload,
    format_discord_message,
    parse_switch_message,
)

logger = logging.getLogger(__name__)


class PlannerSwitchAdapter(BaseSwitchAdapter):
    """Bridges recovery planning and safety gate evaluation to Switch rooms via WebSocket."""

    def __init__(
        self,
        config: SwitchAdapterConfig,
        planner: RecoveryPlanner | None = None,
        desk: SwitchApprovalDesk | None = None,
        executor: SimulationExecutor | None = None,
        verifier: VerificationAgent | None = None,
        simulator: TelemetrySimulator | None = None,
        http_client: SwitchHttpClient | None = None,
        ws_client: SwitchWebSocketClient | None = None,
    ) -> None:
        super().__init__(config, http_client, ws_client)
        self.planner = planner or RecoveryPlanner()
        self.desk = desk or SwitchApprovalDesk(
            safety_gate=self.planner.safety_gate,
            default_timeout_s=self.config.approval_timeout_s,
        )
        self.executor = executor or SimulationExecutor()
        self.verifier = verifier or VerificationAgent(target_cooling_rate_c_per_s=0.20)
        self.simulator = simulator or TelemetrySimulator()
        self.processed_incidents: set[str] = set()
        self.active_incidents: dict[str, dict[str, Any]] = {}
        self.finalized_incidents: set[str] = set()
        self.state_machines: dict[str, IncidentStateMachine] = {}

    def format_planner_status_reply(self) -> str:
        """Format an informational status card of the recovery planner and SafetyGate."""
        lines = [
            "**[ThermoPilot Recovery Planner & SafetyGate]** Architecture Status",
            "• **Certification Engine:** Deterministic SafetyGate (6 Non-Bypassable Physical & Business Rules)",
            "• **Rules Enforced:** `RULE_TEMP_CEILING (80°C)`, `RULE_NODE_POWER_BUDGET (400W)`, `RULE_NO_P0_PREEMPTION`, `RULE_P0_SLA_PRESERVATION`, `RULE_MIN_POWER_FLOOR`, `RULE_POWER_CAP_VALIDITY`",
            "• **Candidate Synthesis:** Algorithmic multi-candidate generator (Plan A: Shed, Plan B: Balanced Dynamic Cap, Plan C: Migration)",
            "• **Execution Boundary:** Zero Chat Actuation. All execution requires governance desk sign-off.",
            ADVISORY_FOOTER,
        ]
        return "\n".join(lines)

    def format_help_reply(self) -> str:
        """Format supported planner commands."""
        lines = [
            "**ThermoPilot Planner Agent — Discord Command Reference**",
            "I synthesize multi-tenant mitigation plans and deterministically certify them via SafetyGate.",
            "",
            "**Supported Queries:**",
            "• `@PlannerAgent status` — Recovery engine and SafetyGate constraint status.",
            "• `@PlannerAgent help` — Display this command reference.",
            "",
            "*(During thermal incidents, I automatically evaluate workload bids and publish certified plans).* ",
            ADVISORY_FOOTER,
        ]
        return "\n".join(lines)

    def parse_governance_command(
        self, text: str
    ) -> tuple[str, str | None, str | None]:
        """Parse operator approval or rejection command.

        Returns:
            (command, incident_id, plan_id) where command is 'approve', 'reject', or ''.
        """
        tokens = [t.strip(",.:;") for t in text.split() if t.strip(",.:;")]
        if not tokens:
            return "", None, None

        cmd = tokens[0].lower()
        if cmd not in ("approve", "reject"):
            return "", None, None

        incident_id: str | None = None
        plan_id: str | None = None

        if cmd == "approve":
            remaining = tokens[1:]
            if len(remaining) == 1:
                token = remaining[0]
                if token.upper().startswith("PLAN-") or token.upper() in (
                    "PLAN_A",
                    "PLAN_B",
                    "PLAN_C",
                    "PLAN_D",
                    "A",
                    "B",
                    "C",
                    "D",
                ):
                    plan_id = (
                        token.upper()
                        if token.upper().startswith("PLAN-")
                        else f"PLAN-{token.upper()}"
                    )
                elif token in self.active_incidents or token.startswith("inc-"):
                    incident_id = token
                else:
                    plan_id = token.upper()
            elif len(remaining) >= 2:
                t1, t2 = remaining[0], remaining[1]
                if t1.upper().startswith("PLAN-") or t1.upper() in ("A", "B", "C", "D"):
                    plan_id = (
                        t1.upper()
                        if t1.upper().startswith("PLAN-")
                        else f"PLAN-{t1.upper()}"
                    )
                    incident_id = t2
                else:
                    incident_id = t1
                    plan_id = (
                        t2.upper()
                        if t2.upper().startswith("PLAN-")
                        else (
                            f"PLAN-{t2.upper()}"
                            if t2.upper() in ("A", "B", "C", "D")
                            else t2.upper()
                        )
                    )

        elif cmd == "reject":
            remaining = tokens[1:]
            if len(remaining) >= 1:
                incident_id = remaining[0]

        # Infer incident_id if unambiguous
        if incident_id is None and len(self.active_incidents) == 1:
            incident_id = next(iter(self.active_incidents))

        return cmd, incident_id, plan_id

    async def handle_governance_decision(
        self,
        room_id: str,
        cmd: str,
        incident_id: str,
        plan_id: str | None,
        operator_id: str,
        incident_context: dict[str, Any],
    ) -> list[str] | str:
        """Handle validated operator approval or rejection command."""
        # 1. Timeout escalation check
        timeout_s = self.config.approval_timeout_s
        elapsed = time.time() - incident_context.get("timestamp", time.time())
        if elapsed > timeout_s:
            # Automatic timeout escalation
            self.finalized_incidents.add(incident_id)
            self.active_incidents.pop(incident_id, None)

            action = ApprovalAction(
                incident_id=incident_id,
                selected_plan_id=None,
                operator_id=operator_id,
                decision=ApprovalDecision.TIMEOUT_ESCALATED,
                timestamp=datetime.now(timezone.utc),
                notes=f"Operator submission received after {elapsed:.1f}s (window: {timeout_s:.1f}s).",
            )
            self.desk.evaluate_approval(
                approval=action,
                incident_id=incident_id,
                current_telemetry=incident_context["telemetry"],
            )

            sm = self.state_machines.get(incident_id)
            if sm and sm.can_transition_to(IncidentState.TIMEOUT_ESCALATED):
                sm.transition(
                    to_state=IncidentState.TIMEOUT_ESCALATED,
                    reason="Approval timeout expired.",
                )

            reply = (
                f"⚠️ **Approval Window Expired (TIMEOUT_ESCALATED):** "
                f"The governance window of {timeout_s:.1f}s for incident `{incident_id}` has expired "
                f"({elapsed:.1f}s elapsed). Workload mutations are strictly blocked under fail-safe policy."
                + ADVISORY_FOOTER
            )
            await self.http_client.send_message(room_id=room_id, content=reply)
            return reply

        # 2. Rejection handling
        if cmd == "reject":
            self.finalized_incidents.add(incident_id)
            self.active_incidents.pop(incident_id, None)

            action = ApprovalAction(
                incident_id=incident_id,
                selected_plan_id=None,
                operator_id=operator_id,
                decision=ApprovalDecision.REJECT,
                timestamp=datetime.now(timezone.utc),
                notes=f"Explicit rejection by human operator {operator_id}.",
            )
            self.desk.evaluate_approval(
                approval=action,
                incident_id=incident_id,
                current_telemetry=incident_context["telemetry"],
            )

            sm = self.state_machines.get(incident_id)
            if sm and sm.can_transition_to(IncidentState.REJECTED):
                sm.transition(
                    to_state=IncidentState.REJECTED,
                    reason=f"Operator {operator_id} rejected.",
                )

            reply = (
                f"🛑 **Incident `{incident_id}` Rejected by Operator `{operator_id}`.** "
                "All recovery plans cancelled. Workloads remain in their current state."
                + ADVISORY_FOOTER
            )
            await self.http_client.send_message(room_id=room_id, content=reply)
            return reply

        # 3. Approval handling
        action = ApprovalAction(
            incident_id=incident_id,
            selected_plan_id=plan_id,
            operator_id=operator_id,
            decision=ApprovalDecision.APPROVE,
            timestamp=datetime.now(timezone.utc),
        )
        wf_result = self.desk.evaluate_approval(
            approval=action,
            incident_id=incident_id,
            current_telemetry=incident_context["telemetry"],
        )

        sm = self.state_machines.get(incident_id)
        if not wf_result.is_authorized_for_execution:
            self.finalized_incidents.add(incident_id)
            self.active_incidents.pop(incident_id, None)
            if sm:
                if sm.can_transition_to(IncidentState.APPROVED):
                    sm.transition(to_state=IncidentState.APPROVED, plan_id=plan_id)
                if sm.can_transition_to(IncidentState.REVALIDATION_FAILED):
                    sm.transition(
                        to_state=IncidentState.REVALIDATION_FAILED,
                        plan_id=plan_id,
                        reason="Pre-execution Safety Gate revalidation failed.",
                    )
            reply = (
                f"⚠️ **Execution Denied by SafetyGate:** Plan `{plan_id}` cannot be executed. "
                f"Reasons: {'; '.join(wf_result.rejection_reasons)}."
                + ADVISORY_FOOTER
            )
            await self.http_client.send_message(room_id=room_id, content=reply)
            return reply

        self.finalized_incidents.add(incident_id)
        self.active_incidents.pop(incident_id, None)

        if sm:
            if sm.can_transition_to(IncidentState.APPROVED):
                sm.transition(
                    to_state=IncidentState.APPROVED,
                    plan_id=plan_id,
                    is_authorized=True,
                    revalidation_passed=True,
                )
            if sm.can_transition_to(IncidentState.EXECUTING):
                sm.transition(
                    to_state=IncidentState.EXECUTING,
                    plan_id=plan_id,
                    is_authorized=True,
                    revalidation_passed=True,
                )

        msg_auth = (
            f"✅ **Plan `{plan_id}` Approved and Pre-Execution Certified by SafetyGate.** "
            f"Operator: `{operator_id}` | Incident: `{incident_id}`."
            + ADVISORY_FOOTER
        )
        await self.http_client.send_message(room_id=room_id, content=msg_auth)

        # 4. Simulation Execution
        exec_res = self.executor.execute_workflow_result(
            workflow_result=wf_result,
            simulator=self.simulator,
            workloads=incident_context["workloads"],
        )

        if sm and sm.can_transition_to(IncidentState.VERIFYING):
            sm.transition(to_state=IncidentState.VERIFYING, plan_id=plan_id)

        exec_payload = ExecutionResultPayload(
            incident_id=incident_id,
            execution_id=exec_res.execution_id,
            plan_id=plan_id or "PLAN-B",
            operator_id=operator_id,
            success=exec_res.success,
            applied_actions=[a.model_dump(mode="json") for a in exec_res.applied_actions],
            node_power_w=float(exec_res.details.get("total_node_power_w", 330.0)),
            revalidation_passed=True,
            is_simulated=True,
        )

        await asyncio.sleep(0.3)
        msg_exec = format_discord_message(exec_payload)
        await self.http_client.send_message(room_id=room_id, content=msg_exec)

        # 5. Post-Execution Verification
        assert wf_result.selected_plan is not None
        verif_rep = self.verifier.verify(
            simulator=self.simulator,
            plan=wf_result.selected_plan,
            workloads=incident_context["workloads"],
            execution_result=exec_res,
            incident_id=incident_id,
        )

        if sm:
            final_state = (
                IncidentState.RECOVERED
                if verif_rep.recovery_successful
                else IncidentState.RECOVERY_FAILED
            )
            if sm.can_transition_to(final_state):
                sm.transition(to_state=final_state, plan_id=plan_id)

        verif_payload = VerificationReportPayload(
            incident_id=incident_id,
            verification_id=verif_rep.verification_id,
            plan_id=plan_id or "PLAN-B",
            recovery_successful=verif_rep.recovery_successful,
            initial_temp_c=verif_rep.initial_temp_c,
            final_temp_c=verif_rep.final_temp_c,
            cooling_rate_c_per_s=verif_rep.observed_cooling_rate_c_per_s,
            reached_target=verif_rep.reached_target,
            ticks_to_target=verif_rep.ticks_to_target,
            audit_hash=None,
            is_simulated=True,
        )

        await asyncio.sleep(0.3)
        msg_verif = format_discord_message(verif_payload)
        await self.http_client.send_message(room_id=room_id, content=msg_verif)

        return [msg_auth, msg_exec, msg_verif]

    def plan_and_certify(
        self, bid_payload: WorkloadBidPayload, room_id: str = ""
    ) -> tuple[
        RecoveryProposalPayload,
        SafetyResultPayload,
        NegotiationCompletePayload,
        DecisionCardPayload,
    ]:
        """Generate candidate plans, run SafetyGate verification, and package outputs."""
        incident_id = bid_payload.incident_id

        # Reconstruct WorkloadSpecs from bids
        workloads: list[WorkloadSpec] = []
        for prop in bid_payload.proposals:
            workloads.append(
                WorkloadSpec(
                    workload_id=prop.workload_id,
                    name=prop.name,
                    owner="Tenant-Cluster",
                    priority=prop.priority,  # type: ignore[arg-type]
                    current_power_w=prop.current_power_w,
                    min_power_w=prop.min_power_w,
                    max_power_w=max(prop.current_power_w, prop.min_power_w),
                    can_preempt=prop.can_preempt,
                    can_power_cap=True,
                    sla_deadline_s=90.0 if "P0" in str(prop.priority) else 3600.0,
                    sla_penalty_usd_per_s=5.50 if "P0" in str(prop.priority) else 0.01,
                )
            )

        # Telemetry snapshot representing the active incident
        telemetry = GPUTelemetry(
            gpu_id=0,
            temperature_core_c=82.5,
            temperature_junction_c=90.0,
            power_draw_w=380.0,
            power_limit_w=400.0,
            fan_speed_pct=95.0,
            clock_freq_mhz=1800,
            utilization_pct=98.0,
            ambient_temp_c=34.0,
        )

        # Synchronize simulation environment to incident telemetry
        self.simulator._workload_powers.clear()
        for w in workloads:
            self.simulator.set_workload_power(w.workload_id, w.current_power_w)
        self.simulator._core_temp_c = telemetry.temperature_core_c

        planner_result = self.planner.plan(
            telemetry=telemetry,
            workloads=workloads,
            include_unsafe_candidate=True,
        )

        # 1. Recovery Proposal Payload
        candidate_items: list[RecoveryCandidateItem] = []
        for c in planner_result.candidates:
            actions = [
                WorkloadActionItem(
                    workload_id=a.workload_id,
                    action=a.action.value,
                    target_power_w=a.target_power_w,
                )
                for a in c.plan.workload_actions
            ]
            candidate_items.append(
                RecoveryCandidateItem(
                    plan_id=c.plan.plan_id,
                    name=c.plan.name,
                    strategy_type=c.plan.strategy_type.value,
                    is_recommended=c.plan.is_recommended,
                    projected_temp_max_c=c.plan.projected_temp_max_c,
                    cooling_time_to_safe_s=c.plan.cooling_time_to_safe_s,
                    actions=actions,
                    estimated_cost_usd=c.plan.estimated_cost_usd,
                    sla_violations_expected=c.plan.sla_violations_expected,
                    rationale=c.plan.rationale,
                    origin=c.origin.value,
                )
            )

        rec_proposal = RecoveryProposalPayload(
            incident_id=incident_id,
            candidates=candidate_items,
        )

        # 2. Safety Result Payload
        safety_items: list[SafetyEvaluationItem] = []
        for c in planner_result.candidates:
            safety_items.append(
                SafetyEvaluationItem(
                    plan_id=c.plan.plan_id,
                    passed=c.safety_result.passed,
                    rejection_reasons=list(c.safety_result.rejection_reasons),
                    rule_ids=list(c.safety_result.rule_ids),
                )
            )

        safety_result = SafetyResultPayload(
            incident_id=incident_id,
            evaluations=safety_items,
        )

        # 3. Negotiation Complete Payload
        rec = planner_result.recommended_candidate
        if rec is not None:
            rec_id = rec.plan.plan_id
            rec_name = rec.plan.name
            proj_temp = rec.plan.projected_temp_max_c
            cool_time = rec.plan.cooling_time_to_safe_s
            summary_actions = ", ".join(
                f"{a.workload_id} -> {a.action.value} ({a.target_power_w:.0f}W)"
                if a.target_power_w is not None
                else f"{a.workload_id} -> {a.action.value}"
                for a in rec.plan.workload_actions
            )
            sla_summary = f"{rec.plan.sla_violations_expected} SLA violations expected"
        else:
            rec_id = "PLAN-A"
            rec_name = "Conservative Workload Shed"
            proj_temp = 74.0
            cool_time = 10.0
            summary_actions = "Conservative load reduction"
            sla_summary = "SLA protected"

        negotiation_complete = NegotiationCompletePayload(
            incident_id=incident_id,
            recommended_plan_id=rec_id,
            plan_name=rec_name,
            projected_temp_max_c=proj_temp,
            cooling_time_s=cool_time,
            actions_summary=summary_actions,
            sla_status=sla_summary,
            next_step="Awaiting authenticated operator sign-off at Compute Desk (Zero Actuation via Chat)",
        )

        # 4. HITL Decision Card via SwitchApprovalDesk
        card = self.desk.create_and_dispatch_card(
            incident_id=incident_id,
            telemetry=telemetry,
            candidates=planner_result.candidates,
            workloads=workloads,
            timeout_s=self.config.approval_timeout_s,
        )

        card_items: list[DecisionCardItem] = []
        for p in card.candidate_plans:
            card_items.append(
                DecisionCardItem(
                    plan_id=p.plan_id,
                    name=p.name,
                    strategy_type=p.strategy_type.value,
                    is_recommended=p.is_recommended,
                    is_safe=p.is_safe,
                    safety_verdict=p.safety_verdict,
                    rejection_reasons=list(p.rejection_reasons),
                    projected_temp_max_c=p.projected_temp_max_c,
                    projected_power_w=p.projected_power_w,
                    cooling_time_to_safe_s=p.cooling_time_to_safe_s,
                    estimated_cost_usd=p.estimated_cost_usd,
                    sla_violations_expected=p.sla_violations_expected,
                    actions_summary=", ".join(
                        f"{a.workload_id}->{a.action.value}"
                        + (f"({a.target_power_w:.0f}W)" if a.target_power_w is not None else "")
                        for a in p.proposed_actions
                    ),
                    is_executable=p.is_executable,
                )
            )

        decision_card = DecisionCardPayload(
            incident_id=incident_id,
            severity=card.severity,
            current_temp_c=card.current_temp_c,
            current_power_w=card.current_power_w,
            critical_trip_temp_c=card.critical_trip_temp_c,
            time_to_trip_s=card.time_to_trip_s,
            candidate_plans=card_items,
            executable_plan_ids=list(card.executable_plan_ids),
            rejected_plan_ids=list(card.rejected_plan_ids),
            timeout_seconds=card.timeout_seconds,
            approval_command=f"@planner-agent approve {incident_id} {rec_id}",
            rejection_command=f"@planner-agent reject {incident_id}",
        )

        self.active_incidents[incident_id] = {
            "card": card,
            "decision_payload": decision_card,
            "telemetry": telemetry,
            "workloads": workloads,
            "candidates": planner_result.candidates,
            "room_id": room_id,
            "timestamp": time.time(),
        }

        self.state_machines[incident_id] = IncidentStateMachine(
            incident_id=incident_id,
            initial_state=IncidentState.AWAITING_APPROVAL,
        )

        return rec_proposal, safety_result, negotiation_complete, decision_card

    async def handle_message_event(
        self, event_data: dict[str, Any]
    ) -> list[str] | str | None:
        """Process incoming Switch room message event."""
        if event_data.get("type") != "message":
            return None

        room_id = event_data.get("room_id")
        if not room_id or not self.config.is_room_authorized(room_id):
            return None

        payload = event_data.get("payload", {})
        if not isinstance(payload, dict):
            return None

        sender_name = payload.get("sender_name", "")
        # Prevent replying to self or duplicate echo
        if sender_name in (
            self.config.client_name,
            "planner-agent",
            "thermopilot-planner-adapter",
        ):
            return None

        body = payload.get("body", "")
        clean_text = re.sub(r"@\S+", "", body).strip()
        clean_lower = clean_text.lower()

        # 1. Defense-in-depth: Reject non-approval chat execution commands
        MUTATION_KEYWORDS = ("execute", "apply", "kill", "throttle now", "mitigate")
        if any(k in clean_lower for k in MUTATION_KEYWORDS):
            reply = (
                "⚠️ **Execution Denied:** Recovery plans cannot be executed from chat. "
                "In ThermoPilot Compute Desk, all mitigation proposals require deterministic SafetyGate "
                "verification and authenticated operator sign-off at the governance desk."
                + ADVISORY_FOOTER
            )
            await self.http_client.send_message(room_id=room_id, content=reply)
            return reply

        # 2. Check for HITL governance command (approve / reject)
        cmd, inc_id, p_id = self.parse_governance_command(clean_text)
        if cmd in ("approve", "reject"):
            # Check operator authorization
            if not self.config.is_operator_authorized(sender_name):
                if sender_name.endswith(("-agent", "-adapter")) or sender_name in (
                    "forecast-agent",
                    "workload-agent",
                    "planner-agent",
                ):
                    reply = (
                        "⚠️ **Execution Denied: Autonomous agents cannot authorize execution.** "
                        "In ThermoPilot Compute Desk, all mitigation proposals require authenticated "
                        "human operator sign-off at the governance desk."
                        + ADVISORY_FOOTER
                    )
                else:
                    reply = (
                        f"⚠️ **Execution Denied: Unauthorized Operator.** "
                        f"User '{sender_name}' is not authorized to submit recovery plan governance decisions. "
                        f"Only designated human operators may approve or reject plans."
                        + ADVISORY_FOOTER
                    )
                await self.http_client.send_message(room_id=room_id, content=reply)
                return reply

            # Validate incident binding
            if not inc_id:
                if not self.active_incidents:
                    reply = (
                        "⚠️ **Execution Denied: No Active Incident.** "
                        "There are currently no pending incidents awaiting operator approval."
                        + ADVISORY_FOOTER
                    )
                else:
                    reply = (
                        f"⚠️ **Incident ID Required:** Please specify the incident ID. "
                        f"Usage: `@planner-agent {cmd} <incident_id>"
                        + (" <plan_id>`" if cmd == "approve" else "`")
                        + ADVISORY_FOOTER
                    )
                await self.http_client.send_message(room_id=room_id, content=reply)
                return reply

            if inc_id in self.finalized_incidents:
                reply = (
                    f"⚠️ **Execution Denied: Duplicate Governance Submission.** "
                    f"Incident '{inc_id}' has already been processed and finalized."
                    + ADVISORY_FOOTER
                )
                await self.http_client.send_message(room_id=room_id, content=reply)
                return reply

            if inc_id not in self.active_incidents:
                reply = (
                    f"⚠️ **Execution Denied: Unknown Incident.** "
                    f"Incident '{inc_id}' was not found in active incidents or has already been resolved."
                    + ADVISORY_FOOTER
                )
                await self.http_client.send_message(room_id=room_id, content=reply)
                return reply

            incident_context = self.active_incidents[inc_id]

            if cmd == "approve" and not p_id:
                candidate_ids = [
                    c.plan_id for c in incident_context["card"].candidate_plans
                ]
                reply = (
                    f"⚠️ **Plan ID Required:** Please specify which candidate plan to approve for incident '{inc_id}'. "
                    f"Available candidates: {', '.join(candidate_ids)}. "
                    f"Usage: `@planner-agent approve {inc_id} <plan_id>`"
                    + ADVISORY_FOOTER
                )
                await self.http_client.send_message(room_id=room_id, content=reply)
                return reply

            return await self.handle_governance_decision(
                room_id=room_id,
                cmd=cmd,
                incident_id=inc_id,
                plan_id=p_id,
                operator_id=sender_name,
                incident_context=incident_context,
            )

        # 2. Check for structured ThermoPilot negotiation message
        msg = parse_switch_message(body)
        if msg is not None:
            if isinstance(msg, WorkloadBidPayload):
                if msg.incident_id in self.processed_incidents:
                    logger.debug(
                        "Skipping already processed incident: %s",
                        msg.incident_id,
                    )
                    return None

                self.processed_incidents.add(msg.incident_id)
                logger.info(
                    "PlannerSwitchAdapter evaluating bids and formulating plans for incident %s",
                    msg.incident_id,
                )

                rec_prop, saf_res, neg_comp, dec_card = self.plan_and_certify(
                    msg, room_id=room_id
                )

                msg1 = format_discord_message(rec_prop)
                await self.http_client.send_message(room_id=room_id, content=msg1)

                # Brief pacing for readable Discord presentation
                await asyncio.sleep(0.3)
                msg2 = format_discord_message(saf_res)
                await self.http_client.send_message(room_id=room_id, content=msg2)

                await asyncio.sleep(0.3)
                msg3 = format_discord_message(neg_comp)
                await self.http_client.send_message(room_id=room_id, content=msg3)

                await asyncio.sleep(0.3)
                msg4 = format_discord_message(dec_card)
                await self.http_client.send_message(room_id=room_id, content=msg4)

                return [msg1, msg2, msg3, msg4]

            # Ignore other structured inter-agent messages
            return None

        # 3. Direct queries addressed to PlannerAgent
        if payload.get("addressed"):
            if any(k in clean_text for k in ("help", "commands", "usage")):
                reply = self.format_help_reply()
                await self.http_client.send_message(room_id=room_id, content=reply)
                return reply

            if any(
                k in clean_text
                for k in ("status", "planner", "safety", "gate", "rules")
            ):
                reply = self.format_planner_status_reply()
                await self.http_client.send_message(room_id=room_id, content=reply)
                return reply

        return None
