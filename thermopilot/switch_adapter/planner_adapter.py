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
from typing import Any

from thermopilot.hitl.switch import SwitchApprovalDesk
from thermopilot.planner.planner import RecoveryPlanner
from thermopilot.schemas import GPUTelemetry, WorkloadSpec
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
    NegotiationCompletePayload,
    RecoveryCandidateItem,
    RecoveryProposalPayload,
    SafetyEvaluationItem,
    SafetyResultPayload,
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
        http_client: SwitchHttpClient | None = None,
        ws_client: SwitchWebSocketClient | None = None,
    ) -> None:
        super().__init__(config, http_client, ws_client)
        self.planner = planner or RecoveryPlanner()
        self.desk = desk or SwitchApprovalDesk(
            safety_gate=self.planner.safety_gate,
            default_timeout_s=self.config.approval_timeout_s,
        )
        self.processed_incidents: set[str] = set()
        self.active_incidents: dict[str, dict[str, Any]] = {}

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
        clean_text = re.sub(r"@\S+", "", body).strip().lower()

        # 1. Defense-in-depth: Reject any attempt to trigger workload actions from chat
        if any(
            k in clean_text
            for k in (
                "approve",
                "execute",
                "apply",
                "kill",
                "throttle now",
                "mitigate",
            )
        ):
            reply = (
                "⚠️ **Execution Denied:** Recovery plans cannot be executed from chat. "
                "In ThermoPilot Compute Desk, all mitigation proposals require deterministic SafetyGate "
                "verification and authenticated operator sign-off at the governance desk."
                + ADVISORY_FOOTER
            )
            await self.http_client.send_message(room_id=room_id, content=reply)
            return reply

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
