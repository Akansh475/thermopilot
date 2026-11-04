"""Workload Owner Switch Agent Adapter.

Bridges tenant workload agents (Alpha VIP and Beta Elastic) to Switch Agent Protocol rooms,
submitting structured proposals and SLA advocacy bids during thermal incidents.

Advisory Only: WorkloadOwner agents have zero actuation authority and cannot mutate
physical hardware or bypass SafetyGate.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from thermopilot.agents.workload_owner import (
    WorkloadOwnerAgent,
    create_competing_workload_agents,
)
from thermopilot.switch_adapter.base import BaseSwitchAdapter
from thermopilot.switch_adapter.client import (
    SwitchHttpClient,
    SwitchWebSocketClient,
)
from thermopilot.switch_adapter.config import SwitchAdapterConfig
from thermopilot.switch_adapter.messages import (
    ADVISORY_FOOTER,
    ThermalAlertPayload,
    WorkloadBidItem,
    WorkloadBidPayload,
    format_discord_message,
    parse_switch_message,
)

logger = logging.getLogger(__name__)


class WorkloadSwitchAdapter(BaseSwitchAdapter):
    """Bridges tenant workload advocacy to Switch Agent Protocol rooms via WebSocket."""

    def __init__(
        self,
        config: SwitchAdapterConfig,
        workloads: dict[str, WorkloadOwnerAgent] | None = None,
        http_client: SwitchHttpClient | None = None,
        ws_client: SwitchWebSocketClient | None = None,
    ) -> None:
        super().__init__(config, http_client, ws_client)
        if workloads is not None:
            self.workloads = workloads
        else:
            all_agents = create_competing_workload_agents()
            self.workloads = {
                wid: agent
                for wid, agent in all_agents.items()
                if wid in ("wl-alpha-inference", "wl-beta-training")
            }
        self.processed_incidents: set[str] = set()

    def format_workload_status_reply(self) -> str:
        """Format an informational status card of registered tenant workloads."""
        lines = [
            "**[ThermoPilot Workload Portfolio]** Active Tenant Workload Profiles",
        ]
        for w in self.workloads.values():
            preempt_badge = (
                "Checkpointable / Preemptible"
                if w.can_preempt
                else "STRICT NON-PREEMPTIBLE"
            )
            lines.append(
                f"• **{w.name}** (`{w.priority.value}` | Owner: `{w.owner}`)\n"
                f"  - Baseline Power: `{w.current_power_w:.0f}W` (Floor: `{w.min_power_w:.0f}W`, Peak: `{w.max_power_w:.0f}W`)\n"
                f"  - Policy: `{preempt_badge}` | SLA Late Fee: `${w.sla_penalty_usd_per_s:.2f}/s`"
            )
        lines.append(ADVISORY_FOOTER)
        return "\n".join(lines)

    def format_help_reply(self) -> str:
        """Format supported workload agent commands."""
        lines = [
            "**ThermoPilot Workload Agent — Discord Command Reference**",
            "I advocate for tenant SLA preservation, electrical power floors, and negotiation bids.",
            "",
            "**Supported Queries:**",
            "• `@WorkloadAgent status` — Active tenant workload specifications and contractual constraints.",
            "• `@WorkloadAgent help` — Display this command reference.",
            "",
            "*(During thermal incidents, I automatically evaluate alerts and publish structured bids for the Planner).* ",
            ADVISORY_FOOTER,
        ]
        return "\n".join(lines)

    def generate_bids_for_incident(self, incident_id: str) -> WorkloadBidPayload:
        """Generate structured negotiation bids for competing workloads."""
        proposals: list[WorkloadBidItem] = []
        for agent in self.workloads.values():
            prop = agent.propose()
            proposals.append(
                WorkloadBidItem(
                    workload_id=prop.workload_id,
                    name=agent.name,
                    priority=prop.priority.value,
                    current_power_w=prop.current_power_w,
                    min_power_w=prop.min_power_w,
                    requested_power_w=prop.requested_power_w,
                    can_preempt=prop.can_preempt,
                    sla_importance_score=prop.sla_importance_score,
                    rationale=prop.rationale,
                    estimated_trade_off=prop.estimated_trade_off,
                )
            )

        # Sort by SLA importance descending (Alpha P0 > Beta P2)
        proposals.sort(key=lambda p: p.sla_importance_score, reverse=True)
        return WorkloadBidPayload(
            incident_id=incident_id,
            proposals=proposals,
        )

    async def handle_message_event(self, event_data: dict[str, Any]) -> str | None:
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
            "workload-agent",
            "thermopilot-workload-adapter",
        ):
            return None

        body = payload.get("body", "")
        clean_text = re.sub(r"@\S+", "", body).strip().lower()

        # 1. Defense-in-depth: Reject any attempt to trigger workload actuation from chat
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
                "⚠️ **Execution Denied:** Workload actuation and plan execution cannot be triggered from chat. "
                "In ThermoPilot Compute Desk, tenant agents only submit advisory bids; all mitigation plans "
                "require deterministic SafetyGate certification and authenticated operator sign-off at the governance desk."
                + ADVISORY_FOOTER
            )
            await self.http_client.send_message(room_id=room_id, content=reply)
            return reply

        # 2. Check for structured ThermoPilot negotiation message
        msg = parse_switch_message(body)
        if msg is not None:
            if isinstance(msg, ThermalAlertPayload):
                if msg.incident_id in self.processed_incidents:
                    logger.debug(
                        "Skipping already processed incident: %s", msg.incident_id
                    )
                    return None

                self.processed_incidents.add(msg.incident_id)
                logger.info(
                    "WorkloadSwitchAdapter generating bids for incident %s",
                    msg.incident_id,
                )
                bid_payload = self.generate_bids_for_incident(msg.incident_id)
                reply = format_discord_message(bid_payload)
                await self.http_client.send_message(room_id=room_id, content=reply)
                return reply

            # Ignore other structured inter-agent messages
            return None

        # 3. Direct queries addressed to WorkloadAgent
        if payload.get("addressed"):
            if any(k in clean_text for k in ("help", "commands", "usage")):
                reply = self.format_help_reply()
                await self.http_client.send_message(room_id=room_id, content=reply)
                return reply

            if any(
                k in clean_text
                for k in ("status", "workload", "bids", "tenant", "portfolio")
            ):
                reply = self.format_workload_status_reply()
                await self.http_client.send_message(room_id=room_id, content=reply)
                return reply

        return None
