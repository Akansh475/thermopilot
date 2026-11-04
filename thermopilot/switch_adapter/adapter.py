"""ThermoPilot Switch Agent Adapter.

Integrates the existing ForecastAgent perception engine with the official
Switch Agent Protocol, exposing advisory GPU thermal trajectory insights
to connected Discord rooms.

Strictly Read-Only & Simulation Guarded:
- Ingests telemetry exclusively from TelemetrySimulator.
- Never mutates workloads or physical hardware.
- Never bypasses SafetyGate.
- Reuses ForecastAgent.observe() without duplicating forecast logic.
- Uses Switch Agent Protocol WebSocket transport for duplex streaming.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from thermopilot.agents.forecast import (
    ForecastAgent,
    ForecastConfig,
    ForecastResult,
    ThermalRiskStatus,
)
from thermopilot.simulator.telemetry import TelemetrySimulator
from thermopilot.switch_adapter.base import BaseSwitchAdapter
from thermopilot.switch_adapter.client import (
    SwitchHttpClient,
    SwitchSSEClient,
    SwitchWebSocketClient,
)
from thermopilot.switch_adapter.config import SwitchAdapterConfig
from thermopilot.switch_adapter.messages import (
    ADVISORY_FOOTER,
    ThermalAlertPayload,
    format_discord_message,
)

logger = logging.getLogger(__name__)


class ForecastSwitchAdapter(BaseSwitchAdapter):
    """Bridges Switch Agent Protocol rooms to the ThermoPilot ForecastAgent via WebSocket."""

    def __init__(
        self,
        config: SwitchAdapterConfig,
        forecast_agent: ForecastAgent | None = None,
        simulator: TelemetrySimulator | None = None,
        http_client: SwitchHttpClient | None = None,
        ws_client: SwitchWebSocketClient | None = None,
        sse_client: SwitchSSEClient | None = None,
    ) -> None:
        super().__init__(config, http_client, ws_client, sse_client)
        self.forecast_agent = forecast_agent or ForecastAgent(ForecastConfig())
        self.simulator = simulator or TelemetrySimulator()

    # -----------------------------------------------------------------------
    # Advisory Forecast Response Synthesis
    # -----------------------------------------------------------------------

    def get_current_forecast(self) -> ForecastResult:
        """Sample fresh simulated telemetry and calculate deterministic forecast."""
        telemetry = self.simulator.current_telemetry()
        return self.forecast_agent.observe(telemetry)

    def format_status_reply(self, forecast: ForecastResult) -> str:
        """Format an instantaneous GPU operational status card for Discord."""
        temp_c = forecast.current_temp_c if forecast.current_temp_c is not None else 0.0
        risk_badge = {
            ThermalRiskStatus.NORMAL: "🟢 NORMAL",
            ThermalRiskStatus.WARNING: "🟡 WARNING",
            ThermalRiskStatus.CRITICAL_RISK: "🔴 CRITICAL RISK",
            ThermalRiskStatus.BREACHED: "🚨 BREACHED",
            ThermalRiskStatus.INSUFFICIENT_DATA: "⚪ INSUFFICIENT DATA",
        }.get(forecast.risk_status, forecast.risk_status.value)

        lines = [
            f"**[ThermoPilot Telemetry Status]** GPU #{forecast.gpu_id}",
            f"• **Core Temperature:** `{temp_c:.1f}°C` (Trip Threshold: `{forecast.critical_threshold_c:.1f}°C`)",
            f"• **Risk Status:** {risk_badge}",
            f"• **Trend:** `{forecast.trend.value}`",
            f"• **Samples Buffered:** `{forecast.sample_count}` samples",
            f"• **Assessment:** {forecast.reason}",
            ADVISORY_FOOTER,
        ]
        return "\n".join(lines)

    def format_trajectory_reply(self, forecast: ForecastResult) -> str:
        """Format a detailed thermal trajectory and Time-to-Throttle report."""
        rate_str = (
            f"{forecast.rate_c_per_s:+.3f}°C/s"
            if forecast.rate_c_per_s is not None
            else "N/A"
        )
        accel_str = (
            f"{forecast.acceleration_c_per_s2:+.4f}°C/s²"
            if forecast.acceleration_c_per_s2 is not None
            else "N/A"
        )
        ttt_str = (
            f"**{forecast.time_to_threshold_s:.1f} seconds**"
            if forecast.time_to_threshold_s is not None
            else "*None (not climbing toward trip)*"
        )

        lines = [
            f"**[ThermoPilot Trajectory Forecast]** GPU #{forecast.gpu_id}",
            f"• **Current Core Temp:** `{forecast.current_temp_c:.1f}°C`",
            f"• **First-Order Rate (dT/dt):** `{rate_str}`",
            f"• **Second-Order Acceleration (d²T/dt²):** `{accel_str}`",
            f"• **Estimated Time-to-Throttle (t_TTT):** {ttt_str}",
            f"• **Incident Trigger Flag:** `{'TRIGGERED' if forecast.trigger_incident else 'OFF'}`",
            f"• **Perception Rationale:** {forecast.reason}",
            ADVISORY_FOOTER,
        ]
        return "\n".join(lines)

    def format_ttt_reply(self, forecast: ForecastResult) -> str:
        """Format a focused Time-to-Throttle countdown indicator."""
        if forecast.time_to_threshold_s is not None:
            msg = (
                f"⏱️ **Time-to-Throttle (t_TTT):** `{forecast.time_to_threshold_s:.1f}s` "
                f"remaining before core temperature reaches the `{forecast.critical_threshold_c:.1f}°C` hardware trip limit."
            )
        else:
            msg = (
                f"⏱️ **Time-to-Throttle:** Nominal. Core temperature ({forecast.current_temp_c:.1f}°C) "
                f"is {forecast.trend.value.lower()} and not on a trajectory to breach {forecast.critical_threshold_c:.1f}°C."
            )
        return msg + ADVISORY_FOOTER

    def format_incident_alert_reply(
        self, incident_id: str | None = None, temp_c: float = 82.5
    ) -> str:
        """Format and record a simulated thermal anomaly alert for multi-agent negotiation."""
        inc_id = incident_id or f"inc-{datetime.now(timezone.utc).strftime('%H%M%S')}"
        self.simulator.trigger_anomaly(target_ambient_c=34.0)
        if self.simulator.core_temp_c < temp_c:
            self.simulator._core_temp_c = temp_c
        telemetry = self.simulator.step()
        forecast = self.forecast_agent.observe(telemetry)

        alert_payload = ThermalAlertPayload(
            incident_id=inc_id,
            gpu_id=forecast.gpu_id,
            current_temp_c=forecast.current_temp_c or temp_c,
            critical_threshold_c=forecast.critical_threshold_c,
            time_to_threshold_s=(
                forecast.time_to_threshold_s
                if forecast.time_to_threshold_s is not None
                else 22.5
            ),
            rate_c_per_s=(
                forecast.rate_c_per_s if forecast.rate_c_per_s is not None else 0.320
            ),
            acceleration_c_per_s2=forecast.acceleration_c_per_s2,
            risk_status="CRITICAL_RISK",
            reason=(
                f"Simulated intake heating anomaly caused rapid thermal runaway to {temp_c:.1f}°C; "
                f"trajectory will breach {forecast.critical_threshold_c:.1f}°C hardware trip limit "
                f"within 25s without coordinated mitigation."
            ),
        )
        return format_discord_message(alert_payload)

    async def trigger_simulated_incident(
        self,
        room_id: str,
        incident_id: str | None = None,
        temp_c: float = 82.5,
    ) -> str:
        """Deterministically trigger and broadcast a simulated thermal incident to the room."""
        content = self.format_incident_alert_reply(
            incident_id=incident_id, temp_c=temp_c
        )
        await self.http_client.send_message(room_id=room_id, content=content)
        return content

    @staticmethod
    def format_help_reply() -> str:
        """Format supported advisory queries."""
        lines = [
            "**ThermoPilot Forecast Agent — Discord Command Reference**",
            "I provide read-only predictive thermal perception for monitored GPU clusters.",
            "",
            "**Supported Queries:**",
            "• `@ForecastAgent status` — Current temperature, power, and thermal risk level.",
            "• `@ForecastAgent forecast` — Polynomial derivative rates and predictive runaway trajectory.",
            "• `@ForecastAgent time-to-trip` (or `ttt`) — Countdown seconds to 85.0°C hardware throttle.",
            "• `@ForecastAgent incident` (or `simulate-incident`) — Trigger simulated thermal incident for multi-agent negotiation.",
            "• `@ForecastAgent help` — Display this command reference.",
            "",
            (
                "*(Note: Workload execution, frequency capping, or mitigation actions cannot be initiated via chat; "
                "all recovery plans require certified SafetyGate validation through the Compute Desk).*"
            ),
            ADVISORY_FOOTER,
        ]
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Message Routing & Handling
    # -----------------------------------------------------------------------

    def generate_advisory_reply(self, message_text: str) -> str:
        """Route user query text to the corresponding advisory response generator."""
        clean_text = re.sub(r"@\S+", "", message_text).strip().lower()

        # Defense-in-depth: Reject any attempt to trigger workload actions
        if any(
            k in clean_text
            for k in ("approve", "execute", "apply", "kill", "throttle now", "mitigate")
        ):
            return (
                "⚠️ **Execution Denied:** Workload actuation and plan execution cannot be triggered from chat. "
                "In ThermoPilot Compute Desk, all mitigation proposals require deterministic SafetyGate "
                "verification and authenticated operator sign-off at the governance desk."
                + ADVISORY_FOOTER
            )

        if any(
            k in clean_text
            for k in (
                "incident",
                "simulate-incident",
                "trigger-incident",
                "anomaly",
                "runaway",
            )
        ):
            return self.format_incident_alert_reply()

        if any(k in clean_text for k in ("help", "commands", "usage")):
            return self.format_help_reply()

        if any(
            k in clean_text
            for k in ("ttt", "time to trip", "time-to-trip", "countdown", "throttle")
        ):
            forecast = self.get_current_forecast()
            return self.format_ttt_reply(forecast)

        if any(
            k in clean_text
            for k in ("forecast", "trajectory", "predict", "derivative", "runaway")
        ):
            forecast = self.get_current_forecast()
            return self.format_trajectory_reply(forecast)

        if any(
            k in clean_text
            for k in ("status", "health", "temp", "temperature", "metrics")
        ):
            forecast = self.get_current_forecast()
            return self.format_status_reply(forecast)

        # Default fallback summary
        forecast = self.get_current_forecast()
        summary = (
            f"**ThermoPilot Advisory:** Core temperature is `{forecast.current_temp_c:.1f}°C` "
            f"({forecast.trend.value}, Risk: `{forecast.risk_status.value}`).\n"
            f"Ask `@ForecastAgent forecast` for trajectory derivatives or `@ForecastAgent help` for commands."
        )
        return summary + ADVISORY_FOOTER

    async def handle_message_event(self, event_data: dict[str, Any]) -> str | None:
        """Process an incoming Switch domain event.

        Enforces:
        - Must be event type 'message'.
        - Must belong to an authorized room ID.
        - Must have payload.addressed == True.
        - Does not reply to itself.
        - Returns the reply text dispatched, or None if skipped.
        """
        if event_data.get("type") != "message":
            return None

        room_id = event_data.get("room_id")
        if not room_id:
            return None

        # 1. Enforce room authorization
        if not self.config.is_room_authorized(room_id):
            logger.info("Ignoring message from unauthorized room: %s", room_id)
            return None

        payload = event_data.get("payload", {})
        if not isinstance(payload, dict):
            return None

        # 2. Enforce addressed-message filter
        if not payload.get("addressed"):
            logger.debug("Skipping unaddressed room chatter in room %s", room_id)
            return None

        sender_name = payload.get("sender_name", "")
        if sender_name == self.config.client_name:
            return None

        body = payload.get("body", "")
        logger.info(
            "Received addressed message in room %s from %s: %r",
            room_id,
            sender_name,
            body,
        )

        reply_content = self.generate_advisory_reply(body)

        try:
            await self.http_client.send_message(room_id=room_id, content=reply_content)
            logger.info("Successfully dispatched advisory reply to room %s", room_id)
            return reply_content
        except Exception as exc:
            logger.error("Failed to dispatch reply to room %s: %s", room_id, exc)
            raise
