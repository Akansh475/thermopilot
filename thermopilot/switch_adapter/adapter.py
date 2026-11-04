"""ThermoPilot Switch Agent Adapter.

Integrates the existing ForecastAgent perception engine with the official
Switch Agent Protocol, exposing advisory GPU thermal trajectory insights
to connected Discord rooms.

Strictly Read-Only & Simulation Guarded:
- Ingests telemetry exclusively from TelemetrySimulator.
- Never mutates workloads or physical hardware.
- Never bypasses SafetyGate.
- Reuses ForecastAgent.observe() without duplicating forecast logic.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from thermopilot.agents.forecast import (
    ForecastAgent,
    ForecastConfig,
    ForecastResult,
    ThermalRiskStatus,
)
from thermopilot.simulator.telemetry import TelemetrySimulator
from thermopilot.switch_adapter.client import (
    SwitchConflictError,
    SwitchHttpClient,
    SwitchSSEClient,
)
from thermopilot.switch_adapter.config import SwitchAdapterConfig

logger = logging.getLogger(__name__)

ADVISORY_FOOTER = (
    "\n\n> 🛡️ **ThermoPilot Compute Desk** | *Advisory Only · Simulated Physics · Zero Hardware Actuation*"
)


class ForecastSwitchAdapter:
    """Bridges Switch Agent Protocol rooms to the ThermoPilot ForecastAgent."""

    def __init__(
        self,
        config: SwitchAdapterConfig,
        forecast_agent: ForecastAgent | None = None,
        simulator: TelemetrySimulator | None = None,
        http_client: SwitchHttpClient | None = None,
        sse_client: SwitchSSEClient | None = None,
    ) -> None:
        self.config = config
        self.forecast_agent = forecast_agent or ForecastAgent(ForecastConfig())
        self.simulator = simulator or TelemetrySimulator()
        self.http_client = http_client or SwitchHttpClient(self.config)
        self.sse_client = sse_client or SwitchSSEClient(self.config)

        self.is_running = False
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._current_cursor: int | None = None
        self._claimed_rooms: set[str] = set()

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

        if any(k in clean_text for k in ("help", "commands", "usage")):
            return self.format_help_reply()

        if any(k in clean_text for k in ("ttt", "time to trip", "time-to-trip", "countdown", "throttle")):
            forecast = self.get_current_forecast()
            return self.format_ttt_reply(forecast)

        if any(k in clean_text for k in ("forecast", "trajectory", "predict", "derivative", "runaway")):
            forecast = self.get_current_forecast()
            return self.format_trajectory_reply(forecast)

        if any(k in clean_text for k in ("status", "health", "temp", "temperature", "metrics")):
            forecast = self.get_current_forecast()
            return self.format_status_reply(forecast)

        # Defense-in-depth: Reject any attempt to trigger workload actions
        if any(k in clean_text for k in ("approve", "execute", "apply", "kill", "throttle now", "mitigate")):
            return (
                "⚠️ **Execution Denied:** Workload actuation and plan execution cannot be triggered from chat. "
                "In ThermoPilot Compute Desk, all mitigation proposals require deterministic SafetyGate "
                "verification and authenticated operator sign-off at the governance desk."
                + ADVISORY_FOOTER
            )

        # Default fallback summary
        forecast = self.get_current_forecast()
        summary = (
            f"**ThermoPilot Advisory:** Core temperature is `{forecast.current_temp_c:.1f}°C` "
            f"({forecast.trend.value}, Risk: `{forecast.risk_status.value}`).\n"
            f"Ask `@ForecastAgent forecast` for trajectory derivatives or `@ForecastAgent help` for commands."
        )
        return summary + ADVISORY_FOOTER

    async def handle_message_event(self, event_data: dict[str, Any]) -> str | None:
        """Process an incoming Switch SSE domain event.

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
        logger.info("Received addressed message in room %s from %s: %r", room_id, sender_name, body)

        reply_content = self.generate_advisory_reply(body)

        try:
            await self.http_client.send_message(room_id=room_id, content=reply_content)
            logger.info("Successfully dispatched advisory reply to room %s", room_id)
            return reply_content
        except Exception as exc:
            logger.error("Failed to dispatch reply to room %s: %s", room_id, exc)
            raise

    # -----------------------------------------------------------------------
    # Heartbeat Worker
    # -----------------------------------------------------------------------

    async def _heartbeat_worker(self) -> None:
        """Maintain persistent connection liveness by beating every 2 seconds."""
        logger.info("Starting Switch heartbeat worker (interval: %.1fs)", self.config.heartbeat_interval_s)
        while self.is_running:
            try:
                await self.http_client.send_heartbeat(
                    connection_id=self.config.connection_id,
                    cursor=self._current_cursor,
                )
                logger.debug("Heartbeat acknowledged by Switch bridge")
            except SwitchConflictError as exc:
                logger.warning("Heartbeat conflict (stream displaced or connection not yet open): %s", exc)
            except Exception as exc:  # noqa: BLE001 - resilience against network glitches
                logger.warning("Heartbeat tick failed: %s", exc)

            try:
                await asyncio.sleep(self.config.heartbeat_interval_s)
            except asyncio.CancelledError:
                break
        logger.info("Heartbeat worker terminated")

    # -----------------------------------------------------------------------
    # Connection Lifecycle & Reconnect Handling
    # -----------------------------------------------------------------------

    async def start(self) -> None:
        """Start the adapter, open the SSE event stream, and launch the heartbeat loop."""
        self.is_running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_worker())

        reconnect_delay = self.config.reconnect_delay_s

        logger.info(
            "Connecting ForecastSwitchAdapter (agent_id: %s, connection_id: %s)",
            self.config.agent_id,
            self.config.connection_id,
        )

        while self.is_running:
            try:
                async for frame in self.sse_client.connect_events(
                    connection_id=self.config.connection_id
                ):
                    if not self.is_running:
                        break

                    # Reset backoff upon receiving frames
                    reconnect_delay = self.config.reconnect_delay_s

                    # Update cursor sequence if present
                    seq = frame.get("id")
                    if seq and seq.isdigit():
                        self._current_cursor = int(seq)

                    data = frame.get("data")
                    if isinstance(data, dict):
                        await self.handle_message_event(data)

            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001 - resilience against network drops
                if not self.is_running:
                    break
                logger.warning("SSE stream disconnected (%s). Reconnecting in %.1fs...", exc, reconnect_delay)
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 1.5, self.config.max_reconnect_delay_s)

        await self.stop()

    async def stop(self) -> None:
        """Gracefully stop background tasks and close client connections."""
        self.is_running = False
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        logger.info("ForecastSwitchAdapter stopped cleanly")
