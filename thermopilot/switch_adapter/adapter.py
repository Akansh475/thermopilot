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
    SwitchAuthError,
    SwitchEvictedError,
    SwitchHttpClient,
    SwitchSSEClient,
    SwitchWebSocketClient,
)
from thermopilot.switch_adapter.config import SwitchAdapterConfig

logger = logging.getLogger(__name__)

ADVISORY_FOOTER = "\n\n> 🛡️ **ThermoPilot Compute Desk** | *Advisory Only · Simulated Physics · Zero Hardware Actuation*"


class ForecastSwitchAdapter:
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
        self.config = config
        self.forecast_agent = forecast_agent or ForecastAgent(ForecastConfig())
        self.simulator = simulator or TelemetrySimulator()
        self.http_client = http_client or SwitchHttpClient(self.config)
        self.ws_client = ws_client or SwitchWebSocketClient(self.config)
        self.sse_client = (
            sse_client  # Maintained for backwards-compatible test fixtures
        )

        self.is_running = False
        self.is_connected = False
        self.current_cursor: int | None = None
        self.connection_generation: int | None = None
        self.active_rooms: list[str] = []
        self.last_heartbeat_time: float = 0.0

    @property
    def _current_cursor(self) -> int | None:
        """Backwards compatibility alias for cursor position."""
        return self.current_cursor

    @_current_cursor.setter
    def _current_cursor(self, val: int | None) -> None:
        self.current_cursor = val

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

    # -----------------------------------------------------------------------
    # Connection Lifecycle & Reconnect Handling
    # -----------------------------------------------------------------------

    async def start(self) -> None:
        """Start the adapter, open the WebSocket event stream, and process events."""
        self.is_running = True
        reconnect_delay = self.config.reconnect_delay_s

        # Resolve declared rooms to claim on connect
        declared_rooms: list[str] = sorted(self.config.authorized_room_ids)
        if not declared_rooms:
            try:
                assigned_rooms = await self.http_client.list_assigned_rooms()
                declared_rooms = [
                    r["room_id"]
                    for r in assigned_rooms
                    if isinstance(r, dict) and "room_id" in r
                ]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not pre-fetch assigned rooms: %s", exc)

        logger.info(
            "Connecting ForecastSwitchAdapter via WebSocket (agent_id: %s, connection_id: %s, rooms: %s)",
            self.config.agent_id,
            self.config.connection_id,
            declared_rooms,
        )

        while self.is_running:
            start_from = (
                str(self.current_cursor) if self.current_cursor is not None else "head"
            )
            try:
                # If sse_client was provided explicitly for legacy test fixtures, use it;
                # otherwise use current WebSocket transport.
                if (
                    self.sse_client is not None
                    and self.ws_client._custom_stream_factory is None
                ):
                    stream = self.sse_client.connect_events(
                        connection_id=self.config.connection_id,
                        start_from=start_from,
                    )
                else:
                    stream = self.ws_client.connect_events(
                        start_from=start_from,
                        rooms=declared_rooms if declared_rooms else None,
                    )

                async for frame in stream:
                    if not self.is_running:
                        break

                    self.is_connected = True
                    reconnect_delay = self.config.reconnect_delay_s

                    event = frame.get("event")
                    seq = frame.get("id")
                    if seq is not None and str(seq).isdigit():
                        self.current_cursor = int(seq)

                    if event == "connection_state":
                        st = frame.get("data", {})
                        if isinstance(st, dict):
                            self.connection_generation = st.get("generation")
                            self.active_rooms = list(st.get("rooms", []))
                            if st.get("cursor") is not None:
                                self.current_cursor = int(st["cursor"])
                        logger.info(
                            "Switch WebSocket connection active (generation: %s, rooms: %s, cursor: %s)",
                            self.connection_generation,
                            self.active_rooms,
                            self.current_cursor,
                        )

                    elif event == "ping":
                        self.last_heartbeat_time = asyncio.get_running_loop().time()
                        logger.debug(
                            "Switch ping acknowledged (pong sent, cursor: %s)",
                            self.current_cursor,
                        )

                    elif event == "message":
                        data = frame.get("data")
                        if isinstance(data, dict):
                            await self.handle_message_event(data)

            except asyncio.CancelledError:
                break
            except SwitchAuthError as exc:
                logger.error(
                    "Non-recoverable authentication failure from Switch bridge: %s", exc
                )
                self.is_running = False
                self.is_connected = False
                break
            except SwitchEvictedError as exc:
                if exc.code == "taken_over":
                    logger.warning(
                        "Connection superseded/taken over by another instance: %s", exc
                    )
                    self.is_running = False
                    self.is_connected = False
                    break
                logger.warning(
                    "Connection evicted (%s). Reconnecting in %.1fs...",
                    exc,
                    reconnect_delay,
                )
                self.is_connected = False
                if not self.is_running:
                    break
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(
                    reconnect_delay * 1.5, self.config.max_reconnect_delay_s
                )
            except Exception as exc:  # noqa: BLE001 - resilience against network drops
                self.is_connected = False
                if not self.is_running:
                    break
                logger.warning(
                    "Switch WebSocket disconnected (%s). Reconnecting in %.1fs (cursor: %s)...",
                    exc,
                    reconnect_delay,
                    self.current_cursor,
                )
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(
                    reconnect_delay * 1.5, self.config.max_reconnect_delay_s
                )

        await self.stop()

    async def stop(self) -> None:
        """Gracefully stop adapter and close WebSocket connection."""
        self.is_running = False
        self.is_connected = False
        await self.ws_client.close()
        logger.info("ForecastSwitchAdapter stopped cleanly")
