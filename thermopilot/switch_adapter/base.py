"""Base Switch Agent Adapter for ThermoPilot.

Provides the shared connection lifecycle, WebSocket streaming, reconnection backoff,
heartbeat acknowledgment, and cursor management for all ThermoPilot Switch agents.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from thermopilot.switch_adapter.client import (
    SwitchAuthError,
    SwitchEvictedError,
    SwitchHttpClient,
    SwitchSSEClient,
    SwitchWebSocketClient,
)
from thermopilot.switch_adapter.config import SwitchAdapterConfig

logger = logging.getLogger(__name__)


class BaseSwitchAdapter(ABC):
    """Abstract base adapter managing WebSocket connection lifecycle and event routing."""

    def __init__(
        self,
        config: SwitchAdapterConfig,
        http_client: SwitchHttpClient | None = None,
        ws_client: SwitchWebSocketClient | None = None,
        sse_client: SwitchSSEClient | None = None,
    ) -> None:
        self.config = config
        self.http_client = http_client or SwitchHttpClient(self.config)
        self.ws_client = ws_client or SwitchWebSocketClient(self.config)
        self.sse_client = sse_client

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

    @abstractmethod
    async def handle_message_event(self, event_data: dict[str, Any]) -> Any | None:
        """Process incoming Switch domain message event."""
        raise NotImplementedError

    async def start(self) -> None:
        """Start adapter, open WebSocket event stream, and process events."""
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
                logger.warning(
                    "[%s] Could not pre-fetch assigned rooms: %s",
                    self.config.client_name,
                    exc,
                )

        logger.info(
            "Connecting %s via WebSocket (agent_id: %s, connection_id: %s, rooms: %s)",
            self.config.client_name,
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
                            "[%s] Switch WebSocket active (generation: %s, rooms: %s, cursor: %s)",
                            self.config.client_name,
                            self.connection_generation,
                            self.active_rooms,
                            self.current_cursor,
                        )

                    elif event == "ping":
                        self.last_heartbeat_time = asyncio.get_running_loop().time()
                        logger.debug(
                            "[%s] Switch ping acknowledged (pong sent, cursor: %s)",
                            self.config.client_name,
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
                    "[%s] Non-recoverable authentication failure from Switch bridge: %s",
                    self.config.client_name,
                    exc,
                )
                self.is_running = False
                self.is_connected = False
                break
            except SwitchEvictedError as exc:
                if exc.code == "taken_over":
                    logger.warning(
                        "[%s] Connection superseded/taken over by another instance: %s",
                        self.config.client_name,
                        exc,
                    )
                    self.is_running = False
                    self.is_connected = False
                    break
                logger.warning(
                    "[%s] Connection evicted (%s). Reconnecting in %.1fs...",
                    self.config.client_name,
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
            except Exception as exc:  # noqa: BLE001
                self.is_connected = False
                if not self.is_running:
                    break
                logger.warning(
                    "[%s] Switch WebSocket disconnected (%s). Reconnecting in %.1fs (cursor: %s)...",
                    self.config.client_name,
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
        logger.info("[%s] Adapter stopped cleanly", self.config.client_name)
