"""Comprehensive unit tests for the ThermoPilot Switch Agent Adapter.

Verifies:
1. SwitchAdapterConfig validation, environment parsing, credential masking, and WebSocket URL building.
2. SwitchHttpClient REST request building, authorization, and error handling.
3. SwitchWebSocketClient frame parsing, automated pong heartbeat responses, cursor tracking, and error mapping.
4. Legacy SwitchSSEClient backward compatibility.
5. ForecastSwitchAdapter message filtering (addressed vs unaddressed, authorized vs unauthorized rooms).
6. Advisory query routing (status, forecast, time-to-trip, help, execution denial).
7. WebSocket streaming lifecycle, clean stop, and ping/pong processing.
8. Reconnection handling and cursor resumption on socket disconnections.
9. Fatal error termination (auth failure, taken_over eviction).
"""

from __future__ import annotations

import unittest
import uuid
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from thermopilot.agents.forecast import ForecastAgent, ForecastConfig
from thermopilot.simulator.telemetry import SimulatorConfig, TelemetrySimulator
from thermopilot.switch_adapter.adapter import ForecastSwitchAdapter
from thermopilot.switch_adapter.client import (
    SwitchAuthError,
    SwitchConflictError,
    SwitchEvictedError,
    SwitchHttpClient,
    SwitchSSEClient,
    SwitchWebSocketClient,
)
from thermopilot.switch_adapter.config import SwitchAdapterConfig


class TestSwitchAdapterConfig(unittest.TestCase):
    """Test configuration validation, environment variable reading, and security masking."""

    def test_valid_config(self) -> None:
        cfg = SwitchAdapterConfig(
            bridge_url="http://localhost:8000",
            agent_id="11111111-2222-3333-4444-555555555555",
            api_key="sw_ak_testsecret12345",
        )
        self.assertEqual(cfg.bridge_url, "http://localhost:8000")
        self.assertEqual(cfg.agent_id, "11111111-2222-3333-4444-555555555555")
        self.assertIn("sw_a...2345", cfg.masked_api_key())
        self.assertNotIn("testsecret", repr(cfg))

    def test_invalid_bridge_url(self) -> None:
        with self.assertRaises(ValueError):
            SwitchAdapterConfig(
                bridge_url="ftp://invalid-host",
                agent_id=str(uuid.uuid4()),
                api_key="secret",
            )

    def test_invalid_connection_id(self) -> None:
        with self.assertRaises(ValueError):
            SwitchAdapterConfig(
                bridge_url="http://localhost:8000",
                agent_id=str(uuid.uuid4()),
                api_key="secret",
                connection_id="not-a-uuid",
            )

    def test_room_authorization(self) -> None:
        cfg = SwitchAdapterConfig(
            bridge_url="http://localhost:8000",
            agent_id=str(uuid.uuid4()),
            api_key="secret",
            authorized_room_ids=frozenset(["room-101", "room-102"]),
        )
        self.assertTrue(cfg.is_room_authorized("room-101"))
        self.assertTrue(cfg.is_room_authorized("room-102"))
        self.assertFalse(cfg.is_room_authorized("room-999"))

    def test_build_ws_url(self) -> None:
        cfg = SwitchAdapterConfig(
            bridge_url="http://127.0.0.1:60824",
            agent_id="agent-1234",
            api_key="secret",
            connection_id="11111111-1111-1111-1111-111111111111",
            scope="all",
            event_filter="addressed",
        )
        url = cfg.build_ws_url(start_from="42", rooms=["room-a", "room-b"])
        self.assertTrue(
            url.startswith("ws://127.0.0.1:60824/agents/agent-1234/connection/ws?")
        )
        self.assertIn("connection_id=11111111-1111-1111-1111-111111111111", url)
        self.assertIn("scope=all", url)
        self.assertIn("filter=addressed", url)
        self.assertIn("start_from=42", url)
        self.assertIn("rooms=room-a%2Croom-b", url)

    def test_from_env(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SWITCH_BRIDGE_URL": "http://127.0.0.1:9000",
                "SWITCH_AGENT_ID": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "SWITCH_API_KEY": "sw_ak_envsecret9999",
                "SWITCH_AUTHORIZED_ROOM_IDS": "room-a,room-b",
                "SWITCH_HEARTBEAT_INTERVAL_S": "1.5",
            },
        ):
            cfg = SwitchAdapterConfig.from_env(env_file=None, connection_id_file=None)
            self.assertEqual(cfg.bridge_url, "http://127.0.0.1:9000")
            self.assertEqual(cfg.agent_id, "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
            self.assertEqual(cfg.api_key, "sw_ak_envsecret9999")
            self.assertEqual(cfg.authorized_room_ids, frozenset(["room-a", "room-b"]))
            self.assertEqual(cfg.heartbeat_interval_s, 1.5)
            self.assertEqual(cfg.client_name, "thermopilot-forecast-adapter")

    def test_from_env_with_roles(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SWITCH_BRIDGE_URL": "http://127.0.0.1:9000",
                "SWITCH_PLANNER_AGENT_ID": "11111111-2222-3333-4444-555555555555",
                "SWITCH_PLANNER_API_KEY": "sw_ak_planner_secret",
                "SWITCH_WORKLOAD_AGENT_ID": "22222222-3333-4444-5555-666666666666",
                "SWITCH_WORKLOAD_API_KEY": "sw_ak_workload_secret",
            },
        ):
            cfg_planner = SwitchAdapterConfig.from_env(
                env_file=None, connection_id_file=None, role="planner"
            )
            self.assertEqual(
                cfg_planner.agent_id, "11111111-2222-3333-4444-555555555555"
            )
            self.assertEqual(cfg_planner.api_key, "sw_ak_planner_secret")
            self.assertEqual(cfg_planner.client_name, "thermopilot-planner-adapter")

            cfg_workload = SwitchAdapterConfig.from_env(
                env_file=None, connection_id_file=None, role="workload"
            )
            self.assertEqual(
                cfg_workload.agent_id, "22222222-3333-4444-5555-666666666666"
            )
            self.assertEqual(cfg_workload.api_key, "sw_ak_workload_secret")
            self.assertEqual(cfg_workload.client_name, "thermopilot-workload-adapter")

    def test_from_env_with_role_missing_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                SwitchAdapterConfig.from_env(
                    env_file=None, connection_id_file=None, role="planner"
                )
            self.assertIn("role 'planner'", str(ctx.exception))


class TestSwitchHttpClient(unittest.IsolatedAsyncioTestCase):
    """Test HTTP client operations, header injection, and response parsing."""

    def setUp(self) -> None:
        self.config = SwitchAdapterConfig(
            bridge_url="http://localhost:8000",
            agent_id="test-agent-1234",
            api_key="sw_ak_mocksecret",
            connection_id=str(uuid.uuid4()),
        )

    async def test_send_heartbeat(self) -> None:
        mock_executor = MagicMock(return_value={"ok": True, "cursor": 100})
        client = SwitchHttpClient(self.config, request_executor=mock_executor)

        res = await client.send_heartbeat(self.config.connection_id, cursor=100)
        self.assertTrue(res["ok"])
        mock_executor.assert_called_once_with(
            "POST",
            f"/agents/{self.config.agent_id}/connection/beat",
            {"connection_id": self.config.connection_id, "cursor": 100},
            None,
        )

    async def test_send_message(self) -> None:
        mock_executor = MagicMock(return_value={"ok": True, "event_id": "$event123"})
        client = SwitchHttpClient(self.config, request_executor=mock_executor)

        res = await client.send_message(
            room_id="room-42", content="Hello from ThermoPilot"
        )
        self.assertEqual(res["event_id"], "$event123")
        mock_executor.assert_called_once_with(
            "POST",
            f"/agents/{self.config.agent_id}/message",
            {"room_id": "room-42", "content": "Hello from ThermoPilot", "metadata": {}},
            None,
        )

    async def test_check_health(self) -> None:
        mock_executor = MagicMock(return_value={"status": "ok"})
        client = SwitchHttpClient(self.config, request_executor=mock_executor)

        is_healthy = await client.check_health()
        self.assertTrue(is_healthy)
        mock_executor.assert_called_once_with("GET", "/health", None, None)

    async def test_connect_to_room(self) -> None:
        mock_executor = MagicMock(return_value={"result": {"name": "Test Room"}})
        client = SwitchHttpClient(self.config, request_executor=mock_executor)

        res = await client.connect_to_room(self.config.connection_id, "room-xyz")
        self.assertEqual(res["result"]["name"], "Test Room")
        mock_executor.assert_called_once_with(
            "POST",
            f"/agents/{self.config.agent_id}/ops/connect_to_room",
            {"room_id": "room-xyz", "include_general_instructions": False},
            {"X-Switch-Connection-Id": self.config.connection_id},
        )

    async def test_list_assigned_rooms(self) -> None:
        mock_executor = MagicMock(
            return_value={
                "result": [
                    {"room_id": "room-1", "name": "General"},
                    {"room_id": "room-2", "name": "Alerts"},
                ]
            }
        )
        client = SwitchHttpClient(self.config, request_executor=mock_executor)

        rooms = await client.list_assigned_rooms()
        self.assertEqual(len(rooms), 2)
        self.assertEqual(rooms[0]["room_id"], "room-1")
        mock_executor.assert_called_once_with(
            "POST",
            f"/agents/{self.config.agent_id}/ops/list_rooms",
            {},
            None,
        )

    async def test_list_rooms_moderation(self) -> None:
        mock_executor = MagicMock(
            return_value={"rooms": [{"id": "r-mod-1", "name": "All Rooms"}]}
        )
        client = SwitchHttpClient(self.config, request_executor=mock_executor)

        rooms = await client.list_rooms()
        self.assertEqual(len(rooms), 1)
        self.assertEqual(rooms[0]["id"], "r-mod-1")
        mock_executor.assert_called_once_with(
            "GET",
            f"/agents/{self.config.agent_id}/moderation/rooms",
            None,
            None,
        )


class TestSwitchWebSocketClient(unittest.IsolatedAsyncioTestCase):
    """Test SwitchWebSocketClient streaming, pong answering, cursor tracking, and error mapping."""

    def setUp(self) -> None:
        self.config = SwitchAdapterConfig(
            bridge_url="http://localhost:8000",
            agent_id="test-agent",
            api_key="secret",
            connection_id=str(uuid.uuid4()),
        )

    async def test_custom_stream_factory_cursor_and_state(self) -> None:
        sample_frames = [
            {
                "event": "connection_state",
                "id": None,
                "data": {
                    "connection_id": self.config.connection_id,
                    "generation": 100,
                    "rooms": ["room-main"],
                    "cursor": 5,
                },
            },
            {
                "event": "message",
                "id": 6,
                "data": {
                    "type": "message",
                    "room_id": "room-main",
                    "payload": {"body": "hi"},
                },
            },
        ]

        async def mock_stream() -> AsyncGenerator[dict[str, Any], None]:
            for frame in sample_frames:
                yield frame

        client = SwitchWebSocketClient(self.config, stream_factory=mock_stream)
        collected = []
        async for frame in client.connect_events(start_from="head"):
            collected.append(frame)

        self.assertEqual(len(collected), 2)
        self.assertEqual(client.generation, 100)
        self.assertEqual(client.rooms, ["room-main"])
        self.assertEqual(client.cursor, 6)

    async def test_refused_frame_raises_auth_error(self) -> None:
        async def mock_refused_stream() -> AsyncGenerator[dict[str, Any], None]:
            yield {
                "event": "refused",
                "data": {"status": 401, "detail": "Invalid token"},
            }

        client = SwitchWebSocketClient(self.config, stream_factory=mock_refused_stream)
        with self.assertRaises(SwitchAuthError):
            async for _ in client.connect_events():
                pass

    async def test_refused_frame_raises_conflict_error(self) -> None:
        async def mock_conflict_stream() -> AsyncGenerator[dict[str, Any], None]:
            yield {
                "event": "refused",
                "data": {"status": 409, "detail": "Superseded"},
            }

        client = SwitchWebSocketClient(self.config, stream_factory=mock_conflict_stream)
        with self.assertRaises(SwitchConflictError):
            async for _ in client.connect_events():
                pass

    async def test_evicted_frame_raises_evicted_error(self) -> None:
        async def mock_evicted_stream() -> AsyncGenerator[dict[str, Any], None]:
            yield {
                "event": "evicted",
                "data": {"code": "taken_over", "reason": "Another socket opened"},
            }

        client = SwitchWebSocketClient(self.config, stream_factory=mock_evicted_stream)
        with self.assertRaises(SwitchEvictedError) as ctx:
            async for _ in client.connect_events():
                pass
        self.assertEqual(ctx.exception.code, "taken_over")


class TestSwitchSSEClient(unittest.IsolatedAsyncioTestCase):
    """Test legacy SSE parsing and backwards compatibility."""

    async def test_custom_stream_factory(self) -> None:
        config = SwitchAdapterConfig(
            bridge_url="http://localhost:8000",
            agent_id="test-agent",
            api_key="secret",
            connection_id=str(uuid.uuid4()),
        )

        sample_frames = [
            {
                "event": "connection_state",
                "id": None,
                "data": {"connection_id": config.connection_id},
            },
            {
                "event": "message",
                "id": "1",
                "data": {"type": "message", "payload": {"body": "hi"}},
            },
        ]

        async def mock_stream() -> AsyncGenerator[dict[str, Any], None]:
            for frame in sample_frames:
                yield frame

        client = SwitchSSEClient(config, stream_factory=mock_stream)
        collected = []
        async for frame in client.connect_events(config.connection_id):
            collected.append(frame)

        self.assertEqual(len(collected), 2)
        self.assertEqual(collected[0]["event"], "connection_state")
        self.assertEqual(collected[1]["data"]["type"], "message")


class TestForecastSwitchAdapter(unittest.IsolatedAsyncioTestCase):
    """Test the complete advisory adapter with simulated telemetry and mock Switch backend."""

    def setUp(self) -> None:
        self.config = SwitchAdapterConfig(
            bridge_url="http://localhost:8000",
            agent_id="agent-uuid-1234",
            api_key="sw_ak_testsecret",
            connection_id=str(uuid.uuid4()),
            authorized_room_ids=frozenset(["room-discord-main"]),
        )
        self.sim = TelemetrySimulator(
            SimulatorConfig(
                power_budget_w=400.0,
                baseline_ambient_c=22.0,
                anomaly_ambient_c=32.0,
                initial_temp_c=72.0,
                noise_std=0.0,
            )
        )
        self.forecast_agent = ForecastAgent(ForecastConfig())
        self.mock_http = MagicMock()
        self.mock_http.send_message = AsyncMock(
            return_value={"ok": True, "event_id": "$mock1"}
        )
        self.mock_http.send_heartbeat = AsyncMock(return_value={"ok": True})

        self.adapter = ForecastSwitchAdapter(
            config=self.config,
            forecast_agent=self.forecast_agent,
            simulator=self.sim,
            http_client=self.mock_http,
        )

    async def test_unaddressed_message_ignored(self) -> None:
        """Unaddressed chatter in room must be completely ignored."""
        event_data = {
            "type": "message",
            "room_id": "room-discord-main",
            "payload": {
                "addressed": False,
                "sender_name": "developer",
                "body": "deploying new microservice",
            },
        }
        res = await self.adapter.handle_message_event(event_data)
        self.assertIsNone(res)
        self.mock_http.send_message.assert_not_called()

    async def test_unauthorized_room_ignored(self) -> None:
        """Addressed message from an unauthorized room must be rejected."""
        event_data = {
            "type": "message",
            "room_id": "room-unauthorized-999",
            "payload": {
                "addressed": True,
                "sender_name": "attacker",
                "body": "@ForecastAgent status",
            },
        }
        res = await self.adapter.handle_message_event(event_data)
        self.assertIsNone(res)
        self.mock_http.send_message.assert_not_called()

    async def test_self_message_ignored(self) -> None:
        """Adapter must never respond to its own messages."""
        event_data = {
            "type": "message",
            "room_id": "room-discord-main",
            "payload": {
                "addressed": True,
                "sender_name": self.config.client_name,
                "body": "Advisory notification",
            },
        }
        res = await self.adapter.handle_message_event(event_data)
        self.assertIsNone(res)
        self.mock_http.send_message.assert_not_called()

    async def test_status_query(self) -> None:
        """User asks '@ForecastAgent status' -> returns current status card."""
        event_data = {
            "type": "message",
            "room_id": "room-discord-main",
            "payload": {
                "addressed": True,
                "sender_name": "operator",
                "body": "@ForecastAgent status",
            },
        }
        reply = await self.adapter.handle_message_event(event_data)
        self.assertIsNotNone(reply)
        self.assertIn("[ThermoPilot Telemetry Status]", reply)
        self.assertIn("72.0°C", reply)
        self.mock_http.send_message.assert_awaited_once_with(
            room_id="room-discord-main", content=reply
        )

    async def test_forecast_query(self) -> None:
        """User asks '@ForecastAgent forecast' -> returns derivative rate & trajectory."""
        event_data = {
            "type": "message",
            "room_id": "room-discord-main",
            "payload": {
                "addressed": True,
                "sender_name": "operator",
                "body": "@ForecastAgent forecast trajectory",
            },
        }
        reply = await self.adapter.handle_message_event(event_data)
        self.assertIsNotNone(reply)
        self.assertIn("[ThermoPilot Trajectory Forecast]", reply)
        self.assertIn("Time-to-Throttle", reply)

    async def test_time_to_trip_query(self) -> None:
        """User asks '@ForecastAgent ttt' -> returns focused countdown."""
        event_data = {
            "type": "message",
            "room_id": "room-discord-main",
            "payload": {
                "addressed": True,
                "sender_name": "operator",
                "body": "@ForecastAgent ttt",
            },
        }
        reply = await self.adapter.handle_message_event(event_data)
        self.assertIsNotNone(reply)
        self.assertIn("Time-to-Throttle", reply)

    async def test_help_query(self) -> None:
        """User asks '@ForecastAgent help' -> returns command reference."""
        event_data = {
            "type": "message",
            "room_id": "room-discord-main",
            "payload": {
                "addressed": True,
                "sender_name": "operator",
                "body": "@ForecastAgent help",
            },
        }
        reply = await self.adapter.handle_message_event(event_data)
        self.assertIsNotNone(reply)
        self.assertIn("Command Reference", reply)
        self.assertIn("@ForecastAgent status", reply)

    async def test_simulate_anomaly_query(self) -> None:
        """User asks '@ForecastAgent simulate-anomaly' -> returns structured perception alert."""
        event_data = {
            "type": "message",
            "room_id": "room-discord-main",
            "payload": {
                "addressed": True,
                "sender_name": "operator",
                "body": "@ForecastAgent simulate-anomaly",
            },
        }
        reply = await self.adapter.handle_message_event(event_data)
        self.assertIsNotNone(reply)
        self.assertIn("Thermal runaway anomaly", reply)
        self.assertIn("SIMULATED TELEMETRY", reply)
        self.assertIn("```thermopilot", reply)

    async def test_execution_attempt_strictly_denied(self) -> None:
        """User attempts to trigger actuation -> strictly denied with SafetyGate reminder."""
        event_data = {
            "type": "message",
            "room_id": "room-discord-main",
            "payload": {
                "addressed": True,
                "sender_name": "operator",
                "body": "@ForecastAgent approve and execute plan b immediately",
            },
        }
        reply = await self.adapter.handle_message_event(event_data)
        self.assertIsNotNone(reply)
        self.assertIn("Execution Denied", reply)
        self.assertIn("SafetyGate", reply)
        self.assertEqual(self.sim.core_temp_c, 72.0)

    async def test_websocket_streaming_lifecycle(self) -> None:
        """Adapter processes WebSocket stream frames including connection_state and ping."""
        frames = [
            {
                "event": "connection_state",
                "id": None,
                "data": {
                    "connection_id": self.config.connection_id,
                    "generation": 12345,
                    "rooms": ["room-discord-main"],
                    "cursor": 10,
                },
            },
            {"event": "ping", "id": None, "data": {}},
            {
                "event": "message",
                "id": 11,
                "data": {
                    "type": "message",
                    "room_id": "room-discord-main",
                    "payload": {
                        "addressed": True,
                        "sender_name": "operator",
                        "body": "@ForecastAgent status",
                    },
                },
            },
        ]

        async def stream_factory() -> AsyncGenerator[dict[str, Any], None]:
            for f in frames:
                yield f
            self.adapter.is_running = False

        self.adapter.ws_client = SwitchWebSocketClient(
            self.config, stream_factory=stream_factory
        )
        await self.adapter.start()

        self.assertEqual(self.adapter.connection_generation, 12345)
        self.assertEqual(self.adapter.active_rooms, ["room-discord-main"])
        self.assertEqual(self.adapter.current_cursor, 11)
        self.assertGreater(self.adapter.last_heartbeat_time, 0.0)
        self.mock_http.send_message.assert_called_once()

    async def test_reconnect_loop_resilience_and_cursor_resumption(self) -> None:
        """Adapter handles socket drop and reconnects with cursor position."""
        call_count = 0
        requested_start_from: list[str] = []

        async def flaky_stream() -> AsyncGenerator[dict[str, Any], None]:
            nonlocal call_count
            call_count += 1
            start_from = (
                str(self.adapter.current_cursor)
                if self.adapter.current_cursor is not None
                else "head"
            )
            requested_start_from.append(start_from)

            if call_count == 1:
                # First connection delivers frame 1 then disconnects
                yield {
                    "event": "message",
                    "id": 1,
                    "data": {
                        "type": "message",
                        "room_id": "room-discord-main",
                        "payload": {
                            "addressed": True,
                            "sender_name": "operator",
                            "body": "@ForecastAgent status",
                        },
                    },
                }
                raise ConnectionResetError("Simulated socket drop")

            # Second connection yields frame 2 and terminates
            yield {
                "event": "message",
                "id": 2,
                "data": {
                    "type": "message",
                    "room_id": "room-discord-main",
                    "payload": {
                        "addressed": True,
                        "sender_name": "operator",
                        "body": "@ForecastAgent ttt",
                    },
                },
            }
            self.adapter.is_running = False

        self.adapter.ws_client = SwitchWebSocketClient(
            self.config, stream_factory=flaky_stream
        )
        self.adapter.config = SwitchAdapterConfig(
            bridge_url=self.config.bridge_url,
            agent_id=self.config.agent_id,
            api_key=self.config.api_key,
            reconnect_delay_s=0.01,
            max_reconnect_delay_s=0.05,
            authorized_room_ids=self.config.authorized_room_ids,
        )

        await self.adapter.start()
        self.assertGreaterEqual(call_count, 2)
        self.assertEqual(requested_start_from[0], "head")
        self.assertEqual(requested_start_from[1], "1")
        self.assertEqual(self.adapter.current_cursor, 2)

    async def test_fatal_auth_error_terminates_cleanly(self) -> None:
        """Non-recoverable authentication errors terminate adapter without infinite loop."""

        async def auth_error_stream() -> AsyncGenerator[dict[str, Any], None]:
            raise SwitchAuthError("4401 Unauthorized")
            yield {}  # unreachable

        self.adapter.ws_client = SwitchWebSocketClient(
            self.config, stream_factory=auth_error_stream
        )
        await self.adapter.start()
        self.assertFalse(self.adapter.is_running)

    async def test_taken_over_eviction_terminates_cleanly(self) -> None:
        """Connection displaced by another instance terminates without infinite loop."""

        async def taken_over_stream() -> AsyncGenerator[dict[str, Any], None]:
            raise SwitchEvictedError("Connection taken over", code="taken_over")
            yield {}  # unreachable

        self.adapter.ws_client = SwitchWebSocketClient(
            self.config, stream_factory=taken_over_stream
        )
        await self.adapter.start()
        self.assertFalse(self.adapter.is_running)

    async def test_room_auto_discovery_and_claiming(self) -> None:
        """Adapter without authorized_room_ids auto-discovers assigned rooms and claims them."""
        open_config = SwitchAdapterConfig(
            bridge_url="http://localhost:8000",
            agent_id="test-agent",
            api_key="secret",
            authorized_room_ids=frozenset(),
        )
        mock_http = MagicMock()
        mock_http.list_assigned_rooms = AsyncMock(
            return_value=[{"room_id": "auto-discovered-room-1", "name": "General"}]
        )

        stream_called_with_rooms = None
        adapter: ForecastSwitchAdapter | None = None

        async def inspect_stream_factory() -> AsyncGenerator[dict[str, Any], None]:
            yield {
                "event": "connection_state",
                "id": None,
                "data": {
                    "connection_id": open_config.connection_id,
                    "generation": 1,
                    "rooms": ["auto-discovered-room-1"],
                    "cursor": 0,
                },
            }
            if adapter is not None:
                adapter.is_running = False

        mock_ws = MagicMock()

        async def mock_connect_events(
            start_from: str | None = None, rooms: Any = None
        ) -> AsyncGenerator[dict[str, Any], None]:
            nonlocal stream_called_with_rooms
            stream_called_with_rooms = rooms
            async for frame in inspect_stream_factory():
                yield frame

        mock_ws.connect_events = mock_connect_events
        mock_ws.close = AsyncMock()
        mock_ws._custom_stream_factory = None

        adapter = ForecastSwitchAdapter(
            config=open_config,
            http_client=mock_http,
            ws_client=mock_ws,
        )
        await adapter.start()

        mock_http.list_assigned_rooms.assert_awaited_once()
        self.assertEqual(stream_called_with_rooms, ["auto-discovered-room-1"])
        self.assertEqual(adapter.active_rooms, ["auto-discovered-room-1"])

    async def test_inbound_mutation_rejection_variations(self) -> None:
        """Any mutation command in chat is rejected without modifying simulator."""
        mutation_queries = [
            "@forecast-agent approve plan a",
            "@forecast-agent execute",
            "@forecast-agent apply mitigation",
            "@forecast-agent kill workload",
            "@forecast-agent throttle now",
            "@forecast-agent mitigate immediately",
            "@forecast-agent please execute the emergency plan",
            "@forecast-agent approve",
        ]
        for query in mutation_queries:
            reply = await self.adapter.handle_message_event(
                {
                    "type": "message",
                    "room_id": "room-discord-main",
                    "payload": {
                        "addressed": True,
                        "sender_name": "attacker",
                        "body": query,
                    },
                }
            )
            self.assertIsNotNone(reply)
            self.assertIn("Execution Denied", reply)
            self.assertIn("SafetyGate", reply)
            self.assertEqual(self.sim.core_temp_c, 72.0)

    async def test_normal_throttle_query_preserved(self) -> None:
        """Advisory 'throttle' and time-to-throttle queries are preserved and not rejected."""
        advisory_queries = [
            "@forecast-agent throttle",
            "@forecast-agent time to throttle",
            "@forecast-agent time to trip",
            "@forecast-agent ttt",
        ]
        for query in advisory_queries:
            reply = await self.adapter.handle_message_event(
                {
                    "type": "message",
                    "room_id": "room-discord-main",
                    "payload": {
                        "addressed": True,
                        "sender_name": "operator",
                        "body": query,
                    },
                }
            )
            self.assertIsNotNone(reply)
            self.assertNotIn("Execution Denied", reply)
            self.assertIn("Time-to-Throttle", reply)
            self.assertEqual(self.sim.core_temp_c, 72.0)


if __name__ == "__main__":
    unittest.main()
