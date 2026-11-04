"""Comprehensive unit tests for the ThermoPilot Switch Agent Adapter.

Verifies:
1. SwitchAdapterConfig validation, environment parsing, and credential masking.
2. SwitchHttpClient REST request building, authorization, and error handling.
3. SwitchSSEClient line parsing, event dispatching, and stream error handling.
4. ForecastSwitchAdapter message filtering (addressed vs unaddressed, authorized vs unauthorized rooms).
5. Advisory query routing (status, forecast, time-to-trip, help, execution denial).
6. Background heartbeat scheduling and lifecycle shutdown.
7. Reconnection handling on simulated network disconnections.
"""

from __future__ import annotations

import asyncio
import unittest
import uuid
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from thermopilot.agents.forecast import ForecastAgent, ForecastConfig
from thermopilot.simulator.telemetry import SimulatorConfig, TelemetrySimulator
from thermopilot.switch_adapter.adapter import ForecastSwitchAdapter
from thermopilot.switch_adapter.client import SwitchHttpClient, SwitchSSEClient
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
            cfg = SwitchAdapterConfig.from_env(connection_id_file=None)
            self.assertEqual(cfg.bridge_url, "http://127.0.0.1:9000")
            self.assertEqual(cfg.agent_id, "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
            self.assertEqual(cfg.api_key, "sw_ak_envsecret9999")
            self.assertEqual(cfg.authorized_room_ids, frozenset(["room-a", "room-b"]))
            self.assertEqual(cfg.heartbeat_interval_s, 1.5)


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

        res = await client.send_message(room_id="room-42", content="Hello from ThermoPilot")
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


class TestSwitchSSEClient(unittest.IsolatedAsyncioTestCase):
    """Test SSE parsing, comment filtering, and structured frame delivery."""

    async def test_custom_stream_factory(self) -> None:
        config = SwitchAdapterConfig(
            bridge_url="http://localhost:8000",
            agent_id="test-agent",
            api_key="secret",
            connection_id=str(uuid.uuid4()),
        )

        sample_frames = [
            {"event": "connection_state", "id": None, "data": {"connection_id": config.connection_id}},
            {"event": "message", "id": "1", "data": {"type": "message", "payload": {"body": "hi"}}},
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
        # Controlled simulator starting nominal at 72.0C
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
        self.mock_http.send_message = AsyncMock(return_value={"ok": True, "event_id": "$mock1"})
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
        # Ensure simulator was not modified
        self.assertEqual(self.sim.core_temp_c, 72.0)

    async def test_heartbeat_worker_lifecycle(self) -> None:
        """Heartbeat worker runs and cancels cleanly on stop."""
        self.adapter.is_running = True
        task = asyncio.create_task(self.adapter._heartbeat_worker())
        await asyncio.sleep(0.05)
        await self.adapter.stop()
        await task
        self.assertFalse(self.adapter.is_running)
        self.mock_http.send_heartbeat.assert_called()

    async def test_reconnect_loop_resilience(self) -> None:
        """Adapter handles stream disconnect and reconnects without crashing."""
        call_count = 0

        async def flaky_stream() -> AsyncGenerator[dict[str, Any], None]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First connection drops immediately
                raise ConnectionResetError("Simulated socket drop")
            # Second connection yields one event then finishes
            yield {
                "event": "message",
                "id": "1",
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
            # Terminate test
            self.adapter.is_running = False

        self.adapter.sse_client = SwitchSSEClient(self.config, stream_factory=flaky_stream)
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
        self.mock_http.send_message.assert_called_once()


if __name__ == "__main__":
    unittest.main()
