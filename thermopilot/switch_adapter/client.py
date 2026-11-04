"""HTTP and WebSocket client implementations for the Switch Agent Protocol.

Provides asynchronous communication with the Switch Agent Bridge:
- Inbound WebSocket duplex streaming (events down, heartbeat pongs up).
- Outbound REST operations (message dispatch, room operations, health probe).
- Deprecated legacy SSE client for backwards compatibility.
"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import AsyncGenerator, Callable, Sequence
from typing import Any

from thermopilot.switch_adapter.config import SwitchAdapterConfig

try:
    import websockets
    import websockets.exceptions
except ImportError:  # pragma: no cover
    websockets = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SwitchClientError(Exception):
    """Base exception for Switch client communication errors."""


class SwitchAuthError(SwitchClientError):
    """Authentication or authorization failure (401/403 or 4401/4403)."""


class SwitchNotFoundError(SwitchClientError):
    """Requested agent, connection, or room was not found (404 or 4404)."""


class SwitchConflictError(SwitchClientError):
    """Conflict error, such as a superseded connection or missing stream (409 or 4409)."""


class SwitchRefusedError(SwitchClientError):
    """Connection or request refused by the Switch bridge."""


class SwitchEvictedError(SwitchConflictError):
    """Connection was displaced or evicted by another instance or heartbeat lapse."""

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Switch HTTP Client (Outbound REST Operations)
# ---------------------------------------------------------------------------


class SwitchHttpClient:
    """Asynchronous HTTP client for Switch Agent Bridge REST endpoints."""

    def __init__(
        self,
        config: SwitchAdapterConfig,
        request_executor: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self._custom_executor = request_executor

    def _sync_request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        """Synchronous HTTP request worker run inside an asyncio thread pool."""
        if self._custom_executor is not None:
            return self._custom_executor(method, path, body, extra_headers)

        url = f"{self.config.bridge_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "User-Agent": f"{self.config.client_name}/{self.config.client_version}",
            "Accept": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)

        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                status = response.status
                raw = response.read().decode("utf-8")
                if not raw.strip():
                    return {"ok": True, "status": status}
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            if exc.code in (401, 403):
                raise SwitchAuthError(
                    f"Switch auth failed ({exc.code}): {err_body}"
                ) from exc
            if exc.code == 404:
                raise SwitchNotFoundError(
                    f"Switch resource not found ({exc.code}): {err_body}"
                ) from exc
            if exc.code == 409:
                raise SwitchConflictError(
                    f"Switch conflict ({exc.code}): {err_body}"
                ) from exc
            raise SwitchClientError(
                f"HTTP error {exc.code} {exc.reason}: {err_body}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise SwitchClientError(f"Connection failure to {url}: {exc}") from exc

    async def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        """Dispatch synchronous HTTP call in an asyncio background worker."""
        return await asyncio.to_thread(
            self._sync_request, method, path, body, extra_headers, timeout_s
        )

    async def check_health(self) -> bool:
        """Probe the Switch bridge GET /health endpoint."""
        try:
            res = await self._request("GET", "/health", timeout_s=5.0)
            return res.get("status") in ("ok", "healthy") or res.get("ok") is True
        except (SwitchClientError, OSError) as exc:
            logger.debug("Health probe failed: %s", exc)
            return False

    async def send_heartbeat(
        self, connection_id: str, cursor: int | None = None
    ) -> dict[str, Any]:
        """Send connection heartbeat to POST /agents/{agent_id}/connection/beat.

        Note: WebSocket transport uses in-band pongs instead of HTTP beats.
        Kept for legacy compatibility.
        """
        path = f"/agents/{self.config.agent_id}/connection/beat"
        body: dict[str, Any] = {"connection_id": connection_id}
        if cursor is not None:
            body["cursor"] = cursor
        return await self._request("POST", path, body=body, timeout_s=5.0)

    async def send_message(
        self, room_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Post a chat reply to Discord/Switch via POST /agents/{agent_id}/message."""
        path = f"/agents/{self.config.agent_id}/message"
        body: dict[str, Any] = {
            "room_id": room_id,
            "content": content,
            "metadata": metadata or {},
        }
        return await self._request("POST", path, body=body, timeout_s=10.0)

    async def connect_to_room(self, connection_id: str, room_id: str) -> dict[str, Any]:
        """Claim a room on the active connection via POST /agents/{id}/ops/connect_to_room."""
        path = f"/agents/{self.config.agent_id}/ops/connect_to_room"
        headers = {"X-Switch-Connection-Id": connection_id}
        body = {
            "room_id": room_id,
            "include_general_instructions": False,
        }
        return await self._request("POST", path, body=body, extra_headers=headers)

    async def list_assigned_rooms(self) -> list[dict[str, Any]]:
        """Fetch rooms this agent is assigned to via POST /agents/{id}/ops/list_rooms."""
        path = f"/agents/{self.config.agent_id}/ops/list_rooms"
        res = await self._request("POST", path, body={}, timeout_s=5.0)
        return res.get("result", [])

    async def list_rooms(self) -> list[dict[str, Any]]:
        """Fetch all tenant rooms visible to this agent via GET /agents/{id}/moderation/rooms."""
        path = f"/agents/{self.config.agent_id}/moderation/rooms"
        res = await self._request("GET", path, timeout_s=5.0)
        return res.get("rooms", [])


# ---------------------------------------------------------------------------
# Switch WebSocket Client (Current Streaming Transport)
# ---------------------------------------------------------------------------


class SwitchWebSocketClient:
    """Asynchronous WebSocket client for the Switch Agent Protocol.

    Provides streaming event reception and automatic pong heartbeat responses
    over a single persistent duplex WebSocket connection.
    """

    def __init__(
        self,
        config: SwitchAdapterConfig,
        stream_factory: Callable[[], AsyncGenerator[dict[str, Any], None]]
        | None = None,
    ) -> None:
        self.config = config
        self._custom_stream_factory = stream_factory
        self.cursor: int | None = None
        self.generation: int | None = None
        self.rooms: list[str] = []
        self._active_socket: Any | None = None

    async def connect_events(
        self,
        start_from: str | None = None,
        rooms: Sequence[str] | frozenset[str] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Establish WebSocket connection to Switch Agent Bridge and yield structured frames.

        Automatically handles server `ping` frames by sending `pong` responses with the
        current cursor position to maintain connection liveness without out-of-band HTTP beats.
        """
        if self._custom_stream_factory is not None:
            async for item in self._custom_stream_factory():
                if isinstance(item, dict):
                    event = item.get("event")
                    frame_id = item.get("id")
                    if frame_id is not None and str(frame_id).isdigit():
                        self.cursor = int(frame_id)
                    if event == "connection_state":
                        st = item.get("data", {})
                        if isinstance(st, dict):
                            self.generation = st.get("generation")
                            self.rooms = list(st.get("rooms", []))
                            if st.get("cursor") is not None:
                                self.cursor = int(st["cursor"])
                    elif event == "refused":
                        ref = item.get("data", {})
                        status = ref.get("status", 400)
                        detail = ref.get("detail", "Refused")
                        if status in (401, 403):
                            raise SwitchAuthError(
                                f"WebSocket connection refused ({status}): {detail}"
                            )
                        if status == 409:
                            raise SwitchConflictError(
                                f"WebSocket connection conflict ({status}): {detail}"
                            )
                        raise SwitchRefusedError(
                            f"WebSocket connection refused ({status}): {detail}"
                        )
                    elif event == "evicted":
                        evict = item.get("data", {})
                        code = evict.get("code", "closed")
                        reason = evict.get("reason", str(evict))
                        raise SwitchEvictedError(
                            f"Connection evicted ({code}): {reason}", code=code
                        )
                yield item
            return

        if websockets is None:
            raise SwitchClientError(
                "The 'websockets' library is required for Switch WebSocket transport. "
                "Install it with 'pip install websockets'."
            )

        ws_url = self.config.build_ws_url(start_from=start_from, rooms=rooms)
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "User-Agent": f"{self.config.client_name}/{self.config.client_version}",
        }

        try:
            async with (
                websockets.connect(
                    ws_url,
                    additional_headers=headers,
                    open_timeout=10.0,
                    close_timeout=5.0,
                    ping_interval=None,  # In-band application pings from Switch core are used
                ) as ws
            ):
                self._active_socket = ws
                try:
                    while True:
                        try:
                            raw = await ws.recv()
                        except (
                            websockets.exceptions.ConnectionClosedOK,
                            websockets.exceptions.ConnectionClosed,
                        ) as exc:
                            code = getattr(
                                exc,
                                "code",
                                getattr(getattr(exc, "rcvd", None), "code", 1000),
                            )
                            reason = getattr(
                                exc,
                                "reason",
                                getattr(getattr(exc, "rcvd", None), "reason", ""),
                            )
                            if code in (4401, 4403):
                                raise SwitchAuthError(
                                    f"WebSocket closed with authentication error ({code}): {reason}"
                                ) from exc
                            if code == 4404:
                                raise SwitchNotFoundError(
                                    f"WebSocket resource not found ({code}): {reason}"
                                ) from exc
                            if code == 4409:
                                raise SwitchConflictError(
                                    f"WebSocket conflict ({code}): {reason}"
                                ) from exc
                            if code >= 4000:
                                raise SwitchClientError(
                                    f"WebSocket closed with status code {code}: {reason}"
                                ) from exc
                            if code in (1000, 1001):
                                break
                            raise SwitchClientError(
                                f"WebSocket connection closed unexpectedly ({code}): {reason}"
                            ) from exc

                        try:
                            frame = (
                                json.loads(raw)
                                if isinstance(raw, (str, bytes))
                                else raw
                            )
                        except json.JSONDecodeError as exc:
                            logger.warning(
                                "Failed to decode JSON frame from Switch: %s", exc
                            )
                            continue

                        if not isinstance(frame, dict):
                            continue

                        event = frame.get("event")

                        # Track event cursor sequence if present
                        frame_id = frame.get("id")
                        if frame_id is not None and str(frame_id).isdigit():
                            self.cursor = int(frame_id)

                        # Handle application ping frames from server
                        if event == "ping":
                            pong_msg: dict[str, Any] = {"type": "pong"}
                            if self.cursor is not None:
                                pong_msg["cursor"] = self.cursor
                            await ws.send(json.dumps(pong_msg))
                            yield frame
                            continue

                        # Handle connection_state frame
                        if event == "connection_state":
                            st = frame.get("data", {})
                            if isinstance(st, dict):
                                self.generation = st.get("generation")
                                self.rooms = list(st.get("rooms", []))
                                if st.get("cursor") is not None:
                                    self.cursor = int(st["cursor"])
                            yield frame
                            continue

                        # Handle refused frame
                        if event == "refused":
                            ref = frame.get("data", {})
                            status = ref.get("status", 400)
                            detail = ref.get("detail", "Refused")
                            if status in (401, 403):
                                raise SwitchAuthError(
                                    f"WebSocket connection refused ({status}): {detail}"
                                )
                            if status == 409:
                                raise SwitchConflictError(
                                    f"WebSocket connection conflict ({status}): {detail}"
                                )
                            raise SwitchRefusedError(
                                f"WebSocket connection refused ({status}): {detail}"
                            )

                        # Handle evicted frame
                        if event == "evicted":
                            evict = frame.get("data", {})
                            code = evict.get("code", "closed")
                            reason = evict.get("reason", str(evict))
                            raise SwitchEvictedError(
                                f"Connection evicted ({code}): {reason}", code=code
                            )

                        yield frame
                finally:
                    self._active_socket = None

        except (
            websockets.exceptions.InvalidStatusCode,
            websockets.exceptions.InvalidHandshake,
        ) as exc:
            status = getattr(exc, "status_code", None)
            if status in (401, 403):
                raise SwitchAuthError(
                    f"WebSocket handshake failed with HTTP {status}: {exc}"
                ) from exc
            if status == 404:
                raise SwitchNotFoundError(
                    f"WebSocket endpoint not found (HTTP {status}): {exc}"
                ) from exc
            if status == 409:
                raise SwitchConflictError(
                    f"WebSocket handshake conflict (HTTP {status}): {exc}"
                ) from exc
            raise SwitchClientError(f"WebSocket handshake failed: {exc}") from exc
        except (OSError, TimeoutError) as exc:
            raise SwitchClientError(f"WebSocket transport error: {exc}") from exc

    async def close(self) -> None:
        """Close active WebSocket connection if open."""
        if self._active_socket is not None:
            try:
                await self._active_socket.close()
            except (OSError, TimeoutError) as exc:
                logger.debug("Error while closing WebSocket: %s", exc)
            self._active_socket = None


# ---------------------------------------------------------------------------
# Switch SSE Client (Deprecated Legacy Transport)
# ---------------------------------------------------------------------------


class SwitchSSEClient:
    """Deprecated Server-Sent Events (SSE) streaming client.

    Maintained for legacy compatibility. Use SwitchWebSocketClient for current
    Switch Agent Protocol installations.
    """

    def __init__(
        self,
        config: SwitchAdapterConfig,
        stream_factory: Callable[[], AsyncGenerator[dict[str, Any], None]]
        | None = None,
    ) -> None:
        self.config = config
        self._custom_stream_factory = stream_factory

    async def connect_events(
        self,
        connection_id: str,
        start_from: str = "head",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Establish SSE connection and yield structured domain and control frames."""
        if self._custom_stream_factory is not None:
            async for item in self._custom_stream_factory():
                yield item
            return

        parsed_url = urllib.parse.urlparse(self.config.bridge_url)
        is_ssl = parsed_url.scheme == "https"
        host = parsed_url.hostname or "localhost"
        port = parsed_url.port or (443 if is_ssl else 80)

        query_params = urllib.parse.urlencode(
            {
                "connection_id": connection_id,
                "scope": self.config.scope,
                "filter": self.config.event_filter,
                "start_from": start_from,
                "client": self.config.client_name,
                "client_version": self.config.client_version,
            }
        )
        path = f"/agents/{self.config.agent_id}/events?{query_params}"

        ssl_ctx = ssl.create_default_context() if is_ssl else None

        reader, writer = await asyncio.open_connection(host, port, ssl=ssl_ctx)

        try:
            http_req = (
                f"GET {path} HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n"
                f"Authorization: Bearer {self.config.api_key}\r\n"
                f"Accept: text/event-stream\r\n"
                f"User-Agent: {self.config.client_name}/{self.config.client_version}\r\n"
                f"Connection: keep-alive\r\n"
                f"\r\n"
            )
            writer.write(http_req.encode("utf-8"))
            await writer.drain()

            status_line = await reader.readline()
            if not status_line:
                raise SwitchClientError(
                    "Empty HTTP response from Switch bridge SSE endpoint"
                )

            parts = status_line.decode("utf-8").strip().split(" ", 2)
            if len(parts) < 2:
                raise SwitchClientError(
                    f"Malformed HTTP status line: {status_line.decode().strip()!r}"
                )

            status_code = int(parts[1])
            if status_code != 200:
                err_lines = []
                while True:
                    line = await reader.readline()
                    if not line or line in (b"\r\n", b"\n"):
                        break
                    err_lines.append(line.decode("utf-8", errors="replace"))
                detail = "".join(err_lines)
                if status_code in (401, 403):
                    raise SwitchAuthError(
                        f"SSE connection refused ({status_code}): {detail}"
                    )
                if status_code == 409:
                    raise SwitchConflictError(
                        f"SSE connection conflict ({status_code}): {detail}"
                    )
                raise SwitchClientError(
                    f"SSE connection failed with HTTP {status_code}: {detail}"
                )

            while True:
                hdr_line = await reader.readline()
                if not hdr_line or hdr_line in (b"\r\n", b"\n"):
                    break

            current_event = "message"
            current_id: str | None = None
            data_buffer: list[str] = []

            while not reader.at_eof():
                line_bytes = await reader.readline()
                if not line_bytes:
                    break

                line = line_bytes.decode("utf-8").rstrip("\r\n")

                if not line:
                    if data_buffer:
                        full_data = "\n".join(data_buffer)
                        try:
                            parsed_payload = json.loads(full_data)
                        except json.JSONDecodeError:
                            parsed_payload = full_data

                        yield {
                            "event": current_event,
                            "id": current_id,
                            "data": parsed_payload,
                        }
                        data_buffer = []
                        current_event = "message"
                        current_id = None
                    continue

                if line.startswith(":"):
                    continue

                if line.startswith("event:"):
                    current_event = line[6:].strip()
                elif line.startswith("id:"):
                    current_id = line[3:].strip()
                elif line.startswith("data:"):
                    data_buffer.append(line[5:].strip())

        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
