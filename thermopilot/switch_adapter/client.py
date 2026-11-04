"""HTTP and SSE client implementations for the Switch Agent Protocol.

Provides asynchronous communication with the Switch Agent Bridge:
- Outbound REST operations (heartbeat, message dispatch, room operations).
- Inbound Server-Sent Events (SSE) streaming with frame parsing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import AsyncGenerator, Callable
from typing import Any

from thermopilot.switch_adapter.config import SwitchAdapterConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SwitchClientError(Exception):
    """Base exception for Switch client communication errors."""


class SwitchAuthError(SwitchClientError):
    """Authentication or authorization failure (401/403)."""


class SwitchNotFoundError(SwitchClientError):
    """Requested agent, connection, or room was not found (404)."""


class SwitchConflictError(SwitchClientError):
    """Conflict error, such as a superseded connection or missing stream (409)."""


# ---------------------------------------------------------------------------
# Switch HTTP Client
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
                raise SwitchAuthError(f"Switch auth failed ({exc.code}): {err_body}") from exc
            if exc.code == 404:
                raise SwitchNotFoundError(f"Switch resource not found ({exc.code}): {err_body}") from exc
            if exc.code == 409:
                raise SwitchConflictError(f"Switch conflict ({exc.code}): {err_body}") from exc
            raise SwitchClientError(f"HTTP error {exc.code} {exc.reason}: {err_body}") from exc
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

        Must be sent every 2 seconds. Switch TTL is 6 seconds.
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

    async def connect_to_room(
        self, connection_id: str, room_id: str
    ) -> dict[str, Any]:
        """Claim a room on the active connection via POST /agents/{id}/ops/connect_to_room."""
        path = f"/agents/{self.config.agent_id}/ops/connect_to_room"
        headers = {"X-Switch-Connection-Id": connection_id}
        body = {
            "room_id": room_id,
            "include_general_instructions": False,
        }
        return await self._request("POST", path, body=body, extra_headers=headers)


# ---------------------------------------------------------------------------
# Switch SSE Client
# ---------------------------------------------------------------------------

class SwitchSSEClient:
    """Asynchronous Server-Sent Events (SSE) streaming client for Switch events."""

    def __init__(
        self,
        config: SwitchAdapterConfig,
        stream_factory: Callable[[], AsyncGenerator[dict[str, Any], None]] | None = None,
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

        query_params = urllib.parse.urlencode({
            "connection_id": connection_id,
            "scope": self.config.scope,
            "filter": self.config.event_filter,
            "start_from": start_from,
            "client": self.config.client_name,
            "client_version": self.config.client_version,
        })
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

            # Read HTTP response status
            status_line = await reader.readline()
            if not status_line:
                raise SwitchClientError("Empty HTTP response from Switch bridge SSE endpoint")

            parts = status_line.decode("utf-8").strip().split(" ", 2)
            if len(parts) < 2:
                raise SwitchClientError(f"Malformed HTTP status line: {status_line.decode().strip()!r}")

            status_code = int(parts[1])
            if status_code != 200:
                # Read response body for error detail
                err_lines = []
                while True:
                    line = await reader.readline()
                    if not line or line in (b"\r\n", b"\n"):
                        break
                    err_lines.append(line.decode("utf-8", errors="replace"))
                detail = "".join(err_lines)
                if status_code in (401, 403):
                    raise SwitchAuthError(f"SSE connection refused ({status_code}): {detail}")
                if status_code == 409:
                    raise SwitchConflictError(f"SSE connection conflict ({status_code}): {detail}")
                raise SwitchClientError(f"SSE connection failed with HTTP {status_code}: {detail}")

            # Consume headers until blank line
            while True:
                hdr_line = await reader.readline()
                if not hdr_line or hdr_line in (b"\r\n", b"\n"):
                    break

            # Parse SSE lines
            current_event = "message"
            current_id: str | None = None
            data_buffer: list[str] = []

            while not reader.at_eof():
                line_bytes = await reader.readline()
                if not line_bytes:
                    break

                line = line_bytes.decode("utf-8").rstrip("\r\n")

                if not line:
                    # Empty line dispatches current buffered SSE event
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
                    # SSE comment line (keepalive)
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
