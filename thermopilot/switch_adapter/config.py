"""Configuration for the ThermoPilot Switch Agent Adapter.

Reads connection parameters, credentials, and room authorization rules from
environment variables or configuration files. Guarantees that credentials
are never logged or serialized in plaintext.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SwitchAdapterConfig(BaseModel):
    """Configuration options for connecting ThermoPilot to Agent Switch."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    bridge_url: str = Field(
        default="http://localhost:8000",
        description="Base URL of the Switch Agent Bridge (e.g. http://localhost:8000)",
    )
    agent_id: str = Field(
        ...,
        description="UUID identifier of the registered Switch agent",
    )
    api_key: str = Field(
        ...,
        description="Bearer API key for the registered Switch agent",
    )
    connection_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Persistent UUID identifier for the client connection instance",
    )
    authorized_room_ids: frozenset[str] = Field(
        default_factory=frozenset,
        description="Set of authorized Discord room IDs; if empty, all rooms the agent belongs to are allowed",
    )
    heartbeat_interval_s: float = Field(
        default=2.0,
        gt=0.0,
        description="Interval in seconds between connection heartbeats (Switch TTL is 6.0s)",
    )
    reconnect_delay_s: float = Field(
        default=2.0,
        gt=0.0,
        description="Initial delay in seconds before attempting to reconnect dropped streams",
    )
    max_reconnect_delay_s: float = Field(
        default=30.0,
        gt=0.0,
        description="Maximum exponential backoff delay in seconds for reconnections",
    )
    client_name: str = Field(
        default="thermopilot-forecast-adapter",
        description="Client declaration identifier sent on the event stream query",
    )
    client_version: str = Field(
        default="0.1.0",
        description="Client version string declared to the Switch bridge",
    )
    scope: str = Field(
        default="all",
        description="Delivery scope: 'all' covers every room the agent belongs to",
    )
    event_filter: str = Field(
        default="all",
        description="Server-side event filter: 'all' or 'addressed'",
    )

    @field_validator("bridge_url")
    @classmethod
    def validate_bridge_url(cls, v: str) -> str:
        parsed = urlparse(v)
        if not parsed.scheme or parsed.scheme not in ("http", "https"):
            raise ValueError(f"bridge_url must start with http:// or https://: {v}")
        return v.rstrip("/")

    @field_validator("connection_id")
    @classmethod
    def validate_connection_id(cls, v: str) -> str:
        try:
            uuid.UUID(v)
        except ValueError as exc:
            raise ValueError(f"connection_id must be a valid UUID string: {v}") from exc
        return v

    def is_room_authorized(self, room_id: str) -> bool:
        """Check whether incoming room_id is allowed to query the Forecast Agent."""
        if not self.authorized_room_ids:
            return True
        return room_id in self.authorized_room_ids

    def masked_api_key(self) -> str:
        """Safe masked representation of api_key for diagnostic logging."""
        if len(self.api_key) <= 8:
            return "***"
        return f"{self.api_key[:4]}...{self.api_key[-4:]}"

    def __repr__(self) -> str:
        return (
            f"SwitchAdapterConfig(bridge_url={self.bridge_url!r}, "
            f"agent_id={self.agent_id!r}, "
            f"api_key={self.masked_api_key()!r}, "
            f"connection_id={self.connection_id!r}, "
            f"authorized_room_ids={list(self.authorized_room_ids)!r}, "
            f"heartbeat_interval_s={self.heartbeat_interval_s})"
        )

    @classmethod
    def from_env(
        cls,
        connection_id_file: Path | str | None = ".switch_connection_id",
        **overrides: Any,
    ) -> SwitchAdapterConfig:
        """Construct configuration from environment variables.

        Supported Environment Variables:
        - SWITCH_BRIDGE_URL: Switch agent bridge base URL (default: http://localhost:8000)
        - SWITCH_AGENT_ID: Switch agent UUID identifier
        - SWITCH_API_KEY: Agent Bearer authentication token
        - SWITCH_CONNECTION_ID: Connection UUID (if unset, loads/saves to connection_id_file)
        - SWITCH_AUTHORIZED_ROOM_IDS: Comma-separated list of permitted room IDs
        - SWITCH_HEARTBEAT_INTERVAL_S: Float seconds between beats (default: 2.0)
        """
        bridge_url = os.environ.get("SWITCH_BRIDGE_URL", "http://localhost:8000")
        agent_id = os.environ.get("SWITCH_AGENT_ID")
        api_key = os.environ.get("SWITCH_API_KEY")

        if not agent_id and "agent_id" not in overrides:
            raise ValueError("SWITCH_AGENT_ID environment variable is required")
        if not api_key and "api_key" not in overrides:
            raise ValueError("SWITCH_API_KEY environment variable is required")

        # Persistent connection ID resolution
        conn_id = os.environ.get("SWITCH_CONNECTION_ID")
        if not conn_id and connection_id_file:
            path = Path(connection_id_file)
            if path.exists():
                try:
                    loaded = path.read_text().strip()
                    uuid.UUID(loaded)
                    conn_id = loaded
                except (OSError, ValueError):
                    conn_id = None
            if not conn_id:
                conn_id = str(uuid.uuid4())
                try:
                    path.write_text(conn_id)
                except OSError:
                    pass

        rooms_raw = os.environ.get("SWITCH_AUTHORIZED_ROOM_IDS", "")
        authorized_rooms = (
            frozenset(r.strip() for r in rooms_raw.split(",") if r.strip())
            if rooms_raw
            else frozenset()
        )

        heartbeat_s = float(os.environ.get("SWITCH_HEARTBEAT_INTERVAL_S", "2.0"))

        data: dict[str, Any] = {
            "bridge_url": bridge_url,
            "agent_id": agent_id or overrides.get("agent_id"),
            "api_key": api_key or overrides.get("api_key"),
            "authorized_room_ids": authorized_rooms,
            "heartbeat_interval_s": heartbeat_s,
        }
        if conn_id:
            data["connection_id"] = conn_id

        data.update(overrides)
        return cls(**data)
