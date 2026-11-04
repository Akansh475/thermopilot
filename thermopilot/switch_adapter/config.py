"""Configuration for the ThermoPilot Switch Agent Adapter.

Reads connection parameters, credentials, and room authorization rules from
environment variables or configuration files. Guarantees that credentials
are never logged or serialized in plaintext.
"""

from __future__ import annotations

import os
import urllib.parse
import uuid
from collections.abc import Sequence
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
    authorized_operators: frozenset[str] = Field(
        default_factory=frozenset,
        description="Set of authorized operator usernames; if empty, any authenticated human is allowed",
    )
    approval_timeout_s: float = Field(
        default=30.0,
        gt=0.0,
        description="Approval governance window in seconds before fail-safe timeout escalation",
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

    def is_operator_authorized(self, operator_id: str) -> bool:
        """Check whether operator_id is an authorized human operator (agents strictly rejected)."""
        if not operator_id or not operator_id.strip():
            return False

        clean_id = operator_id.strip().lower()
        # Autonomous agents are strictly prohibited from acting as human approval authorities
        forbidden_agents = {
            "forecast-agent",
            "workload-agent",
            "planner-agent",
            "thermopilot-forecast-adapter",
            "thermopilot-workload-adapter",
            "thermopilot-planner-adapter",
            self.client_name.lower(),
        }
        if clean_id in forbidden_agents or clean_id.endswith(("-agent", "-adapter")):
            return False

        if not self.authorized_operators:
            return True

        allowed = {op.lower() for op in self.authorized_operators}
        return clean_id in allowed

    def build_ws_url(
        self,
        start_from: str | None = None,
        rooms: Sequence[str] | frozenset[str] | None = None,
    ) -> str:
        """Construct the official Switch Agent Protocol WebSocket URL with query parameters."""
        parsed = urlparse(self.bridge_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        netloc = parsed.netloc

        params: dict[str, str] = {
            "connection_id": self.connection_id,
            "scope": self.scope,
            "filter": self.event_filter,
            "start_from": start_from or "head",
            "client": self.client_name,
            "client_version": self.client_version,
        }
        if rooms:
            params["rooms"] = ",".join(sorted(rooms))

        query_str = urllib.parse.urlencode(params)
        return f"{scheme}://{netloc}/agents/{self.agent_id}/connection/ws?{query_str}"

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
        env_file: Path | str | None = ".env",
        connection_id_file: Path | str | None = ".switch_connection_id",
        role: str | None = None,
        **overrides: Any,
    ) -> SwitchAdapterConfig:
        """Construct configuration from environment variables.

        Supported Environment Variables:
        - SWITCH_BRIDGE_URL: Switch agent bridge base URL (default: http://localhost:8000)
        - SWITCH_AGENT_ID: Switch agent UUID identifier (or SWITCH_{ROLE}_AGENT_ID if role specified)
        - SWITCH_API_KEY: Agent Bearer authentication token (or SWITCH_{ROLE}_API_KEY)
        - SWITCH_CONNECTION_ID: Connection UUID (if unset, loads/saves to connection_id_file)
        - SWITCH_AUTHORIZED_ROOM_IDS: Comma-separated list of permitted room IDs
        - SWITCH_HEARTBEAT_INTERVAL_S: Float seconds between beats (default: 2.0)
        """
        env_dict: dict[str, str] = {}
        if env_file:
            path = Path(env_file)
            if path.exists():
                try:
                    for line in path.read_text().splitlines():
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            env_dict[k.strip()] = v.strip().strip("\"'")
                except OSError:
                    pass

        bridge_url = os.environ.get("SWITCH_BRIDGE_URL") or env_dict.get(
            "SWITCH_BRIDGE_URL", "http://localhost:8000"
        )
        if role and role.lower() != "forecast":
            prefix = f"SWITCH_{role.upper()}_"
            agent_id = os.environ.get(f"{prefix}AGENT_ID") or env_dict.get(
                f"{prefix}AGENT_ID"
            )
            api_key = os.environ.get(f"{prefix}API_KEY") or env_dict.get(
                f"{prefix}API_KEY"
            )
            if connection_id_file == ".switch_connection_id":
                connection_id_file = f".switch_{role.lower()}_connection_id"
        else:
            agent_id = (
                os.environ.get("SWITCH_FORECAST_AGENT_ID")
                or env_dict.get("SWITCH_FORECAST_AGENT_ID")
                or os.environ.get("SWITCH_AGENT_ID")
                or env_dict.get("SWITCH_AGENT_ID")
            )
            api_key = (
                os.environ.get("SWITCH_FORECAST_API_KEY")
                or env_dict.get("SWITCH_FORECAST_API_KEY")
                or os.environ.get("SWITCH_API_KEY")
                or env_dict.get("SWITCH_API_KEY")
            )
            if role and connection_id_file == ".switch_connection_id":
                connection_id_file = f".switch_{role.lower()}_connection_id"

        if not agent_id and "agent_id" not in overrides:
            role_hint = f" for role '{role}'" if role else ""
            raise ValueError(
                f"SWITCH_AGENT_ID environment variable is required{role_hint}"
            )
        if not api_key and "api_key" not in overrides:
            role_hint = f" for role '{role}'" if role else ""
            raise ValueError(
                f"SWITCH_API_KEY environment variable is required{role_hint}"
            )

        # Persistent connection ID resolution
        conn_id = os.environ.get("SWITCH_CONNECTION_ID") or env_dict.get(
            "SWITCH_CONNECTION_ID"
        )
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

        rooms_raw = os.environ.get("SWITCH_AUTHORIZED_ROOM_IDS") or env_dict.get(
            "SWITCH_AUTHORIZED_ROOM_IDS", ""
        )
        authorized_rooms = (
            frozenset(r.strip() for r in rooms_raw.split(",") if r.strip())
            if rooms_raw
            else frozenset()
        )

        heartbeat_s = float(
            os.environ.get("SWITCH_HEARTBEAT_INTERVAL_S")
            or env_dict.get("SWITCH_HEARTBEAT_INTERVAL_S", "2.0")
        )

        client_name = (
            f"thermopilot-{role.lower()}-adapter"
            if role
            else "thermopilot-forecast-adapter"
        )

        operators_raw = os.environ.get("SWITCH_AUTHORIZED_OPERATORS") or env_dict.get(
            "SWITCH_AUTHORIZED_OPERATORS", ""
        )
        authorized_operators = (
            frozenset(op.strip() for op in operators_raw.split(",") if op.strip())
            if operators_raw
            else frozenset()
        )

        approval_timeout_s = float(
            os.environ.get("SWITCH_APPROVAL_TIMEOUT_S")
            or env_dict.get("SWITCH_APPROVAL_TIMEOUT_S", "30.0")
        )

        data: dict[str, Any] = {
            "bridge_url": bridge_url,
            "agent_id": agent_id or overrides.get("agent_id"),
            "api_key": api_key or overrides.get("api_key"),
            "authorized_room_ids": authorized_rooms,
            "authorized_operators": authorized_operators,
            "approval_timeout_s": approval_timeout_s,
            "heartbeat_interval_s": heartbeat_s,
            "client_name": client_name,
        }
        if conn_id:
            data["connection_id"] = conn_id

        data.update(overrides)
        return cls(**data)
