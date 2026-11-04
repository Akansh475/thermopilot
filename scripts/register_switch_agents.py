#!/usr/bin/env python3
"""Registration script for ThermoPilot multi-agent Switch integration.

Registers two additional Switch agents:
1. planner-agent (Recovery Planning & Synthesis)
2. workload-agent (Tenant Workload SLA Advocacy)

Assigns both agents to the target Discord room (97764de6-af0f-49b6-9685-2888bcd26629).
Safely stores their agent IDs and API keys in .env (mode 0600) idempotently without
modifying existing forecast-agent credentials.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any

DISCORD_ROOM_ID = "97764de6-af0f-49b6-9685-2888bcd26629"
SWITCH_CORE_ENV_PATH = Path(
    "/Users/alex/Library/Application Support/switchdash/local-switch-server/native/run/switch-core.env"
)
LOCAL_ENV_PATH = Path(".env")
LOCAL_ENV_TMP_PATH = Path(".env.tmp")


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse key-value pairs from an environment file."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip("\"'")
    except OSError as exc:
        print(f"Warning: Could not read {path}: {exc}", file=sys.stderr)
    return env


def get_gateway_opener(
    bridge_url: str, server_env: dict[str, str]
) -> urllib.request.OpenerDirector | None:
    """Log in to Switch Gateway as admin to query agents and assign rooms."""
    email = server_env.get("GATEWAY_ADMIN_EMAIL")
    password = server_env.get("GATEWAY_ADMIN_PASSWORD")
    if not email or not password:
        return None

    cj = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    login_url = f"{bridge_url}/gateway/auth/login"
    login_payload = json.dumps({"email": email, "password": password}).encode("utf-8")
    req = urllib.request.Request(
        login_url,
        data=login_payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with opener.open(req, timeout=10) as resp:
            if resp.status == 200:
                return opener
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: Gateway admin login failed: {exc}", file=sys.stderr)
    return None


def fetch_existing_gateway_agents(
    bridge_url: str, opener: urllib.request.OpenerDirector
) -> dict[str, dict[str, Any]]:
    """Query gateway for currently registered agents by name."""
    url = f"{bridge_url}/gateway/agents"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with opener.open(req, timeout=10) as resp:
            agents = json.loads(resp.read().decode("utf-8"))
            return {a["name"]: a for a in agents if isinstance(a, dict) and "name" in a}
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: Failed to fetch gateway agents: {exc}", file=sys.stderr)
        return {}


def register_agent(
    bridge_url: str,
    reg_token: str,
    name: str,
    description: str,
    metadata: dict[str, Any],
    tools: list[dict[str, Any]],
) -> tuple[str, str]:
    """Register a new agent via the Switch Agent Protocol POST /agents."""
    payload = {
        "name": name,
        "description": description,
        "connector_type": "custom",
        "overwrite": False,
        "integration_profile": {
            "connection_model": "always_on",
            "message_exchange": True,
            "pre_invocation_mediation": [],
            "post_invocation_mediation": [],
            "event_reporting": [],
            "command_capabilities": {
                "reset": "unsupported",
                "compact": "unsupported",
                "interrupt": "unsupported",
            },
        },
        "tools": tools,
        "models": [],
        "metadata": metadata,
    }

    req = urllib.request.Request(
        f"{bridge_url}/agents",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {reg_token}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=15) as resp:
        if resp.status not in (200, 201):
            raise RuntimeError(
                f"Registration of {name} failed with HTTP status {resp.status}"
            )
        data = json.loads(resp.read().decode("utf-8"))
        agent_id = data.get("id")
        api_key = data.get("api_key")
        if not agent_id or not api_key:
            raise RuntimeError(
                f"Registration response for {name} missing id or api_key"
            )
        return str(agent_id), str(api_key)


def assign_agent_to_room(
    bridge_url: str,
    opener: urllib.request.OpenerDirector,
    room_id: str,
    agent_id: str,
) -> bool:
    """Assign agent to target room via Gateway room service."""
    url = f"{bridge_url}/gateway/rooms/{room_id}/agents"
    body = json.dumps({"agent_ids": [agent_id]}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with opener.open(req, timeout=10) as resp:
            return resp.status in (200, 201)
    except Exception as exc:  # noqa: BLE001
        print(
            f"Warning: Failed to assign agent {agent_id} to room {room_id}: {exc}",
            file=sys.stderr,
        )
        return False


def verify_agent_room_assignment(
    bridge_url: str,
    api_key: str,
    agent_id: str,
    target_room_id: str,
) -> bool:
    """Verify that agent is assigned to target room via POST /agents/{id}/ops/list_rooms."""
    url = f"{bridge_url}/agents/{agent_id}/ops/list_rooms"
    req = urllib.request.Request(
        url,
        data=b"{}",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            result = data.get("result", [])
            assigned_ids = [r.get("room_id") for r in result if isinstance(r, dict)]
            return target_room_id in assigned_ids
    except Exception as exc:  # noqa: BLE001
        print(
            f"Warning: Failed to verify room assignment for {agent_id}: {exc}",
            file=sys.stderr,
        )
        return False


def save_env_safely(
    env_path: Path, tmp_path: Path, env_updates: dict[str, str]
) -> None:
    """Atomically update .env with mode 0600, preserving existing keys and formatting."""
    existing_lines: list[str] = []
    if env_path.exists():
        existing_lines = env_path.read_text(encoding="utf-8").splitlines()

    updated_keys: set[str] = set()
    new_lines: list[str] = []

    for line in existing_lines:
        trimmed = line.strip()
        if trimmed and not trimmed.startswith("#") and "=" in trimmed:
            k, _ = trimmed.split("=", 1)
            k = k.strip()
            if k in env_updates:
                new_lines.append(f"{k}={env_updates[k]}")
                updated_keys.add(k)
                continue
        new_lines.append(line)

    for k, v in env_updates.items():
        if k not in updated_keys:
            new_lines.append(f"{k}={v}")

    content = "\n".join(new_lines).strip() + "\n"

    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, env_path)
        os.chmod(env_path, 0o600)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def main() -> int:
    """Main registration runner."""
    print("=" * 60)
    print("ThermoPilot Switch Multi-Agent Registration (Milestone 1)")
    print("=" * 60)

    local_env = parse_env_file(LOCAL_ENV_PATH)
    server_env = parse_env_file(SWITCH_CORE_ENV_PATH)

    bridge_url = local_env.get("SWITCH_BRIDGE_URL") or "http://127.0.0.1:60824"
    reg_token = local_env.get("SWITCH_REGISTRATION_TOKEN") or server_env.get(
        "AGENT_REGISTRATION_TOKEN"
    )

    if not reg_token:
        print(
            "Error: SWITCH_REGISTRATION_TOKEN not found in .env or switch-core.env",
            file=sys.stderr,
        )
        return 1

    opener = get_gateway_opener(bridge_url, server_env)
    if not opener:
        print(
            "Error: Could not authenticate to Switch Gateway admin API", file=sys.stderr
        )
        return 1

    existing_agents = fetch_existing_gateway_agents(bridge_url, opener)

    agent_specs = [
        {
            "name": "planner-agent",
            "description": "ThermoPilot Recovery Planner Agent (synthesizes certified candidate recovery plans for thermal runaway mitigation)",
            "env_id_key": "SWITCH_PLANNER_AGENT_ID",
            "env_api_key": "SWITCH_PLANNER_API_KEY",
            "metadata": {
                "app": "thermopilot",
                "subsystem": "planner",
                "agent_role": "recovery_planner",
            },
            "tools": [
                {
                    "name": "generate_recovery_plans",
                    "description": "Generate candidate recovery plans (Plan A Conservative Shed, Plan B Balanced Cap, Plan C Migration)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "telemetry": {
                                "type": "object",
                                "description": "Current GPU telemetry",
                            },
                            "workload_bids": {
                                "type": "object",
                                "description": "Tenant workload bids and constraints",
                            },
                        },
                    },
                }
            ],
        },
        {
            "name": "workload-agent",
            "description": "ThermoPilot Workload Advocate Agent (represents competing tenant workloads and negotiates SLA-preserving power/preemption bids)",
            "env_id_key": "SWITCH_WORKLOAD_AGENT_ID",
            "env_api_key": "SWITCH_WORKLOAD_API_KEY",
            "metadata": {
                "app": "thermopilot",
                "subsystem": "workload_advocacy",
                "agent_role": "workload_owner",
            },
            "tools": [
                {
                    "name": "submit_workload_bid",
                    "description": "Submit a structured bid and proposal for tenant workloads (Alpha VIP, Beta Elastic, Gamma Analytics)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "workload_id": {"type": "string"},
                            "requested_action": {"type": "string"},
                            "importance_score": {"type": "number"},
                        },
                    },
                }
            ],
        },
    ]

    env_updates: dict[str, str] = {}
    results: list[dict[str, Any]] = []

    for spec in agent_specs:
        name = spec["name"]
        id_key = spec["env_id_key"]
        api_key_key = spec["env_api_key"]

        existing_id = local_env.get(id_key)
        existing_key = local_env.get(api_key_key)

        agent_id: str | None = None
        api_key: str | None = None

        if name in existing_agents and existing_id and existing_key:
            # Idempotent: already registered on server and credentials present in .env
            server_agent = existing_agents[name]
            if server_agent.get("id") == existing_id:
                print(
                    f"ℹ️  Agent '{name}' is already registered with matching ID in .env. Skipping registration."
                )
                agent_id = existing_id
                api_key = existing_key
            else:
                print(
                    f"Warning: Local {id_key} does not match server ID for '{name}'. Updating ID.",
                    file=sys.stderr,
                )
                agent_id = str(server_agent.get("id"))
                api_key = existing_key
        elif name in existing_agents:
            # Exists on server but not in .env; cannot recover hash-only API key without re-registration
            server_agent = existing_agents[name]
            print(f"ℹ️  Agent '{name}' exists on server ({server_agent.get('id')}).")
            if existing_key:
                agent_id = str(server_agent.get("id"))
                api_key = existing_key
            else:
                print(f"Re-registering '{name}' to obtain active API key...")
                agent_id, api_key = register_agent(
                    bridge_url=bridge_url,
                    reg_token=reg_token,
                    name=name,
                    description=spec["description"],
                    metadata=spec["metadata"],
                    tools=spec["tools"],
                )
        else:
            print(f"Registering new agent '{name}' on Switch...")
            agent_id, api_key = register_agent(
                bridge_url=bridge_url,
                reg_token=reg_token,
                name=name,
                description=spec["description"],
                metadata=spec["metadata"],
                tools=spec["tools"],
            )
            print(f"✅ Successfully registered '{name}'.")

        assert agent_id is not None
        assert api_key is not None

        env_updates[id_key] = agent_id
        env_updates[api_key_key] = api_key

        # Assign to Discord-linked room
        print(f"Assigning '{name}' to room {DISCORD_ROOM_ID}...")
        assigned = assign_agent_to_room(bridge_url, opener, DISCORD_ROOM_ID, agent_id)
        if assigned:
            print(f"✅ Assigned '{name}' to room {DISCORD_ROOM_ID}.")
        else:
            print(
                f"⚠️  Assignment call for '{name}' returned non-200, verifying status..."
            )

        verified = verify_agent_room_assignment(
            bridge_url, api_key, agent_id, DISCORD_ROOM_ID
        )
        print(
            f"Room membership verified via ops/list_rooms: {'YES' if verified else 'PENDING'}"
        )

        results.append(
            {
                "name": name,
                "agent_id": agent_id,
                "room_assigned": verified,
            }
        )

    # Persist in .env safely
    print("\nUpdating .env safely...")
    save_env_safely(LOCAL_ENV_PATH, LOCAL_ENV_TMP_PATH, env_updates)
    mode = oct(LOCAL_ENV_PATH.stat().st_mode & 0o777)
    print(f"✅ Saved credentials to {LOCAL_ENV_PATH} (file permissions: {mode}).")

    # Confirm forecast-agent is untouched and active
    forecast_id = local_env.get("SWITCH_AGENT_ID")
    forecast_key = local_env.get("SWITCH_API_KEY")
    if forecast_id and forecast_key:
        forecast_verified = verify_agent_room_assignment(
            bridge_url, forecast_key, forecast_id, DISCORD_ROOM_ID
        )
        print(
            f"✅ Verified forecast-agent ({forecast_id}) remains operational in room: {forecast_verified}"
        )

    print("\n" + "=" * 60)
    print("REGISTRATION SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"Agent Name:    {r['name']}")
        print(f"Agent ID:      {r['agent_id']}")
        print(f"Room Assigned: {r['room_assigned']} ({DISCORD_ROOM_ID})")
        print("-" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
