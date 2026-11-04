# ThermoPilot–Switch Integration Guide

> Connecting the ThermoPilot Thermal Forecast Agent to Switch and team chat channels (Discord).

---

## 1. Overview & Architecture

The **ThermoPilot Switch Adapter** (`thermopilot.switch_adapter`) connects the existing deterministic [`ForecastAgent`](file:///Users/alex/Desktop/Thermo_plot/thermopilot/agents/forecast.py) to **Agent Switch** using the official **Switch Agent Protocol**.

```
   ┌─────────────────────────────────────────────────────────┐
   │             Discord Room / Channel                     │
   └───────────────────────────▲─────────────────────────────┘
                               │
                               ▼
   ┌─────────────────────────────────────────────────────────┐
   │                    Agent Switch                         │
   └───────────────────────────▲─────────────────────────────┘
                               │  Switch Agent Protocol
                               │  (Outbound HTTP / Inbound SSE)
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│             ThermoPilot Switch Agent Adapter (Async Client)              │
│  - Persistent connection_id generation & heartbeat (every 2.0s)          │
│  - SSE event stream listener (GET /agents/{id}/events)                   │
│  - Addressed-message filter & room authorization checker                 │
│  - Direct chat reply dispatcher (POST /agents/{id}/message)              │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   ThermoPilot Perception Engine                          │
│  - ForecastAgent.observe(telemetry)                                      │
│  - Reads telemetry exclusively from TelemetrySimulator                   │
│  - Evaluates dT/dt rate, acceleration d²T/dt², and Time-to-Throttle      │
│  - STRICTLY READ-ONLY & ADVISORY (No workload mutation or hardware touch)│
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Official Wire Protocol Summary

* **Downstream Delivery:** One Server-Sent Events (SSE) stream (`GET /agents/{id}/events?connection_id=...&scope=all&filter=all`).
* **Upstream Operations:** Standard HTTP POST requests (`POST /agents/{id}/message`, `POST /agents/{id}/connection/beat`).
* **Client Heartbeat:** Sent every **2.0 seconds** to maintain connection liveness (Switch TTL is 6.0 seconds).
* **Network Simplicity:** ThermoPilot runs as an **outbound client**. No open incoming ports, public webhooks, or ngrok tunnels are required.

---

## 3. Environment Variables & Configuration

The adapter reads its configuration from the environment (or `.env` file):

| Variable | Description | Required? | Default |
| :--- | :--- | :--- | :--- |
| `SWITCH_BRIDGE_URL` | Base URL of the Switch Agent Bridge | No | `http://localhost:8000` |
| `SWITCH_AGENT_ID` | UUID identifier of the registered agent | **Yes** | — |
| `SWITCH_API_KEY` | Bearer token returned at agent registration | **Yes** | — |
| `SWITCH_AUTHORIZED_ROOM_IDS` | Comma-separated list of permitted Discord room IDs | No | All member rooms |
| `SWITCH_CONNECTION_ID` | Persistent UUID for connection continuity | No | Auto-generated & saved to `.switch_connection_id` |
| `SWITCH_HEARTBEAT_INTERVAL_S` | Interval between heartbeat pings in seconds | No | `2.0` |

> [!SECURITY]
> The adapter masks `SWITCH_API_KEY` in logs and diagnostic outputs to prevent credential leakage. Never commit `.env` or credential files to source control.

---

## 4. Setup & Registration Instructions

### Step 1: Register the Forecast Agent with Switch
Registration is performed **once and out-of-band** against the Switch Agent Bridge using the deployment registration token:

```bash
curl -X POST "$SWITCH_BRIDGE_URL/agents" \
  -H "Authorization: Bearer $AGENT_REGISTRATION_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "forecast-agent",
    "description": "ThermoPilot GPU Thermal Forecast Agent (deterministic predictive perception & Time-to-Throttle estimation)",
    "connector_type": "custom",
    "overwrite": true,
    "integration_profile": {
      "connection_model": "always_on",
      "message_exchange": true,
      "pre_invocation_mediation": [],
      "post_invocation_mediation": [],
      "event_reporting": []
    }
  }'
```

Save the returned `id` and `api_key` to your environment:
```bash
export SWITCH_AGENT_ID="<id_from_response>"
export SWITCH_API_KEY="<api_key_from_response>"
export SWITCH_BRIDGE_URL="http://localhost:8000"
```

### Step 2: Invite the Agent to your Discord Channel
In your connected Discord channel, add the agent to the room:
```text
!invite-agent @forecast-agent
```
*(Or use Discord slash command: `/invite-agent agent:forecast-agent`)*

---

## 5. Starting and Stopping the Adapter

### Starting the Adapter
Launch the standalone asynchronous adapter from the repository root:

```bash
python -m thermopilot.switch_adapter
```

**Expected Startup Output:**
```text
2026-11-01 14:30:00 [INFO] thermopilot.switch_adapter: Connecting ForecastSwitchAdapter (agent_id: ..., connection_id: ...)
2026-11-01 14:30:00 [INFO] thermopilot.switch_adapter.adapter: Starting Switch heartbeat worker (interval: 2.0s)
```

### Stopping the Adapter
Send a standard interrupt (`Ctrl+C` / `SIGINT` or `SIGTERM`). The adapter cleanly terminates the heartbeat task and closes the SSE stream connection:
```text
2026-11-01 14:31:00 [INFO] thermopilot.switch_adapter: Shutdown signal received. Stopping adapter...
2026-11-01 14:31:00 [INFO] thermopilot.switch_adapter.adapter: ForecastSwitchAdapter stopped cleanly
```

---

## 6. Supported Discord Chat Queries

Users in authorized Discord rooms can mention the agent (`@forecast-agent` or `@ForecastAgent`):

* **`@ForecastAgent status`**
  Returns instantaneous GPU telemetry (core temperature, junction temperature, power draw, ambient intake, and categorical risk status).
* **`@ForecastAgent forecast`**
  Returns first-order rate ($\frac{dT}{dt}$ in °C/s), second-order thermal acceleration ($\frac{d^2T}{dt^2}$ in °C/s²), trajectory direction, and Time-to-Throttle ($t_{TTT}$).
* **`@ForecastAgent time-to-trip` (or `ttt`)**
  Returns the estimated countdown seconds before reaching the critical $85.0^\circ\text{C}$ hardware trip limit.
* **`@ForecastAgent help`**
  Displays the command reference.

---

## 7. Safety & Simulation Disclosures

* **Purely Advisory:** The Switch adapter operates strictly as a read-only perception interface.
* **Non-Bypassable SafetyGate:** Any actuation commands (e.g. `approve`, `execute`, `throttle`) sent via chat are explicitly rejected with an advisory notice that workload recovery plans require deterministic [`SafetyGate`](file:///Users/alex/Desktop/Thermo_plot/thermopilot/safety/gate.py) verification at the Compute Desk.
* **Simulated Telemetry Only:** Telemetry is generated via [`TelemetrySimulator`](file:///Users/alex/Desktop/Thermo_plot/thermopilot/simulator/telemetry.py) (Newtonian cooling model, forward Euler integration). No physical hardware is touched.
