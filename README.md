# ThermoPilot Compute Desk

> **Turning GPU thermal incidents into cost-aware, human-approved business decisions.**
> Built for the **Flip the Switch Hackathon 2026** — **100% Complete & Verified (277 Tests Passing)**.

---

## 1. Executive Summary & Problem Statement

Modern AI data centers and GPU clusters operate under intense power and thermal constraints. As multi-tenant clusters run high-density training and inference workloads, thermal spikes are inevitable due to ambient temperature shifts, cooling zone imbalances, or unexpected workload superposition.

Today, cluster management handles thermal incidents through blunt, automated mechanisms:
- **Blind Hardware Throttling:** Clock speeds are uniformly slashed, degrading throughput and causing silent SLA breaches across both mission-critical and low-priority jobs.
- **Abrupt Job Evacuation / Killing:** High-priority batch runs or distributed training jobs are killed or preempted without understanding downstream contractual penalties or recovery costs.
- **Disconnection from Business Priorities:** Infrastructure controllers have zero visibility into customer SLAs, revenue impact, or workload deadlines. Engineering teams are alerted only *after* jobs fail or hardware overheats.

### The Solution: ThermoPilot Compute Desk
**ThermoPilot Compute Desk** is an autonomous, multi-agent workload optimization and thermal governance system. When thermal anomalies or overheating trajectories are detected, ThermoPilot coordinates competing workload-owner agents, formulates trade-off recovery plans with an LLM-powered planner (Llama), validates every plan through a deterministic, non-negotiable safety gate, and presents human operators with cost-aware decision cards via **Switch** and team chat.

```
[ GPU Cluster Telemetry (Simulated) ] 
                 │
                 ▼
┌─────────────────────────────────┐
│     Thermal Forecast Agent      │ ──► Anticipates thermal runaway at 81.5°C before 85°C hardware trip
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│     Workload-Owner Agents       │ ──► Agent Alpha (VIP SLA) vs. Beta (Training) vs. Gamma (Analytics)
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│     Llama Recovery Planner      │ ──► Generates 3 feasible plans (A, B, C) + evaluates unsafe candidate (D)
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│   Deterministic SafetyGate      │ ──► Hard physical & SLA checks (Certifies A, B, C; Blocks D)
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ Switch Human-in-the-Loop Desk   │ ──► Cost-aware decision cards presented to on-call operator
│                                 │     (Strict Policy: Timeout NEVER mutates workloads automatically)
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ Simulation Executor & Verifier  │ ──► Revalidates safety, applies power caps/pauses, verifies 75°C target
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ Cryptographic Audit Ledger      │ ──► Append-only SHA-256 hash chain verifying complete lifecycle integrity
└─────────────────────────────────┘
```

---

## 2. Simulation Boundary Disclosure

> [!IMPORTANT]
> **SIMULATION-ONLY ENVIRONMENT:**
> ThermoPilot Compute Desk operates strictly within a discrete-time simulation environment. All telemetry, core temperatures, junction hotspot calculations, and workload actuations are simulated mathematical models (Euler forward integration, Newton cooling $\tau=15.0\text{s}$, $R_{th}=0.14^\circ\text{C}/\text{W}$).
> **No physical GPUs, NVML drivers, DCGM interfaces, host power rails, or cloud infrastructure are touched or altered.**
> The demonstration does not claim real-world energy savings or physical hardware performance.

---

## 3. Quick Start & Demonstration Commands

Requires Python 3.10+ (tested on Python 3.14):

```bash
# Clone the repository
cd Thermo_plot

# Install dependencies
pip install pydantic
```

### Demonstration Modes

| Mode | Command Line | Description |
| :--- | :--- | :--- |
| **Switch Multi-Agent Runner** | `python -m thermopilot.switch_adapter` | Launches all 3 Switch agents (`forecast`, `workload`, `planner`) with live WebSocket connections to Discord. |
| **Automated 60s Demo** | `python -m thermopilot.cli --auto` | Runs complete 8-stage hackathon presentation in terminal. |
| **Interactive Control Desk** | `python -m thermopilot.cli --interactive` | Interactive REPL with operator hotkeys (`[1]`, `[2]`, `[3]`, `[R]`, `[T]`). |
| **Web Control Room Dashboard** | `python -m thermopilot.cli --web --port 8080` | Launches local browser GUI at `http://localhost:8080`. |
| **SafetyGate Block Proof** | `python -m thermopilot.cli --attempt-plan-d` | Demonstrates that Plan D is strictly blocked from approval/execution. |
| **Operator Rejection Scenario** | `python -m thermopilot.cli --reject` | Proves operator rejection leaves workloads completely unmutated. |
| **Operator Timeout Escalation** | `python -m thermopilot.cli --timeout` | Proves approval timeout escalates without modifying workloads. |
| **Reset Simulator State** | `python -m thermopilot.cli --reset` | Cleans audit log and restores 81.5°C initial conditions. |

### Switch Multi-Agent & Discord Live Demo Setup

1. **Keep Switch Console Open:** The **Switch Console** desktop application must remain open and running (`http://127.0.0.1:60824`) to maintain the local bridge and WebSocket routing.
2. **Room Assignment:** Ensure all three agents (`forecast-agent`, `workload-agent`, `planner-agent`) are registered and connected to the target Discord room (e.g. `Akansh Server: general`).
3. **Start the Multi-Agent Switch Runner:**
   ```bash
   python -m thermopilot.switch_adapter
   ```
   Maintains independent WebSocket connections, credential isolation, heartbeat tracking, and message dispatch for all three agents.
4. **Simulation Boundary Reminder:** All GPU telemetry, runaway trajectories, and workload power-cap actuations are discrete-time mathematical simulations with zero physical hardware access.

### Demonstrating the Four Core Scenarios

#### Scenario A: Successful Recovery Lifecycle
1. Trigger the simulated anomaly in Discord:
   ```text
   @forecast-agent simulate-anomaly
   ```
2. The multi-agent workflow initiates automatically:
   - `forecast-agent` posts a thermal runaway perception alert (`81.5°C`, Time-to-Trip `23.3s`, marked `[SIMULATED TELEMETRY]`).
   - `workload-agent` calculates urgency and submits competing tenant bids (VIP P0 non-preemption vs. elastic P2 training).
   - `planner-agent` evaluates candidates via SafetyGate, certifies Plan B, blocks Plan D, and publishes a Decision Card.
3. Authorize the certified plan within the **30-second governance window** using the **actual incident ID** from the decision card:
   ```text
   @planner-agent approve <incident_id> PLAN-B
   ```
4. Outcome: SafetyGate revalidates immediately pre-execution, `SimulationExecutor` modulates simulated workloads, and `VerificationAgent` confirms cooling to $\le 75.0^\circ\text{C}$ safe envelope with zero SLA violations and publishes the report with an append-only SHA-256 cryptographic `audit_hash`.

#### Scenario B: Unsafe Plan Blocked by SafetyGate
1. After an anomaly alert, attempt to approve the uncertified/unsafe Plan D:
   ```text
   @planner-agent approve <incident_id> PLAN-D
   ```
2. Outcome: SafetyGate strictly blocks execution (`Execution Denied by SafetyGate`). Zero workloads are mutated, and the incident transitions to `REVALIDATION_FAILED`.  
   *(CLI direct proof: `python -m thermopilot.cli --attempt-plan-d`)*

#### Scenario C: Explicit Human Operator Rejection
1. Reject the recovery proposals:
   ```text
   @planner-agent reject <incident_id>
   ```
2. Outcome: Recovery plans are cancelled. Workloads remain in their current state, and the incident transitions to `REJECTED`.  
   *(CLI direct proof: `python -m thermopilot.cli --reject`)*

#### Scenario D: Approval Timeout Escalation
1. Trigger an anomaly with `@forecast-agent simulate-anomaly` and allow the **30-second governance window** to elapse without submitting an approval.
2. If an approval is submitted after expiration:
   ```text
   @planner-agent approve <incident_id> PLAN-B
   ```
3. Outcome: Fail-safe policy triggers automatic timeout escalation (`TIMEOUT_ESCALATED`). Workload mutations are strictly blocked to prevent uncoordinated execution.  
   *(CLI direct proof: `python -m thermopilot.cli --timeout`)*

### Running Quality Checks & Tests

```bash
# Run complete test suite (277 tests)
pytest

# Run Ruff linter
ruff check .
```

---

## 4. Key Subsystems & Architecture

1. **Discrete-Time Telemetry Simulator (`thermopilot/simulator/`):**
   Forward Euler integration with Newton cooling, 400W total node budget, configurable stochastic noise, $2.5\times$ demo time-scaling, and intake ambient heating anomalies ($22^\circ\text{C} \to 32^\circ\text{C}$).
2. **Thermal Forecast Agent (`thermopilot/agents/forecast.py`):**
   Rolling polynomial regression detecting runaway acceleration ($\frac{d^2T}{dt^2} > 0$) and computing Time-to-Trip ($t_{TTT}$) toward the $85.0^\circ\text{C}$ hardware trip limit.
3. **Workload-Owner Agents (`thermopilot/agents/workload_owner.py`):**
   Autonomous tenant agents advocating for competing workloads (Alpha P0 VIP trading, Gamma P1 production analytics, Beta P2 diffusion training) calculating contractual urgency scores and mitigation proposals.
4. **Recovery Planner & Llama Integration (`thermopilot/planner/`):**
   Candidate plan synthesis (Llama prompt pipeline with instant deterministic fallback). Generates Plans A, B, C and evaluates baseline Plan D.
5. **Deterministic SafetyGate (`thermopilot/safety/gate.py`):**
   Non-negotiable physical and SLA boundary enforcement ($T \le 80.0^\circ\text{C}$, $P \le 400\text{W}$, P0 non-preemption, zero P0 SLA penalties, min power floors). Certifies Plans A, B, C; permanently blocks Plan D.
6. **Switch HITL Approval Desk (`thermopilot/hitl/switch.py`):**
   Decision card formatting, executable plan dispatch, and operator sign-off validation with strict fail-safe timeout escalation (timeout never mutates workloads).
7. **Simulation Executor & Verifier (`thermopilot/executor/`):**
   Atomic simulator actuation (`POWER_CAP`, `PAUSE`, `MIGRATE`), pre-execution SafetyGate revalidation, and 30-tick post-execution thermal decay verification ($\le 75.0^\circ\text{C}$ recovery target, cooling rate $\ge 0.5^\circ\text{C}/\text{s}$, zero SLA violations).
8. **Append-Only Cryptographic Audit Ledger & State Machine (`thermopilot/audit/`):**
   SHA-256 cryptographic hash-chain ledger binding every lifecycle event (`prev_hash`), detecting tampering, and enforcing deterministic 12-state incident progression.
9. **Compute Desk UI & Web Dashboard (`thermopilot/ui/`):**
   Rich ANSI terminal control desk and zero-dependency local web dashboard (`http://localhost:8080`) with real-time SVG thermal trajectory charts and audit trail viewers.

---

## 5. Thermal Threshold Hierarchy

| Threshold | Value | Meaning & Responsible Component |
| :--- | :--- | :--- |
| **$T_{base}$ (Steady State)** | $70.0^\circ\text{C} - 72.0^\circ\text{C}$ | Normal baseline operating temperature under dual-job load. |
| **$T_{safe\_target}$ (Recovery Target)** | $\le 75.0^\circ\text{C}$ | Target threshold where an active thermal incident is verified as resolved. |
| **$T_{ceiling}$ (Safety Gate Limit)** | $\le 80.0^\circ\text{C}$ | Hard physical constraint: Safety Gate rejects any plan with projected steady-state $>80.0^\circ\text{C}$. |
| **$T_{warn}$ (Incident Trigger)** | $81.5^\circ\text{C}$ | Forecast Agent trigger threshold upon positive thermal acceleration ($\frac{d^2T}{dt^2} > 0$). |
| **$T_{crit}$ (Hardware Trip)** | $85.0^\circ\text{C}$ | Hardware thermal throttle / emergency shutdown trip point. Intercepted early by ThermoPilot. |
| **$T_{runaway}$ (Projected Peak)** | $88.0^\circ\text{C}$ | Projected unmitigated temperature peak if zero intervention occurs. |

---

## 6. Repository Layout

```
Thermo_plot/
├── README.md                      # Project overview, architecture, CLI guide & test instructions
├── pyproject.toml                 # Packaging metadata & dependency specifications
├── docs/
│   ├── architecture.md            # Detailed system architecture, data models & safety boundaries
│   ├── demo-plan.md               # 60-second hackathon demonstration plan
│   └── demo-script.md             # Timed narration script, operator cues, screen states & fallbacks
├── thermopilot/
│   ├── __init__.py                # Package initialization
│   ├── cli.py                     # Top-level CLI command entrypoint
│   ├── schemas.py                 # Immutable Pydantic data models & state enumerations
│   ├── simulator/                 # Discrete-time thermal telemetry physics simulator
│   ├── agents/                    # Forecast perception agent & workload-owner negotiation agents
│   ├── planner/                   # Recovery planner (Llama prompt pipeline + deterministic fallback)
│   ├── safety/                    # Deterministic SafetyGate rule evaluator
│   ├── hitl/                      # Switch HITL approval desk & decision card dispatch
│   ├── switch_adapter/            # Multi-agent Switch runtime & WebSocket protocol adapters
│   ├── executor/                  # Simulation plan executor & post-execution verification agent
│   ├── audit/                     # Cryptographic append-only JSONL audit ledger & state machine
│   └── ui/                        # Compute Desk ANSI terminal interface & local web control room
└── tests/                         # Complete pytest test suite (277 tests, 100% passing)
```

---

## 7. Hackathon Evaluation Rubric Compliance

- **Thermal Safety:** Critical hardware trip ($85.0^\circ\text{C}$) never reached. Target safe recovery ($\le 75.0^\circ\text{C}$) reached in 3 ticks ($7.5\text{s}$ simulated).
- **Business SLA Protection:** VIP financial job (Alpha P0) completed with zero SLA violations and zero power floor breaches.
- **Human Governance:** Switch decision cards present clear economic trade-offs. Timeout escalation preserves workloads without unauthorized mutations.
- **Mathematical Safety Gate:** Deterministic gate certified Plans A, B, and C while completely blocking naive Plan D.
- **Auditability:** Complete incident lifecycle persisted in append-only JSONL with verifiable SHA-256 cryptographic chain.
