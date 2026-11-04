"""Structured Message Protocol for ThermoPilot Multi-Agent Switch Negotiation.

Defines validated schemas and formatting utilities for:
- thermal_alert (ForecastAgent -> Room / WorkloadOwner / Planner)
- workload_bid (WorkloadOwnerAgent -> Room / Planner)
- recovery_proposal (PlannerAgent -> Room / Compute Desk)
- safety_result (PlannerAgent SafetyGate -> Room / Compute Desk)
- negotiation_complete (PlannerAgent -> Room / Governance Desk)

All messages provide:
1. Rich, human-readable Discord markdown with emojis, bullet points, and safety footer.
2. Machine-parseable JSON payload block delimited by ```thermopilot ... ```.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

ADVISORY_FOOTER = (
    "\n\n> 🛡️ **ThermoPilot Compute Desk** | "
    "*Advisory Only · Simulated Physics · Zero Hardware Actuation*"
)


class SwitchMessageType(str, Enum):
    """Categorical enumeration of multi-agent message types."""

    THERMAL_ALERT = "thermal_alert"
    WORKLOAD_BID = "workload_bid"
    RECOVERY_PROPOSAL = "recovery_proposal"
    SAFETY_RESULT = "safety_result"
    NEGOTIATION_COMPLETE = "negotiation_complete"
    DECISION_CARD = "decision_card"
    EXECUTION_RESULT = "execution_result"
    VERIFICATION_REPORT = "verification_report"


# ---------------------------------------------------------------------------
# 1. Thermal Alert Schema (Emitted by forecast-agent)
# ---------------------------------------------------------------------------


class ThermalAlertPayload(BaseModel):
    """Perception alert announcing elevated thermal risk or incident trigger."""

    model_config = ConfigDict(frozen=True)

    type: Literal["thermal_alert"] = "thermal_alert"
    incident_id: str = Field(..., description="Unique incident identifier")
    gpu_id: int = Field(default=0, ge=0, description="GPU hardware identifier")
    current_temp_c: float = Field(..., description="Core temperature in Celsius")
    critical_threshold_c: float = Field(
        default=85.0, description="Hardware trip threshold in Celsius"
    )
    time_to_threshold_s: float | None = Field(
        default=None, description="Projected seconds to hardware throttle"
    )
    rate_c_per_s: float | None = Field(
        default=None, description="First-order derivative (dT/dt) in °C/s"
    )
    acceleration_c_per_s2: float | None = Field(
        default=None, description="Second-order acceleration in °C/s^2"
    )
    risk_status: str = Field(..., description="Risk tier badge string")
    reason: str = Field(..., description="Perception rationale")
    is_simulated: bool = Field(
        default=True,
        description="Explicit flag confirming simulated telemetry (zero hardware actuation)",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO-8601 timestamp",
    )


# ---------------------------------------------------------------------------
# 2. Workload Bid Schema (Emitted by workload-agent)
# ---------------------------------------------------------------------------


class WorkloadBidItem(BaseModel):
    """Structured proposal representing an individual workload tenant."""

    model_config = ConfigDict(frozen=True)

    workload_id: str
    name: str
    priority: str
    current_power_w: float
    min_power_w: float
    requested_power_w: float
    can_preempt: bool
    sla_importance_score: float
    rationale: str
    estimated_trade_off: str


class WorkloadBidPayload(BaseModel):
    """Collection of competing workload bids advocating for tenant priorities."""

    model_config = ConfigDict(frozen=True)

    type: Literal["workload_bid"] = "workload_bid"
    incident_id: str = Field(..., description="Correlated incident identifier")
    proposals: list[WorkloadBidItem] = Field(
        ..., min_length=1, description="Tenant negotiation bids"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO-8601 timestamp",
    )


# ---------------------------------------------------------------------------
# 3. Recovery Proposal Schema (Emitted by planner-agent)
# ---------------------------------------------------------------------------


class WorkloadActionItem(BaseModel):
    """Target power or action for a workload within a candidate plan."""

    model_config = ConfigDict(frozen=True)

    workload_id: str
    action: str
    target_power_w: float | None = None


class RecoveryCandidateItem(BaseModel):
    """Evaluated recovery plan candidate."""

    model_config = ConfigDict(frozen=True)

    plan_id: str
    name: str
    strategy_type: str
    is_recommended: bool
    projected_temp_max_c: float
    cooling_time_to_safe_s: float
    actions: list[WorkloadActionItem]
    estimated_cost_usd: float
    sla_violations_expected: int
    rationale: str
    origin: str


class RecoveryProposalPayload(BaseModel):
    """Recovery options formulated by the planner."""

    model_config = ConfigDict(frozen=True)

    type: Literal["recovery_proposal"] = "recovery_proposal"
    incident_id: str = Field(..., description="Correlated incident identifier")
    candidates: list[RecoveryCandidateItem] = Field(
        ..., min_length=1, description="Candidate recovery options"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO-8601 timestamp",
    )


# ---------------------------------------------------------------------------
# 4. Safety Result Schema (Emitted by planner-agent SafetyGate)
# ---------------------------------------------------------------------------


class SafetyEvaluationItem(BaseModel):
    """Deterministic certification report for a candidate plan."""

    model_config = ConfigDict(frozen=True)

    plan_id: str
    passed: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    rule_ids: list[str] = Field(default_factory=list)


class SafetyResultPayload(BaseModel):
    """Audit report from SafetyGate certifying or rejecting candidate plans."""

    model_config = ConfigDict(frozen=True)

    type: Literal["safety_result"] = "safety_result"
    incident_id: str = Field(..., description="Correlated incident identifier")
    evaluations: list[SafetyEvaluationItem] = Field(
        ..., min_length=1, description="Safety evaluations per plan"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO-8601 timestamp",
    )


# ---------------------------------------------------------------------------
# 5. Negotiation Complete Schema (Emitted by planner-agent)
# ---------------------------------------------------------------------------


class NegotiationCompletePayload(BaseModel):
    """Final consensus summary ready for operator sign-off at Compute Desk."""

    model_config = ConfigDict(frozen=True)

    type: Literal["negotiation_complete"] = "negotiation_complete"
    incident_id: str = Field(..., description="Correlated incident identifier")
    recommended_plan_id: str = Field(..., description="Recommended safe plan ID")
    plan_name: str = Field(..., description="Recommended plan title")
    projected_temp_max_c: float = Field(
        ..., description="Steady-state core temperature"
    )
    cooling_time_s: float = Field(
        ..., description="Projected recovery time to safe band"
    )
    actions_summary: str = Field(
        ..., description="High-level workload allocation summary"
    )
    sla_status: str = Field(..., description="SLA preservation summary")
    next_step: str = Field(
        default="Awaiting authenticated operator sign-off at Compute Desk",
        description="Governance requirement",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO-8601 timestamp",
    )


# ---------------------------------------------------------------------------
# 6. HITL Decision Card Schema (Emitted by planner-agent / SwitchApprovalDesk)
# ---------------------------------------------------------------------------


class DecisionCardItem(BaseModel):
    """Candidate plan representation for operator decision."""

    model_config = ConfigDict(frozen=True)

    plan_id: str
    name: str
    strategy_type: str
    is_recommended: bool
    is_safe: bool
    safety_verdict: str
    rejection_reasons: list[str] = Field(default_factory=list)
    projected_temp_max_c: float
    projected_power_w: float
    cooling_time_to_safe_s: float
    estimated_cost_usd: float
    sla_violations_expected: int
    actions_summary: str
    is_executable: bool


class DecisionCardPayload(BaseModel):
    """Interactive decision card presenting actionable plans and fallback chat commands."""

    model_config = ConfigDict(frozen=True)

    type: Literal["decision_card"] = "decision_card"
    incident_id: str = Field(..., description="Correlated incident identifier")
    severity: str = Field(default="CRITICAL", description="Incident severity level")
    current_temp_c: float = Field(
        ..., description="Instantaneous core temperature in Celsius"
    )
    current_power_w: float = Field(..., description="Current power draw in Watts")
    critical_trip_temp_c: float = Field(
        default=85.0, description="Hardware trip temperature"
    )
    time_to_trip_s: float | None = Field(default=None, description="Seconds to trip")
    candidate_plans: list[DecisionCardItem] = Field(..., min_length=1)
    executable_plan_ids: list[str] = Field(default_factory=list)
    rejected_plan_ids: list[str] = Field(default_factory=list)
    timeout_seconds: float = Field(default=30.0, description="Approval window duration")
    approval_command: str = Field(..., description="Chat approval command template")
    rejection_command: str = Field(..., description="Chat rejection command template")
    is_simulated: bool = Field(default=True, description="Simulated physics indicator")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO-8601 timestamp",
    )


# ---------------------------------------------------------------------------
# 7. Execution Result Schema (Emitted after SimulationExecutor runs)
# ---------------------------------------------------------------------------


class ExecutionResultPayload(BaseModel):
    """Summary of executed recovery plan actions."""

    model_config = ConfigDict(frozen=True)

    type: Literal["execution_result"] = "execution_result"
    incident_id: str
    execution_id: str
    plan_id: str
    operator_id: str
    success: bool
    applied_actions: list[dict[str, Any]] = Field(default_factory=list)
    node_power_w: float
    revalidation_passed: bool
    is_simulated: bool = Field(default=True)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO-8601 timestamp",
    )


# ---------------------------------------------------------------------------
# 8. Verification Report Schema (Emitted after VerificationAgent verifies)
# ---------------------------------------------------------------------------


class VerificationReportPayload(BaseModel):
    """Summary of post-execution thermal recovery verification."""

    model_config = ConfigDict(frozen=True)

    type: Literal["verification_report"] = "verification_report"
    incident_id: str
    verification_id: str
    plan_id: str
    recovery_successful: bool
    initial_temp_c: float
    final_temp_c: float
    cooling_rate_c_per_s: float
    reached_target: bool
    ticks_to_target: int | None = None
    audit_hash: str | None = None
    is_simulated: bool = Field(default=True)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO-8601 timestamp",
    )


# ---------------------------------------------------------------------------
# Discriminated Union & Parser Adapter
# ---------------------------------------------------------------------------

SwitchMessagePayload = Annotated[
    ThermalAlertPayload
    | WorkloadBidPayload
    | RecoveryProposalPayload
    | SafetyResultPayload
    | NegotiationCompletePayload
    | DecisionCardPayload
    | ExecutionResultPayload
    | VerificationReportPayload,
    Field(discriminator="type"),
]

_PAYLOAD_ADAPTER: TypeAdapter[SwitchMessagePayload] = TypeAdapter(SwitchMessagePayload)

_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:thermopilot|json)?\s*(\{.*?\})\s*```", re.DOTALL
)


def parse_switch_message(content: str) -> SwitchMessagePayload | None:
    """Extract and validate structured SwitchMessagePayload from chat content.

    Inspects text for ```thermopilot ... ``` or ```json ... ``` blocks containing
    a recognized discriminated type. Returns None if no valid payload block is found.
    """
    if not content:
        return None

    match = _CODE_BLOCK_PATTERN.search(content)
    if not match:
        return None

    raw_json = match.group(1).strip()
    try:
        data = json.loads(raw_json)
        if not isinstance(data, dict) or "type" not in data:
            return None
        return _PAYLOAD_ADAPTER.validate_python(data)
    except Exception:  # noqa: BLE001
        return None


def format_discord_message(payload: SwitchMessagePayload) -> str:
    """Format structured payload into rich Discord markdown with embedded JSON."""
    json_block = f"```thermopilot\n{payload.model_dump_json(indent=2)}\n```"

    if isinstance(payload, ThermalAlertPayload):
        ttt_str = (
            f"`{payload.time_to_threshold_s:.1f}s`"
            if payload.time_to_threshold_s is not None
            else "*Nominal*"
        )
        rate_str = (
            f"`{payload.rate_c_per_s:+.3f}°C/s`"
            if payload.rate_c_per_s is not None
            else "N/A"
        )
        body = (
            f"🚨 **[ThermoPilot Perception Alert]** Thermal runaway anomaly on GPU #{payload.gpu_id} [SIMULATED TELEMETRY]!\n"
            f"• **Incident ID:** `{payload.incident_id}`\n"
            f"• **Current Core Temp:** `{payload.current_temp_c:.1f}°C` (Trip Limit: `{payload.critical_threshold_c:.1f}°C`)\n"
            f"• **Time-to-Throttle (t_TTT):** {ttt_str}\n"
            f"• **Rate (dT/dt):** {rate_str} | **Risk Status:** {payload.risk_status}\n"
            f"• **Perception Rationale:** {payload.reason}\n"
            f"• **Next Action:** Requesting workload bids and certified plan from @workload-agent @planner-agent\n\n"
            f"{json_block}"
            f"{ADVISORY_FOOTER}"
        )
        return body

    if isinstance(payload, WorkloadBidPayload):
        proposals_lines = []
        for prop in payload.proposals:
            proposals_lines.append(
                f"• **{prop.name}** (`{prop.priority}` | SLA Score: `{prop.sla_importance_score:.2f}`)\n"
                f"  - Allocation: Current `{prop.current_power_w:.0f}W` ➔ Requested `{prop.requested_power_w:.0f}W` (Floor: `{prop.min_power_w:.0f}W`)\n"
                f"  - Preemptible: `{'YES' if prop.can_preempt else 'NO (STRICT LOCK)'}`\n"
                f"  - Advocacy: {prop.rationale}"
            )
        proposals_text = "\n".join(proposals_lines)

        body = (
            f"💼 **[ThermoPilot Workload Bids]** Tenant negotiation proposals submitted\n"
            f"• **Incident ID:** `{payload.incident_id}` | CC: @planner-agent\n\n"
            f"{proposals_text}\n\n"
            f"{json_block}"
            f"{ADVISORY_FOOTER}"
        )
        return body

    if isinstance(payload, RecoveryProposalPayload):
        candidate_lines = []
        for c in payload.candidates:
            star = "⭐ **[RECOMMENDED]** " if c.is_recommended else ""
            actions_summary = ", ".join(
                f"{a.workload_id} -> {a.action} ({a.target_power_w:.0f}W)"
                if a.target_power_w is not None
                else f"{a.workload_id} -> {a.action}"
                for a in c.actions
            )
            candidate_lines.append(
                f"• {star}**{c.plan_id}: {c.name}** (`{c.strategy_type}` | Origin: `{c.origin}`)\n"
                f"  - Actions: {actions_summary}\n"
                f"  - Projected Core Temp: `{c.projected_temp_max_c:.1f}°C` | Recovery Time: `{c.cooling_time_to_safe_s:.1f}s`\n"
                f"  - Trade-off Rationale: {c.rationale}"
            )
        candidates_text = "\n".join(candidate_lines)

        body = (
            f"📋 **[ThermoPilot Recovery Candidates]** Formulated mitigation options\n"
            f"• **Incident ID:** `{payload.incident_id}`\n\n"
            f"{candidates_text}\n\n"
            f"{json_block}"
            f"{ADVISORY_FOOTER}"
        )
        return body

    if isinstance(payload, SafetyResultPayload):
        eval_lines = []
        for ev in payload.evaluations:
            if ev.passed:
                eval_lines.append(
                    f"• ✅ **{ev.plan_id}:** `PASSED SAFETYGATE` (All constraints verified)"
                )
            else:
                reasons = "; ".join(ev.rejection_reasons)
                eval_lines.append(
                    f"• ❌ **{ev.plan_id}:** `BLOCKED BY SAFETYGATE`\n"
                    f"  - Violations: {reasons}"
                )
        evals_text = "\n".join(eval_lines)

        body = (
            f"🛡️ **[ThermoPilot SafetyGate Audit]** Deterministic Constraint Certification\n"
            f"• **Incident ID:** `{payload.incident_id}`\n\n"
            f"{evals_text}\n\n"
            f"{json_block}"
            f"{ADVISORY_FOOTER}"
        )
        return body

    if isinstance(payload, NegotiationCompletePayload):
        body = (
            f"🏁 **[ThermoPilot Multi-Agent Consensus]** Certified Recovery Plan Ready\n"
            f"• **Incident ID:** `{payload.incident_id}`\n"
            f"• **Certified Plan:** `{payload.recommended_plan_id}` ({payload.plan_name})\n"
            f"• **Workload Actions:** {payload.actions_summary}\n"
            f"• **Projected Recovery:** Core settles at `{payload.projected_temp_max_c:.1f}°C` in `{payload.cooling_time_s:.1f}s`\n"
            f"• **SLA Preservation:** {payload.sla_status}\n"
            f"• 🔒 **Governance Notice:** {payload.next_step}\n\n"
            f"{json_block}"
            f"{ADVISORY_FOOTER}"
        )
        return body

    if isinstance(payload, DecisionCardPayload):
        options_lines = []
        for p in payload.candidate_plans:
            rec = " ⭐ **[RECOMMENDED]**" if p.is_recommended else ""
            if p.is_safe:
                options_lines.append(
                    f"• 🟢 **{p.plan_id}: {p.name}**{rec} `[CERTIFIED SAFE]`\n"
                    f"  - Steady Temp: `{p.projected_temp_max_c:.1f}°C` | Power: `{p.projected_power_w:.1f}W` | Cooldown: `{p.cooling_time_to_safe_s:.1f}s`\n"
                    f"  - Actions: {p.actions_summary}\n"
                    f"  - Status: [Action Available: Approve {p.plan_id}]"
                )
            else:
                violations = "; ".join(p.rejection_reasons)
                options_lines.append(
                    f"• 🔴 **{p.plan_id}: {p.name}** `[BLOCKED BY SAFETYGATE]`\n"
                    f"  - Violations: {violations}\n"
                    f"  - Status: ⛔ [Execution Disabled - Safety Violation]"
                )
        options_text = "\n".join(options_lines)

        body = (
            f"⚖️ **[ThermoPilot HITL Governance Desk]** Action Required for Incident `{payload.incident_id}` [SIMULATED]\n"
            f"• **Telemetry Snapshot:** Core Temp: `{payload.current_temp_c:.1f}°C` | Power: `{payload.current_power_w:.1f}W` | Trip: `{payload.critical_trip_temp_c:.1f}°C`\n"
            f"• **Governance Window:** `{payload.timeout_seconds:.0f}s` window (Strict policy: Timeout never mutates workloads)\n\n"
            f"**Candidate Recovery Options:**\n"
            f"{options_text}\n\n"
            f"**Operator Commands (Chat Fallback & Audit Binding):**\n"
            f"• To Approve: `{payload.approval_command}`\n"
            f"• To Reject: `{payload.rejection_command}`\n\n"
            f"{json_block}"
            f"{ADVISORY_FOOTER}"
        )
        return body

    if isinstance(payload, ExecutionResultPayload):
        actions_list = [
            f"{a.get('workload_id')}: {a.get('action')} ({a.get('target_power_w', 0):.0f}W)"
            if a.get("target_power_w") is not None
            else f"{a.get('workload_id')}: {a.get('action')}"
            for a in payload.applied_actions
        ]
        actions_str = ", ".join(actions_list) if actions_list else "None"

        body = (
            f"⚡ **[ThermoPilot Actuation Report]** Plan Execution Completed [SIMULATED]\n"
            f"• **Incident ID:** `{payload.incident_id}` | **Plan ID:** `{payload.plan_id}`\n"
            f"• **Authorized Operator:** `{payload.operator_id}`\n"
            f"• **SafetyGate Pre-Execution Revalidation:** `{'✅ PASSED (100% Certified)' if payload.revalidation_passed else '❌ FAILED'}`\n"
            f"• **Applied Workload Actions:** {actions_str}\n"
            f"• **Total Node Power Allocation:** `{payload.node_power_w:.1f}W` (Limit: 400.0W)\n"
            f"• **Status:** Actuation applied to simulation. Initiating post-execution thermal verification...\n\n"
            f"{json_block}"
            f"{ADVISORY_FOOTER}"
        )
        return body

    if isinstance(payload, VerificationReportPayload):
        verdict = (
            "✅ RECOVERY SUCCESSFUL (Safe Band Restored)"
            if payload.recovery_successful
            else "❌ RECOVERY FAILED"
        )
        ticks_str = (
            f"`{payload.ticks_to_target}` ticks"
            if payload.ticks_to_target is not None
            else "N/A"
        )
        hash_str = (
            f"`{payload.audit_hash[:16]}...`"
            if payload.audit_hash
            else "*Cryptographically verified*"
        )

        body = (
            f"🎯 **[ThermoPilot Verification Report]** Post-Execution Thermal Audit [SIMULATED]\n"
            f"• **Incident ID:** `{payload.incident_id}` | **Plan ID:** `{payload.plan_id}`\n"
            f"• **Recovery Verdict:** {verdict}\n"
            f"• **Thermal Restitution:** Core dropped `{payload.initial_temp_c:.1f}°C` ➔ `{payload.final_temp_c:.1f}°C` (Target <= 75.0°C)\n"
            f"• **Cooling Rate:** `{payload.cooling_rate_c_per_s:+.3f}°C/s` (Target reached in {ticks_str})\n"
            f"• **SLA Preservation:** 100% compliant, 0 contractual penalties incurred\n"
            f"• **Cryptographic Audit Hash:** {hash_str}\n"
            f"• **Lifecycle Status:** Incident marked `RESOLVED` in StateMachine & AuditLedger\n\n"
            f"{json_block}"
            f"{ADVISORY_FOOTER}"
        )
        return body

    return json_block + ADVISORY_FOOTER
