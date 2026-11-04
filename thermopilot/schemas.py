"""Pydantic data contracts for ThermoPilot Compute Desk.

Defines validated schemas for GPU telemetry, workload specifications, recovery
plan options, deterministic safety evaluations, operator approvals, and audit events.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# 1. Telemetry Contracts
# ---------------------------------------------------------------------------

class GPUTelemetry(BaseModel):
    """Real-time physical telemetry emitted by a monitored GPU node."""
    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the telemetry reading",
    )
    gpu_id: int = Field(default=0, ge=0, description="GPU hardware identifier")
    temperature_core_c: float = Field(
        ..., description="GPU core silicon temperature in Celsius"
    )
    temperature_junction_c: float = Field(
        ..., description="Hotspot / memory junction temperature in Celsius"
    )
    power_draw_w: float = Field(
        ..., ge=0.0, description="Instantaneous power draw in Watts"
    )
    power_limit_w: float = Field(
        ..., ge=0.0, description="Configured thermal design power (TDP) limit in Watts"
    )
    fan_speed_pct: float = Field(
        ..., ge=0.0, le=100.0, description="Chassis / GPU fan speed percentage"
    )
    clock_freq_mhz: int = Field(
        ..., ge=0, description="Current core operating clock frequency in MHz"
    )
    utilization_pct: float = Field(
        ..., ge=0.0, le=100.0, description="Compute engine utilization percentage"
    )
    ambient_temp_c: float = Field(
        ..., description="Intake ambient rack air temperature in Celsius"
    )


# ---------------------------------------------------------------------------
# 2. Workload Specifications & Node Constraints
# ---------------------------------------------------------------------------

class WorkloadPriority(str, Enum):
    """Priority tiers reflecting contractual and business urgency."""
    P0_CRITICAL = "P0_CRITICAL"   # Mission-critical / VIP (zero preemption, strict SLA)
    P1_HIGH = "P1_HIGH"           # Production priority
    P2_FLEXIBLE = "P2_FLEXIBLE"   # Checkpointable / batch training


class WorkloadSpec(BaseModel):
    """Contractual and operational profile of an active workload tenant."""
    model_config = ConfigDict(frozen=True)

    workload_id: str = Field(..., description="Unique workload identifier")
    name: str = Field(..., description="Human-readable workload name")
    owner: str = Field(..., description="Department or tenant identifier")
    priority: WorkloadPriority = Field(
        ..., description="Workload priority tier"
    )
    current_power_w: float = Field(
        ..., ge=0.0, description="Current allocated power budget in Watts"
    )
    min_power_w: float = Field(
        ..., ge=0.0, description="Minimum operational power floor below which job degrades or fails"
    )
    max_power_w: float = Field(
        ..., ge=0.0, description="Maximum peak power allocation in Watts"
    )
    can_preempt: bool = Field(
        default=False, description="Whether the workload can be paused or checkpoint-evicted"
    )
    can_power_cap: bool = Field(
        default=True, description="Whether the workload supports frequency / power capping"
    )
    sla_deadline_s: float = Field(
        ..., ge=0.0, description="Remaining seconds to contractual SLA completion deadline"
    )
    sla_penalty_usd_per_s: float = Field(
        ..., ge=0.0, description="Financial penalty per second beyond SLA deadline"
    )

    @model_validator(mode="after")
    def validate_power_bounds(self) -> "WorkloadSpec":
        if self.min_power_w > self.max_power_w:
            raise ValueError(
                f"min_power_w ({self.min_power_w}W) cannot exceed max_power_w ({self.max_power_w}W)"
            )
        return self


class NodeConstraints(BaseModel):
    """Physical and operational boundary constraints for a GPU compute node."""
    model_config = ConfigDict(frozen=True)

    max_temp_c: float = Field(
        default=80.0,
        gt=0.0,
        description="Maximum steady-state GPU core temperature in Celsius (RULE_TEMP_CEILING)",
    )
    power_budget_w: float = Field(
        default=400.0,
        gt=0.0,
        description="Maximum electrical power envelope in Watts (RULE_NODE_POWER_BUDGET)",
    )


# ---------------------------------------------------------------------------
# 3. Recovery Plans & Actions
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """Discrete actuation actions applicable to a workload."""
    POWER_CAP = "POWER_CAP"       # Restrict power limit to target_power_w
    PAUSE = "PAUSE"               # Save checkpoint and pause execution
    MIGRATE = "MIGRATE"           # Transfer workload execution to secondary node
    RESUME = "RESUME"             # Unpause previously suspended workload


class WorkloadAction(BaseModel):
    """Specific mitigation action targeted at an individual workload."""
    model_config = ConfigDict(frozen=True)

    workload_id: str = Field(..., description="Target workload identifier")
    action: ActionType = Field(..., description="Action to execute")
    target_power_w: float | None = Field(
        default=None, ge=0.0, description="Target power cap in Watts (required for POWER_CAP)"
    )

    @model_validator(mode="after")
    def validate_action_fields(self) -> "WorkloadAction":
        if self.action == ActionType.POWER_CAP and self.target_power_w is None:
            raise ValueError("target_power_w is required when action is POWER_CAP")
        return self


class WorkloadProposal(BaseModel):
    """Structured negotiation bid and rationale produced by a workload-owner agent."""
    model_config = ConfigDict(frozen=True)

    workload_id: str = Field(..., description="Target workload identifier")
    priority: WorkloadPriority = Field(..., description="Workload priority tier")
    current_power_w: float = Field(..., ge=0.0, description="Currently allocated power budget in Watts")
    min_power_w: float = Field(..., ge=0.0, description="Minimum operational power floor in Watts")
    sla_importance_score: float = Field(
        ..., ge=0.0, le=1.0, description="Normalized SLA/performance importance score (0.0 to 1.0)"
    )
    requested_power_w: float | None = Field(
        default=None, ge=0.0, description="Advocated target power cap in Watts (if applicable)"
    )
    preferred_action: ActionType = Field(
        ..., description="Agent's preferred mitigation action (e.g. POWER_CAP, PAUSE, MIGRATE)"
    )
    can_preempt: bool = Field(
        default=False, description="Whether the workload can tolerate preemption or pausing"
    )
    can_power_cap: bool = Field(
        default=True, description="Whether the workload supports power/frequency capping"
    )
    rationale: str = Field(..., description="Concise advocacy rationale justifying the proposal")
    estimated_trade_off: str = Field(..., description="Estimated performance, cost, or SLA trade-off")


class NegotiationEvidence(BaseModel):
    """Consolidated negotiation evidence and structured proposals from workload agents."""
    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the negotiation session",
    )
    proposals: list[WorkloadProposal] = Field(
        default_factory=list, description="List of structured proposals from workload agents"
    )

    @property
    def workload_ids(self) -> list[str]:
        """List of all workload identifiers with submitted proposals."""
        return [p.workload_id for p in self.proposals]

    def get_proposal(self, workload_id: str) -> WorkloadProposal | None:
        """Retrieve proposal for a specific workload ID."""
        for p in self.proposals:
            if p.workload_id == workload_id:
                return p
        return None


class StrategyType(str, Enum):
    """High-level recovery strategy category."""
    CONSERVATIVE_SHED = "CONSERVATIVE_SHED"       # Plan A: Pause lower priority, protect VIP
    BALANCED_POWER_CAP = "BALANCED_POWER_CAP"     # Plan B: Coordinated power capping
    MIGRATION_OFFLOAD = "MIGRATION_OFFLOAD"       # Plan C: Checkpoint export and offload
    UNSAFE_THROTTLE = "UNSAFE_THROTTLE"           # Plan D: Naive equal cut (Unsafe candidate)


class PlanOrigin(str, Enum):
    """Origin provenance of a recovery plan candidate."""
    LLM = "LLM"
    FALLBACK = "FALLBACK"


class RecoveryPlanOption(BaseModel):
    """A proposed recovery plan option generated by the planner."""
    model_config = ConfigDict(frozen=True)

    plan_id: str = Field(..., description="Unique plan identifier (e.g. PLAN-A, PLAN-B)")
    name: str = Field(..., description="Short descriptive plan title")
    strategy_type: StrategyType = Field(..., description="High-level strategy classification")
    is_recommended: bool = Field(
        default=False, description="Whether the planner flags this as the optimal business trade-off"
    )
    projected_temp_max_c: float = Field(
        ..., description="Estimated steady-state maximum temperature in Celsius under this plan"
    )
    cooling_time_to_safe_s: float = Field(
        ..., ge=0.0, description="Projected seconds until core temperature returns below safe limit"
    )
    workload_actions: list[WorkloadAction] = Field(
        ..., min_length=1, description="List of concrete actions to apply across workloads"
    )
    estimated_cost_usd: float = Field(
        ..., ge=0.0, description="Total projected cost including penalties and migration fees"
    )
    sla_violations_expected: int = Field(
        default=0, ge=0, description="Number of expected contractual SLA breaches"
    )
    rationale: str = Field(
        ..., description="Planner explanation outlining the trade-off calculus"
    )
    origin: PlanOrigin | None = Field(
        default=None, description="Candidate plan origin provenance (LLM or FALLBACK)"
    )


# ---------------------------------------------------------------------------
# 4. Deterministic Safety Gate Evaluation
# ---------------------------------------------------------------------------

class SafetyRuleId(str, Enum):
    """Deterministic safety rule identifiers enforced by SafetyGate."""
    RULE_TEMP_CEILING = "RULE_TEMP_CEILING"
    RULE_NODE_POWER_BUDGET = "RULE_NODE_POWER_BUDGET"
    RULE_NO_P0_PREEMPTION = "RULE_NO_P0_PREEMPTION"
    RULE_P0_SLA_PRESERVATION = "RULE_P0_SLA_PRESERVATION"
    RULE_MIN_POWER_FLOOR = "RULE_MIN_POWER_FLOOR"
    RULE_POWER_CAP_VALIDITY = "RULE_POWER_CAP_VALIDITY"


class RuleEvaluation(BaseModel):
    """Outcome of an individual deterministic constraint check."""
    model_config = ConfigDict(frozen=True)

    rule_name: str = Field(..., description="Name or identifier of the safety rule evaluated")
    passed: bool = Field(..., description="True if the rule condition was satisfied")
    detail: str = Field(..., description="Human-readable explanation of the rule evaluation")
    observed_value: float | str | int | bool = Field(..., description="Measured or computed value")
    limit_value: float | str | int | bool = Field(..., description="Hard threshold constraint")
    rule_id: str | None = Field(
        default=None, description="Explicit rule identifier (alias of rule_name)"
    )

    @model_validator(mode="before")
    @classmethod
    def populate_rule_id(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "rule_name" in data and not data.get("rule_id"):
                data["rule_id"] = str(data["rule_name"])
            elif "rule_id" in data and not data.get("rule_name"):
                data["rule_name"] = str(data["rule_id"])
        return data


class SafetyGateResult(BaseModel):
    """Aggregate safety certification report for a candidate recovery plan."""
    model_config = ConfigDict(frozen=True)

    plan_id: str = Field(..., description="Identifier of the evaluated recovery plan")
    passed: bool = Field(
        ..., description="True if all deterministic rules passed; False if rejected"
    )
    evaluations: list[RuleEvaluation] = Field(
        default_factory=list, description="Individual rule verification details"
    )
    rejection_reasons: list[str] = Field(
        default_factory=list, description="List of explicit failure reasons if rejected"
    )

    @property
    def rule_ids(self) -> list[str]:
        """List of all evaluated rule identifiers."""
        return [e.rule_name for e in self.evaluations]


class EvaluatedPlanCandidate(BaseModel):
    """A candidate recovery plan coupled with its deterministic Safety Gate evaluation."""
    model_config = ConfigDict(frozen=True)

    plan: RecoveryPlanOption = Field(..., description="The candidate recovery plan option")
    origin: PlanOrigin = Field(..., description="Provenance origin: LLM or FALLBACK")
    safety_result: SafetyGateResult = Field(..., description="Safety Gate verification report")

    @property
    def is_safe(self) -> bool:
        """Whether this candidate plan passed all deterministic safety gate rules."""
        return self.safety_result.passed


class PlannerResult(BaseModel):
    """Consolidated outcome of recovery planning session."""
    model_config = ConfigDict(frozen=True)

    success: bool = Field(..., description="True if planning completed and candidates were evaluated")
    candidates: list[EvaluatedPlanCandidate] = Field(
        default_factory=list, description="Evaluated candidate recovery plans"
    )
    generation_source: PlanOrigin | None = Field(
        default=None, description="Source that generated the candidates (LLM or FALLBACK)"
    )
    error_message: str | None = Field(
        default=None, description="Error explanation if candidate generation failed"
    )
    negotiation_evidence: NegotiationEvidence | None = Field(
        default=None, description="Structured negotiation evidence ingested during planning"
    )

    @property
    def safe_candidates(self) -> list[EvaluatedPlanCandidate]:
        """Filter to only candidates that passed the deterministic Safety Gate."""
        return [c for c in self.candidates if c.is_safe]

    @property
    def recommended_candidate(self) -> EvaluatedPlanCandidate | None:
        """Return recommended plan if safe, or the first safe plan, or None."""
        for c in self.safe_candidates:
            if c.plan.is_recommended:
                return c
        return self.safe_candidates[0] if self.safe_candidates else None


# ---------------------------------------------------------------------------
# 5. Approvals & Human Governance
# ---------------------------------------------------------------------------

class ApprovalDecision(str, Enum):
    """Operator decision or escalation outcome."""
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    TIMEOUT_ESCALATED = "TIMEOUT_ESCALATED"  # Alarms raised; NEVER mutates workloads automatically


class ApprovalAction(BaseModel):
    """Governance action emitted by human operator or desk timeout escalation."""
    model_config = ConfigDict(frozen=True)

    incident_id: str = Field(..., description="Active thermal incident identifier")
    selected_plan_id: str | None = Field(
        default=None, description="Identifier of the approved plan (None if rejected or timed out)"
    )
    operator_id: str = Field(
        ..., description="Identifier of the on-call operator or escalation subsystem"
    )
    decision: ApprovalDecision = Field(
        ..., description="Outcome decision (APPROVE, REJECT, or TIMEOUT_ESCALATED)"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the approval action",
    )
    notes: str | None = Field(
        default=None, description="Optional operator rationale or escalation context"
    )

    @model_validator(mode="after")
    def validate_plan_selection(self) -> "ApprovalAction":
        if self.decision == ApprovalDecision.APPROVE and not self.selected_plan_id:
            raise ValueError("selected_plan_id is required when decision is APPROVE")
        if self.decision in (ApprovalDecision.TIMEOUT_ESCALATED, ApprovalDecision.REJECT) and self.selected_plan_id is not None:
            raise ValueError(
                f"selected_plan_id must be None when decision is {self.decision.value}. "
                "Workload mutations cannot be authorized on timeout escalation or rejection."
            )
        return self


class CandidatePlanSummary(BaseModel):
    """Summary of a candidate recovery plan for operator decision card display."""
    model_config = ConfigDict(frozen=True)

    plan_id: str = Field(..., description="Unique plan identifier (e.g. PLAN-A, PLAN-B)")
    name: str = Field(..., description="Short descriptive plan title")
    strategy_type: StrategyType = Field(..., description="High-level strategy category")
    is_recommended: bool = Field(
        default=False, description="Whether the planner flags this as the optimal business trade-off"
    )
    is_safe: bool = Field(..., description="True if plan passed all Safety Gate deterministic rules")
    safety_verdict: str = Field(
        ..., description="Certification status: CERTIFIED_SAFE or REJECTED_BY_SAFETY_GATE"
    )
    rejection_reasons: list[str] = Field(
        default_factory=list, description="Explicit failure reasons if rejected by Safety Gate"
    )
    projected_temp_max_c: float = Field(
        ..., description="Estimated steady-state maximum temperature in Celsius under this plan"
    )
    projected_power_w: float = Field(
        ..., ge=0.0, description="Estimated total node electrical power draw under this plan in Watts"
    )
    cooling_time_to_safe_s: float = Field(
        ..., ge=0.0, description="Projected seconds until temperature returns below safe limit"
    )
    estimated_cost_usd: float = Field(
        ..., ge=0.0, description="Total projected cost including SLA penalties and migration fees"
    )
    sla_violations_expected: int = Field(
        default=0, ge=0, description="Number of expected contractual SLA breaches"
    )
    proposed_actions: list[WorkloadAction] = Field(
        default_factory=list, description="List of proposed workload mitigation actions"
    )
    is_executable: bool = Field(
        ..., description="True if plan is certified safe and operator can execute it"
    )
    rationale: str = Field(..., description="Trade-off rationale explaining the plan")

    @model_validator(mode="after")
    def validate_executable_safety(self) -> "CandidatePlanSummary":
        if not self.is_safe and self.is_executable:
            raise ValueError(
                f"Candidate plan '{self.plan_id}' failed safety gate certification and cannot be marked executable."
            )
        return self


class DecisionCard(BaseModel):
    """Interactive decision card presented to human operator via Switch or CLI desk."""
    model_config = ConfigDict(frozen=True)

    incident_id: str = Field(..., description="Active thermal incident identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of incident card generation",
    )
    gpu_id: int = Field(default=0, ge=0, description="Target GPU identifier")
    severity: str = Field(default="CRITICAL", description="Incident severity level (e.g. CRITICAL, WARNING)")
    current_temp_c: float = Field(..., description="Current GPU core temperature in Celsius")
    current_power_w: float = Field(..., ge=0.0, description="Current instantaneous power draw in Watts")
    critical_trip_temp_c: float = Field(
        default=85.0, description="Hardware shutdown trip threshold in Celsius"
    )
    time_to_trip_s: float | None = Field(
        default=None, ge=0.0, description="Projected seconds until hardware trip threshold (t_TTT)"
    )
    candidate_plans: list[CandidatePlanSummary] = Field(
        ..., min_length=1, description="List of evaluated recovery plan candidates"
    )
    executable_plan_ids: list[str] = Field(
        default_factory=list, description="Plan IDs certified safe and available for approval"
    )
    rejected_plan_ids: list[str] = Field(
        default_factory=list, description="Plan IDs rejected by Safety Gate (cannot be approved)"
    )
    timeout_seconds: float = Field(
        default=15.0, gt=0.0, description="Governance window before automatic timeout escalation"
    )

    @property
    def has_executable_plans(self) -> bool:
        """Whether at least one candidate plan passed safety certification."""
        return len(self.executable_plan_ids) > 0


class WorkflowStatus(str, Enum):
    """Operational status of the human approval workflow."""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    TIMEOUT_ESCALATED = "TIMEOUT_ESCALATED"
    REVALIDATION_FAILED = "REVALIDATION_FAILED"
    INVALID_SUBMISSION = "INVALID_SUBMISSION"


class ApprovalWorkflowResult(BaseModel):
    """Outcome of the human-in-the-loop approval and pre-execution safety verification."""
    model_config = ConfigDict(frozen=True)

    incident_id: str = Field(..., description="Thermal incident identifier")
    status: WorkflowStatus = Field(..., description="Outcome status of the approval workflow")
    decision: ApprovalDecision | None = Field(
        default=None, description="Operator decision (APPROVE, REJECT, or TIMEOUT_ESCALATED)"
    )
    selected_plan: RecoveryPlanOption | None = Field(
        default=None, description="The validated recovery plan approved for execution (if authorized)"
    )
    is_authorized_for_execution: bool = Field(
        default=False, description="Strict boolean flag indicating if execution is fully authorized"
    )
    revalidation_result: SafetyGateResult | None = Field(
        default=None, description="Safety Gate revalidation report evaluated immediately before authorization"
    )
    rejection_reasons: list[str] = Field(
        default_factory=list, description="Rejection reasons if approval was invalid or authorization denied"
    )
    action: ApprovalAction | None = Field(
        default=None, description="The underlying approval action recorded in governance"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of workflow resolution",
    )

    @model_validator(mode="after")
    def validate_authorization_consistency(self) -> "ApprovalWorkflowResult":
        if self.is_authorized_for_execution:
            if self.status != WorkflowStatus.APPROVED:
                raise ValueError("is_authorized_for_execution cannot be True unless status is APPROVED.")
            if self.selected_plan is None:
                raise ValueError("is_authorized_for_execution cannot be True when selected_plan is None.")
            if self.decision != ApprovalDecision.APPROVE:
                raise ValueError("is_authorized_for_execution cannot be True unless decision is APPROVE.")
        return self


# ---------------------------------------------------------------------------
# 6. Audit Trail & Event Ledger
# ---------------------------------------------------------------------------

class AuditEventType(str, Enum):
    """Categorized lifecycle events recorded in the immutable audit ledger."""
    INCIDENT_TRIGGERED = "INCIDENT_TRIGGERED"
    INCIDENT_DETECTED = "INCIDENT_DETECTED"
    FORECAST_COMPLETED = "FORECAST_COMPLETED"
    WORKLOAD_BIDS_COLLECTED = "WORKLOAD_BIDS_COLLECTED"
    PLANNER_PLANS_GENERATED = "PLANNER_PLANS_GENERATED"
    SAFETY_GATE_EVALUATED = "SAFETY_GATE_EVALUATED"
    DECISION_CARD_DISPATCHED = "DECISION_CARD_DISPATCHED"
    OPERATOR_DECISION_RECORDED = "OPERATOR_DECISION_RECORDED"
    TIMEOUT_ESCALATION_RECORDED = "TIMEOUT_ESCALATION_RECORDED"
    PRE_EXECUTION_REVALIDATED = "PRE_EXECUTION_REVALIDATED"
    PLAN_EXECUTED = "PLAN_EXECUTED"
    EXECUTION_DENIED = "EXECUTION_DENIED"
    VERIFICATION_COMPLETED = "VERIFICATION_COMPLETED"
    STATE_TRANSITION = "STATE_TRANSITION"


class AuditEvent(BaseModel):
    """Cryptographically auditable append-only ledger entry."""
    model_config = ConfigDict(frozen=True)

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique UUID for this audit entry",
    )
    incident_id: str = Field(..., description="Associated thermal incident ID")
    event_type: AuditEventType = Field(..., description="Type of lifecycle event")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of event occurrence",
    )
    sequence_num: int = Field(
        default=0, ge=0, description="Sequential sequence index in ledger hash chain"
    )
    prev_hash: str | None = Field(
        default=None, description="Cryptographic SHA-256 hash of preceding ledger event"
    )
    hash: str = Field(
        default="", description="Cryptographic SHA-256 hash of this event"
    )
    is_simulated: bool = Field(
        default=True, description="Marker indicating if event pertains to simulated infrastructure"
    )
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Structured event payload conforming to lifecycle schemas"
    )


class IncidentState(str, Enum):
    """Lifecycle states of a thermal incident in the deterministic state machine."""
    DETECTED = "DETECTED"
    FORECASTED = "FORECASTED"
    PLANNING = "PLANNING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    TIMEOUT_ESCALATED = "TIMEOUT_ESCALATED"
    REVALIDATION_FAILED = "REVALIDATION_FAILED"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    RECOVERED = "RECOVERED"
    RECOVERY_FAILED = "RECOVERY_FAILED"


class StateTransitionRecord(BaseModel):
    """Immutable audit record of a state machine transition."""
    model_config = ConfigDict(frozen=True)

    transition_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier UUID for this state transition",
    )
    incident_id: str = Field(..., description="Associated thermal incident ID")
    from_state: IncidentState = Field(..., description="State before transition")
    to_state: IncidentState = Field(..., description="State after transition")
    plan_id: str | None = Field(default=None, description="Selected recovery plan ID if applicable")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of state transition",
    )
    reason: str | None = Field(default=None, description="Human-readable transition rationale")
    details: dict[str, Any] = Field(default_factory=dict, description="Operational transition metadata")


class LedgerVerificationReport(BaseModel):
    """Audit ledger cryptographic chain verification report."""
    model_config = ConfigDict(frozen=True)

    is_valid: bool = Field(..., description="True if ledger hash chain is completely unbroken and valid")
    total_events: int = Field(..., ge=0, description="Total number of events validated")
    corrupted_event_index: int | None = Field(
        default=None, description="First line index where corruption/tampering was detected"
    )
    errors: list[str] = Field(default_factory=list, description="Itemized integrity failure explanations")


# ---------------------------------------------------------------------------
# 7. Execution & Verification Contracts
# ---------------------------------------------------------------------------

class ExecutionStatus(str, Enum):
    """Operational status of recovery plan execution."""
    SUCCESS = "SUCCESS"
    REJECTED_UNAUTHORIZED = "REJECTED_UNAUTHORIZED"
    REJECTED_SAFETY_VIOLATION = "REJECTED_SAFETY_VIOLATION"
    REJECTED_INVALID_PLAN = "REJECTED_INVALID_PLAN"
    REJECTED_STALE_APPROVAL = "REJECTED_STALE_APPROVAL"
    FAILED_EXECUTION = "FAILED_EXECUTION"


class ExecutionResult(BaseModel):
    """Immutable record of an execution attempt on the simulator."""
    model_config = ConfigDict(frozen=True)

    execution_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique execution identifier UUID",
    )
    incident_id: str = Field(..., description="Target thermal incident ID")
    plan_id: str = Field(..., description="Executed or evaluated recovery plan ID")
    success: bool = Field(..., description="Whether execution was authorized and completed successfully")
    status: ExecutionStatus = Field(..., description="Detailed execution outcome status")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the execution attempt",
    )
    applied_actions: list[WorkloadAction] = Field(
        default_factory=list, description="Workload actions applied to simulator"
    )
    initial_telemetry: GPUTelemetry | None = Field(
        default=None, description="Simulator telemetry snapshot immediately prior to execution"
    )
    post_execution_telemetry: GPUTelemetry | None = Field(
        default=None, description="Simulator telemetry snapshot immediately following execution"
    )
    revalidation_result: SafetyGateResult | None = Field(
        default=None, description="Safety Gate revalidation report evaluated immediately before execution"
    )
    rejection_reasons: list[str] = Field(
        default_factory=list, description="Explicit failure or rejection reasons if unexecuted"
    )
    is_simulated: bool = Field(
        default=True, description="Strict boolean tag confirming this execution affected ONLY simulated state"
    )
    details: dict[str, Any] = Field(
        default_factory=dict, description="Operational metadata and diagnostics"
    )


class VerificationCheck(BaseModel):
    """Outcome of an individual post-execution verification check."""
    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Descriptive identifier of the check")
    passed: bool = Field(..., description="Whether the verification check passed")
    observed_value: Any = Field(..., description="Measured or computed physical value")
    threshold_value: Any = Field(..., description="Expected safe physical boundary")
    detail: str = Field(..., description="Human-readable explanation of check outcome")


class VerificationReport(BaseModel):
    """Comprehensive post-execution verification analysis over the recovery window."""
    model_config = ConfigDict(frozen=True)

    verification_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique verification analysis identifier UUID",
    )
    incident_id: str = Field(..., description="Associated thermal incident ID")
    plan_id: str = Field(..., description="Evaluated recovery plan ID")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of verification completion",
    )
    recovery_successful: bool = Field(
        ..., description="Whether the system achieved safe thermal recovery without SLA or safety violations"
    )
    is_simulated: bool = Field(
        default=True, description="Confirmation that verification was performed in a simulated environment"
    )
    initial_temp_c: float = Field(..., description="GPU core temperature in Celsius prior to recovery window")
    final_temp_c: float = Field(..., description="GPU core temperature in Celsius at end of recovery window")
    min_temp_c: float = Field(..., description="Minimum temperature observed during recovery window")
    max_temp_c: float = Field(..., description="Peak temperature observed during recovery window")
    temp_delta_c: float = Field(..., description="Temperature change (initial - final) in Celsius")
    target_temp_c: float = Field(
        default=75.0, description="Target safe recovery temperature ceiling in Celsius"
    )
    reached_target: bool = Field(
        ..., description="Whether core temperature reached or exceeded safe target (<= 75.0°C)"
    )
    approaching_target: bool = Field(
        default=False, description="Whether core temperature is monotonically decreasing toward target"
    )
    ticks_to_target: int | None = Field(
        default=None, description="Simulation ticks elapsed until safe recovery target was first achieved"
    )
    sim_time_to_target_s: float | None = Field(
        default=None, description="Simulated seconds elapsed until safe recovery target was first achieved"
    )
    observed_cooling_rate_c_per_s: float = Field(
        ..., description="Observed thermal decay rate in °C per simulated second"
    )
    target_cooling_rate_c_per_s: float = Field(
        default=0.5, description="Benchmark minimum cooling rate in °C per simulated second"
    )
    meets_cooling_rate_target: bool = Field(
        ..., description="Whether observed cooling rate meets or exceeds target rate"
    )
    checks: list[VerificationCheck] = Field(
        default_factory=list, description="Detailed list of individual verification checks"
    )
    failure_reasons: list[str] = Field(
        default_factory=list, description="Honest, un-fabricated failure explanations if recovery did not succeed"
    )
    temperature_trajectory: list[float] = Field(
        default_factory=list, description="Monitored temperature readings across simulation ticks"
    )
    ticks_monitored: int = Field(..., description="Total simulation ticks monitored (up to 30)")
    simulated_duration_s: float = Field(..., description="Total simulated seconds observed during window")
    summary: str = Field(..., description="Human-readable verification narrative")


__all__ = [
    "ActionType",
    "ApprovalAction",
    "ApprovalDecision",
    "ApprovalWorkflowResult",
    "AuditEvent",
    "AuditEventType",
    "CandidatePlanSummary",
    "DecisionCard",
    "EvaluatedPlanCandidate",
    "ExecutionResult",
    "ExecutionStatus",
    "GPUTelemetry",
    "IncidentState",
    "LedgerVerificationReport",
    "NegotiationEvidence",
    "NodeConstraints",
    "PlanOrigin",
    "PlannerResult",
    "RecoveryPlanOption",
    "RuleEvaluation",
    "SafetyGateResult",
    "SafetyRuleId",
    "StateTransitionRecord",
    "StrategyType",
    "VerificationCheck",
    "VerificationReport",
    "WorkflowStatus",
    "WorkloadAction",
    "WorkloadPriority",
    "WorkloadProposal",
    "WorkloadSpec",
]
