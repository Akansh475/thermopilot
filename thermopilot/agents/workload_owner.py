"""Workload-Owner Negotiation Agents for ThermoPilot Compute Desk.

Models autonomous tenant agents that advocate for competing computational workloads.
Each agent evaluates active thermal alerts, calculates deterministic SLA urgency scores,
and emits structured negotiation proposals (bids) for the recovery planner.

Advisory Only: Agents have zero actuation authority, cannot modify simulator state,
and cannot bypass or weaken deterministic Safety Gate constraints.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

from thermopilot.agents.forecast import ForecastResult, ThermalRiskStatus
from thermopilot.schemas import (
    ActionType,
    GPUTelemetry,
    NegotiationEvidence,
    WorkloadPriority,
    WorkloadProposal,
    WorkloadSpec,
)


class WorkloadOwnerAgent:
    """Deterministic workload-owner agent advocating for an individual tenant."""

    def __init__(self, spec: WorkloadSpec) -> None:
        """Initialize agent with its contractual workload profile.

        Args:
            spec: Validated WorkloadSpec instance.
        """
        if not isinstance(spec, WorkloadSpec):
            raise TypeError(f"Expected WorkloadSpec instance, got {type(spec).__name__}")
        self.spec = spec

    # -----------------------------------------------------------------------
    # Workload Properties
    # -----------------------------------------------------------------------

    @property
    def workload_id(self) -> str:
        """Unique identifier of the tenant workload."""
        return self.spec.workload_id

    @property
    def name(self) -> str:
        """Human-readable workload title."""
        return self.spec.name

    @property
    def owner(self) -> str:
        """Tenant or organizational unit."""
        return self.spec.owner

    @property
    def priority(self) -> WorkloadPriority:
        """Priority tier reflecting business urgency."""
        return self.spec.priority

    @property
    def current_power_w(self) -> float:
        """Current electrical power allocation in Watts."""
        return self.spec.current_power_w

    @property
    def min_power_w(self) -> float:
        """Hard minimum operational power floor in Watts."""
        return self.spec.min_power_w

    @property
    def max_power_w(self) -> float:
        """Maximum peak power allocation in Watts."""
        return self.spec.max_power_w

    @property
    def can_preempt(self) -> bool:
        """Whether the workload supports eviction or pausing."""
        return self.spec.can_preempt

    @property
    def can_power_cap(self) -> bool:
        """Whether the workload supports frequency / power capping."""
        return self.spec.can_power_cap

    @property
    def sla_deadline_s(self) -> float:
        """Remaining seconds to contractual SLA completion deadline."""
        return self.spec.sla_deadline_s

    @property
    def sla_penalty_usd_per_s(self) -> float:
        """Financial penalty per second beyond SLA deadline."""
        return self.spec.sla_penalty_usd_per_s

    # -----------------------------------------------------------------------
    # Deterministic Scoring & Advocacy
    # -----------------------------------------------------------------------

    def compute_sla_importance(self) -> float:
        """Compute normalized SLA/performance criticality score in [0.0, 1.0].

        Deterministic formula synthesizing priority tier, financial late penalty rate,
        and contractual deadline urgency:
        - P0_CRITICAL base: 0.85
        - P1_HIGH base: 0.50
        - P2_FLEXIBLE base: 0.10
        Plus penalty rate scaling (up to +0.10) and deadline urgency scaling (up to +0.05).
        """
        if self.spec.priority == WorkloadPriority.P0_CRITICAL:
            base_score = 0.85
        elif self.spec.priority == WorkloadPriority.P1_HIGH:
            base_score = 0.50
        else:
            base_score = 0.10

        # Financial penalty factor (scaled up to $5.50/s reference max rate)
        penalty_factor = min(0.10, (self.spec.sla_penalty_usd_per_s / 5.50) * 0.10)

        # Deadline proximity factor (shorter deadlines score higher)
        deadline_factor = 0.05 * (1.0 - min(1.0, self.spec.sla_deadline_s / 600.0))

        score = min(1.0, max(0.0, base_score + penalty_factor + deadline_factor))
        return round(score, 4)

    def propose(
        self,
        telemetry: GPUTelemetry | None = None,
        forecast: ForecastResult | None = None,
    ) -> WorkloadProposal:
        """Generate a structured negotiation proposal advocating for this workload.

        Args:
            telemetry: Current physical GPU telemetry reading.
            forecast: Perception forecast indicating incident risk and t_TTT.

        Returns:
            Validated WorkloadProposal capturing requested action, power, and rationale.
        """
        importance = self.compute_sla_importance()
        is_thermal_elevated = (
            forecast is not None
            and forecast.risk_status
            in (
                ThermalRiskStatus.CRITICAL_RISK,
                ThermalRiskStatus.BREACHED,
                ThermalRiskStatus.WARNING,
            )
        ) or (telemetry is not None and telemetry.temperature_core_c >= 80.0)

        # -------------------------------------------------------------------
        # 1. P0_CRITICAL Advocate Logic
        # -------------------------------------------------------------------
        if self.spec.priority == WorkloadPriority.P0_CRITICAL or not self.spec.can_preempt:
            preferred_action = ActionType.POWER_CAP

            if is_thermal_elevated:
                # Accepts modest reduction down to safe buffer above min power floor (e.g. 220W for Alpha)
                buffer = 0.40 * (self.spec.max_power_w - self.spec.min_power_w)
                requested_power = max(
                    self.spec.min_power_w,
                    min(self.spec.current_power_w, self.spec.min_power_w + buffer),
                )
            else:
                requested_power = self.spec.current_power_w

            requested_power = round(requested_power, 1)
            concession_w = self.spec.current_power_w - requested_power

            rationale = (
                f"P0_CRITICAL VIP workload '{self.name}': Mission-critical SLA deadline in "
                f"{self.spec.sla_deadline_s:.0f}s with ${self.spec.sla_penalty_usd_per_s:.2f}/s "
                f"(${self.spec.sla_penalty_usd_per_s * 3600:.0f}/hr) late fee. Workload is non-preemptible; "
                f"strictly rejects any PAUSE or eviction. Requests guaranteed power allocation of "
                f"{requested_power:.0f}W (safely above {self.spec.min_power_w:.0f}W floor) to preserve deadline."
            )
            estimated_trade_off = (
                f"Zero SLA breach risk avoiding ${self.spec.sla_penalty_usd_per_s * 60:.0f}/min in penalties; "
                f"contributes {concession_w:.0f}W electrical power concession toward node thermal recovery."
            )

        # -------------------------------------------------------------------
        # 2. P1_HIGH Advocate Logic
        # -------------------------------------------------------------------
        elif self.spec.priority == WorkloadPriority.P1_HIGH:
            preferred_action = ActionType.POWER_CAP

            if is_thermal_elevated:
                # Can tolerate moderate capping (e.g. 80% current or down to 100W)
                target = max(self.spec.min_power_w, self.spec.current_power_w * 0.80)
                requested_power = round(target, 1)
            else:
                requested_power = self.spec.current_power_w

            concession_w = self.spec.current_power_w - requested_power
            rationale = (
                f"P1_HIGH Production workload '{self.name}': Contractual SLA penalty of "
                f"${self.spec.sla_penalty_usd_per_s:.2f}/s (${self.spec.sla_penalty_usd_per_s * 3600:.0f}/hr). "
                f"Accepts moderate dynamic power capping down to {requested_power:.0f}W "
                f"(above {self.spec.min_power_w:.0f}W floor) to assist cluster cooling, but requests "
                f"priority preservation over lower-tier flexible training jobs."
            )
            estimated_trade_off = (
                f"Adds modest ~15-20% execution delay without contractual SLA breach; "
                f"frees {concession_w:.0f}W electrical power envelope for VIP protection."
            )

        # -------------------------------------------------------------------
        # 3. P2_FLEXIBLE Advocate Logic
        # -------------------------------------------------------------------
        else:
            if is_thermal_elevated:
                # Highly flexible: can tolerate capping to 160W, checkpoint-pausing, or migration
                requested_power = round(max(self.spec.min_power_w, 160.0), 1)
                preferred_action = ActionType.POWER_CAP
                concession_w = self.spec.current_power_w - requested_power
                rationale = (
                    f"P2_FLEXIBLE Exploratory workload '{self.name}': Elastic batch training with active "
                    f"checkpointing and minimal ${self.spec.sla_penalty_usd_per_s:.2f}/s late penalty. "
                    f"Can accept power capping down to {requested_power:.0f}W, checkpoint-pause (0W), "
                    f"or migration off-node to alleviate thermal runaway in exchange for cluster credits."
                )
                estimated_trade_off = (
                    f"Frees up to {self.spec.current_power_w:.0f}W if paused (or {concession_w:.0f}W if capped) "
                    f"with negligible business penalty (${self.spec.sla_penalty_usd_per_s * 3600:.2f}/hr)."
                )
            else:
                requested_power = self.spec.current_power_w
                preferred_action = ActionType.POWER_CAP
                rationale = (
                    f"P2_FLEXIBLE workload '{self.name}': Operating nominally. Prefers retaining current "
                    f"{requested_power:.0f}W allocation while thermal headroom permits."
                )
                estimated_trade_off = "Nominal training throughput maintained."

        return WorkloadProposal(
            workload_id=self.spec.workload_id,
            priority=self.spec.priority,
            current_power_w=self.spec.current_power_w,
            min_power_w=self.spec.min_power_w,
            sla_importance_score=importance,
            requested_power_w=requested_power,
            preferred_action=preferred_action,
            can_preempt=self.spec.can_preempt,
            can_power_cap=self.spec.can_power_cap,
            rationale=rationale,
            estimated_trade_off=estimated_trade_off,
        )


# ---------------------------------------------------------------------------
# Multi-Agent Negotiation Collection & Factories
# ---------------------------------------------------------------------------

def collect_workload_proposals(
    workloads: Sequence[WorkloadSpec | WorkloadOwnerAgent] | Mapping[str, WorkloadSpec | WorkloadOwnerAgent],
    telemetry: GPUTelemetry | None = None,
    forecast: ForecastResult | None = None,
) -> NegotiationEvidence:
    """Collect structured negotiation proposals from all active workload-owner agents.

    Args:
        workloads: Sequence or Mapping of WorkloadSpec or WorkloadOwnerAgent instances.
        telemetry: Optional current telemetry snapshot.
        forecast: Optional forecast perception result.

    Returns:
        NegotiationEvidence containing all proposals ordered by SLA importance descending.
    """
    agents: list[WorkloadOwnerAgent] = []

    if isinstance(workloads, Mapping):
        items = workloads.values()
    elif isinstance(workloads, (list, tuple, set)):
        items = workloads
    else:
        raise TypeError(f"Expected sequence or mapping of workloads, got {type(workloads).__name__}")

    for item in items:
        if isinstance(item, WorkloadOwnerAgent):
            agents.append(item)
        elif isinstance(item, WorkloadSpec):
            agents.append(WorkloadOwnerAgent(item))
        else:
            raise TypeError(f"Invalid workload item of type {type(item).__name__}")

    proposals: list[WorkloadProposal] = []
    for agent in agents:
        proposal = agent.propose(telemetry=telemetry, forecast=forecast)
        proposals.append(proposal)

    # Sort deterministically by SLA importance descending (P0 > P1 > P2)
    proposals.sort(key=lambda p: (p.sla_importance_score, -p.current_power_w), reverse=True)

    return NegotiationEvidence(
        timestamp=datetime.now(timezone.utc),
        proposals=proposals,
    )


def create_competing_workload_agents() -> dict[str, WorkloadOwnerAgent]:
    """Factory creating the reference three competing workload-owner agents.

    Returns:
        Dict mapping workload IDs to their respective WorkloadOwnerAgent:
        - 'wl-alpha-inference': P0_CRITICAL Financial Trading Batch (rigid SLA, non-preemptible)
        - 'wl-gamma-analytics': P1_HIGH Production Analytics Pipeline (moderate SLA, can cap)
        - 'wl-beta-training': P2_FLEXIBLE Diffusion Fine-Tuning (checkpointable, elastic)
    """
    alpha = WorkloadSpec(
        workload_id="wl-alpha-inference",
        name="Financial Trading Batch",
        owner="Trading-Prod",
        priority=WorkloadPriority.P0_CRITICAL,
        current_power_w=250.0,
        min_power_w=200.0,
        max_power_w=250.0,
        can_preempt=False,
        can_power_cap=True,
        sla_deadline_s=90.0,
        sla_penalty_usd_per_s=5.50,
    )
    gamma = WorkloadSpec(
        workload_id="wl-gamma-analytics",
        name="Production Analytics Pipeline",
        owner="Analytics-Infra",
        priority=WorkloadPriority.P1_HIGH,
        current_power_w=120.0,
        min_power_w=80.0,
        max_power_w=140.0,
        can_preempt=True,
        can_power_cap=True,
        sla_deadline_s=600.0,
        sla_penalty_usd_per_s=1.00,
    )
    beta = WorkloadSpec(
        workload_id="wl-beta-training",
        name="Diffusion Fine-Tuning",
        owner="Research-Exploratory",
        priority=WorkloadPriority.P2_FLEXIBLE,
        current_power_w=150.0,
        min_power_w=150.0,
        max_power_w=200.0,
        can_preempt=True,
        can_power_cap=True,
        sla_deadline_s=3600.0,
        sla_penalty_usd_per_s=0.01,
    )

    return {
        alpha.workload_id: WorkloadOwnerAgent(alpha),
        gamma.workload_id: WorkloadOwnerAgent(gamma),
        beta.workload_id: WorkloadOwnerAgent(beta),
    }
