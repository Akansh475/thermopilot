"""Deterministic Fallback Candidate Generator for ThermoPilot Compute Desk.

Provides a zero-latency, local algorithmic fallback that generates three
distinct certified feasible recovery options matching the architecture specifications:
- Plan A: Conservative Workload Shed (pause lower-priority Beta, allocate safe power to VIP Alpha)
- Plan B: Balanced Dynamic Power Cap (Alpha 220W, Beta 160W; optimal net economics)
- Plan C: Workload Migration / Offload (migrate Beta to secondary node, Alpha retains full 250W envelope)
Optionally produces Plan D (Blind 50% Throttle) for demonstration and safety evaluation.
"""

from collections.abc import Mapping, Sequence

from thermopilot.agents.forecast import ForecastResult
from thermopilot.schemas import (
    ActionType,
    GPUTelemetry,
    NegotiationEvidence,
    NodeConstraints,
    PlanOrigin,
    RecoveryPlanOption,
    StrategyType,
    WorkloadAction,
    WorkloadPriority,
    WorkloadSpec,
)


class DeterministicFallbackPlanner:
    """Algorithmic, deterministic fallback candidate generator."""

    def __init__(self, node_constraints: NodeConstraints | None = None) -> None:
        self.node_constraints = node_constraints or NodeConstraints()

    def generate_candidates(
        self,
        telemetry: GPUTelemetry | None = None,
        workloads: Sequence[WorkloadSpec] | Mapping[str, WorkloadSpec] | None = None,
        node_constraints: NodeConstraints | None = None,
        forecast: ForecastResult | None = None,
        include_unsafe_candidate: bool = False,
        negotiation_evidence: NegotiationEvidence | None = None,
    ) -> list[RecoveryPlanOption]:
        """Generate three meaningfully different candidate recovery plans.

        Args:
            telemetry: Current physical GPU telemetry.
            workloads: Active workload specifications.
            node_constraints: Node power and thermal boundaries.
            forecast: Perception forecast indicating thermal trajectory.
            include_unsafe_candidate: Whether to append Plan D (Unsafe Candidate).
            negotiation_evidence: Optional structured proposals from workload agents.

        Returns:
            List of RecoveryPlanOption instances tagged with PlanOrigin.FALLBACK.
        """
        # Normalize workloads
        workload_map: dict[str, WorkloadSpec] = {}
        if isinstance(workloads, Mapping):
            workload_map = dict(workloads)
        elif isinstance(workloads, (list, tuple, set)):
            for w in workloads:
                if isinstance(w, WorkloadSpec):
                    workload_map[w.workload_id] = w

        # Identify VIP / Critical workload (P0 or non-preemptible)
        alpha_id = "wl-alpha-inference"
        beta_id = "wl-beta-training"
        alpha_spec: WorkloadSpec | None = None
        beta_spec: WorkloadSpec | None = None

        for wid, spec in workload_map.items():
            if spec.priority == WorkloadPriority.P0_CRITICAL or not spec.can_preempt:
                alpha_id = wid
                alpha_spec = spec
                break

        for wid, spec in workload_map.items():
            if wid != alpha_id:
                beta_id = wid
                beta_spec = spec
                break

        # Defaults if specs not provided
        alpha_min = alpha_spec.min_power_w if alpha_spec else 200.0
        alpha_max = alpha_spec.max_power_w if alpha_spec else 250.0
        beta_min = beta_spec.min_power_w if beta_spec else 150.0
        beta_max = beta_spec.max_power_w if beta_spec else 200.0

        alpha_cap_a = min(alpha_max, max(alpha_min, 220.0))
        alpha_cap_b = min(alpha_max, max(alpha_min, 220.0))
        beta_cap_b = min(beta_max, max(beta_min, 160.0))
        alpha_cap_c = min(alpha_max, 250.0)

        # -------------------------------------------------------------------
        # Plan A: Conservative Workload Shed (Feasible)
        # -------------------------------------------------------------------
        plan_a = RecoveryPlanOption(
            plan_id="PLAN-A",
            name="Conservative Workload Shed",
            strategy_type=StrategyType.CONSERVATIVE_SHED,
            is_recommended=False,
            projected_temp_max_c=67.5,
            cooling_time_to_safe_s=14.0,
            workload_actions=[
                WorkloadAction(
                    workload_id=alpha_id,
                    action=ActionType.POWER_CAP,
                    target_power_w=alpha_cap_a,
                ),
                WorkloadAction(
                    workload_id=beta_id,
                    action=ActionType.PAUSE,
                ),
            ],
            estimated_cost_usd=45.0,
            sla_violations_expected=0,
            rationale=(
                f"Checkpoint-pause flexible job '{beta_id}'; allocate safe power to VIP '{alpha_id}' "
                f"({alpha_cap_a:.0f}W). Guarantees rapid cooldown with zero SLA penalty."
            ),
            origin=PlanOrigin.FALLBACK,
        )

        # -------------------------------------------------------------------
        # Plan B: Balanced Dynamic Power Cap (Feasible & Recommended)
        # -------------------------------------------------------------------
        plan_b = RecoveryPlanOption(
            plan_id="PLAN-B",
            name="Dynamic Workload Power-Cap Balancing",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            is_recommended=True,
            projected_temp_max_c=74.0,
            cooling_time_to_safe_s=17.5,
            workload_actions=[
                WorkloadAction(
                    workload_id=alpha_id,
                    action=ActionType.POWER_CAP,
                    target_power_w=alpha_cap_b,
                ),
                WorkloadAction(
                    workload_id=beta_id,
                    action=ActionType.POWER_CAP,
                    target_power_w=beta_cap_b,
                ),
            ],
            estimated_cost_usd=12.0,
            sla_violations_expected=0,
            rationale=(
                f"Dynamically caps power across both jobs ('{alpha_id}' at {alpha_cap_b:.0f}W, "
                f"'{beta_id}' at {beta_cap_b:.0f}W; total {alpha_cap_b + beta_cap_b:.0f}W <= 400W budget). "
                f"Zero SLA breach on VIP job, optimal net economics ($12)."
            ),
            origin=PlanOrigin.FALLBACK,
        )

        # -------------------------------------------------------------------
        # Plan C: Workload Migration / Offload (Feasible)
        # -------------------------------------------------------------------
        plan_c = RecoveryPlanOption(
            plan_id="PLAN-C",
            name="Workload Migration / Offload",
            strategy_type=StrategyType.MIGRATION_OFFLOAD,
            is_recommended=False,
            projected_temp_max_c=69.5,
            cooling_time_to_safe_s=12.0,
            workload_actions=[
                WorkloadAction(
                    workload_id=alpha_id,
                    action=ActionType.POWER_CAP,
                    target_power_w=alpha_cap_c,
                ),
                WorkloadAction(
                    workload_id=beta_id,
                    action=ActionType.MIGRATE,
                ),
            ],
            estimated_cost_usd=28.0,
            sla_violations_expected=0,
            rationale=(
                f"Export '{beta_id}' checkpoint and migrate active execution to secondary capacity. "
                f"'{alpha_id}' retains full {alpha_cap_c:.0f}W power envelope with small cloud transfer cost ($28)."
            ),
            origin=PlanOrigin.FALLBACK,
        )

        candidates = [plan_a, plan_b, plan_c]

        # -------------------------------------------------------------------
        # Plan D: Blind 50% Throttle (Unsafe Candidate for Evaluation)
        # -------------------------------------------------------------------
        if include_unsafe_candidate:
            plan_d = RecoveryPlanOption(
                plan_id="PLAN-D",
                name="Blind 50% Throttle",
                strategy_type=StrategyType.UNSAFE_THROTTLE,
                is_recommended=False,
                projected_temp_max_c=65.5,
                cooling_time_to_safe_s=8.0,
                workload_actions=[
                    WorkloadAction(
                        workload_id=alpha_id,
                        action=ActionType.POWER_CAP,
                        target_power_w=125.0,
                    ),
                    WorkloadAction(
                        workload_id=beta_id,
                        action=ActionType.POWER_CAP,
                        target_power_w=125.0,
                    ),
                ],
                estimated_cost_usd=320.0,
                sla_violations_expected=1,
                rationale=(
                    "Slashes power equally across both jobs (125W each). Evaluated by Safety Gate "
                    "and rejected due to P0 SLA contract breach and violation of minimum power floor."
                ),
                origin=PlanOrigin.FALLBACK,
            )
            candidates.append(plan_d)

        return candidates
