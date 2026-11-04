"""Orchestrating Recovery Planner for ThermoPilot Compute Desk.

Coordinates candidate generation (via Llama with deterministic algorithmic fallback)
and validates all candidates through the deterministic Safety Gate. Guarantees clear
separation between AI candidate proposal and hard safety certification.
"""

from collections.abc import Mapping, Sequence

from thermopilot.agents.forecast import ForecastResult
from thermopilot.agents.workload_owner import collect_workload_proposals
from thermopilot.planner.fallback import DeterministicFallbackPlanner
from thermopilot.planner.interface import (
    BaseRecoveryPlanner,
    PlannerError,
)
from thermopilot.planner.llama import LlamaRecoveryPlanner
from thermopilot.safety.gate import SafetyGate
from thermopilot.schemas import (
    EvaluatedPlanCandidate,
    GPUTelemetry,
    NegotiationEvidence,
    NodeConstraints,
    PlannerResult,
    PlanOrigin,
    RecoveryPlanOption,
    WorkloadSpec,
)


class RecoveryPlanner(BaseRecoveryPlanner):
    """Integrated recovery planner coordinating candidate generation and safety evaluation."""

    def __init__(
        self,
        safety_gate: SafetyGate | None = None,
        llama_planner: LlamaRecoveryPlanner | None = None,
        fallback_planner: DeterministicFallbackPlanner | None = None,
        node_constraints: NodeConstraints | None = None,
    ) -> None:
        self.safety_gate = safety_gate or SafetyGate(node_constraints)
        self.llama_planner = llama_planner
        self.fallback_planner = fallback_planner or DeterministicFallbackPlanner(node_constraints)
        self.node_constraints = node_constraints or self.safety_gate.node_constraints

    def plan(
        self,
        telemetry: GPUTelemetry,
        workloads: Sequence[WorkloadSpec] | Mapping[str, WorkloadSpec],
        forecast: ForecastResult | None = None,
        node_constraints: NodeConstraints | None = None,
        use_fallback_on_failure: bool = True,
        include_unsafe_candidate: bool = False,
        negotiation_evidence: NegotiationEvidence | None = None,
    ) -> PlannerResult:
        """Produce, certify, and package candidate recovery plans.

        Args:
            telemetry: Real-time GPU telemetry reading.
            workloads: Active workload specifications.
            forecast: Perception forecast indicating thermal trajectory.
            node_constraints: Node power and temperature envelope constraints.
            use_fallback_on_failure: Whether to seamlessly invoke deterministic fallback
                if Llama is unconfigured, unreachable, or returns malformed output.
            include_unsafe_candidate: Whether to include Plan D in fallback candidate list
                to demonstrate and test explicit Safety Gate rejection.
            negotiation_evidence: Optional structured proposals and bids from workload agents.
                If None and active workloads are provided, bids are collected automatically.

        Returns:
            PlannerResult containing evaluated candidates with origin tracking, gate certification,
            and associated negotiation evidence.
        """
        active_constraints = node_constraints or self.node_constraints
        candidates: list[RecoveryPlanOption] = []
        generation_source: PlanOrigin | None = None
        generation_error: str | None = None

        # Collect advisory negotiation evidence if not explicitly passed
        active_evidence = negotiation_evidence
        if active_evidence is None and workloads:
            try:
                active_evidence = collect_workload_proposals(workloads, telemetry, forecast)
            except (TypeError, ValueError, AttributeError):
                active_evidence = None

        # -------------------------------------------------------------------
        # 1. Candidate Generation Phase (Llama or Deterministic Fallback)
        # -------------------------------------------------------------------
        if self.llama_planner is not None:
            try:
                candidates = self.llama_planner.generate_candidates(
                    telemetry=telemetry,
                    workloads=workloads,
                    node_constraints=active_constraints,
                    forecast=forecast,
                    negotiation_evidence=active_evidence,
                )
                generation_source = PlanOrigin.LLM
            except PlannerError as exc:
                generation_error = str(exc)
                if not use_fallback_on_failure:
                    return PlannerResult(
                        success=False,
                        candidates=[],
                        generation_source=None,
                        error_message=f"Llama candidate generation failed: {exc}",
                        negotiation_evidence=active_evidence,
                    )
                # Controlled fallback to deterministic candidate generator
                candidates = self.fallback_planner.generate_candidates(
                    telemetry=telemetry,
                    workloads=workloads,
                    node_constraints=active_constraints,
                    forecast=forecast,
                    include_unsafe_candidate=include_unsafe_candidate,
                    negotiation_evidence=active_evidence,
                )
                generation_source = PlanOrigin.FALLBACK
        else:
            if not use_fallback_on_failure:
                return PlannerResult(
                    success=False,
                    candidates=[],
                    generation_source=None,
                    error_message=(
                        "Llama planner is not configured or provided, and deterministic "
                        "fallback candidate generation is disabled."
                    ),
                    negotiation_evidence=active_evidence,
                )
            candidates = self.fallback_planner.generate_candidates(
                telemetry=telemetry,
                workloads=workloads,
                node_constraints=active_constraints,
                forecast=forecast,
                include_unsafe_candidate=include_unsafe_candidate,
                negotiation_evidence=active_evidence,
            )
            generation_source = PlanOrigin.FALLBACK

        # -------------------------------------------------------------------
        # 2. Deterministic Safety Gate Evaluation Phase
        # -------------------------------------------------------------------
        # AI candidate generation is strictly separated from safety certification.
        # Every candidate MUST pass through the deterministic gate.
        evaluated_candidates: list[EvaluatedPlanCandidate] = []
        for candidate in candidates:
            # Ensure candidate has origin tag
            plan_with_origin = (
                candidate
                if candidate.origin is not None
                else candidate.model_copy(update={"origin": generation_source})
            )

            safety_result = self.safety_gate.evaluate(
                plan=plan_with_origin,
                telemetry=telemetry,
                workloads=workloads,
                node_constraints=active_constraints,
            )

            evaluated_candidates.append(
                EvaluatedPlanCandidate(
                    plan=plan_with_origin,
                    origin=generation_source,
                    safety_result=safety_result,
                )
            )

        return PlannerResult(
            success=True,
            candidates=evaluated_candidates,
            generation_source=generation_source,
            error_message=generation_error,
            negotiation_evidence=active_evidence,
        )
