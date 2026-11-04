"""Llama-Powered Recovery Planner for ThermoPilot Compute Desk.

Synthesizes telemetry, workload bids, and node constraints to prompt Llama
for strategic recovery candidates. Formats structured prompts, strictly validates
LLM JSON outputs, and tags candidate provenance.
"""

import json
import re
from collections.abc import Mapping, Sequence

from pydantic import ValidationError

from thermopilot.agents.forecast import ForecastResult
from thermopilot.planner.interface import (
    LlamaClientConfig,
    LlamaClientProtocol,
    LlamaConfigurationError,
    LlamaUnavailableError,
    MalformedPlanError,
)
from thermopilot.schemas import (
    GPUTelemetry,
    NegotiationEvidence,
    NodeConstraints,
    PlanOrigin,
    RecoveryPlanOption,
    WorkloadSpec,
)

SYSTEM_PROMPT = """You are ThermoPilot Recovery Planner, an autonomous AI thermal governor for a high-density GPU cluster.
Your responsibility is to analyze active thermal runaway incidents and formulate 3 distinct recovery plans:
1. Plan A (CONSERVATIVE_SHED): Checkpoint-pause lower-priority batch workloads; protect VIP workloads.
2. Plan B (BALANCED_POWER_CAP): Dynamically cap power allocations across jobs to stay within node budget (recommended).
3. Plan C (MIGRATION_OFFLOAD): Checkpoint and migrate flexible workload off-node; preserve VIP workload capacity.

STRICT CONSTRAINTS:
- Steady-state max temperature must not exceed 80.0°C.
- Total node power must not exceed 400.0W.
- P0_CRITICAL workloads must NEVER be preempted or paused, and must never breach SLA.
- Workloads must remain at or above their configured minimum power floor (min_power_w).
- Return ONLY a valid JSON array of objects matching the RecoveryPlanOption schema. No conversational filler.
"""


class LlamaRecoveryPlanner:
    """Llama-powered recovery plan candidate generator."""

    def __init__(
        self,
        client: LlamaClientProtocol | None = None,
        config: LlamaClientConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or LlamaClientConfig()

    def is_configured(self) -> bool:
        """Check whether a valid client or endpoint configuration exists."""
        if self.client is not None:
            return True
        return bool(self.config.api_key or self.config.endpoint_url)

    def generate_candidates(
        self,
        telemetry: GPUTelemetry | None,
        workloads: Sequence[WorkloadSpec] | Mapping[str, WorkloadSpec] | None,
        node_constraints: NodeConstraints | None = None,
        forecast: ForecastResult | None = None,
        negotiation_evidence: NegotiationEvidence | None = None,
    ) -> list[RecoveryPlanOption]:
        """Query Llama to generate recovery plan candidates.

        Args:
            telemetry: Current GPUTelemetry snapshot.
            workloads: Active WorkloadSpec entries.
            node_constraints: Node boundaries.
            forecast: Perception forecast results.
            negotiation_evidence: Optional structured bids from workload-owner agents.

        Returns:
            List of validated RecoveryPlanOption instances tagged with PlanOrigin.LLM.

        Raises:
            LlamaConfigurationError: If Llama client/provider is not configured.
            LlamaUnavailableError: If Llama client call fails or times out.
            MalformedPlanError: If Llama output cannot be parsed into valid plan options.
        """
        if not self.is_configured():
            raise LlamaConfigurationError(
                "Llama provider is not configured. Provide an API key, endpoint URL, or a configured client."
            )

        if self.client is None:
            raise LlamaUnavailableError(
                f"Llama client connection to '{self.config.endpoint_url}' failed: Client transport not initialized."
            )

        prompt = self._format_prompt(
            telemetry, workloads, node_constraints, forecast, negotiation_evidence
        )

        try:
            raw_response = self.client.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
        except Exception as exc:
            raise LlamaUnavailableError(f"Llama inference call failed: {exc}") from exc

        return self._parse_llm_response(raw_response)

    def _format_prompt(
        self,
        telemetry: GPUTelemetry | None,
        workloads: Sequence[WorkloadSpec] | Mapping[str, WorkloadSpec] | None,
        node_constraints: NodeConstraints | None,
        forecast: ForecastResult | None,
        negotiation_evidence: NegotiationEvidence | None = None,
    ) -> str:
        """Construct structured user prompt with real-time operational context."""
        constraints = node_constraints or NodeConstraints()

        # Telemetry context
        if telemetry is not None:
            telem_str = (
                f"GPU Core Temp: {telemetry.temperature_core_c:.1f}°C, "
                f"Junction: {telemetry.temperature_junction_c:.1f}°C, "
                f"Power Draw: {telemetry.power_draw_w:.1f}W / Limit: {telemetry.power_limit_w:.1f}W, "
                f"Ambient: {telemetry.ambient_temp_c:.1f}°C, Fan: {telemetry.fan_speed_pct:.1f}%"
            )
        else:
            telem_str = "Telemetry: [UNAVAILABLE]"

        # Forecast context
        if forecast is not None:
            ttt = f"{forecast.time_to_threshold_s:.1f}s" if forecast.time_to_threshold_s else "N/A"
            fc_str = (
                f"Trend: {forecast.trend.value}, Risk: {forecast.risk_status.value}, "
                f"Time-to-Throttle (t_TTT): {ttt}, Rate: {forecast.rate_c_per_s}°C/s"
            )
        else:
            fc_str = "Forecast: [UNAVAILABLE]"

        # Workload context
        wl_list = []
        if isinstance(workloads, Mapping):
            wl_items = workloads.values()
        elif isinstance(workloads, (list, tuple, set)):
            wl_items = workloads
        else:
            wl_items = []

        for w in wl_items:
            if isinstance(w, WorkloadSpec):
                wl_list.append(
                    f"- {w.workload_id} ({w.name}): Priority={w.priority.value}, "
                    f"Current={w.current_power_w:.0f}W, MinFloor={w.min_power_w:.0f}W, "
                    f"Max={w.max_power_w:.0f}W, Preemptible={w.can_preempt}, "
                    f"Deadline={w.sla_deadline_s:.0f}s, Penalty=${w.sla_penalty_usd_per_s:.2f}/s"
                )

        wl_str = "\n".join(wl_list) if wl_list else "No active workloads"

        # Negotiation evidence context
        neg_str = ""
        if negotiation_evidence is not None and negotiation_evidence.proposals:
            bids = []
            for p in negotiation_evidence.proposals:
                req_p = f"{p.requested_power_w:.0f}W" if p.requested_power_w is not None else "N/A"
                bids.append(
                    f"- [{p.priority.value}] {p.workload_id}: Preferred Action: {p.preferred_action.value} "
                    f"(Requested: {req_p}, SLA Urgency: {p.sla_importance_score:.2f})\n"
                    f"  Rationale: {p.rationale}\n"
                    f"  Trade-off: {p.estimated_trade_off}"
                )
            neg_str = "\n\nWORKLOAD NEGOTIATION EVIDENCE & BIDS:\n" + "\n".join(bids)

        return f"""INCIDENT CONTEXT:
Node Power Budget: {constraints.power_budget_w:.1f}W
Max Safe Core Temp: {constraints.max_temp_c:.1f}°C

CURRENT TELEMETRY:
{telem_str}

THERMAL FORECAST:
{fc_str}

ACTIVE WORKLOADS:
{wl_str}{neg_str}

Formulate 3 recovery plans (PLAN-A, PLAN-B, PLAN-C) adhering to all physical and contractual constraints.
Return JSON array only:"""

    def _parse_llm_response(self, raw_response: str) -> list[RecoveryPlanOption]:
        """Extract, parse, and validate RecoveryPlanOption array from Llama output."""
        if not raw_response or not raw_response.strip():
            raise MalformedPlanError("Llama returned an empty response.")

        clean_text = raw_response.strip()

        # Extract markdown json block if present
        json_match = re.search(r"```(?:json)?\s*(\[\s*\{.*\}\s*\])\s*```", clean_text, re.DOTALL)
        if json_match:
            clean_text = json_match.group(1).strip()
        else:
            # Look for JSON array brackets
            bracket_match = re.search(r"(\[\s*\{.*\}\s*\])", clean_text, re.DOTALL)
            if bracket_match:
                clean_text = bracket_match.group(1).strip()

        try:
            data = json.loads(clean_text)
        except Exception as exc:
            snippet = clean_text[:120].replace("\n", " ")
            raise MalformedPlanError(f"Failed to parse Llama output as JSON: {snippet}... ({exc})") from exc

        if not isinstance(data, list):
            raise MalformedPlanError(
                f"Expected JSON array of recovery plan candidates, got {type(data).__name__}."
            )

        if len(data) == 0:
            raise MalformedPlanError("Llama returned an empty candidate plan list.")

        candidates: list[RecoveryPlanOption] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise MalformedPlanError(
                    f"Candidate at index {i} is not a JSON object: {type(item).__name__}."
                )
            try:
                plan = RecoveryPlanOption.model_validate(item)
                # Stamp provenance origin
                plan = plan.model_copy(update={"origin": PlanOrigin.LLM})
                candidates.append(plan)
            except ValidationError as exc:
                raise MalformedPlanError(
                    f"Candidate plan schema validation failed at index {i}: {exc}"
                ) from exc

        return candidates
