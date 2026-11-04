"""Interfaces, protocols, configurations, and exception hierarchy for ThermoPilot planners."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from thermopilot.agents.forecast import ForecastResult
from thermopilot.schemas import (
    GPUTelemetry,
    NegotiationEvidence,
    NodeConstraints,
    PlannerResult,
    WorkloadSpec,
)


class PlannerError(Exception):
    """Base exception for all recovery planner errors."""


class LlamaConfigurationError(PlannerError):
    """Raised when Llama provider or model configuration is missing or invalid."""


class LlamaUnavailableError(PlannerError):
    """Raised when Llama service/endpoint is unreachable, times out, or fails."""


class MalformedPlanError(PlannerError):
    """Raised when candidate plan output from LLM is malformed, invalid JSON, or schema-invalid."""


class LlamaClientConfig(BaseModel):
    """Configuration settings for connecting to a Llama LLM provider."""
    model_config = ConfigDict(frozen=True)

    endpoint_url: str | None = Field(
        default=None, description="HTTP/REST endpoint URL for Llama model completions"
    )
    api_key: str | None = Field(
        default=None, description="API authorization key for the Llama service"
    )
    model: str = Field(
        default="llama-3-70b-instruct", description="Identifier of the Llama model"
    )
    timeout_s: float = Field(
        default=2.5, gt=0.0, description="Inference timeout window in seconds"
    )
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="Sampling temperature for deterministic generation"
    )


@runtime_checkable
class LlamaClientProtocol(Protocol):
    """Protocol interface that all Llama client implementations must fulfill."""

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Submit a prompt to Llama and return the raw generated text response."""
        ...


class CallableLlamaClient:
    """Wrapper that adapts a Python callable into a LlamaClientProtocol instance."""

    def __init__(self, fn: Callable[[str, str | None], str]) -> None:
        self._fn = fn

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return self._fn(prompt, system_prompt)


class BaseRecoveryPlanner(ABC):
    """Abstract interface for multi-agent recovery planners."""

    @abstractmethod
    def plan(
        self,
        telemetry: GPUTelemetry,
        workloads: Sequence[WorkloadSpec] | Mapping[str, WorkloadSpec],
        forecast: ForecastResult | None = None,
        node_constraints: NodeConstraints | None = None,
        use_fallback_on_failure: bool = True,
        negotiation_evidence: NegotiationEvidence | None = None,
    ) -> PlannerResult:
        """Produce and certify candidate recovery plans."""
        ...
