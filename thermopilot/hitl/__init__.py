"""Human-in-the-Loop (HITL) Switch approval and governance module for ThermoPilot Compute Desk."""

from thermopilot.hitl.switch import (
    InvalidApprovalError,
    MockConsoleSwitchClient,
    SwitchApprovalDesk,
    SwitchClient,
    SwitchConfigurationError,
    SwitchError,
    SwitchTimeoutError,
    WebhookSwitchClient,
    WebhookSwitchClientConfig,
    apply_plan_to_simulator_if_authorized,
)
from thermopilot.schemas import (
    ApprovalWorkflowResult,
    CandidatePlanSummary,
    DecisionCard,
    WorkflowStatus,
)

__all__ = [
    "ApprovalWorkflowResult",
    "CandidatePlanSummary",
    "DecisionCard",
    "InvalidApprovalError",
    "MockConsoleSwitchClient",
    "SwitchApprovalDesk",
    "SwitchClient",
    "SwitchConfigurationError",
    "SwitchError",
    "SwitchTimeoutError",
    "WebhookSwitchClient",
    "WebhookSwitchClientConfig",
    "WorkflowStatus",
    "apply_plan_to_simulator_if_authorized",
]
