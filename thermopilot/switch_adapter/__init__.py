"""Switch Agent Protocol adapter package for ThermoPilot Compute Desk."""

from thermopilot.switch_adapter.adapter import ForecastSwitchAdapter
from thermopilot.switch_adapter.base import BaseSwitchAdapter
from thermopilot.switch_adapter.client import (
    SwitchAuthError,
    SwitchClientError,
    SwitchConflictError,
    SwitchEvictedError,
    SwitchHttpClient,
    SwitchNotFoundError,
    SwitchRefusedError,
    SwitchSSEClient,
    SwitchWebSocketClient,
)
from thermopilot.switch_adapter.config import SwitchAdapterConfig
from thermopilot.switch_adapter.planner_adapter import PlannerSwitchAdapter
from thermopilot.switch_adapter.runner import MultiAgentSwitchRunner
from thermopilot.switch_adapter.workload_adapter import WorkloadSwitchAdapter

__all__ = [
    "BaseSwitchAdapter",
    "ForecastSwitchAdapter",
    "MultiAgentSwitchRunner",
    "PlannerSwitchAdapter",
    "SwitchAdapterConfig",
    "SwitchAuthError",
    "SwitchClientError",
    "SwitchConflictError",
    "SwitchEvictedError",
    "SwitchHttpClient",
    "SwitchNotFoundError",
    "SwitchRefusedError",
    "SwitchSSEClient",
    "SwitchWebSocketClient",
    "WorkloadSwitchAdapter",
]
