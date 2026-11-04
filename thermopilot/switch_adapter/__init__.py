"""Switch Agent Protocol adapter package for ThermoPilot Compute Desk."""

from thermopilot.switch_adapter.adapter import ForecastSwitchAdapter
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

__all__ = [
    "ForecastSwitchAdapter",
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
]
