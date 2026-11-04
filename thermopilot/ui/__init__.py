"""ThermoPilot Compute Desk UI and Demo Module.

Provides:
- Terminal Compute Desk (ANSI formatted control room interface)
- Local Web Control Room Dashboard (zero-dependency HTTP/SVG interface)
- Repeatable Demo CLI Runner (60-second hackathon scenario and test driver)
"""

from typing import Any

__all__ = ["ComputeDesk", "DemoRunner"]


def __getattr__(name: str) -> Any:
    """Lazy-load module attributes to prevent runpy cyclic import warnings."""
    if name == "ComputeDesk":
        from thermopilot.ui.desk import ComputeDesk
        return ComputeDesk
    if name == "DemoRunner":
        from thermopilot.ui.demo import DemoRunner
        return DemoRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
