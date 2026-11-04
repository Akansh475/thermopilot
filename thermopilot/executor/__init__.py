"""Simulation Executor & Verification Agent for ThermoPilot Compute Desk.

Exposes the core execution actuator and post-execution verification agent for
applying safety-certified recovery plans to the discrete-time telemetry simulator.
"""

from thermopilot.executor.executor import (
    ExecutionDeniedError,
    ExecutionSafetyError,
    ExecutionValidationError,
    ExecutorError,
    SimulationExecutor,
)
from thermopilot.executor.verifier import (
    PostExecutionVerificationAgent,
    VerificationAgent,
)
from thermopilot.schemas import (
    ExecutionResult,
    ExecutionStatus,
    VerificationCheck,
    VerificationReport,
)

__all__ = [
    "ExecutionDeniedError",
    "ExecutionResult",
    "ExecutionSafetyError",
    "ExecutionStatus",
    "ExecutionValidationError",
    "ExecutorError",
    "PostExecutionVerificationAgent",
    "SimulationExecutor",
    "VerificationAgent",
    "VerificationCheck",
    "VerificationReport",
]
