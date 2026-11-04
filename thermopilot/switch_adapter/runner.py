"""Multi-Agent Switch Runtime for ThermoPilot Compute Desk.

Coordinates independent WebSocket connections, room event routing, and negotiation lifecycles
for:
1. forecast-agent (perception & thermal anomaly alerts)
2. workload-agent (tenant SLA & electrical floor advocacy)
3. planner-agent (candidate plan formulation & SafetyGate certification)
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from thermopilot.switch_adapter.adapter import ForecastSwitchAdapter
from thermopilot.switch_adapter.config import SwitchAdapterConfig
from thermopilot.switch_adapter.planner_adapter import PlannerSwitchAdapter
from thermopilot.switch_adapter.workload_adapter import WorkloadSwitchAdapter

logger = logging.getLogger(__name__)


class MultiAgentSwitchRunner:
    """Orchestrates concurrent, independent WebSocket connections for all three Switch agents."""

    def __init__(
        self,
        forecast_adapter: ForecastSwitchAdapter,
        workload_adapter: WorkloadSwitchAdapter,
        planner_adapter: PlannerSwitchAdapter,
    ) -> None:
        self.forecast_adapter = forecast_adapter
        self.workload_adapter = workload_adapter
        self.planner_adapter = planner_adapter

        self._tasks: list[asyncio.Task[Any]] = []
        self._is_running = False

    @property
    def is_running(self) -> bool:
        """True if the runner has active running adapters."""
        return self._is_running

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> MultiAgentSwitchRunner:
        """Instantiate runner from environment variables and credentials in .env."""
        cfg_f = SwitchAdapterConfig.from_env(role="forecast", env_file=env_file)
        cfg_w = SwitchAdapterConfig.from_env(role="workload", env_file=env_file)
        cfg_p = SwitchAdapterConfig.from_env(role="planner", env_file=env_file)

        return cls(
            forecast_adapter=ForecastSwitchAdapter(config=cfg_f),
            workload_adapter=WorkloadSwitchAdapter(config=cfg_w),
            planner_adapter=PlannerSwitchAdapter(config=cfg_p),
        )

    async def start(self) -> None:
        """Start all three adapters concurrently."""
        self._is_running = True
        logger.info(
            "Starting ThermoPilot Multi-Agent Switch Runner (3 agents: forecast, workload, planner)"
        )

        self._tasks = [
            asyncio.create_task(
                self.forecast_adapter.start(), name="task-forecast-adapter"
            ),
            asyncio.create_task(
                self.workload_adapter.start(), name="task-workload-adapter"
            ),
            asyncio.create_task(
                self.planner_adapter.start(), name="task-planner-adapter"
            ),
        ]

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop all three adapters and cancel tasks."""
        if not self._is_running:
            return

        self._is_running = False
        logger.info("Stopping ThermoPilot Multi-Agent Switch Runner...")

        stop_coros = [
            self.forecast_adapter.stop(),
            self.workload_adapter.stop(),
            self.planner_adapter.stop(),
        ]
        await asyncio.gather(*stop_coros, return_exceptions=True)

        for task in self._tasks:
            if not task.done():
                task.cancel()

        self._tasks.clear()
        logger.info("Multi-Agent Switch Runner stopped cleanly.")

    async def trigger_simulated_incident(
        self,
        room_id: str,
        incident_id: str | None = None,
        temp_c: float = 82.5,
    ) -> str:
        """Trigger and broadcast a simulated thermal runaway anomaly from forecast-agent."""
        logger.info(
            "Dispatching simulated thermal incident to room %s (incident_id=%s, temp=%.1f°C)",
            room_id,
            incident_id,
            temp_c,
        )
        return await self.forecast_adapter.trigger_simulated_incident(
            room_id=room_id,
            incident_id=incident_id,
            temp_c=temp_c,
        )
