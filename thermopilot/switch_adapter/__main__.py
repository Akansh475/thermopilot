"""Executable entrypoint for running the ThermoPilot Switch Agent Adapter."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

from thermopilot.switch_adapter.adapter import ForecastSwitchAdapter
from thermopilot.switch_adapter.config import SwitchAdapterConfig


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


async def main() -> None:
    setup_logging()
    logger = logging.getLogger("thermopilot.switch_adapter")

    try:
        config = SwitchAdapterConfig.from_env()
    except (ValueError, KeyError, OSError) as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    adapter = ForecastSwitchAdapter(config=config)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received. Stopping adapter...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except (NotImplementedError, RuntimeError):
            pass

    adapter_task = asyncio.create_task(adapter.start())

    try:
        await stop_event.wait()
    finally:
        await adapter.stop()
        adapter_task.cancel()
        try:
            await adapter_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
