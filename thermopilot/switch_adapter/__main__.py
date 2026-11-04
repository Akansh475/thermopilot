"""Executable entrypoint for running the ThermoPilot Switch Agent Adapters."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from thermopilot.switch_adapter.adapter import ForecastSwitchAdapter
from thermopilot.switch_adapter.config import SwitchAdapterConfig
from thermopilot.switch_adapter.runner import MultiAgentSwitchRunner


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


async def main() -> None:
    setup_logging()
    logger = logging.getLogger("thermopilot.switch_adapter")

    parser = argparse.ArgumentParser(
        description="Run ThermoPilot Switch Agent Adapter(s)"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run only forecast-agent instead of all 3 multi-agent adapters",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to environment file (default: .env)",
    )
    args = parser.parse_args()

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received. Stopping...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except (NotImplementedError, RuntimeError):
            pass

    if args.single:
        try:
            config = SwitchAdapterConfig.from_env(env_file=args.env_file)
        except (ValueError, KeyError, OSError) as exc:
            logger.error("Configuration error: %s", exc)
            sys.exit(1)

        runner = ForecastSwitchAdapter(config=config)
    else:
        try:
            runner = MultiAgentSwitchRunner.from_env(env_file=args.env_file)
        except (ValueError, KeyError, OSError) as exc:
            logger.error("Multi-agent configuration error: %s", exc)
            sys.exit(1)

    runner_task = asyncio.create_task(runner.start())

    try:
        await stop_event.wait()
    finally:
        await runner.stop()
        runner_task.cancel()
        try:
            await runner_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
