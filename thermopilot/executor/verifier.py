"""Post-Execution Verification Agent for ThermoPilot Compute Desk.

Monitors physical temperature decay and SLA preservation in the telemetry simulator
post-execution. Verifies whether thermal recovery reaches the target safe threshold
(<= 75.0°C) at or above the target cooling rate (>= 0.5°C/s simulated).

GOVERNANCE & REPORTING HONESTY:
1. Honest Reporting: Never fabricates measurements or forces a successful outcome.
   If cooling stalls, power exceeds budget, or SLAs degrade, reports failures explicitly.
2. Simulated Labeling: Clearly marks all findings, metrics, and trajectories as SIMULATED.
3. Multi-Check Verification: Validates safe recovery target (<= 75°C), cooling rate
   (>= 0.5°C/s), physical ceiling (<= 80°C), hardware trip (< 85°C), node power budget
   (<= 400W), workload floors, and P0 SLA protections over up to 30 discrete simulation ticks.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from thermopilot.schemas import (
    ExecutionResult,
    GPUTelemetry,
    NodeConstraints,
    RecoveryPlanOption,
    VerificationCheck,
    VerificationReport,
    WorkloadPriority,
    WorkloadSpec,
)
from thermopilot.simulator.telemetry import TelemetrySimulator


class VerificationAgent:
    """Post-execution verification agent validating simulated thermal recovery trajectories."""

    def __init__(
        self,
        target_temp_c: float = 75.0,
        target_cooling_rate_c_per_s: float = 0.5,
        ceiling_temp_c: float = 80.0,
        critical_temp_c: float = 85.0,
        default_max_ticks: int = 30,
    ) -> None:
        """Initialize VerificationAgent.

        Args:
            target_temp_c: Target safe temperature ceiling in Celsius (default: 75.0°C).
            target_cooling_rate_c_per_s: Target cooling rate in °C/s simulated (default: 0.5°C/s).
            ceiling_temp_c: Deterministic Safety Gate ceiling in Celsius (default: 80.0°C).
            critical_temp_c: Hardware trip / shutdown threshold in Celsius (default: 85.0°C).
            default_max_ticks: Default number of simulation ticks to monitor post-execution (default: 30).
        """
        self.target_temp_c = target_temp_c
        self.target_cooling_rate_c_per_s = target_cooling_rate_c_per_s
        self.ceiling_temp_c = ceiling_temp_c
        self.critical_temp_c = critical_temp_c
        self.default_max_ticks = default_max_ticks

    @staticmethod
    def _normalize_workloads(
        workloads: Sequence[WorkloadSpec] | Mapping[str, WorkloadSpec] | Any,
    ) -> dict[str, WorkloadSpec]:
        """Normalize workload specifications into a mapping keyed by workload_id."""
        workload_map: dict[str, WorkloadSpec] = {}
        if isinstance(workloads, Mapping):
            for item in workloads.values():
                if isinstance(item, WorkloadSpec):
                    workload_map[item.workload_id] = item
        elif isinstance(workloads, (list, tuple, set)):
            for item in workloads:
                if isinstance(item, WorkloadSpec):
                    workload_map[item.workload_id] = item
        return workload_map

    def verify(
        self,
        simulator: TelemetrySimulator,
        plan: RecoveryPlanOption,
        workloads: Sequence[WorkloadSpec] | Mapping[str, WorkloadSpec],
        execution_result: ExecutionResult | None = None,
        incident_id: str | None = None,
        max_ticks: int | None = None,
        target_temp_c: float | None = None,
        target_cooling_rate_c_per_s: float | None = None,
        ceiling_temp_c: float | None = None,
        critical_temp_c: float | None = None,
        node_constraints: NodeConstraints | None = None,
        stop_on_target: bool = False,
    ) -> VerificationReport:
        """Advance simulator for up to 30 ticks and verify thermal recovery against safety contracts.

        Args:
            simulator: TelemetrySimulator on which the plan was executed.
            plan: RecoveryPlanOption that was executed.
            workloads: Active workload specifications.
            execution_result: Optional prior ExecutionResult from SimulationExecutor.
            incident_id: Optional incident identifier.
            max_ticks: Maximum simulation ticks to advance and monitor (default: 30).
            target_temp_c: Override for safe recovery temperature target (default: 75.0°C).
            target_cooling_rate_c_per_s: Override for target cooling rate (default: 0.5°C/s).
            ceiling_temp_c: Override for temperature ceiling limit (default: 80.0°C).
            critical_temp_c: Override for hardware trip limit (default: 85.0°C).
            node_constraints: Optional node constraints for power/temp thresholds.
            stop_on_target: If True, stops advancing immediately upon reaching safe target.

        Returns:
            VerificationReport containing measured trajectory, individual check evaluations,
            and honest recovery outcome verdict.

        Raises:
            TypeError: If simulator is not a TelemetrySimulator instance.
        """
        if not isinstance(simulator, TelemetrySimulator):
            raise TypeError(
                f"VerificationAgent strictly monitors TelemetrySimulator instances. "
                f"Received invalid target of type '{type(simulator).__name__}'."
            )

        # Thresholds
        eff_target_temp = target_temp_c if target_temp_c is not None else self.target_temp_c
        eff_target_rate = (
            target_cooling_rate_c_per_s
            if target_cooling_rate_c_per_s is not None
            else self.target_cooling_rate_c_per_s
        )
        eff_ceiling_temp = (
            node_constraints.max_temp_c
            if node_constraints is not None
            else (ceiling_temp_c if ceiling_temp_c is not None else self.ceiling_temp_c)
        )
        eff_power_budget = (
            node_constraints.power_budget_w
            if node_constraints is not None
            else simulator.power_budget_w
        )
        eff_critical_temp = critical_temp_c if critical_temp_c is not None else self.critical_temp_c
        effective_max_ticks = max_ticks if max_ticks is not None else self.default_max_ticks

        target_incident_id = (
            incident_id
            or (execution_result.incident_id if execution_result else None)
            or "INC-SIMULATED-001"
        )
        plan_id = (
            plan.plan_id
            if plan is not None
            else (execution_result.plan_id if execution_result else "UNKNOWN-PLAN")
        )

        workload_map = self._normalize_workloads(workloads)

        # -------------------------------------------------------------------
        # 1. Record Initial State Before Window Advance
        # -------------------------------------------------------------------
        if execution_result is not None and execution_result.initial_telemetry is not None:
            initial_temp_c = execution_result.initial_telemetry.temperature_core_c
        else:
            initial_temp_c = simulator.core_temp_c

        start_sim_time_s = simulator.sim_time_s
        immediate_temp_c = simulator.core_temp_c

        # Trajectory list starting with immediate temperature post-plan
        temperature_trajectory: list[float] = [immediate_temp_c]
        ticks_monitored = 0
        reached_target = immediate_temp_c <= eff_target_temp
        ticks_to_target: int | None = 0 if reached_target else None
        sim_time_to_target_s: float | None = 0.0 if reached_target else None

        # -------------------------------------------------------------------
        # 2. Advance Simulator for up to 30 Ticks
        # -------------------------------------------------------------------
        for tick in range(1, effective_max_ticks + 1):
            telem: GPUTelemetry = simulator.step()
            ticks_monitored += 1
            current_temp = telem.temperature_core_c
            temperature_trajectory.append(current_temp)

            if current_temp <= eff_target_temp and not reached_target:
                reached_target = True
                ticks_to_target = tick
                sim_time_to_target_s = round(simulator.sim_time_s - start_sim_time_s, 2)
                if stop_on_target:
                    break

        # -------------------------------------------------------------------
        # 3. Compute Physical Decay & Trajectory Metrics
        # -------------------------------------------------------------------
        final_temp_c = simulator.core_temp_c
        min_temp_c = min(temperature_trajectory)
        max_temp_c = max(temperature_trajectory)
        temp_delta_c = round(initial_temp_c - final_temp_c, 2)
        total_sim_duration_s = round(simulator.sim_time_s - start_sim_time_s, 2)

        if min_temp_c <= eff_target_temp:
            reached_target = True

        approaching_target = final_temp_c < initial_temp_c

        # Calculate observed cooling rate in °C / simulated second
        if total_sim_duration_s > 0.0:
            if reached_target and sim_time_to_target_s is not None and sim_time_to_target_s > 0.0:
                # Active decay rate from initial to reaching target
                target_reading = (
                    temperature_trajectory[ticks_to_target]
                    if (ticks_to_target is not None and ticks_to_target < len(temperature_trajectory))
                    else eff_target_temp
                )
                rate_to_target = (initial_temp_c - target_reading) / sim_time_to_target_s
                rate_overall = (initial_temp_c - final_temp_c) / total_sim_duration_s
                observed_cooling_rate = max(rate_to_target, rate_overall)
            else:
                observed_cooling_rate = (initial_temp_c - final_temp_c) / total_sim_duration_s
        else:
            observed_cooling_rate = 0.0

        observed_cooling_rate = max(0.0, round(observed_cooling_rate, 3))
        meets_cooling_rate_target = (observed_cooling_rate >= eff_target_rate) and (temp_delta_c > 0.0)

        # -------------------------------------------------------------------
        # 4. Perform Deterministic Verification Checks
        # -------------------------------------------------------------------
        checks: list[VerificationCheck] = []
        failure_reasons: list[str] = []

        # Check 1: Safe recovery target (<= 75.0°C)
        target_check_passed = reached_target and (final_temp_c <= eff_target_temp)
        if target_check_passed:
            target_detail = (
                f"Core temperature reached safe target ({final_temp_c:.1f}°C <= {eff_target_temp:.1f}°C) "
                f"in {sim_time_to_target_s or 0.0:.1f}s ({ticks_to_target or 0} ticks)."
            )
        else:
            target_detail = (
                f"Core temperature did not achieve safe target: final {final_temp_c:.1f}°C, "
                f"minimum {min_temp_c:.1f}°C exceeds target {eff_target_temp:.1f}°C."
            )
            failure_reasons.append(
                f"Thermal recovery target ({eff_target_temp:.1f}°C) was not achieved; lowest observed: {min_temp_c:.1f}°C (final: {final_temp_c:.1f}°C)."
            )

        checks.append(
            VerificationCheck(
                name="CHECK_SAFE_RECOVERY_TARGET",
                passed=target_check_passed,
                observed_value=final_temp_c,
                threshold_value=eff_target_temp,
                detail=target_detail,
            )
        )

        # Check 2: Observed cooling rate (>= 0.5°C/s target)
        rate_check_passed = meets_cooling_rate_target
        if rate_check_passed:
            rate_detail = (
                f"Observed cooling rate {observed_cooling_rate:.2f}°C/s meets or exceeds "
                f"target {eff_target_rate:.2f}°C/s."
            )
        else:
            rate_detail = (
                f"Observed cooling rate {observed_cooling_rate:.2f}°C/s is below "
                f"required {eff_target_rate:.2f}°C/s target."
            )
            failure_reasons.append(
                f"Observed cooling rate ({observed_cooling_rate:.2f}°C/s) failed to meet target ({eff_target_rate:.2f}°C/s)."
            )

        checks.append(
            VerificationCheck(
                name="CHECK_COOLING_RATE",
                passed=rate_check_passed,
                observed_value=observed_cooling_rate,
                threshold_value=eff_target_rate,
                detail=rate_detail,
            )
        )

        # Check 3: Safety temperature ceiling (<= 80.0°C)
        ceiling_passed = final_temp_c <= eff_ceiling_temp
        if ceiling_passed:
            ceiling_detail = f"Final temperature ({final_temp_c:.1f}°C) satisfies safety ceiling ({eff_ceiling_temp:.1f}°C)."
        else:
            ceiling_detail = f"Final temperature ({final_temp_c:.1f}°C) exceeds safety ceiling ({eff_ceiling_temp:.1f}°C)."
            failure_reasons.append(
                f"Final temperature ({final_temp_c:.1f}°C) exceeds safety ceiling ({eff_ceiling_temp:.1f}°C)."
            )

        checks.append(
            VerificationCheck(
                name="CHECK_TEMPERATURE_CEILING",
                passed=ceiling_passed,
                observed_value=final_temp_c,
                threshold_value=eff_ceiling_temp,
                detail=ceiling_detail,
            )
        )

        # Check 4: Critical hardware trip threshold (< 85.0°C peak)
        trip_passed = max_temp_c < eff_critical_temp
        if trip_passed:
            trip_detail = f"Peak temperature ({max_temp_c:.1f}°C) remained below critical hardware trip threshold ({eff_critical_temp:.1f}°C)."
        else:
            trip_detail = f"Peak temperature ({max_temp_c:.1f}°C) reached or exceeded hardware trip threshold ({eff_critical_temp:.1f}°C)."
            failure_reasons.append(
                f"Peak temperature ({max_temp_c:.1f}°C) reached or exceeded hardware trip limit ({eff_critical_temp:.1f}°C)."
            )

        checks.append(
            VerificationCheck(
                name="CHECK_HARDWARE_TRIP_LIMIT",
                passed=trip_passed,
                observed_value=max_temp_c,
                threshold_value=eff_critical_temp,
                detail=trip_detail,
            )
        )

        # Check 5: Node electrical power budget (<= 400.0W)
        effective_power_draw_w = simulator.effective_power_draw_w
        power_budget_passed = effective_power_draw_w <= eff_power_budget
        if power_budget_passed:
            power_detail = f"Total node electrical power ({effective_power_draw_w:.1f}W) within budget ({eff_power_budget:.1f}W)."
        else:
            power_detail = f"Total node power ({effective_power_draw_w:.1f}W) exceeds budget ({eff_power_budget:.1f}W)."
            failure_reasons.append(
                f"Total node power ({effective_power_draw_w:.1f}W) exceeds budget ({eff_power_budget:.1f}W)."
            )

        checks.append(
            VerificationCheck(
                name="CHECK_NODE_POWER_BUDGET",
                passed=power_budget_passed,
                observed_value=effective_power_draw_w,
                threshold_value=eff_power_budget,
                detail=power_detail,
            )
        )

        # Check 6: Active workload power floors
        floor_violations: list[str] = []
        for wid, spec in workload_map.items():
            curr_p = simulator.get_workload_power(wid)
            if curr_p > 0.0 and curr_p < spec.min_power_w:
                floor_violations.append(
                    f"'{wid}' power ({curr_p:.1f}W) below minimum floor ({spec.min_power_w:.1f}W)"
                )

        floor_passed = len(floor_violations) == 0
        if floor_passed:
            floor_detail = "All active workloads satisfy configured minimum power floors."
        else:
            floor_detail = f"Workloads violating power floors: {'; '.join(floor_violations)}."
            failure_reasons.append(floor_detail)

        checks.append(
            VerificationCheck(
                name="CHECK_WORKLOAD_POWER_FLOORS",
                passed=floor_passed,
                observed_value="VALID" if floor_passed else "FLOOR_VIOLATION",
                threshold_value="VALID",
                detail=floor_detail,
            )
        )

        # Check 7: P0 SLA protection
        p0_issues: list[str] = []
        for wid, spec in workload_map.items():
            if spec.priority == WorkloadPriority.P0_CRITICAL or not spec.can_preempt:
                curr_p = simulator.get_workload_power(wid)
                if curr_p <= 0.0:
                    p0_issues.append(f"P0 workload '{wid}' was preempted or removed from simulator")
                elif curr_p < spec.min_power_w:
                    p0_issues.append(f"P0 workload '{wid}' power ({curr_p:.1f}W) below min floor ({spec.min_power_w:.1f}W)")

        if plan.sla_violations_expected > 0:
            p0_issues.append(f"Plan specifies {plan.sla_violations_expected} expected SLA violations")

        p0_passed = len(p0_issues) == 0
        if p0_passed:
            p0_detail = "P0 workload SLA integrity preserved (active, above floor, 0 SLA penalties)."
        else:
            p0_detail = f"P0 SLA protection violations: {'; '.join(p0_issues)}."
            failure_reasons.append(p0_detail)

        checks.append(
            VerificationCheck(
                name="CHECK_P0_SLA_PROTECTION",
                passed=p0_passed,
                observed_value="PRESERVED" if p0_passed else "BREACHED",
                threshold_value="PRESERVED",
                detail=p0_detail,
            )
        )

        # -------------------------------------------------------------------
        # 5. Formulate Honest Aggregate Verdict & Narrative
        # -------------------------------------------------------------------
        recovery_successful = all(c.passed for c in checks) and len(failure_reasons) == 0

        if recovery_successful:
            summary = (
                f"[SIMULATED ENVIRONMENT] Thermal envelope successfully restored. "
                f"Core temperature cooled from {initial_temp_c:.1f}°C to {final_temp_c:.1f}°C "
                f"(delta: -{temp_delta_c:.1f}°C) over {total_sim_duration_s:.1f}s simulated ({ticks_monitored} ticks). "
                f"Safe target (<= {eff_target_temp:.1f}°C) reached in {sim_time_to_target_s:.1f}s "
                f"with cooling rate {observed_cooling_rate:.2f}°C/s (target: >= {eff_target_rate:.2f}°C/s). "
                f"Node power ({effective_power_draw_w:.1f}W), workload floors, and P0 SLA protections fully satisfied."
            )
        else:
            summary = (
                f"[SIMULATED ENVIRONMENT] Thermal recovery FAILED. Violations detected: "
                f"{'; '.join(failure_reasons)}. Final temp: {final_temp_c:.1f}°C (target: <= {eff_target_temp:.1f}°C), "
                f"cooling rate: {observed_cooling_rate:.2f}°C/s (target: >= {eff_target_rate:.2f}°C/s)."
            )

        return VerificationReport(
            incident_id=target_incident_id,
            plan_id=plan_id,
            timestamp=datetime.now(timezone.utc),
            recovery_successful=recovery_successful,
            is_simulated=True,
            initial_temp_c=initial_temp_c,
            final_temp_c=final_temp_c,
            min_temp_c=min_temp_c,
            max_temp_c=max_temp_c,
            temp_delta_c=temp_delta_c,
            target_temp_c=eff_target_temp,
            reached_target=reached_target,
            approaching_target=approaching_target,
            ticks_to_target=ticks_to_target,
            sim_time_to_target_s=sim_time_to_target_s,
            observed_cooling_rate_c_per_s=observed_cooling_rate,
            target_cooling_rate_c_per_s=eff_target_rate,
            meets_cooling_rate_target=meets_cooling_rate_target,
            checks=checks,
            failure_reasons=failure_reasons,
            temperature_trajectory=temperature_trajectory,
            ticks_monitored=ticks_monitored,
            simulated_duration_s=total_sim_duration_s,
            summary=summary,
        )


# Backward-compatible alias
PostExecutionVerificationAgent = VerificationAgent
