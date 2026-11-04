"""Unit tests for the Post-Execution Verification Agent.

Verifies:
1. Recording actual measurements before and after plan execution.
2. Advancing simulator for up to 30 ticks and recording temperature trajectory.
3. Measuring whether temperature reaches or approaches the 75.0°C recovery target.
4. Calculating observed cooling rate and verifying whether it meets the >= 0.5°C/s target.
5. Checking node electrical power budget, workload power floors, P0 SLA protections, and temperature limits.
6. Honest failure reporting: never fabricates measurements or forces successful outcomes.
7. Clear labeling of all findings as simulated (is_simulated=True, [SIMULATED ENVIRONMENT]).
8. Early stopping option (stop_on_target=True).
9. End-to-end integration: Executor execution followed by VerificationAgent analysis.
"""

import unittest
from datetime import datetime, timezone

from thermopilot.executor.executor import SimulationExecutor
from thermopilot.executor.verifier import (
    PostExecutionVerificationAgent,
    VerificationAgent,
)
from thermopilot.safety.gate import SafetyGate
from thermopilot.schemas import (
    ActionType,
    ApprovalAction,
    ApprovalDecision,
    ExecutionResult,
    ExecutionStatus,
    NodeConstraints,
    PlanOrigin,
    RecoveryPlanOption,
    StrategyType,
    WorkloadAction,
    WorkloadPriority,
    WorkloadSpec,
)
from thermopilot.simulator.telemetry import SimulatorConfig, TelemetrySimulator


def make_test_workloads() -> dict[str, WorkloadSpec]:
    """Helper returning standard Alpha (P0) and Beta (P2) workloads."""
    alpha = WorkloadSpec(
        workload_id="wl-alpha-inference",
        name="Financial Trading Inference",
        owner="Trading-Prod",
        priority=WorkloadPriority.P0_CRITICAL,
        current_power_w=230.0,
        min_power_w=200.0,
        max_power_w=250.0,
        can_preempt=False,
        can_power_cap=True,
        sla_deadline_s=90.0,
        sla_penalty_usd_per_s=5.50,
    )
    beta = WorkloadSpec(
        workload_id="wl-beta-training",
        name="Diffusion Fine-Tuning",
        owner="Research-Exploratory",
        priority=WorkloadPriority.P2_FLEXIBLE,
        current_power_w=150.0,
        min_power_w=100.0,
        max_power_w=180.0,
        can_preempt=True,
        can_power_cap=True,
        sla_deadline_s=600.0,
        sla_penalty_usd_per_s=0.20,
    )
    return {alpha.workload_id: alpha, beta.workload_id: beta}


class TestVerificationAgent(unittest.TestCase):
    """Test suite for the Post-Execution Verification Agent."""

    def setUp(self) -> None:
        self.incident_id = "INC-20261027-001"
        self.workloads = make_test_workloads()
        self.verifier = VerificationAgent(
            target_temp_c=75.0,
            target_cooling_rate_c_per_s=0.5,
            ceiling_temp_c=80.0,
            critical_temp_c=85.0,
            default_max_ticks=30,
        )

        # Standard successful recovery plan (Plan A: 220W Alpha, Beta paused)
        self.plan_a = RecoveryPlanOption(
            plan_id="PLAN-A",
            name="Conservative Workload Shed",
            strategy_type=StrategyType.CONSERVATIVE_SHED,
            is_recommended=False,
            projected_temp_max_c=68.0,
            cooling_time_to_safe_s=14.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=220.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.PAUSE),
            ],
            estimated_cost_usd=45.0,
            sla_violations_expected=0,
            rationale="Pause flexible training job Beta; allocate safe power to VIP Alpha.",
            origin=PlanOrigin.FALLBACK,
        )

    # -----------------------------------------------------------------------
    # 1. Successful Recovery Verification Tests
    # -----------------------------------------------------------------------

    def test_successful_thermal_recovery_verification(self) -> None:
        """Verify successful recovery trajectory reaching safe target <= 75°C with rate >= 0.5°C/s."""
        # Initial simulator under anomaly at 81.5°C
        sim = TelemetrySimulator(
            SimulatorConfig(
                initial_temp_c=81.5,
                time_scale=2.5,
                noise_std=0.0,
                default_workloads={"wl-alpha-inference": 220.0},  # Plan A applied: Beta paused
            )
        )
        sim.trigger_anomaly(32.0)

        # At 220W with 32°C ambient: T_ss = 32 + (220 * 0.14) = 62.8°C
        # Core temperature should smoothly decay well below 75.0°C
        report = self.verifier.verify(
            simulator=sim,
            plan=self.plan_a,
            workloads=self.workloads,
            incident_id=self.incident_id,
            max_ticks=30,
        )

        self.assertTrue(report.recovery_successful)
        self.assertTrue(report.is_simulated)
        self.assertEqual(report.plan_id, "PLAN-A")
        self.assertEqual(report.incident_id, self.incident_id)

        # Trajectory checks
        self.assertEqual(report.initial_temp_c, 81.5)
        self.assertLess(report.final_temp_c, 75.0)
        self.assertTrue(report.reached_target)
        self.assertIsNotNone(report.ticks_to_target)
        self.assertIsNotNone(report.sim_time_to_target_s)
        self.assertLessEqual(report.ticks_to_target, 10)  # Reaches in ~3-4 ticks

        # Cooling rate checks: must meet >= 0.5°C/s
        self.assertTrue(report.meets_cooling_rate_target)
        self.assertGreaterEqual(report.observed_cooling_rate_c_per_s, 0.5)

        # 30 ticks monitored
        self.assertEqual(report.ticks_monitored, 30)
        self.assertEqual(len(report.temperature_trajectory), 31)  # initial + 30 ticks
        self.assertAlmostEqual(report.simulated_duration_s, 75.0, delta=0.5)  # 30 * 2.5s = 75.0s

        # All 7 checks must pass
        self.assertEqual(len(report.checks), 7)
        self.assertTrue(all(c.passed for c in report.checks))
        self.assertEqual(len(report.failure_reasons), 0)

        # Clear simulated labeling
        self.assertIn("[SIMULATED ENVIRONMENT]", report.summary)
        self.assertIn("successfully restored", report.summary.lower())

    def test_alias_class_matches_verification_agent(self) -> None:
        """Verify PostExecutionVerificationAgent is an identical alias for VerificationAgent."""
        self.assertIs(PostExecutionVerificationAgent, VerificationAgent)

    def test_stop_on_target_option(self) -> None:
        """Verify stop_on_target=True terminates simulation advance immediately when target reached."""
        sim = TelemetrySimulator(
            SimulatorConfig(
                initial_temp_c=81.5,
                time_scale=2.5,
                noise_std=0.0,
                default_workloads={"wl-alpha-inference": 220.0},
            )
        )
        sim.trigger_anomaly(32.0)

        report = self.verifier.verify(
            simulator=sim,
            plan=self.plan_a,
            workloads=self.workloads,
            max_ticks=30,
            stop_on_target=True,
        )

        self.assertTrue(report.recovery_successful)
        self.assertTrue(report.reached_target)
        self.assertLess(report.ticks_monitored, 30)
        self.assertEqual(report.ticks_monitored, report.ticks_to_target)

    # -----------------------------------------------------------------------
    # 2. Honest Failure Reporting Tests (Never Fabricate Outcomes)
    # -----------------------------------------------------------------------

    def test_honest_failure_reporting_when_temperature_does_not_cool(self) -> None:
        """Verify honest reporting: if power remains high under anomaly, failure is reported without fabrication."""
        # Unmitigated power: 380W under 32°C anomaly -> steady state 85.2°C
        sim = TelemetrySimulator(
            SimulatorConfig(
                initial_temp_c=81.5,
                time_scale=2.5,
                noise_std=0.0,
                default_workloads={"wl-alpha-inference": 230.0, "wl-beta-training": 150.0},
            )
        )
        sim.trigger_anomaly(32.0)

        naive_plan = RecoveryPlanOption(
            plan_id="PLAN-INEFFECTIVE",
            name="Ineffective Plan",
            strategy_type=StrategyType.BALANCED_POWER_CAP,
            projected_temp_max_c=85.0,
            cooling_time_to_safe_s=0.0,
            workload_actions=[
                WorkloadAction(workload_id="wl-alpha-inference", action=ActionType.POWER_CAP, target_power_w=230.0),
                WorkloadAction(workload_id="wl-beta-training", action=ActionType.POWER_CAP, target_power_w=150.0),
            ],
            estimated_cost_usd=0.0,
            rationale="Maintains 380W load",
        )

        report = self.verifier.verify(
            simulator=sim,
            plan=naive_plan,
            workloads=self.workloads,
            max_ticks=30,
        )

        # STRICT ASSERTIONS: must report failure honestly
        self.assertFalse(report.recovery_successful)
        self.assertFalse(report.reached_target)
        self.assertFalse(report.meets_cooling_rate_target)
        self.assertGreater(report.final_temp_c, 81.5)  # Actually heated up
        self.assertEqual(report.observed_cooling_rate_c_per_s, 0.0)

        # Explicit failure reasons present
        self.assertGreater(len(report.failure_reasons), 0)
        failure_text = "; ".join(report.failure_reasons).lower()
        self.assertIn("target (75.0°c) was not achieved", failure_text)
        self.assertIn("cooling rate", failure_text)

        # Summary explicitly states failure
        self.assertIn("FAILED", report.summary)
        self.assertIn("[SIMULATED ENVIRONMENT]", report.summary)

    def test_honest_failure_reporting_hardware_trip_violation(self) -> None:
        """Verify honest reporting when peak temperature breaches the 85.0°C trip limit."""
        sim = TelemetrySimulator(
            SimulatorConfig(
                initial_temp_c=84.5,
                time_scale=2.5,
                noise_std=0.0,
                default_workloads={"wl-alpha-inference": 250.0, "wl-beta-training": 150.0},
            )
        )
        sim.trigger_anomaly(32.0)

        report = self.verifier.verify(
            simulator=sim,
            plan=self.plan_a,
            workloads=self.workloads,
            max_ticks=10,
        )

        self.assertFalse(report.recovery_successful)
        self.assertGreaterEqual(report.max_temp_c, 85.0)

        trip_check = next((c for c in report.checks if c.name == "CHECK_HARDWARE_TRIP_LIMIT"), None)
        self.assertIsNotNone(trip_check)
        self.assertFalse(trip_check.passed)

    def test_honest_failure_reporting_power_budget_violation(self) -> None:
        """Verify honest reporting when total power exceeds node budget."""
        sim = TelemetrySimulator(
            SimulatorConfig(
                initial_temp_c=74.0,
                power_budget_w=400.0,
                default_workloads={"wl-alpha-inference": 220.0, "wl-beta-training": 120.0},  # 340W total
            )
        )

        # Verification against a tighter 300W node constraint: 340W exceeds 300W budget
        report = self.verifier.verify(
            simulator=sim,
            plan=self.plan_a,
            workloads=self.workloads,
            node_constraints=NodeConstraints(power_budget_w=300.0),
            max_ticks=5,
        )

        self.assertFalse(report.recovery_successful)
        power_check = next((c for c in report.checks if c.name == "CHECK_NODE_POWER_BUDGET"), None)
        self.assertIsNotNone(power_check)
        self.assertFalse(power_check.passed)

    def test_honest_failure_reporting_p0_preemption_violation(self) -> None:
        """Verify honest reporting if P0 workload is missing from simulator."""
        sim = TelemetrySimulator(
            SimulatorConfig(
                initial_temp_c=72.0,
                default_workloads={"wl-beta-training": 150.0},  # Alpha was removed!
            )
        )

        report = self.verifier.verify(
            simulator=sim,
            plan=self.plan_a,
            workloads=self.workloads,
            max_ticks=5,
        )

        self.assertFalse(report.recovery_successful)
        p0_check = next((c for c in report.checks if c.name == "CHECK_P0_SLA_PROTECTION"), None)
        self.assertIsNotNone(p0_check)
        self.assertFalse(p0_check.passed)
        self.assertIn("preempted", p0_check.detail.lower())

    def test_honest_failure_reporting_workload_power_floor_violation(self) -> None:
        """Verify honest reporting if workload power is below min_power_w floor."""
        # Alpha floor is 200W, but running at 160W
        sim = TelemetrySimulator(
            SimulatorConfig(
                initial_temp_c=72.0,
                default_workloads={"wl-alpha-inference": 160.0},
            )
        )

        report = self.verifier.verify(
            simulator=sim,
            plan=self.plan_a,
            workloads=self.workloads,
            max_ticks=5,
        )

        self.assertFalse(report.recovery_successful)
        floor_check = next((c for c in report.checks if c.name == "CHECK_WORKLOAD_POWER_FLOORS"), None)
        self.assertIsNotNone(floor_check)
        self.assertFalse(floor_check.passed)

    def test_reject_non_simulator_target(self) -> None:
        """Verify TypeError is raised if simulator is not TelemetrySimulator."""
        with self.assertRaises(TypeError):
            self.verifier.verify(
                simulator=object(),  # Invalid non-simulator target
                plan=self.plan_a,
                workloads=self.workloads,
            )

    # -----------------------------------------------------------------------
    # 3. End-to-End Executor & Verifier Integration Pipeline
    # -----------------------------------------------------------------------

    def test_end_to_end_executor_and_verifier_pipeline(self) -> None:
        """Verify complete pipeline: approval -> execution on simulator -> verification report."""
        sim = TelemetrySimulator(
            SimulatorConfig(
                initial_temp_c=81.5,
                time_scale=2.5,
                default_workloads={"wl-alpha-inference": 230.0, "wl-beta-training": 150.0},
            )
        )
        sim.trigger_anomaly(32.0)

        # Safety gate evaluation
        gate = SafetyGate()
        safety_res = gate.evaluate(self.plan_a, sim.current_telemetry(), self.workloads)
        self.assertTrue(safety_res.passed)

        # Operator approval
        approval = ApprovalAction(
            incident_id=self.incident_id,
            selected_plan_id="PLAN-A",
            operator_id="operator-alex",
            decision=ApprovalDecision.APPROVE,
            timestamp=datetime.now(timezone.utc),
        )

        # Step 1: Execute
        executor = SimulationExecutor(safety_gate=gate)
        exec_result: ExecutionResult = executor.execute(
            plan=self.plan_a,
            approval=approval,
            safety_result=safety_res,
            simulator=sim,
            workloads=self.workloads,
            incident_id=self.incident_id,
            reference_time=approval.timestamp,
        )
        self.assertTrue(exec_result.success)
        self.assertEqual(exec_result.status, ExecutionStatus.SUCCESS)

        # Step 2: Verify post-execution trajectory
        report = self.verifier.verify(
            simulator=sim,
            plan=self.plan_a,
            workloads=self.workloads,
            execution_result=exec_result,
            incident_id=self.incident_id,
            max_ticks=30,
        )

        self.assertTrue(report.recovery_successful)
        self.assertTrue(report.reached_target)
        self.assertLess(report.final_temp_c, 75.0)
        self.assertGreaterEqual(report.observed_cooling_rate_c_per_s, 0.5)
        self.assertEqual(report.ticks_monitored, 30)
        self.assertTrue(report.is_simulated)


if __name__ == "__main__":
    unittest.main()
