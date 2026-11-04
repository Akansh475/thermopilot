"""Repeatable Demonstration Runner and CLI Driver for ThermoPilot Compute Desk.

Provides:
- Deterministic 8-stage demonstration execution
- Simulator and coordinator reset capability
- Interactive terminal control desk
- Automated 60-second hackathon demonstration script runner
- State dictionary export for local web and REST clients

Strictly simulation-only: All telemetry and actuations are simulated.
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thermopilot.agents.forecast import ForecastResult, ThermalRiskStatus, ThermalTrend
from thermopilot.agents.workload_owner import collect_workload_proposals
from thermopilot.audit.coordinator import IncidentLifecycleCoordinator
from thermopilot.audit.ledger import AuditLedger
from thermopilot.audit.state_machine import IncidentStateMachine
from thermopilot.executor.executor import ExecutionDeniedError, SimulationExecutor
from thermopilot.executor.verifier import VerificationAgent
from thermopilot.hitl.switch import SwitchApprovalDesk
from thermopilot.planner.fallback import DeterministicFallbackPlanner
from thermopilot.planner.planner import RecoveryPlanner
from thermopilot.safety.gate import SafetyGate
from thermopilot.schemas import (
    ApprovalAction,
    ApprovalDecision,
    ApprovalWorkflowResult,
    DecisionCard,
    ExecutionResult,
    GPUTelemetry,
    IncidentState,
    NegotiationEvidence,
    NodeConstraints,
    PlannerResult,
    VerificationReport,
    WorkloadPriority,
    WorkloadSpec,
)
from thermopilot.simulator.telemetry import SimulatorConfig, TelemetrySimulator
from thermopilot.ui.desk import ComputeDesk


def create_demo_workloads() -> dict[str, WorkloadSpec]:
    """Construct reference tenant workloads for demonstration.

    - Alpha: VIP P0 Financial Trading Batch (non-preemptible, rigid SLA)
    - Beta: Elastic P2 Diffusion Fine-Tuning (checkpointable, elastic)
    - Gamma: P1 Production Analytics (moderate SLA, can cap)
    """
    alpha = WorkloadSpec(
        workload_id="wl-alpha-inference",
        name="Financial Trading Batch",
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
    gamma = WorkloadSpec(
        workload_id="wl-gamma-analytics",
        name="Production Analytics",
        owner="Analytics-Infra",
        priority=WorkloadPriority.P1_HIGH,
        current_power_w=0.0,
        min_power_w=80.0,
        max_power_w=140.0,
        can_preempt=True,
        can_power_cap=True,
        sla_deadline_s=1200.0,
        sla_penalty_usd_per_s=1.00,
    )
    return {
        alpha.workload_id: alpha,
        beta.workload_id: beta,
        gamma.workload_id: gamma,
    }


def create_demo_simulator() -> TelemetrySimulator:
    """Instantiate reference physics simulator for the demo thermal emergency.

    Starts at 81.5°C under +10°C ambient anomaly (32.0°C intake).
    Total initial power: 230W (Alpha) + 150W (Beta) = 380W on a 400W budget.
    """
    config = SimulatorConfig(
        power_budget_w=400.0,
        baseline_ambient_c=22.0,
        anomaly_ambient_c=32.0,
        initial_temp_c=81.5,
        time_scale=2.5,
        noise_std=0.0,
        seed=42,
        default_workloads={
            "wl-alpha-inference": 230.0,
            "wl-beta-training": 150.0,
        },
    )
    sim = TelemetrySimulator(config)
    sim.trigger_anomaly(target_ambient_c=32.0)
    return sim


class DemoRunner:
    """Manages the full lifecycle and reproducible demonstration execution."""

    def __init__(
        self,
        incident_id: str = "INC-20261027-001",
        ledger_path: Path | str | None = None,
        use_color: bool | None = None,
    ) -> None:
        """Initialize DemoRunner.

        Args:
            incident_id: Identifier for the demonstration incident.
            ledger_path: Path to write the cryptographic audit JSONL ledger.
            use_color: Whether to use terminal ANSI colors.
        """
        self.incident_id = incident_id
        if ledger_path is None:
            audit_dir = Path(".audit")
            audit_dir.mkdir(parents=True, exist_ok=True)
            self.ledger_path = audit_dir / f"demo_audit_{incident_id.lower()}.jsonl"
        else:
            self.ledger_path = Path(ledger_path)
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

        self.desk = ComputeDesk(use_color=use_color)
        self.reset()

    # -----------------------------------------------------------------------
    # Reset & State Initialization
    # -----------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the simulator, audit ledger, and governance state machine to initial conditions."""
        # Clean or initialize audit ledger
        if self.ledger_path.exists():
            try:
                self.ledger_path.unlink()
            except OSError:
                pass

        self.ledger = AuditLedger(self.ledger_path)
        self.workloads = create_demo_workloads()
        self.simulator = create_demo_simulator()
        self.node_constraints = NodeConstraints(
            power_budget_w=400.0,
            thermal_ceiling_c=80.0,
            recovery_target_temp_c=75.0,
            critical_trip_temp_c=85.0,
        )

        self.safety_gate = SafetyGate(self.node_constraints)
        self.fallback_planner = DeterministicFallbackPlanner(self.node_constraints)
        self.planner = RecoveryPlanner(
            safety_gate=self.safety_gate,
            fallback_planner=self.fallback_planner,
            node_constraints=self.node_constraints,
        )
        self.switch_desk = SwitchApprovalDesk()
        self.executor = SimulationExecutor()
        self.verifier = VerificationAgent()

        self.state_machine = IncidentStateMachine(
            incident_id=self.incident_id,
            initial_state=IncidentState.DETECTED,
            ledger=self.ledger,
        )
        self.coordinator = IncidentLifecycleCoordinator(
            incident_id=self.incident_id,
            ledger=self.ledger,
            state_machine=self.state_machine,
            desk=self.switch_desk,
            executor=self.executor,
            verifier=self.verifier,
        )

        # Cached stage artifacts
        self.last_telemetry: GPUTelemetry = self.simulator.current_telemetry()
        self.last_forecast: ForecastResult | None = None
        self.last_negotiation: NegotiationEvidence | None = None
        self.last_planner_result: PlannerResult | None = None
        self.last_decision_card: DecisionCard | None = None
        self.last_workflow_result: ApprovalWorkflowResult | None = None
        self.last_execution_result: ExecutionResult | None = None
        self.last_verification_report: VerificationReport | None = None
        self.historical_telemetry: list[GPUTelemetry] = [self.last_telemetry]

    # -----------------------------------------------------------------------
    # Step-by-Step Execution
    # -----------------------------------------------------------------------

    def step_1_detect(self) -> GPUTelemetry:
        """Stage 1: Detect thermal incident approaching critical threshold."""
        self.last_telemetry = self.simulator.current_telemetry()
        self.coordinator.record_incident_detected(self.last_telemetry)
        return self.last_telemetry

    def step_2_forecast(self) -> ForecastResult:
        """Stage 2: Perception agent evaluates runaway trajectory and countdown."""
        # Use calibrated thermal forecast based on active ambient surge
        rate = 0.15
        time_to_trip = max(1.0, (85.0 - self.last_telemetry.temperature_core_c) / rate)
        self.last_forecast = ForecastResult(
            timestamp=self.last_telemetry.timestamp,
            gpu_id=0,
            current_temp_c=self.last_telemetry.temperature_core_c,
            critical_threshold_c=85.0,
            trend=ThermalTrend.RISING,
            rate_c_per_s=rate,
            acceleration_c_per_s2=0.002,
            time_to_threshold_s=round(time_to_trip, 1),
            confidence_score=0.98,
            risk_status=ThermalRiskStatus.CRITICAL_RISK,
            trigger_incident=True,
            reason=f"Thermal runaway detected: {self.last_telemetry.temperature_core_c:.1f}°C rising at +{rate:.2f}°C/s. Hardware trip in {time_to_trip:.1f}s.",
        )
        self.coordinator.record_forecast_completed(self.last_forecast)
        return self.last_forecast

    def step_3_plan(self) -> tuple[PlannerResult, NegotiationEvidence]:
        """Stage 3: Collect workload bids, generate recovery candidates, and evaluate SafetyGate."""
        # Collect tenant negotiation bids
        active_workloads = {wid: spec for wid, spec in self.workloads.items() if spec.current_power_w > 0}
        self.last_negotiation = collect_workload_proposals(
            self.workloads, self.last_telemetry, self.last_forecast
        )

        # Plan with fallback candidate generator including Plan D (Unsafe candidate)
        self.last_planner_result = self.planner.plan(
            telemetry=self.last_telemetry,
            workloads=active_workloads,
            forecast=self.last_forecast,
            node_constraints=self.node_constraints,
            include_unsafe_candidate=True,
            negotiation_evidence=self.last_negotiation,
        )

        self.coordinator.record_planning_completed(
            self.last_planner_result, self.last_negotiation
        )
        return self.last_planner_result, self.last_negotiation

    def step_4_dispatch_card(self) -> DecisionCard:
        """Stage 4: Dispatch Switch decision card with certified and blocked plans."""
        if not self.last_planner_result:
            raise RuntimeError("Cannot dispatch card: planning step not completed.")

        active_workloads = {wid: spec for wid, spec in self.workloads.items() if spec.current_power_w > 0}
        self.last_decision_card = self.coordinator.dispatch_decision_card(
            telemetry=self.last_telemetry,
            candidates=self.last_planner_result.candidates,
            workloads=active_workloads,
            forecast=self.last_forecast,
            node_constraints=self.node_constraints,
            timeout_s=15.0,
        )
        return self.last_decision_card

    def step_5_operator_decision(
        self,
        decision: ApprovalDecision = ApprovalDecision.APPROVE,
        plan_id: str = "PLAN-A",
        operator_id: str = "ops-primary-77",
        reason: str = "Approved by on-call operator via Switch Desk.",
    ) -> ApprovalWorkflowResult:
        """Stage 5: Human operator reviews decision card and submits authorized decision."""
        if self.last_decision_card is None:
            raise RuntimeError("No decision card available for approval.")

        # Blocked plan check: UI pre-check prevents submitting blocked Plan D
        if decision == ApprovalDecision.APPROVE and plan_id in self.last_decision_card.rejected_plan_ids:
            # We record or return a denied submission
            raise ValueError(
                f"Cannot approve '{plan_id}': Plan is BLOCKED by SafetyGate and non-executable."
            )

        approval_ts = (
            self.last_decision_card.timestamp
            if self.last_decision_card
            else datetime.now(timezone.utc)
        )

        action = ApprovalAction(
            incident_id=self.incident_id,
            operator_id=operator_id,
            decision=decision,
            selected_plan_id=plan_id if decision == ApprovalDecision.APPROVE else None,
            timestamp=approval_ts,
            reason=reason,
        )

        self.last_workflow_result = self.coordinator.evaluate_approval_decision(
            approval=action,
            current_telemetry=self.simulator.current_telemetry(),
            node_constraints=self.node_constraints,
        )
        return self.last_workflow_result

    def step_6_execute(self) -> ExecutionResult:
        """Stage 6: Pre-execution SafetyGate revalidation and simulator actuation."""
        if self.last_workflow_result is None or not self.last_workflow_result.is_authorized_for_execution:
            raise ExecutionDeniedError("Cannot execute: plan has not been authorized by operator.")

        active_workloads = {wid: spec for wid, spec in self.workloads.items() if spec.current_power_w > 0}
        self.last_execution_result = self.coordinator.execute_plan(
            workflow_result=self.last_workflow_result,
            simulator=self.simulator,
            workloads=active_workloads,
            node_constraints=self.node_constraints,
        )
        return self.last_execution_result

    def step_7_advance_cooldown(self, ticks: int = 15) -> list[GPUTelemetry]:
        """Stage 7: Advance simulator ticks and record thermal decay trajectory."""
        readings: list[GPUTelemetry] = []
        for _ in range(ticks):
            telem = self.simulator.step()
            readings.append(telem)
            self.historical_telemetry.append(telem)
            self.last_telemetry = telem
        return readings

    def step_8_verify(self, max_ticks: int = 25) -> VerificationReport:
        """Stage 8: Run Verification Agent checks and verify audit hash chain."""
        if self.last_workflow_result is None or self.last_workflow_result.selected_plan is None:
            raise RuntimeError("Cannot verify: no executed plan.")

        active_workloads = {wid: spec for wid, spec in self.workloads.items() if spec.current_power_w > 0}
        self.last_verification_report = self.coordinator.verify_recovery(
            simulator=self.simulator,
            plan=self.last_workflow_result.selected_plan,
            workloads=active_workloads,
            execution_result=self.last_execution_result,
            max_ticks=max_ticks,
            node_constraints=self.node_constraints,
        )
        self.last_telemetry = self.simulator.current_telemetry()
        self.historical_telemetry.append(self.last_telemetry)
        return self.last_verification_report

    # -----------------------------------------------------------------------
    # Automated Demonstration Runner
    # -----------------------------------------------------------------------

    def run_auto_demo(
        self,
        plan_id: str = "PLAN-A",
        delay_s: float = 0.0,
        print_output: bool = True,
    ) -> dict[str, Any]:
        """Run complete 8-stage hackathon demonstration automatically.

        Args:
            plan_id: Plan to approve (PLAN-A or PLAN-B). Default PLAN-A.
            delay_s: Optional seconds to pause between steps for live presentation.
            print_output: Whether to print formatted Compute Desk screens to console.

        Returns:
            Dictionary summarizing execution metrics and verification results.
        """
        self.reset()

        # Step 1: Detection
        self.step_1_detect()
        if print_output:
            self._print_banner("STAGE 1: THERMAL EMERGENCY DETECTED")
            print(self.desk.render_screen(
                incident_id=self.incident_id,
                state=self.state_machine.current_state,
                telemetry=self.last_telemetry,
                workloads=self.workloads,
                candidates=[],
                ledger=self.ledger,
            ))
        if delay_s > 0:
            time.sleep(delay_s)

        # Step 2: Forecast
        self.step_2_forecast()
        if print_output:
            self._print_banner("STAGE 2: PERCEPTION & THERMAL RUNAWAY FORECAST")
            print(self.desk.render_telemetry_panel(
                telemetry=self.last_telemetry,
                forecast=self.last_forecast,
            ))
        if delay_s > 0:
            time.sleep(delay_s)

        # Step 3: Negotiation & Planning
        self.step_3_plan()
        if print_output:
            self._print_banner("STAGE 3: WORKLOAD TENANT NEGOTIATION & SAFETY GATE EVALUATION")
            print(self.desk.render_workloads_panel(
                workloads=self.workloads,
                negotiation=self.last_negotiation,
            ))
            if self.last_planner_result:
                print(self.desk.render_plans_panel(self.last_planner_result.candidates))
        if delay_s > 0:
            time.sleep(delay_s)

        # Step 4: Decision Card Dispatch
        self.step_4_dispatch_card()
        if print_output:
            self._print_banner("STAGE 4: SWITCH DECISION CARD DISPATCHED")
            print(self.desk.render_operator_controls(card=self.last_decision_card))
        if delay_s > 0:
            time.sleep(delay_s)

        # Step 5: Operator Sign-off
        self.step_5_operator_decision(
            decision=ApprovalDecision.APPROVE,
            plan_id=plan_id,
            operator_id="ops-lead-alex",
            reason=f"Approved safe recovery option '{plan_id}' via Switch interface.",
        )
        if print_output:
            self._print_banner(f"STAGE 5: OPERATOR EXPLICIT APPROVAL ({plan_id})")
            print(self.desk.render_operator_controls(workflow_result=self.last_workflow_result))
        if delay_s > 0:
            time.sleep(delay_s)

        # Step 6: Simulator Actuation
        self.step_6_execute()
        if print_output:
            self._print_banner("STAGE 6: PRE-EXECUTION REVALIDATION & SIMULATOR ACTUATION")
            print(f"  Actions applied: {[a.action.value for a in self.last_execution_result.applied_actions if self.last_execution_result]}")
            print(f"  New Node Power:  {self.simulator.effective_power_draw_w:.1f}W (Clamped to budget)")
        if delay_s > 0:
            time.sleep(delay_s)

        # Step 7 & 8: Verification Agent advances simulator and monitors trajectory
        self.step_8_verify(max_ticks=25)
        report = self.ledger.verify_chain()
        is_chain_valid = report.is_valid

        if print_output:
            self._print_banner("STAGE 7: SIMULATOR ADVANCES & NEWTON THERMAL DECAY")
            print(self.desk.render_telemetry_panel(telemetry=self.last_telemetry))
        if delay_s > 0:
            time.sleep(delay_s)

        if print_output:
            self._print_banner("STAGE 8: POST-EXECUTION VERIFICATION & AUDIT HASH CHAIN")
            print(self.desk.render_verification_panel(report=self.last_verification_report))
            print(self.desk.render_audit_panel(self.ledger, limit=8))

        return {
            "incident_id": self.incident_id,
            "final_state": self.state_machine.current_state.value,
            "initial_temp_c": self.last_verification_report.initial_temp_c if self.last_verification_report else None,
            "final_temp_c": self.last_telemetry.temperature_core_c,
            "recovery_successful": self.last_verification_report.recovery_successful if self.last_verification_report else False,
            "observed_cooling_rate_c_per_s": self.last_verification_report.observed_cooling_rate_c_per_s if self.last_verification_report else 0.0,
            "audit_chain_valid": is_chain_valid,
            "audit_events_count": len(self.ledger.read_events()),
        }

    # -----------------------------------------------------------------------
    # Interactive CLI Mode
    # -----------------------------------------------------------------------

    def run_interactive_cli(self) -> None:
        """Run interactive step-by-step operator experience in console."""
        self.reset()
        self.step_1_detect()
        self.step_2_forecast()
        self.step_3_plan()
        self.step_4_dispatch_card()

        while True:
            # Render full desk screen
            print("\033[H\033[J", end="")  # Clear terminal
            candidates = self.last_planner_result.candidates if self.last_planner_result else []
            print(self.desk.render_screen(
                incident_id=self.incident_id,
                state=self.state_machine.current_state,
                telemetry=self.last_telemetry,
                workloads=self.workloads,
                candidates=candidates,
                ledger=self.ledger,
                forecast=self.last_forecast,
                negotiation=self.last_negotiation,
                card=self.last_decision_card,
                workflow_result=self.last_workflow_result,
                verification_report=self.last_verification_report,
                exec_res=self.last_execution_result,
            ))

            state = self.state_machine.current_state
            if state in (IncidentState.RECOVERED, IncidentState.REJECTED, IncidentState.TIMEOUT_ESCALATED, IncidentState.RECOVERY_FAILED):
                print(f"\n[DEMO COMPLETE: Final state '{state.value}']. Press 'R' to reset, or 'Q' to quit.")
                choice = input("Option > ").strip().upper()
                if choice == "R":
                    self.reset()
                    self.step_1_detect()
                    self.step_2_forecast()
                    self.step_3_plan()
                    self.step_4_dispatch_card()
                    continue
                elif choice == "Q":
                    break
                else:
                    continue

            print("\nEnter Choice: [1] Approve Plan A  [2] Approve Plan B  [3] Approve Plan C  [4] Attempt Plan D (Blocked)")
            print("              [R] Reject All  [T] Simulate Timeout  [Q] Quit")
            try:
                choice = input("Operator Choice > ").strip().upper()
            except (EOFError, KeyboardInterrupt):
                break

            if choice in ("1", "A"):
                self.step_5_operator_decision(ApprovalDecision.APPROVE, plan_id="PLAN-A")
                self.step_6_execute()
                self.step_7_advance_cooldown(ticks=15)
                self.step_8_verify()
            elif choice in ("2", "B"):
                self.step_5_operator_decision(ApprovalDecision.APPROVE, plan_id="PLAN-B")
                self.step_6_execute()
                self.step_7_advance_cooldown(ticks=15)
                self.step_8_verify()
            elif choice in ("3", "C"):
                self.step_5_operator_decision(ApprovalDecision.APPROVE, plan_id="PLAN-C")
                self.step_6_execute()
                self.step_7_advance_cooldown(ticks=15)
                self.step_8_verify()
            elif choice in ("4", "D"):
                print("\n[DENIED] Plan D is rejected by SafetyGate! Pre-execution authorization fails.")
                try:
                    self.step_5_operator_decision(ApprovalDecision.APPROVE, plan_id="PLAN-D")
                except ValueError as exc:
                    print(f"Error: {exc}")
                time.sleep(2.0)
            elif choice == "R":
                self.step_5_operator_decision(ApprovalDecision.REJECT, reason="Operator explicit veto")
            elif choice == "T":
                self.step_5_operator_decision(ApprovalDecision.TIMEOUT_ESCALATED, reason="Operator window expired")
            elif choice == "Q":
                break

    # -----------------------------------------------------------------------
    # State Serialization for Web / API
    # -----------------------------------------------------------------------

    def get_state_dict(self) -> dict[str, Any]:
        """Serialize complete current system state for web dashboards and external clients."""
        events = self.ledger.read_events()
        chain_report = self.ledger.verify_chain()
        is_chain_valid = chain_report.is_valid
        chain_reason = "; ".join(chain_report.errors) if chain_report.errors else "Chain intact"

        candidates_data = []
        if self.last_planner_result:
            for c in self.last_planner_result.candidates:
                candidates_data.append({
                    "plan_id": c.plan.plan_id,
                    "name": c.plan.name,
                    "strategy_type": c.plan.strategy_type.value,
                    "is_recommended": c.plan.is_recommended,
                    "projected_temp_max_c": c.plan.projected_temp_max_c,
                    "cooling_time_to_safe_s": c.plan.cooling_time_to_safe_s,
                    "estimated_cost_usd": c.plan.estimated_cost_usd,
                    "sla_violations_expected": c.plan.sla_violations_expected,
                    "is_safe": c.safety_result.passed,
                    "rejection_reasons": c.safety_result.rejection_reasons,
                    "workload_actions": [a.model_dump(mode="json") for a in c.plan.workload_actions],
                    "rationale": c.plan.rationale,
                })

        workloads_data = []
        for wid, w in self.workloads.items():
            workloads_data.append({
                "workload_id": wid,
                "name": w.name,
                "owner": w.owner,
                "priority": w.priority.value,
                "current_power_w": self.simulator.get_workload_power(wid),
                "min_power_w": w.min_power_w,
                "max_power_w": w.max_power_w,
                "can_preempt": w.can_preempt,
                "can_power_cap": w.can_power_cap,
                "sla_deadline_s": w.sla_deadline_s,
                "sla_penalty_usd_per_s": w.sla_penalty_usd_per_s,
            })

        history_data = [
            {
                "tick": i,
                "temp_core_c": t.temperature_core_c,
                "power_draw_w": t.power_draw_w,
                "timestamp": t.timestamp.isoformat(),
            }
            for i, t in enumerate(self.historical_telemetry)
        ]

        return {
            "incident_id": self.incident_id,
            "state": self.state_machine.current_state.value,
            "is_simulated": True,
            "telemetry": self.last_telemetry.model_dump(mode="json"),
            "forecast": self.last_forecast.model_dump(mode="json") if self.last_forecast else None,
            "workloads": workloads_data,
            "candidates": candidates_data,
            "decision_card": self.last_decision_card.model_dump(mode="json") if self.last_decision_card else None,
            "workflow_result": self.last_workflow_result.model_dump(mode="json") if self.last_workflow_result else None,
            "verification_report": self.last_verification_report.model_dump(mode="json") if self.last_verification_report else None,
            "execution_result": self.last_execution_result.model_dump(mode="json") if self.last_execution_result else None,
            "audit_ledger": {
                "chain_valid": is_chain_valid,
                "chain_reason": chain_reason,
                "events_count": len(events),
                "recent_events": [e.model_dump(mode="json") for e in events[-10:]],
            },
            "history": history_data,
            "node_constraints": self.node_constraints.model_dump(mode="json"),
        }

    def _print_banner(self, title: str) -> None:
        """Helper to print banner separator."""
        print(f"\n>>> {title}")
        print("-" * 88)


# ---------------------------------------------------------------------------
# CLI Argument Dispatcher
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI Entry point for demo runner."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ThermoPilot Compute Desk: Operations Control Room & Demo CLI"
    )
    parser.add_argument(
        "--auto", action="store_true", help="Run automated 60-second hackathon demonstration."
    )
    parser.add_argument(
        "--plan",
        choices=["PLAN-A", "PLAN-B", "PLAN-C"],
        default="PLAN-A",
        help="Select recovery plan to approve in automated mode (default: PLAN-A).",
    )
    parser.add_argument(
        "--reject", action="store_true", help="Demonstrate operator explicit rejection."
    )
    parser.add_argument(
        "--timeout", action="store_true", help="Demonstrate operator timeout escalation."
    )
    parser.add_argument(
        "--attempt-plan-d",
        action="store_true",
        help="Demonstrate that Plan D cannot be executed via any approval.",
    )
    parser.add_argument(
        "--reset", action="store_true", help="Reset simulator and audit ledger to clean initial state."
    )
    parser.add_argument(
        "--web", action="store_true", help="Launch the local web dashboard server."
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Web dashboard port (default: 8080)."
    )
    parser.add_argument(
        "--delay", type=float, default=0.0, help="Delay in seconds between steps during automated demo."
    )
    parser.add_argument(
        "--plain", action="store_true", help="Disable ANSI color codes in terminal output."
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Launch interactive terminal control desk."
    )

    args = parser.parse_args()

    if args.web:
        from thermopilot.ui.web import launch_web_dashboard
        launch_web_dashboard(port=args.port)
        return

    runner = DemoRunner(use_color=not args.plain)

    if args.reset:
        runner.reset()
        print("[SUCCESS] Telemetry simulator and audit ledger reset to initial incident conditions.")
        return

    if args.reject:
        runner.reset()
        runner.step_1_detect()
        runner.step_2_forecast()
        runner.step_3_plan()
        runner.step_4_dispatch_card()
        res = runner.step_5_operator_decision(ApprovalDecision.REJECT, reason="Operator explicit veto")
        print(f"[REJECTION DEMO] Operator decision: {res.status.value}. Simulator workloads unchanged.")
        print(f"Final state: {runner.state_machine.current_state.value}")
        return

    if args.timeout:
        runner.reset()
        runner.step_1_detect()
        runner.step_2_forecast()
        runner.step_3_plan()
        runner.step_4_dispatch_card()
        res = runner.step_5_operator_decision(ApprovalDecision.TIMEOUT_ESCALATED, reason="Approval window expired")
        print(f"[TIMEOUT DEMO] Operator status: {res.status.value}. Workloads protected from mutation.")
        print(f"Final state: {runner.state_machine.current_state.value}")
        return

    if args.attempt_plan_d:
        runner.reset()
        runner.step_1_detect()
        runner.step_2_forecast()
        runner.step_3_plan()
        runner.step_4_dispatch_card()
        print("[PLAN D SAFETY TEST] Attempting to approve Plan D...")
        try:
            runner.step_5_operator_decision(ApprovalDecision.APPROVE, plan_id="PLAN-D")
            print("[ERROR] Plan D was approved! SafetyGate failed!")
            sys.exit(1)
        except ValueError as exc:
            print(f"[SUCCESS] SafetyGate blocked Plan D as expected: {exc}")
            print(f"State machine remains safely in: {runner.state_machine.current_state.value}")
            return

    if args.interactive:
        runner.run_interactive_cli()
        return

    # Default to automated demo run
    res = runner.run_auto_demo(plan_id=args.plan, delay_s=args.delay, print_output=True)
    print("\n" + "=" * 88)
    print(f"DEMO COMPLETE | Result: {res['final_state']} | Temp: {res['final_temp_c']}°C | Chain: {'VALID' if res['audit_chain_valid'] else 'INVALID'}")
    print("=" * 88)


if __name__ == "__main__":
    main()
