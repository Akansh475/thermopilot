"""Compute Desk: Terminal Operations Control Room for ThermoPilot.

Renders live simulated telemetry, workload tenant bidding, recovery plan trade-offs,
SafetyGate verdicts, human operator approval actions, post-execution verification,
and cryptographic audit ledger verification.

Strictly simulation-only: labeled prominently as simulated data with zero physical GPU access.
"""

import os
import sys
from collections.abc import Mapping, Sequence

from thermopilot.agents.forecast import ForecastResult
from thermopilot.audit.ledger import AuditLedger
from thermopilot.schemas import (
    ApprovalWorkflowResult,
    DecisionCard,
    EvaluatedPlanCandidate,
    ExecutionResult,
    GPUTelemetry,
    IncidentState,
    NegotiationEvidence,
    RecoveryPlanOption,
    VerificationReport,
    WorkloadSpec,
)


class Color:
    """Terminal ANSI color codes with safe disabling."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


class ComputeDesk:
    """Terminal Operations Control Room renderer and interface."""

    def __init__(self, use_color: bool | None = None) -> None:
        """Initialize ComputeDesk.

        Args:
            use_color: Whether to use ANSI color codes. If None, auto-detected from stdout and env.
        """
        if use_color is not None:
            self.use_color = use_color
        else:
            self.use_color = (
                sys.stdout.isatty()
                and os.environ.get("NO_COLOR") is None
                and os.environ.get("TERM") != "dumb"
            )

    def _c(self, code: str, text: str) -> str:
        """Apply color code if coloring is enabled."""
        if not self.use_color:
            return text
        return f"{code}{text}{Color.RESET}"

    # -----------------------------------------------------------------------
    # 1. Header Banner
    # -----------------------------------------------------------------------

    def render_header(
        self,
        incident_id: str = "INC-20261027-001",
        state: IncidentState = IncidentState.DETECTED,
        node_id: str = "node-gpu-cluster-alpha-0",
    ) -> str:
        """Render control room header with incident ID and simulation markers."""
        sim_tag = self._c(Color.BOLD + Color.YELLOW, "[SIMULATED ENVIRONMENT - NO PHYSICAL HARDWARE ACCESSED]")
        title = self._c(Color.BOLD + Color.CYAN, "THERMOPILOT COMPUTE DESK")
        sub = self._c(Color.DIM, "Autonomous, Cost-Aware GPU Thermal Governance & HITL Approval")

        state_color = Color.GREEN
        if state in (IncidentState.DETECTED, IncidentState.FORECASTED, IncidentState.PLANNING):
            state_color = Color.YELLOW
        elif state == IncidentState.AWAITING_APPROVAL:
            state_color = Color.MAGENTA
        elif state in (IncidentState.EXECUTING, IncidentState.VERIFYING):
            state_color = Color.CYAN
        elif state in (IncidentState.REJECTED, IncidentState.TIMEOUT_ESCALATED, IncidentState.RECOVERY_FAILED, IncidentState.REVALIDATION_FAILED):
            state_color = Color.RED

        state_badge = self._c(Color.BOLD + state_color, f"[{state.value.upper()}]")

        lines = [
            "=" * 88,
            f"  {title} | {sub}",
            f"  Incident: {self._c(Color.BOLD, incident_id)} | Node: {node_id} | Status: {state_badge}",
            f"  Safety Notice: {sim_tag}",
            "=" * 88,
        ]
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # 2. Live Telemetry Panel
    # -----------------------------------------------------------------------

    def render_telemetry_panel(
        self,
        telemetry: GPUTelemetry,
        forecast: ForecastResult | None = None,
        critical_trip_temp_c: float = 85.0,
        thermal_ceiling_c: float = 80.0,
        recovery_target_c: float = 75.0,
        power_budget_w: float = 400.0,
    ) -> str:
        """Render live simulated telemetry gauges and thermal trajectory."""
        temp = telemetry.temperature_core_c
        power = telemetry.power_draw_w

        # Temperature status color
        if temp >= critical_trip_temp_c:
            temp_str = self._c(Color.BOLD + Color.RED, f"{temp:.1f}°C [CRITICAL TRIP EXCEEDED]")
        elif temp >= thermal_ceiling_c:
            temp_str = self._c(Color.BOLD + Color.RED, f"{temp:.1f}°C [ABOVE SAFETY CEILING]")
        elif temp > recovery_target_c:
            temp_str = self._c(Color.BOLD + Color.YELLOW, f"{temp:.1f}°C [ELEVATED - RUNAWAY RISK]")
        else:
            temp_str = self._c(Color.BOLD + Color.GREEN, f"{temp:.1f}°C [NOMINAL / SAFE]")

        # Power status color
        p_pct = min(100.0, (power / power_budget_w) * 100.0)
        power_bar = self._render_progress_bar(p_pct, width=20)
        power_str = f"{power:.1f}W / {power_budget_w:.1f}W ({p_pct:.0f}%) [{power_bar}]"

        # Time-to-threshold
        if forecast and forecast.time_to_threshold_s is not None:
            ttt_color = Color.RED if forecast.time_to_threshold_s < 25.0 else Color.YELLOW
            ttt_str = self._c(Color.BOLD + ttt_color, f"{forecast.time_to_threshold_s:.1f}s to {critical_trip_temp_c:.1f}°C Trip")
            trend_str = f"{forecast.trend.value.upper()} ({forecast.rate_c_per_s:+.2f}°C/s)"
        else:
            ttt_str = self._c(Color.GREEN, "Nominal / Equilibrium")
            trend_str = "STABLE"

        anomaly_badge = (
            self._c(Color.BOLD + Color.RED, "[ANOMALY ACTIVE: +10°C INTAKE SURGE]")
            if telemetry.ambient_temp_c >= 28.0
            else self._c(Color.GREEN, "[NOMINAL]")
        )

        lines = [
            f" [LIVE SIMULATED TELEMETRY]  Timestamp: {telemetry.timestamp.strftime('%H:%M:%S UTC')}",
            f"  Core Temp:       {temp_str}  (Ceiling: {thermal_ceiling_c:.1f}°C, Target: {recovery_target_c:.1f}°C, Trip: {critical_trip_temp_c:.1f}°C)",
            f"  Node Power:      {power_str}",
            f"  Ambient Intake:  {telemetry.ambient_temp_c:.1f}°C {anomaly_badge}",
            f"  Time-to-Trip:    {ttt_str} | Thermal Trend: {trend_str}",
            f"  Fan Speed:       {telemetry.fan_speed_pct:.0f}% | Clock: {telemetry.clock_freq_mhz} MHz | Utilization: {telemetry.utilization_pct:.0f}%",
            "-" * 88,
        ]
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # 3. Workload Tenant Bidding Panel
    # -----------------------------------------------------------------------

    def render_workloads_panel(
        self,
        workloads: Sequence[WorkloadSpec] | Mapping[str, WorkloadSpec],
        negotiation: NegotiationEvidence | None = None,
    ) -> str:
        """Render competing tenant workloads, contractual SLAs, and negotiation bids."""
        workload_map: dict[str, WorkloadSpec] = (
            dict(workloads) if isinstance(workloads, Mapping) else {w.workload_id: w for w in workloads}
        )

        # Proposal lookups
        bids_by_wid = {}
        if negotiation and negotiation.proposals:
            for prop in negotiation.proposals:
                bids_by_wid[prop.workload_id] = prop

        lines = [
            " [WORKLOAD-OWNER AGENTS] Competing Tenant Priorities & Negotiation Bids:",
            "  " + f"{'Workload / Tenant':<28} {'Tier':<12} {'Power':<10} {'Floor':<8} {'SLA Deadline':<14} {'Preempt':<9} {'Urgency':<8}",
            "  " + "-" * 84,
        ]

        for wid, spec in workload_map.items():
            tier_color = Color.RED if "P0" in spec.priority.value else (Color.YELLOW if "P1" in spec.priority.value else Color.CYAN)
            tier_str = self._c(Color.BOLD + tier_color, spec.priority.value[:11])
            preempt_str = "No (VIP)" if not spec.can_preempt else "Yes"
            p_str = f"{spec.current_power_w:.0f}W"
            floor_str = f"{spec.min_power_w:.0f}W"
            sla_str = f"{spec.sla_deadline_s:.0f}s (${spec.sla_penalty_usd_per_s:.2f}/s)"

            prop = bids_by_wid.get(wid)
            urgency_str = f"{prop.sla_importance_score:.2f}" if prop else "N/A"

            title_str = f"{spec.name[:24]} ({wid})"
            lines.append(f"  {title_str:<28} {tier_str:<12} {p_str:<10} {floor_str:<8} {sla_str:<14} {preempt_str:<9} {urgency_str:<8}")

            if prop:
                pref_action = prop.preferred_action.value
                cap_target = f" -> {prop.requested_power_w:.0f}W" if prop.requested_power_w else ""
                bid_summary = f"    └─ Proposal: {pref_action}{cap_target} | Bid: {prop.rationale}"
                lines.append(self._c(Color.DIM, bid_summary[:86]))

        lines.append("-" * 88)
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # 4. Recovery Planner & SafetyGate Evaluation Panel
    # -----------------------------------------------------------------------

    def render_plans_panel(
        self,
        candidates: Sequence[EvaluatedPlanCandidate] | Sequence[RecoveryPlanOption],
    ) -> str:
        """Render recovery candidates and SafetyGate PASS/BLOCK decisions."""
        lines = [
            " [RECOVERY PLANNER & SAFETY GATE] Candidate Evaluation & Trade-offs:",
            "  " + f"{'Plan':<8} {'Name':<26} {'T_proj':<9} {'Cooldown':<10} {'SLA Breaches':<14} {'Cost':<8} {'Safety Gate Status':<20}",
            "  " + "-" * 84,
        ]

        for item in candidates:
            if isinstance(item, EvaluatedPlanCandidate):
                plan = item.plan
                safety = item.safety_result
                is_safe = safety.passed
                rejection_reasons = safety.rejection_reasons
            else:
                plan = item
                is_safe = plan.is_safe
                rejection_reasons = plan.rejection_reasons

            rec_marker = "*" if plan.is_recommended else " "
            pid_str = f"{rec_marker}{plan.plan_id}"

            if is_safe:
                gate_badge = self._c(Color.BOLD + Color.GREEN, "[PASS - CERTIFIED]")
            else:
                gate_badge = self._c(Color.BOLD + Color.RED, "[BLOCKED BY GATE]")

            temp_proj = f"{plan.projected_temp_max_c:.1f}°C"
            cool_str = f"{plan.cooling_time_to_safe_s:.0f}s"
            sla_str = f"{plan.sla_violations_expected} breaches"
            cost_str = f"${plan.estimated_cost_usd:.0f}"

            lines.append(
                f"  {pid_str:<8} {plan.name[:25]:<26} {temp_proj:<9} {cool_str:<10} {sla_str:<14} {cost_str:<8} {gate_badge:<20}"
            )

            # Print actions and rationale
            actions_summary = ", ".join(
                f"{a.workload_id}:{a.action.value}" + (f"({a.target_power_w:.0f}W)" if a.target_power_w is not None else "")
                for a in plan.workload_actions
            )
            lines.append(self._c(Color.DIM, f"    └─ Actions: {actions_summary}"))

            if not is_safe and rejection_reasons:
                for reason in rejection_reasons:
                    block_msg = f"    [X] REJECTION REASON: {reason}"
                    lines.append(self._c(Color.RED, block_msg))

        lines.append("-" * 88)
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # 5. Switch HITL Operator Action Panel
    # -----------------------------------------------------------------------

    def render_operator_controls(
        self,
        card: DecisionCard | None = None,
        workflow_result: ApprovalWorkflowResult | None = None,
    ) -> str:
        """Render Switch Human-in-the-Loop decision controls and timeout policy."""
        lines = [
            " [SWITCH HITL APPROVAL DESK] Human Governance Sign-Off:",
        ]

        if workflow_result is not None:
            status_color = Color.GREEN if workflow_result.is_authorized_for_execution else Color.RED
            status_str = self._c(Color.BOLD + status_color, f"Status: {workflow_result.status.value.upper()}")
            lines.append(f"  {status_str}")
            if workflow_result.selected_plan:
                lines.append(f"  Authorized Plan: {self._c(Color.BOLD, workflow_result.selected_plan.plan_id)} ({workflow_result.selected_plan.name})")
            if workflow_result.rejection_reasons:
                lines.append(self._c(Color.RED, f"  Rejection / Denial: {'; '.join(workflow_result.rejection_reasons)}"))
        elif card is not None:
            lines.append(
                f"  Awaiting Operator Sign-Off (Timeout: {card.timeout_seconds:.0f}s - Policy: Timeout NEVER mutates workloads)"
            )
            lines.append(f"  Executable Plans: {', '.join(card.executable_plan_ids)}")
            if card.rejected_plan_ids:
                blocked_str = self._c(Color.RED, f"BLOCKED: {', '.join(card.rejected_plan_ids)} (Execution button disabled)")
                lines.append(f"  {blocked_str}")
            lines.append("  Operator Hotkeys:")
            lines.append("    [1 / A] Approve PLAN-A: Conservative Workload Shed")
            lines.append("    [2 / B] Approve PLAN-B: Balanced Dynamic Power Cap (RECOMMENDED)")
            lines.append("    [3 / C] Approve PLAN-C: Workload Migration / Offload")
            lines.append(f"    [4 / D] {self._c(Color.RED, 'Attempt PLAN-D: Blind 50% Throttle (BLOCKED - CANNOT EXECUTE)')}")
            lines.append("    [R] Reject All Plans")
            lines.append("    [T] Simulate Operator Timeout Escalation")
        else:
            lines.append("  No active decision card dispatched.")

        lines.append("-" * 88)
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # 6. Post-Execution Verification Panel
    # -----------------------------------------------------------------------

    def render_verification_panel(
        self,
        report: VerificationReport | None = None,
        exec_res: ExecutionResult | None = None,
    ) -> str:
        """Render post-execution thermal decay verification and SLA validation."""
        lines = [
            " [VERIFICATION AGENT] Post-Execution Thermal Trajectory & SLA Verification:",
        ]

        if report is not None:
            verdict_color = Color.GREEN if report.recovery_successful else Color.RED
            verdict_badge = self._c(
                Color.BOLD + verdict_color,
                "[RECOVERY SUCCESSFUL - 75°C TARGET ACHIEVED]"
                if report.recovery_successful
                else "[RECOVERY FAILED - TARGET NOT MET]",
            )
            delta_t = report.final_temp_c - report.initial_temp_c
            target_status = "REACHED" if report.reached_target else "NOT REACHED"
            ticks_str = f"{report.ticks_to_target} ticks" if report.ticks_to_target is not None else "N/A"

            lines.append(f"  Verdict: {verdict_badge}")
            lines.append(
                f"  Temp Trajectory:  {report.initial_temp_c:.1f}°C -> {report.final_temp_c:.1f}°C (Delta: {delta_t:+.1f}°C)"
            )
            lines.append(
                f"  Cooling Rate:     {report.observed_cooling_rate_c_per_s:.2f}°C/s simulated (Target >= 0.50°C/s)"
            )
            lines.append(
                f"  Target Check:     75.0°C Envelope {target_status} in {ticks_str} ({report.sim_time_to_target_s or 0.0:.1f}s simulated)"
            )
            sla_badge = (
                self._c(Color.GREEN, "PASS (0 Violations)")
                if report.recovery_successful
                else self._c(Color.RED, "VIOLATIONS REPORTED")
            )
            lines.append(f"  VIP SLA Guard:    {sla_badge}")

            if report.failure_reasons:
                for reason in report.failure_reasons:
                    lines.append(self._c(Color.RED, f"    [!] Reason: {reason}"))
        elif exec_res is not None:
            lines.append(f"  Actions applied: {len(exec_res.applied_actions)} workload operations. Verifying thermal trajectory...")
        else:
            lines.append("  Awaiting execution completion.")

        lines.append("-" * 88)
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # 7. Cryptographic Audit Trail Panel
    # -----------------------------------------------------------------------

    def render_audit_panel(self, ledger: AuditLedger, limit: int = 5) -> str:
        """Render immutable audit ledger trail and cryptographic hash-chain status."""
        events = ledger.read_events()
        report = ledger.verify_chain()
        is_valid = report.is_valid
        reason = "; ".join(report.errors) if report.errors else "Chain intact"

        chain_color = Color.GREEN if is_valid else Color.RED
        chain_status = self._c(
            Color.BOLD + chain_color,
            f"[CHAIN VERIFIED: SHA-256 HASH INTEGRITY INTACT ({len(events)} EVENTS)]"
            if is_valid
            else f"[CHAIN TAMPERED / CORRUPTED: {reason}]",
        )

        lines = [
            f" [CRYPTOGRAPHIC AUDIT LEDGER] Append-Only Event Trail | {chain_status}",
            "  " + f"{'Seq':<5} {'Event Type':<28} {'Sim':<6} {'Hash Prefix':<14} {'Timestamp':<20}",
            "  " + "-" * 84,
        ]

        # Show most recent events
        recent = events[-limit:] if len(events) > limit else events
        for ev in recent:
            sim_marker = "[SIM]" if ev.is_simulated else "[REAL]"
            hash_prefix = (ev.hash[:12] + "..") if ev.hash else "None"
            ts_str = ev.timestamp.strftime("%H:%M:%S.%f")[:12]
            lines.append(f"  {ev.sequence_num:<5} {ev.event_type.value:<28} {sim_marker:<6} {hash_prefix:<14} {ts_str:<20}")

        lines.append("=" * 88)
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # 8. Full Control Room Screen
    # -----------------------------------------------------------------------

    def render_screen(
        self,
        incident_id: str,
        state: IncidentState,
        telemetry: GPUTelemetry,
        workloads: Sequence[WorkloadSpec] | Mapping[str, WorkloadSpec],
        candidates: Sequence[EvaluatedPlanCandidate] | Sequence[RecoveryPlanOption],
        ledger: AuditLedger,
        forecast: ForecastResult | None = None,
        negotiation: NegotiationEvidence | None = None,
        card: DecisionCard | None = None,
        workflow_result: ApprovalWorkflowResult | None = None,
        verification_report: VerificationReport | None = None,
        exec_res: ExecutionResult | None = None,
    ) -> str:
        """Compose and render complete Compute Desk operations control room."""
        sections = [
            self.render_header(incident_id=incident_id, state=state),
            self.render_telemetry_panel(telemetry=telemetry, forecast=forecast),
            self.render_workloads_panel(workloads=workloads, negotiation=negotiation),
            self.render_plans_panel(candidates=candidates),
            self.render_operator_controls(card=card, workflow_result=workflow_result),
            self.render_verification_panel(report=verification_report, exec_res=exec_res),
            self.render_audit_panel(ledger=ledger),
        ]
        return "\n".join(sections)

    # -----------------------------------------------------------------------
    # Helper Utilities
    # -----------------------------------------------------------------------

    def _render_progress_bar(self, percentage: float, width: int = 20) -> str:
        """Render a terminal text progress bar."""
        clamped = max(0.0, min(100.0, percentage))
        filled = round((clamped / 100.0) * width)
        empty = width - filled
        bar = "#" * filled + "-" * empty
        return bar
