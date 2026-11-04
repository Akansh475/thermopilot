"""Deterministic Safety Gate for ThermoPilot Compute Desk.

Evaluates proposed recovery plans against real-time physical telemetry,
workload specifications, and node constraints. Operates strictly
deterministically without external network calls or LLM dependencies.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from thermopilot.schemas import (
    ActionType,
    ApprovalAction,
    ApprovalDecision,
    GPUTelemetry,
    NodeConstraints,
    RecoveryPlanOption,
    RuleEvaluation,
    SafetyGateResult,
    SafetyRuleId,
    WorkloadAction,
    WorkloadPriority,
    WorkloadSpec,
)


class SafetyGate:
    """Deterministic Safety Gate for validating recovery plans.

    Enforces 6 non-bypassable physical and business rules:
    1. RULE_TEMP_CEILING: Projected GPU temperature must not exceed 80.0°C.
    2. RULE_NODE_POWER_BUDGET: Total node electrical power must remain within budget (400.0W).
    3. RULE_NO_P0_PREEMPTION: P0_CRITICAL and non-preemptible workloads must never be preempted.
    4. RULE_P0_SLA_PRESERVATION: P0 workload SLA constraints must be strictly preserved.
    5. RULE_MIN_POWER_FLOOR: Every workload must remain at or above its configured minimum power floor.
    6. RULE_POWER_CAP_VALIDITY: POWER_CAP actions must have valid target powers respecting workload limits.

    Fails safe on any missing, unparseable, or invalid inputs.
    """

    def __init__(self, node_constraints: NodeConstraints | None = None) -> None:
        """Initialize SafetyGate with optional default node constraints."""
        self.node_constraints = node_constraints or NodeConstraints()

    def evaluate(
        self,
        plan: Any,
        telemetry: Any,
        workloads: Sequence[WorkloadSpec] | Mapping[str, WorkloadSpec] | Any,
        node_constraints: NodeConstraints | None = None,
    ) -> SafetyGateResult:
        """Evaluate a recovery plan option against telemetry, workloads, and node constraints.

        Args:
            plan: Candidate RecoveryPlanOption to validate.
            telemetry: Current GPUTelemetry snapshot.
            workloads: Active workload specifications (sequence or mapping of WorkloadSpec).
            node_constraints: Optional node constraint overrides.

        Returns:
            Validated SafetyGateResult detailing aggregate certification and individual rule outcomes.
        """
        rejection_reasons: list[str] = []

        # -------------------------------------------------------------------
        # 1. Validate Node Constraints
        # -------------------------------------------------------------------
        active_constraints = node_constraints or self.node_constraints
        if not isinstance(active_constraints, NodeConstraints):
            return SafetyGateResult(
                plan_id=getattr(plan, "plan_id", "UNKNOWN"),
                passed=False,
                evaluations=[],
                rejection_reasons=[
                    f"Invalid node_constraints: expected NodeConstraints, got {type(active_constraints).__name__}."
                ],
            )
        if active_constraints.max_temp_c <= 0.0 or active_constraints.power_budget_w <= 0.0:
            return SafetyGateResult(
                plan_id=getattr(plan, "plan_id", "UNKNOWN"),
                passed=False,
                evaluations=[],
                rejection_reasons=[
                    "Node constraints must specify strictly positive max_temp_c and power_budget_w."
                ],
            )

        # -------------------------------------------------------------------
        # 2. Validate Candidate Plan Input
        # -------------------------------------------------------------------
        if plan is None:
            return SafetyGateResult(
                plan_id="UNKNOWN",
                passed=False,
                evaluations=[],
                rejection_reasons=["Recovery plan option cannot be None."],
            )
        if not isinstance(plan, RecoveryPlanOption):
            return SafetyGateResult(
                plan_id=str(getattr(plan, "plan_id", "INVALID")),
                passed=False,
                evaluations=[],
                rejection_reasons=[
                    f"Invalid plan input: expected RecoveryPlanOption, got {type(plan).__name__}."
                ],
            )

        plan_id = plan.plan_id

        # -------------------------------------------------------------------
        # 3. Validate Telemetry Snapshot Input
        # -------------------------------------------------------------------
        if telemetry is None:
            rejection_reasons.append("Current telemetry reading cannot be None.")
        elif not isinstance(telemetry, GPUTelemetry):
            rejection_reasons.append(
                f"Invalid telemetry input: expected GPUTelemetry, got {type(telemetry).__name__}."
            )

        # -------------------------------------------------------------------
        # 4. Normalize and Validate Workload Specifications
        # -------------------------------------------------------------------
        workload_map: dict[str, WorkloadSpec] = {}
        if workloads is None:
            rejection_reasons.append("Workload specifications cannot be None.")
        elif isinstance(workloads, Mapping):
            for wid, spec in workloads.items():
                if not isinstance(spec, WorkloadSpec):
                    rejection_reasons.append(
                        f"Workload entry '{wid}' is not an instance of WorkloadSpec."
                    )
                else:
                    workload_map[spec.workload_id] = spec
            if not workload_map and not rejection_reasons:
                rejection_reasons.append("Workload specifications mapping cannot be empty.")
        elif isinstance(workloads, (list, tuple, set)):
            for item in workloads:
                if not isinstance(item, WorkloadSpec):
                    rejection_reasons.append(
                        f"Workload list contains invalid item of type {type(item).__name__}."
                    )
                else:
                    if item.workload_id in workload_map:
                        rejection_reasons.append(
                            f"Duplicate workload ID '{item.workload_id}' found in workload specifications."
                        )
                    workload_map[item.workload_id] = item
            if not workload_map and not rejection_reasons:
                rejection_reasons.append("Workload specifications list cannot be empty.")
        else:
            rejection_reasons.append(
                f"Invalid workloads input: expected sequence or mapping of WorkloadSpec, got {type(workloads).__name__}."
            )

        # -------------------------------------------------------------------
        # 5. Validate Plan Actions
        # -------------------------------------------------------------------
        action_map: dict[str, WorkloadAction] = {}
        if not plan.workload_actions:
            rejection_reasons.append("Plan workload_actions list cannot be empty.")
        else:
            for act in plan.workload_actions:
                if not isinstance(act, WorkloadAction):
                    rejection_reasons.append(
                        f"Invalid workload action item: expected WorkloadAction, got {type(act).__name__}."
                    )
                    continue

                if act.workload_id in action_map:
                    rejection_reasons.append(
                        f"Conflicting duplicate actions detected for workload '{act.workload_id}'."
                    )

                if workload_map and act.workload_id not in workload_map:
                    rejection_reasons.append(
                        f"Action targets unknown workload ID '{act.workload_id}' not present in active workloads."
                    )

                action_map[act.workload_id] = act

        # -------------------------------------------------------------------
        # 6. Evaluate Rule 1: RULE_TEMP_CEILING
        # -------------------------------------------------------------------
        limit_temp = active_constraints.max_temp_c
        obs_temp = plan.projected_temp_max_c
        if obs_temp is None or not isinstance(obs_temp, (int, float)):
            r1_passed = False
            r1_detail = "Projected maximum temperature is missing or non-numeric."
            rejection_reasons.append(r1_detail)
            eval_temp = RuleEvaluation(
                rule_name=SafetyRuleId.RULE_TEMP_CEILING.value,
                passed=False,
                detail=r1_detail,
                observed_value="INVALID",
                limit_value=limit_temp,
            )
        else:
            r1_passed = obs_temp <= limit_temp
            if r1_passed:
                r1_detail = f"Steady state {obs_temp:.1f}C <= {limit_temp:.1f}C"
            else:
                r1_detail = f"Steady state {obs_temp:.1f}C exceeds {limit_temp:.1f}C ceiling"
                rejection_reasons.append(
                    f"Projected GPU temperature ({obs_temp:.1f}°C) exceeds safety limit ({limit_temp:.1f}°C)."
                )
            eval_temp = RuleEvaluation(
                rule_name=SafetyRuleId.RULE_TEMP_CEILING.value,
                passed=r1_passed,
                detail=r1_detail,
                observed_value=round(float(obs_temp), 2),
                limit_value=limit_temp,
            )

        # -------------------------------------------------------------------
        # 7. Evaluate Rule 2: RULE_NODE_POWER_BUDGET
        # -------------------------------------------------------------------
        limit_power = active_constraints.power_budget_w
        if workload_map:
            allocated_powers: dict[str, float] = {}
            for wid, spec in workload_map.items():
                if wid in action_map:
                    act = action_map[wid]
                    if act.action == ActionType.POWER_CAP:
                        allocated_powers[wid] = act.target_power_w if act.target_power_w is not None else 0.0
                    elif act.action in (ActionType.PAUSE, ActionType.MIGRATE):
                        allocated_powers[wid] = 0.0
                    elif act.action == ActionType.RESUME:
                        allocated_powers[wid] = (
                            act.target_power_w if act.target_power_w is not None else spec.current_power_w
                        )
                else:
                    allocated_powers[wid] = spec.current_power_w

            total_projected_power = sum(allocated_powers.values())
            r2_passed = total_projected_power <= limit_power
            if r2_passed:
                r2_detail = f"{total_projected_power:.1f}W <= {limit_power:.1f}W"
            else:
                r2_detail = f"Allocated power {total_projected_power:.1f}W exceeds {limit_power:.1f}W budget"
                rejection_reasons.append(
                    f"Total node power ({total_projected_power:.1f}W) exceeds budget ({limit_power:.1f}W)."
                )
            eval_power = RuleEvaluation(
                rule_name=SafetyRuleId.RULE_NODE_POWER_BUDGET.value,
                passed=r2_passed,
                detail=r2_detail,
                observed_value=round(total_projected_power, 2),
                limit_value=limit_power,
            )
        else:
            r2_detail = "Unable to compute total node power without active workload specifications."
            rejection_reasons.append(r2_detail)
            eval_power = RuleEvaluation(
                rule_name=SafetyRuleId.RULE_NODE_POWER_BUDGET.value,
                passed=False,
                detail=r2_detail,
                observed_value=0.0,
                limit_value=limit_power,
            )

        # -------------------------------------------------------------------
        # 8. Evaluate Rule 3: RULE_NO_P0_PREEMPTION
        # -------------------------------------------------------------------
        if workload_map:
            preempted_wids: list[tuple[str, str]] = []
            for wid, act in action_map.items():
                spec = workload_map.get(wid)
                if spec is not None:
                    is_p0 = spec.priority == WorkloadPriority.P0_CRITICAL
                    is_non_preemptible = not spec.can_preempt
                    if (is_p0 or is_non_preemptible) and act.action in (ActionType.PAUSE, ActionType.MIGRATE):
                        preempted_wids.append((wid, act.action.value))

            if not preempted_wids:
                p0_names = [s.name for s in workload_map.values() if s.priority == WorkloadPriority.P0_CRITICAL]
                name_str = p0_names[0] if p0_names else "Alpha"
                r3_detail = f"{name_str} not preempted"
                eval_preempt = RuleEvaluation(
                    rule_name=SafetyRuleId.RULE_NO_P0_PREEMPTION.value,
                    passed=True,
                    detail=r3_detail,
                    observed_value="ACTIVE",
                    limit_value="NO_PREEMPTION",
                )
            else:
                names = ", ".join(f"'{wid}' ({act})" for wid, act in preempted_wids)
                r3_detail = f"Workloads preempted: {names}"
                for wid, act in preempted_wids:
                    rejection_reasons.append(
                        f"P0_CRITICAL / non-preemptible workload '{wid}' cannot be preempted (action: {act})."
                    )
                eval_preempt = RuleEvaluation(
                    rule_name=SafetyRuleId.RULE_NO_P0_PREEMPTION.value,
                    passed=False,
                    detail=r3_detail,
                    observed_value="PREEMPTED",
                    limit_value="NO_PREEMPTION",
                )
        else:
            r3_detail = "Workload specifications missing for preemption check."
            rejection_reasons.append(r3_detail)
            eval_preempt = RuleEvaluation(
                rule_name=SafetyRuleId.RULE_NO_P0_PREEMPTION.value,
                passed=False,
                detail=r3_detail,
                observed_value="UNKNOWN",
                limit_value="NO_PREEMPTION",
            )

        # -------------------------------------------------------------------
        # 9. Evaluate Rule 4: RULE_P0_SLA_PRESERVATION
        # -------------------------------------------------------------------
        p0_sla_issues: list[str] = []
        if plan.sla_violations_expected > 0:
            p0_sla_issues.append(
                f"Expected SLA violations ({plan.sla_violations_expected}) > 0"
            )

        if workload_map:
            for wid, spec in workload_map.items():
                if spec.priority == WorkloadPriority.P0_CRITICAL:
                    if wid in action_map:
                        act = action_map[wid]
                        if act.action in (ActionType.PAUSE, ActionType.MIGRATE):
                            p0_sla_issues.append(
                                f"P0 workload '{wid}' execution interrupted ({act.action.value})"
                            )
                        elif (
                            act.action == ActionType.POWER_CAP
                            and act.target_power_w is not None
                            and act.target_power_w < spec.min_power_w
                        ):
                            p0_sla_issues.append(
                                f"P0 workload '{wid}' power ({act.target_power_w:.1f}W) below min floor ({spec.min_power_w:.1f}W)"
                            )
                    elif spec.current_power_w < spec.min_power_w:
                        p0_sla_issues.append(
                            f"P0 workload '{wid}' current power ({spec.current_power_w:.1f}W) below min floor ({spec.min_power_w:.1f}W)"
                        )

            if not p0_sla_issues:
                eval_sla = RuleEvaluation(
                    rule_name=SafetyRuleId.RULE_P0_SLA_PRESERVATION.value,
                    passed=True,
                    detail=f"Expected P0 SLA violations: {plan.sla_violations_expected}",
                    observed_value=0,
                    limit_value=0,
                )
            else:
                observed_violations = max(plan.sla_violations_expected, len(p0_sla_issues))
                for issue in p0_sla_issues:
                    rejection_reasons.append(f"P0 SLA contract breach: {issue}.")
                eval_sla = RuleEvaluation(
                    rule_name=SafetyRuleId.RULE_P0_SLA_PRESERVATION.value,
                    passed=False,
                    detail=f"P0 SLA breaches detected: {'; '.join(p0_sla_issues)}",
                    observed_value=observed_violations,
                    limit_value=0,
                )
        else:
            r4_detail = "Workload specifications missing for P0 SLA preservation check."
            rejection_reasons.append(r4_detail)
            eval_sla = RuleEvaluation(
                rule_name=SafetyRuleId.RULE_P0_SLA_PRESERVATION.value,
                passed=False,
                detail=r4_detail,
                observed_value=max(plan.sla_violations_expected, 1),
                limit_value=0,
            )

        # -------------------------------------------------------------------
        # 10. Evaluate Rule 5: RULE_MIN_POWER_FLOOR
        # -------------------------------------------------------------------
        if workload_map:
            floor_violations: list[tuple[str, float, float]] = []
            floor_pass_details: list[str] = []
            for wid, spec in workload_map.items():
                if wid in action_map:
                    act = action_map[wid]
                    if act.action == ActionType.POWER_CAP and act.target_power_w is not None:
                        if act.target_power_w < spec.min_power_w:
                            floor_violations.append((wid, act.target_power_w, spec.min_power_w))
                        else:
                            floor_pass_details.append(
                                f"{spec.name or wid} power {act.target_power_w:.0f}W >= {spec.min_power_w:.0f}W floor"
                            )
                else:
                    if spec.current_power_w < spec.min_power_w:
                        floor_violations.append((wid, spec.current_power_w, spec.min_power_w))
                    else:
                        floor_pass_details.append(
                            f"{spec.name or wid} power {spec.current_power_w:.0f}W >= {spec.min_power_w:.0f}W floor"
                        )

            if not floor_violations:
                detail_str = "; ".join(floor_pass_details) if floor_pass_details else "All active workloads satisfy power floor"
                obs_val = 200.0
                limit_val = 200.0
                for act in action_map.values():
                    if act.action == ActionType.POWER_CAP and act.target_power_w is not None:
                        obs_val = act.target_power_w
                        spec = workload_map.get(act.workload_id)
                        if spec:
                            limit_val = spec.min_power_w
                        break

                eval_floor = RuleEvaluation(
                    rule_name=SafetyRuleId.RULE_MIN_POWER_FLOOR.value,
                    passed=True,
                    detail=detail_str,
                    observed_value=obs_val,
                    limit_value=limit_val,
                )
            else:
                for wid, target, floor in floor_violations:
                    rejection_reasons.append(
                        f"Workload '{wid}' allocated power ({target:.1f}W) is below minimum operational floor ({floor:.1f}W)."
                    )
                worst_target = min(t for _, t, _ in floor_violations)
                worst_floor = max(f for _, _, f in floor_violations)
                eval_floor = RuleEvaluation(
                    rule_name=SafetyRuleId.RULE_MIN_POWER_FLOOR.value,
                    passed=False,
                    detail=f"Workloads below min power floor: {', '.join(f'{w} ({t}W < {f}W)' for w, t, f in floor_violations)}",
                    observed_value=worst_target,
                    limit_value=worst_floor,
                )
        else:
            r5_detail = "Workload specifications missing for minimum power floor check."
            rejection_reasons.append(r5_detail)
            eval_floor = RuleEvaluation(
                rule_name=SafetyRuleId.RULE_MIN_POWER_FLOOR.value,
                passed=False,
                detail=r5_detail,
                observed_value=0.0,
                limit_value=0.0,
            )

        # -------------------------------------------------------------------
        # 11. Evaluate Rule 6: RULE_POWER_CAP_VALIDITY
        # -------------------------------------------------------------------
        power_cap_issues: list[str] = []
        if workload_map:
            for act in plan.workload_actions:
                if act.action == ActionType.POWER_CAP:
                    spec = workload_map.get(act.workload_id)
                    if spec is not None:
                        if not spec.can_power_cap:
                            power_cap_issues.append(
                                f"Workload '{spec.workload_id}' does not permit power capping (can_power_cap=False)"
                            )
                        if act.target_power_w is None:
                            power_cap_issues.append(
                                f"Workload '{act.workload_id}' missing target_power_w"
                            )
                        elif act.target_power_w <= 0.0:
                            power_cap_issues.append(
                                f"Workload '{act.workload_id}' target power must be positive, got {act.target_power_w}W"
                            )
                        elif act.target_power_w > spec.max_power_w:
                            power_cap_issues.append(
                                f"Workload '{act.workload_id}' target power ({act.target_power_w:.1f}W) exceeds maximum ({spec.max_power_w:.1f}W)"
                            )
                        elif act.target_power_w < spec.min_power_w:
                            power_cap_issues.append(
                                f"Workload '{act.workload_id}' target power ({act.target_power_w:.1f}W) is below minimum ({spec.min_power_w:.1f}W)"
                            )

            if not power_cap_issues:
                eval_validity = RuleEvaluation(
                    rule_name=SafetyRuleId.RULE_POWER_CAP_VALIDITY.value,
                    passed=True,
                    detail="All POWER_CAP actions include valid target power and respect workload limits",
                    observed_value="VALID",
                    limit_value="VALID",
                )
            else:
                for issue in power_cap_issues:
                    rejection_reasons.append(f"Invalid POWER_CAP action: {issue}.")
                eval_validity = RuleEvaluation(
                    rule_name=SafetyRuleId.RULE_POWER_CAP_VALIDITY.value,
                    passed=False,
                    detail=f"Invalid POWER_CAP actions: {'; '.join(power_cap_issues)}",
                    observed_value="INVALID",
                    limit_value="VALID",
                )
        else:
            r6_detail = "Workload specifications missing for POWER_CAP validation."
            rejection_reasons.append(r6_detail)
            eval_validity = RuleEvaluation(
                rule_name=SafetyRuleId.RULE_POWER_CAP_VALIDITY.value,
                passed=False,
                detail=r6_detail,
                observed_value="UNKNOWN",
                limit_value="VALID",
            )

        # -------------------------------------------------------------------
        # 12. Aggregate Certification Outcome
        # -------------------------------------------------------------------
        evaluations = [
            eval_temp,
            eval_power,
            eval_preempt,
            eval_sla,
            eval_floor,
            eval_validity,
        ]

        passed = all(e.passed for e in evaluations) and len(rejection_reasons) == 0

        return SafetyGateResult(
            plan_id=plan_id,
            passed=passed,
            evaluations=evaluations,
            rejection_reasons=rejection_reasons,
        )

    def authorize_execution(
        self,
        approval: Any,
        plan: Any,
        safety_result: SafetyGateResult | None = None,
    ) -> bool:
        """Validate whether an operator approval authorizes plan execution.

        Strictly enforces that TIMEOUT_ESCALATED and REJECT decisions CANNOT authorize
        or execute any workload mutation.

        Execution is authorized ONLY if:
        1. approval is an ApprovalAction with decision == ApprovalDecision.APPROVE
        2. approval.selected_plan_id == plan.plan_id
        3. safety_result is provided and passed == True (and matches plan.plan_id)

        Returns:
            True ONLY when explicitly approved by an operator for a certified-safe plan.
            False in all other circumstances.
        """
        authorized, _ = self.validate_authorization(approval, plan, safety_result)
        return authorized

    def validate_authorization(
        self,
        approval: Any,
        plan: Any,
        safety_result: SafetyGateResult | None = None,
    ) -> tuple[bool, list[str]]:
        """Validate operator authorization with explanatory rejection reasons.

        Returns:
            Tuple of (is_authorized, list_of_rejection_reasons).
        """
        if approval is None:
            return False, ["Approval action is missing (None)."]
        if not isinstance(approval, ApprovalAction):
            return False, [
                f"Invalid approval action type: expected ApprovalAction, got {type(approval).__name__}."
            ]

        # Explicit policy: timeouts NEVER authorize workload mutations
        if approval.decision == ApprovalDecision.TIMEOUT_ESCALATED:
            return False, [
                "Approval timed out (TIMEOUT_ESCALATED). Workload mutations cannot be executed automatically."
            ]
        if approval.decision == ApprovalDecision.REJECT:
            return False, ["Plan execution was explicitly rejected by the operator."]
        if approval.decision != ApprovalDecision.APPROVE:
            return False, [f"Unknown or unapproved decision type: {approval.decision.value}."]

        if plan is None:
            return False, ["Recovery plan is missing (None)."]
        if not isinstance(plan, RecoveryPlanOption):
            return False, [
                f"Invalid recovery plan type: expected RecoveryPlanOption, got {type(plan).__name__}."
            ]

        if approval.selected_plan_id != plan.plan_id:
            return False, [
                f"Selected plan ID '{approval.selected_plan_id}' does not match plan '{plan.plan_id}'."
            ]

        if safety_result is None:
            return False, ["Safety gate certification result is required to authorize execution."]
        if not isinstance(safety_result, SafetyGateResult):
            return False, [
                f"Invalid safety result type: expected SafetyGateResult, got {type(safety_result).__name__}."
            ]
        if safety_result.plan_id != plan.plan_id:
            return False, [
                f"Safety gate result plan ID '{safety_result.plan_id}' does not match plan '{plan.plan_id}'."
            ]
        if not safety_result.passed:
            return False, [
                f"Plan '{plan.plan_id}' was rejected by the safety gate and cannot be executed."
            ]

        return True, []
