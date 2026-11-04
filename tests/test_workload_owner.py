"""Unit tests for the Workload-Owner Negotiation Agents.

Verifies:
1. Deterministic SLA importance scoring (P0 > P1 > P2).
2. Structured proposal generation capturing required fields:
   - Workload ID and priority
   - Minimum power requirement and current allocation
   - SLA/performance importance score
   - Requested power allocation / preferred action
   - Rationale and estimated trade-offs
3. Multi-workload competition between P0_CRITICAL (Alpha), P1_HIGH (Gamma), and P2_FLEXIBLE (Beta).
4. Strict preservation of configured minimum power floors across all proposals.
5. P0 preemption rejection (Alpha strictly resists pauses or eviction).
6. Deterministic repeatability across multiple calls.
7. Input validation and error handling on invalid workload data.
8. Advisory-only guarantee and preservation of simulator state (no actuation authority).
9. Integration with RecoveryPlanner (ingestion, prompt formatting, Safety Gate enforcement).
"""

import unittest
from datetime import datetime, timezone

from thermopilot.agents.forecast import ForecastResult, ThermalRiskStatus, ThermalTrend
from thermopilot.agents.workload_owner import (
    WorkloadOwnerAgent,
    collect_workload_proposals,
    create_competing_workload_agents,
)
from thermopilot.planner.interface import CallableLlamaClient
from thermopilot.planner.llama import LlamaRecoveryPlanner
from thermopilot.planner.planner import RecoveryPlanner
from thermopilot.safety.gate import SafetyGate
from thermopilot.schemas import (
    ActionType,
    GPUTelemetry,
    NegotiationEvidence,
    NodeConstraints,
    WorkloadPriority,
    WorkloadProposal,
)
from thermopilot.simulator.telemetry import SimulatorConfig, TelemetrySimulator


def make_test_telemetry(core_temp_c: float = 81.5, power_draw_w: float = 380.0) -> GPUTelemetry:
    """Helper to construct standard test telemetry."""
    return GPUTelemetry(
        timestamp=datetime(2026, 10, 27, 10, 0, 0, tzinfo=timezone.utc),
        gpu_id=0,
        temperature_core_c=core_temp_c,
        temperature_junction_c=core_temp_c + 8.0,
        power_draw_w=power_draw_w,
        power_limit_w=400.0,
        fan_speed_pct=92.0,
        clock_freq_mhz=1845,
        utilization_pct=98.0,
        ambient_temp_c=32.0,
    )


def make_test_forecast(risk_status: ThermalRiskStatus = ThermalRiskStatus.CRITICAL_RISK) -> ForecastResult:
    """Helper to construct forecast alert."""
    return ForecastResult(
        timestamp=datetime(2026, 10, 27, 10, 0, 0, tzinfo=timezone.utc),
        gpu_id=0,
        current_temp_c=81.5,
        critical_threshold_c=85.0,
        trend=ThermalTrend.RISING,
        rate_c_per_s=0.15,
        acceleration_c_per_s2=0.002,
        time_to_threshold_s=23.3,
        risk_status=risk_status,
        trigger_incident=True,
        reason="Critical thermal runaway imminent",
        sample_count=10,
    )


class TestWorkloadOwnerAgent(unittest.TestCase):
    """Test suite for individual WorkloadOwnerAgent advocacy and deterministic scoring."""

    def setUp(self) -> None:
        self.competing_agents = create_competing_workload_agents()
        self.alpha_agent = self.competing_agents["wl-alpha-inference"]
        self.gamma_agent = self.competing_agents["wl-gamma-analytics"]
        self.beta_agent = self.competing_agents["wl-beta-training"]
        self.telemetry = make_test_telemetry()
        self.forecast = make_test_forecast()

    # -----------------------------------------------------------------------
    # 1. Deterministic SLA Importance Scoring Tests
    # -----------------------------------------------------------------------

    def test_compute_sla_importance_ranking(self) -> None:
        """Verify deterministic SLA importance ranking: P0_CRITICAL > P1_HIGH > P2_FLEXIBLE."""
        alpha_score = self.alpha_agent.compute_sla_importance()
        gamma_score = self.gamma_agent.compute_sla_importance()
        beta_score = self.beta_agent.compute_sla_importance()

        # Check ranking
        self.assertGreater(alpha_score, gamma_score)
        self.assertGreater(gamma_score, beta_score)

        # Expected score ranges:
        # Alpha (P0, $5.50/s, 90s): ~0.99
        self.assertGreaterEqual(alpha_score, 0.90)
        self.assertLessEqual(alpha_score, 1.0)

        # Gamma (P1, $1.00/s, 600s): ~0.50 - 0.70
        self.assertGreaterEqual(gamma_score, 0.50)
        self.assertLess(gamma_score, 0.85)

        # Beta (P2, $0.01/s, 3600s): ~0.10
        self.assertGreaterEqual(beta_score, 0.10)
        self.assertLess(beta_score, 0.30)

    def test_compute_sla_importance_deterministic_repeatability(self) -> None:
        """Verify identical workload profiles produce bit-exact scores across calls."""
        score1 = self.alpha_agent.compute_sla_importance()
        score2 = self.alpha_agent.compute_sla_importance()
        self.assertEqual(score1, score2)

    # -----------------------------------------------------------------------
    # 2. P0_CRITICAL Alpha Advocacy Tests
    # -----------------------------------------------------------------------

    def test_p0_critical_agent_strongly_rejects_preemption(self) -> None:
        """Verify P0 Alpha strictly rejects preemption and requests power above floor."""
        proposal = self.alpha_agent.propose(telemetry=self.telemetry, forecast=self.forecast)

        self.assertIsInstance(proposal, WorkloadProposal)
        self.assertEqual(proposal.workload_id, "wl-alpha-inference")
        self.assertEqual(proposal.priority, WorkloadPriority.P0_CRITICAL)
        self.assertEqual(proposal.preferred_action, ActionType.POWER_CAP)
        self.assertFalse(proposal.can_preempt)

        # Minimum power floor must be preserved
        self.assertEqual(proposal.min_power_w, 200.0)
        self.assertIsNotNone(proposal.requested_power_w)
        self.assertGreaterEqual(proposal.requested_power_w, 200.0)
        self.assertLessEqual(proposal.requested_power_w, 250.0)

        # Rationale must advocate for VIP trading SLA
        self.assertIn("non-preemptible", proposal.rationale.lower())
        self.assertIn("strictly rejects any pause", proposal.rationale.lower())
        self.assertIn("zero sla breach risk", proposal.estimated_trade_off.lower())

    # -----------------------------------------------------------------------
    # 3. P1_HIGH Gamma Advocacy Tests
    # -----------------------------------------------------------------------

    def test_p1_high_agent_tolerates_moderate_cap(self) -> None:
        """Verify P1 Gamma accepts moderate capping while requesting priority over P2."""
        proposal = self.gamma_agent.propose(telemetry=self.telemetry, forecast=self.forecast)

        self.assertEqual(proposal.workload_id, "wl-gamma-analytics")
        self.assertEqual(proposal.priority, WorkloadPriority.P1_HIGH)
        self.assertEqual(proposal.preferred_action, ActionType.POWER_CAP)

        # Must respect 80W floor
        self.assertEqual(proposal.min_power_w, 80.0)
        self.assertIsNotNone(proposal.requested_power_w)
        self.assertGreaterEqual(proposal.requested_power_w, 80.0)
        self.assertLessEqual(proposal.requested_power_w, 120.0)

        # Rationale asserts priority over P2
        self.assertIn("p1_high", proposal.rationale.lower())
        self.assertIn("priority", proposal.rationale.lower())

    # -----------------------------------------------------------------------
    # 4. P2_FLEXIBLE Beta Advocacy Tests
    # -----------------------------------------------------------------------

    def test_p2_flexible_agent_offers_pause_or_elastic_cap(self) -> None:
        """Verify P2 Beta highlights elasticity, checkpointing, and willingness to shed."""
        proposal = self.beta_agent.propose(telemetry=self.telemetry, forecast=self.forecast)

        self.assertEqual(proposal.workload_id, "wl-beta-training")
        self.assertEqual(proposal.priority, WorkloadPriority.P2_FLEXIBLE)
        self.assertTrue(proposal.can_preempt)

        # Must respect 150W floor
        self.assertEqual(proposal.min_power_w, 150.0)
        self.assertGreaterEqual(proposal.requested_power_w, 150.0)

        # Rationale highlights checkpointing and flexibility
        self.assertIn("checkpoint", proposal.rationale.lower())
        self.assertIn("elastic", proposal.rationale.lower())

    # -----------------------------------------------------------------------
    # 5. Multi-Workload Competition & Proposal Collection Tests
    # -----------------------------------------------------------------------

    def test_collect_workload_proposals_ordering(self) -> None:
        """Verify collected proposals are sorted deterministically by SLA importance descending."""
        evidence = collect_workload_proposals(
            workloads=self.competing_agents,
            telemetry=self.telemetry,
            forecast=self.forecast,
        )

        self.assertIsInstance(evidence, NegotiationEvidence)
        self.assertEqual(len(evidence.proposals), 3)

        # Order must be Alpha (P0) -> Gamma (P1) -> Beta (P2)
        self.assertEqual(evidence.proposals[0].workload_id, "wl-alpha-inference")
        self.assertEqual(evidence.proposals[1].workload_id, "wl-gamma-analytics")
        self.assertEqual(evidence.proposals[2].workload_id, "wl-beta-training")

        self.assertEqual(
            evidence.workload_ids,
            ["wl-alpha-inference", "wl-gamma-analytics", "wl-beta-training"],
        )

        # get_proposal helper
        alpha_prop = evidence.get_proposal("wl-alpha-inference")
        self.assertIsNotNone(alpha_prop)
        self.assertEqual(alpha_prop.priority, WorkloadPriority.P0_CRITICAL)

        self.assertIsNone(evidence.get_proposal("nonexistent-workload"))

    def test_collect_workload_proposals_from_specs(self) -> None:
        """Verify collection works directly with WorkloadSpec instances."""
        specs = [agent.spec for agent in self.competing_agents.values()]
        evidence = collect_workload_proposals(specs, self.telemetry, self.forecast)
        self.assertEqual(len(evidence.proposals), 3)
        self.assertEqual(evidence.proposals[0].workload_id, "wl-alpha-inference")

    # -----------------------------------------------------------------------
    # 6. Error Handling & Input Validation Tests
    # -----------------------------------------------------------------------

    def test_invalid_workload_spec_rejected(self) -> None:
        """Verify passing non-WorkloadSpec raises TypeError."""
        with self.assertRaises(TypeError):
            WorkloadOwnerAgent("not_a_spec")  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            collect_workload_proposals(["not_a_spec"])  # type: ignore[list-item]

        with self.assertRaises(TypeError):
            collect_workload_proposals(12345)  # type: ignore[arg-type]

    # -----------------------------------------------------------------------
    # 7. Advisory-Only & Simulator State Preservation Tests
    # -----------------------------------------------------------------------

    def test_agent_has_no_actuation_authority(self) -> None:
        """Verify agent is advisory-only and cannot mutate simulator state or hardware."""
        sim = TelemetrySimulator(
            SimulatorConfig(
                power_budget_w=400.0,
                baseline_ambient_c=22.0,
                default_workloads={"wl-alpha": 230.0, "wl-beta": 150.0},
            )
        )
        initial_temp = sim.core_temp_c
        initial_powers = sim.workload_powers

        # Run proposal generations
        for agent in self.competing_agents.values():
            agent.propose(telemetry=sim.current_telemetry(), forecast=self.forecast)

        # Verify simulator state is strictly unmodified
        self.assertEqual(sim.core_temp_c, initial_temp)
        self.assertEqual(sim.workload_powers, initial_powers)
        self.assertEqual(sim.tick_count, 0)

        # Verify agent class has no execute/mutate methods
        self.assertFalse(hasattr(self.alpha_agent, "execute"))
        self.assertFalse(hasattr(self.alpha_agent, "set_power"))
        self.assertFalse(hasattr(self.alpha_agent, "apply_action"))

    # -----------------------------------------------------------------------
    # 8. Planner Integration Tests with Negotiation Evidence
    # -----------------------------------------------------------------------

    def test_recovery_planner_ingests_and_returns_negotiation_evidence(self) -> None:
        """Verify RecoveryPlanner captures negotiation evidence in PlannerResult."""
        safety_gate = SafetyGate(NodeConstraints(max_temp_c=80.0, power_budget_w=400.0))
        planner = RecoveryPlanner(safety_gate=safety_gate)

        workloads = {w.workload_id: w.spec for w in self.competing_agents.values()}

        # 1. Automatic collection during plan()
        result_auto = planner.plan(
            telemetry=self.telemetry,
            workloads=workloads,
            forecast=self.forecast,
        )
        self.assertTrue(result_auto.success)
        self.assertIsNotNone(result_auto.negotiation_evidence)
        self.assertEqual(len(result_auto.negotiation_evidence.proposals), 3)
        self.assertEqual(
            result_auto.negotiation_evidence.proposals[0].workload_id,
            "wl-alpha-inference",
        )

        # 2. Explicit collection passed in
        explicit_evidence = collect_workload_proposals(workloads, self.telemetry, self.forecast)
        result_explicit = planner.plan(
            telemetry=self.telemetry,
            workloads=workloads,
            forecast=self.forecast,
            negotiation_evidence=explicit_evidence,
        )
        self.assertTrue(result_explicit.success)
        self.assertEqual(result_explicit.negotiation_evidence, explicit_evidence)

    def test_llama_prompt_contains_negotiation_evidence(self) -> None:
        """Verify Llama prompt formatter includes structured negotiation evidence."""
        captured_prompts: list[str] = []

        def mock_client(prompt: str, system_prompt: str | None) -> str:
            captured_prompts.append(prompt)
            # Return valid JSON
            return (
                '[{"plan_id": "PLAN-A", "name": "Plan", "strategy_type": "CONSERVATIVE_SHED",'
                '"projected_temp_max_c": 70.0, "cooling_time_to_safe_s": 10.0, "estimated_cost_usd": 10.0,'
                '"sla_violations_expected": 0, "rationale": "Test",'
                '"workload_actions": [{"workload_id": "wl-alpha-inference", "action": "POWER_CAP", "target_power_w": 220.0}]}]'
            )

        llama_planner = LlamaRecoveryPlanner(client=CallableLlamaClient(mock_client))
        workloads = {w.workload_id: w.spec for w in self.competing_agents.values()}
        evidence = collect_workload_proposals(workloads, self.telemetry, self.forecast)

        llama_planner.generate_candidates(
            telemetry=self.telemetry,
            workloads=workloads,
            forecast=self.forecast,
            negotiation_evidence=evidence,
        )

        self.assertEqual(len(captured_prompts), 1)
        prompt = captured_prompts[0]
        self.assertIn("WORKLOAD NEGOTIATION EVIDENCE & BIDS:", prompt)
        self.assertIn("wl-alpha-inference", prompt)
        self.assertIn("wl-gamma-analytics", prompt)
        self.assertIn("wl-beta-training", prompt)
        self.assertIn("P0_CRITICAL", prompt)

    def test_advisory_proposals_still_strictly_enforced_by_safety_gate(self) -> None:
        """Verify Safety Gate remains final arbiter; unsafe plans are rejected regardless of agent bids."""
        safety_gate = SafetyGate(NodeConstraints(max_temp_c=80.0, power_budget_w=400.0))
        planner = RecoveryPlanner(safety_gate=safety_gate)

        # 1. Standard node workloads (Alpha + Beta): Plans A, B, C pass; Plan D is rejected
        standard_workloads = {
            "wl-alpha-inference": self.alpha_agent.spec,
            "wl-beta-training": self.beta_agent.spec,
        }

        # Run planner with include_unsafe_candidate=True to inject Plan D
        result = planner.plan(
            telemetry=self.telemetry,
            workloads=standard_workloads,
            forecast=self.forecast,
            include_unsafe_candidate=True,
        )

        self.assertTrue(result.success)
        # 4 candidates generated (Plans A, B, C, D)
        self.assertEqual(len(result.candidates), 4)

        # Plan D must be REJECTED by Safety Gate
        plan_d = next(c for c in result.candidates if c.plan.plan_id == "PLAN-D")
        self.assertFalse(plan_d.is_safe)
        self.assertFalse(plan_d.safety_result.passed)

        # Safe candidates only contains certified plans (Plans A, B, C)
        self.assertEqual(len(result.safe_candidates), 3)
        self.assertNotIn("PLAN-D", [c.plan.plan_id for c in result.safe_candidates])

        # 2. When 3 competing workloads are on a 400W node and exceed budget (500W),
        # Safety Gate strictly enforces node budget and rejects Plan B
        three_workloads = {w.workload_id: w.spec for w in self.competing_agents.values()}
        result_three = planner.plan(
            telemetry=self.telemetry,
            workloads=three_workloads,
            forecast=self.forecast,
        )
        self.assertTrue(result_three.success)
        # Plan B has Alpha 220W + Beta 160W + Gamma 120W = 500W > 400W budget
        plan_b = next(c for c in result_three.candidates if c.plan.plan_id == "PLAN-B")
        self.assertFalse(plan_b.is_safe)
        self.assertIn("exceeds budget", "; ".join(plan_b.safety_result.rejection_reasons).lower())


if __name__ == "__main__":
    unittest.main()
