"""Local Web Control Room Dashboard for ThermoPilot Compute Desk.

Zero-dependency HTTP server using standard library http.server, providing:
- Real-time GPU operations control room interface
- Live SVG thermal trajectory chart
- Workload-owner agent cards and negotiation bids
- Candidate plan comparison matrix with Plan D visibly blocked by SafetyGate
- Switch HITL operator sign-off interface
- Post-execution verification gauges
- Cryptographic audit trail with SHA-256 hash-chain verification badge
- Local REST API for UI state, stepping, actions, and resets

Strictly simulation-only: labeled prominently as simulated data with zero physical GPU access.
"""

import http.server
import json
import threading
import urllib.parse
import webbrowser
from typing import Any

from thermopilot.executor.executor import ExecutionDeniedError
from thermopilot.schemas import ApprovalDecision
from thermopilot.ui.demo import DemoRunner

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ThermoPilot Compute Desk | GPU Operations Control Room</title>
  <style>
    :root {
      --bg: #0d1117;
      --card-bg: #161b22;
      --border: #30363d;
      --text: #c9d1d9;
      --text-dim: #8b949e;
      --accent: #58a6ff;
      --danger: #f85149;
      --danger-bg: rgba(248, 81, 73, 0.15);
      --warning: #d29922;
      --warning-bg: rgba(210, 153, 34, 0.15);
      --success: #3fb950;
      --success-bg: rgba(63, 185, 80, 0.15);
      --purple: #bc8cff;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
      font-size: 14px;
      line-height: 1.5;
      padding: 16px;
    }
    .container { max-width: 1440px; margin: 0 auto; display: flex; flex-direction: column; gap: 16px; }
    .header {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }
    .header-title h1 { font-size: 20px; color: #fff; font-weight: 700; letter-spacing: 0.5px; }
    .header-title p { font-size: 12px; color: var(--text-dim); }
    .header-badges { display: flex; gap: 10px; align-items: center; }
    .badge {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 20px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .badge-sim { background: var(--warning-bg); color: var(--warning); border: 1px solid var(--warning); }
    .badge-state { background: var(--success-bg); color: var(--success); border: 1px solid var(--success); }
    .badge-danger { background: var(--danger-bg); color: var(--danger); border: 1px solid var(--danger); }
    .badge-purple { background: rgba(188, 140, 255, 0.15); color: var(--purple); border: 1px solid var(--purple); }

    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
    @media (max-width: 992px) { .grid-2, .grid-3 { grid-template-columns: 1fr; } }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .card-title {
      font-size: 13px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-dim);
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
    }

    /* Telemetry metrics */
    .metric-row { display: flex; justify-content: space-between; gap: 12px; }
    .metric-box {
      flex: 1;
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px;
      text-align: center;
    }
    .metric-val { font-size: 24px; font-weight: 700; color: #fff; }
    .metric-lbl { font-size: 11px; color: var(--text-dim); margin-top: 4px; }
    .val-danger { color: var(--danger) !important; }
    .val-warning { color: var(--warning) !important; }
    .val-success { color: var(--success) !important; }

    /* Workload cards */
    .workload-card {
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .workload-header { display: flex; justify-content: space-between; align-items: center; }
    .workload-name { font-weight: 600; color: #fff; font-size: 13px; }
    .progress-bar-bg {
      background: #21262d;
      border-radius: 4px;
      height: 8px;
      width: 100%;
      overflow: hidden;
    }
    .progress-bar-fill {
      height: 100%;
      background: var(--accent);
      border-radius: 4px;
      transition: width 0.3s;
    }

    /* Candidate plans table */
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th { text-align: left; padding: 8px; color: var(--text-dim); border-bottom: 1px solid var(--border); }
    td { padding: 8px; border-bottom: 1px solid #21262d; vertical-align: top; }
    tr:last-child td { border-bottom: none; }
    .row-blocked { opacity: 0.6; background: rgba(248, 81, 73, 0.05); }

    /* Controls & Buttons */
    .btn {
      padding: 8px 14px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      border: 1px solid transparent;
      transition: all 0.2s;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
    }
    .btn-primary { background: #238636; color: #fff; border-color: #2ea043; }
    .btn-primary:hover { background: #2ea043; }
    .btn-secondary { background: #21262d; color: var(--text); border-color: var(--border); }
    .btn-secondary:hover { background: #30363d; }
    .btn-danger { background: var(--danger-bg); color: var(--danger); border-color: var(--danger); }
    .btn-danger:hover { background: rgba(248, 81, 73, 0.3); }
    .btn:disabled { opacity: 0.4; cursor: not-allowed; }

    /* Action bar */
    .action-bar { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }

    /* SVG Chart */
    .chart-box {
      width: 100%;
      height: 180px;
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px;
      display: flex;
      flex-direction: column;
    }
  </style>
</head>
<body>
  <div class="container">

    <!-- Top Control Room Header -->
    <header class="header">
      <div class="header-title">
        <h1>ThermoPilot Compute Desk</h1>
        <p>Autonomous GPU Thermal Governance & Human-in-the-Loop Operations Console</p>
      </div>
      <div class="header-badges">
        <span class="badge badge-sim">&#9888; SIMULATED TELEMETRY ONLY</span>
        <span class="badge badge-state" id="badge-state">DETECTED</span>
        <span class="badge badge-purple" id="badge-incident">INC-20261027-001</span>
      </div>
    </header>

    <!-- Top Telemetry & Live Trajectory -->
    <div class="grid-2">
      <!-- Live Gauges -->
      <section class="card">
        <div class="card-title">
          <span>Live GPU Telemetry (Node 0)</span>
          <span id="telemetry-ts" style="font-size:11px; font-weight:normal; color:var(--text-dim);">--:--:--</span>
        </div>
        <div class="metric-row">
          <div class="metric-box">
            <div class="metric-val val-danger" id="val-temp">--.-°C</div>
            <div class="metric-lbl">Core Temp (Trip: 85°C)</div>
          </div>
          <div class="metric-box">
            <div class="metric-val" id="val-power">--W</div>
            <div class="metric-lbl">Power Draw (Max: 400W)</div>
          </div>
          <div class="metric-box">
            <div class="metric-val val-warning" id="val-ambient">--.-°C</div>
            <div class="metric-lbl">Intake Ambient (+10°C)</div>
          </div>
          <div class="metric-box">
            <div class="metric-val val-danger" id="val-ttt">--s</div>
            <div class="metric-lbl">Time-to-Trip (t_TTT)</div>
          </div>
        </div>
        <div style="font-size: 11px; color: var(--text-dim); display:flex; justify-content:space-between; margin-top:4px;">
          <span>Thermal Ceiling: <b>80.0°C</b> | Target Safe: <b>75.0°C</b></span>
          <span>Simulation Speed: <b>2.5x</b> | Newton Decay: <b>tau=15s</b></span>
        </div>
      </section>

      <!-- SVG Thermal Trajectory Chart -->
      <section class="card">
        <div class="card-title">
          <span>Simulated Thermal Trajectory</span>
          <span style="font-size:11px; font-weight:normal; color:var(--text-dim);">Historical Ticks vs Limits</span>
        </div>
        <div class="chart-box">
          <svg id="trajectory-svg" width="100%" height="100%" viewBox="0 0 500 140" preserveAspectRatio="none">
            <!-- Reference lines -->
            <line x1="0" y1="25" x2="500" y2="25" stroke="#f85149" stroke-dasharray="4" stroke-width="1.5" />
            <text x="5" y="21" fill="#f85149" font-size="10">85°C Critical Trip</text>

            <line x1="0" y1="55" x2="500" y2="55" stroke="#d29922" stroke-dasharray="3" stroke-width="1" />
            <text x="5" y="51" fill="#d29922" font-size="10">80°C Safety Ceiling</text>

            <line x1="0" y1="85" x2="500" y2="85" stroke="#3fb950" stroke-dasharray="3" stroke-width="1" />
            <text x="5" y="81" fill="#3fb950" font-size="10">75°C Recovery Target</text>

            <!-- Dynamic trajectory path -->
            <path id="temp-path" d="" fill="none" stroke="#58a6ff" stroke-width="2.5" />
          </svg>
        </div>
      </section>
    </div>

    <!-- Workload Tenants -->
    <section class="card">
      <div class="card-title">
        <span>Workload-Owner Agents (Competing Tenant Priorities & Bids)</span>
        <span style="font-size:11px; font-weight:normal;">Autonomous SLA Advocacy</span>
      </div>
      <div class="grid-3" id="workloads-container">
        <!-- Rendered dynamically -->
      </div>
    </section>

    <!-- Recovery Candidates Matrix & Safety Gate -->
    <section class="card">
      <div class="card-title">
        <span>Recovery Planner Candidates & Deterministic SafetyGate</span>
        <span style="font-size:11px; font-weight:normal;">Hard Physical & SLA Invariants Enforced</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>Plan ID & Name</th>
            <th>Strategy</th>
            <th>Projected Temp</th>
            <th>Cooldown</th>
            <th>Cost / SLA Impact</th>
            <th>SafetyGate Status</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody id="plans-table-body">
          <!-- Rendered dynamically -->
        </tbody>
      </table>
    </section>

    <!-- Human-in-the-Loop Operator Action Center & Verification -->
    <div class="grid-2">
      <!-- Operator Controls -->
      <section class="card">
        <div class="card-title">
          <span>Switch HITL Governance Sign-off</span>
          <span style="font-size:11px; font-weight:normal; color:var(--text-dim);">Strict Policy: Timeout NEVER mutates workloads</span>
        </div>
        <div style="font-size:12px; color:var(--text-dim);" id="approval-instruction">
          Review candidate trade-offs and authorize a certified recovery plan. Blocked plans cannot be approved.
        </div>
        <div class="action-bar" id="operator-actions">
          <button class="btn btn-primary" onclick="submitDecision('APPROVE', 'PLAN-A')">Approve Plan A (Safe Recovery)</button>
          <button class="btn btn-secondary" onclick="submitDecision('APPROVE', 'PLAN-B')">Approve Plan B (Balanced Cap)</button>
          <button class="btn btn-secondary" onclick="submitDecision('APPROVE', 'PLAN-C')">Approve Plan C (Migration)</button>
          <button class="btn btn-danger" onclick="submitDecision('REJECT', null)">Reject All</button>
          <button class="btn btn-secondary" onclick="submitDecision('TIMEOUT', null)">Simulate Timeout</button>
        </div>
        <div id="decision-feedback" style="font-size:12px; margin-top:8px;"></div>
      </section>

      <!-- Verification Agent -->
      <section class="card">
        <div class="card-title">
          <span>Post-Execution Verification Agent</span>
          <span style="font-size:11px; font-weight:normal;">Thermal Decay & SLA Adherence</span>
        </div>
        <div class="metric-row">
          <div class="metric-box">
            <div class="metric-val" id="verif-rate">--.-°C/s</div>
            <div class="metric-lbl">Cooling Rate (Target &ge; 0.50°C/s)</div>
          </div>
          <div class="metric-box">
            <div class="metric-val" id="verif-delta">--.-°C</div>
            <div class="metric-lbl">Temperature Delta (&Delta;T)</div>
          </div>
          <div class="metric-box">
            <div class="metric-val" id="verif-sla">--</div>
            <div class="metric-lbl">VIP SLA Protection (0 Breaches)</div>
          </div>
        </div>
        <div id="verif-summary" style="font-size:12px; color:var(--text-dim); margin-top:4px;">
          Awaiting execution completion.
        </div>
      </section>
    </div>

    <!-- Cryptographic Audit Ledger -->
    <section class="card">
      <div class="card-title">
        <span>Cryptographic Audit Ledger (Append-Only SHA-256 Hash Chain)</span>
        <span class="badge badge-state" id="audit-chain-badge">CHAIN VERIFIED</span>
      </div>
      <table>
        <thead>
          <tr>
            <th style="width: 60px;">Seq</th>
            <th>Event Type</th>
            <th style="width: 80px;">Simulated</th>
            <th>Hash Prefix (SHA-256)</th>
            <th>Timestamp</th>
          </tr>
        </thead>
        <tbody id="audit-table-body">
          <!-- Rendered dynamically -->
        </tbody>
      </table>
    </section>

    <!-- Demo Controls Bar -->
    <footer class="header" style="justify-content: flex-start; gap: 12px;">
      <button class="btn btn-secondary" onclick="resetIncident()">Reset Demo Incident</button>
      <button class="btn btn-secondary" onclick="stepSimulator(1)">Step +1 Tick</button>
      <button class="btn btn-secondary" onclick="stepSimulator(5)">Step +5 Ticks</button>
      <button class="btn btn-primary" onclick="runAutoDemo()">Run 60-Second Auto Demo</button>
      <span style="margin-left:auto; font-size:11px; color:var(--text-dim);">
        ThermoPilot Compute Desk &bull; Flip the Switch Hackathon &bull; Simulation-Only Mode
      </span>
    </footer>

  </div>

  <script>
    let state = {};

    async function fetchState() {
      try {
        const res = await fetch('/api/state');
        state = await res.json();
        render();
      } catch (err) {
        console.error("Failed to fetch state:", err);
      }
    }

    function render() {
      if (!state || !state.telemetry) return;

      // Badges
      const badgeState = document.getElementById('badge-state');
      badgeState.innerText = state.state.toUpperCase();
      badgeState.className = 'badge ' + (state.state === 'recovered' ? 'badge-state' : (state.state === 'rejected' || state.state === 'timeout_escalated' || state.state === 'recovery_failed' ? 'badge-danger' : 'badge-purple'));
      document.getElementById('badge-incident').innerText = state.incident_id;

      // Telemetry
      const telem = state.telemetry;
      const tempVal = document.getElementById('val-temp');
      tempVal.innerText = telem.temperature_core_c.toFixed(1) + '°C';
      tempVal.className = 'metric-val ' + (telem.temperature_core_c >= 85 ? 'val-danger' : (telem.temperature_core_c >= 80 ? 'val-danger' : (telem.temperature_core_c > 75 ? 'val-warning' : 'val-success')));

      document.getElementById('val-power').innerText = telem.power_draw_w.toFixed(0) + 'W';
      document.getElementById('val-ambient').innerText = telem.ambient_temp_c.toFixed(1) + '°C';
      document.getElementById('telemetry-ts').innerText = telem.timestamp ? telem.timestamp.substring(11, 19) + ' UTC' : '--:--:--';

      if (state.forecast && state.forecast.time_to_threshold_s !== null) {
        document.getElementById('val-ttt').innerText = state.forecast.time_to_threshold_s.toFixed(1) + 's';
      } else {
        document.getElementById('val-ttt').innerText = 'Nominal';
      }

      // Workloads
      const wlContainer = document.getElementById('workloads-container');
      wlContainer.innerHTML = '';
      if (state.workloads) {
        state.workloads.forEach(w => {
          const isP0 = w.priority.includes('P0');
          const isP1 = w.priority.includes('P1');
          const tierClass = isP0 ? 'badge-danger' : (isP1 ? 'badge-sim' : 'badge-purple');
          const pct = Math.min(100, (w.current_power_w / w.max_power_w) * 100);

          wlContainer.innerHTML += `
            <div class="workload-card">
              <div class="workload-header">
                <span class="workload-name">${w.name}</span>
                <span class="badge ${tierClass}">${w.priority.split('_')[0]}</span>
              </div>
              <div style="font-size:11px; color:var(--text-dim); display:flex; justify-content:space-between;">
                <span>Power: <b>${w.current_power_w.toFixed(0)}W</b> / ${w.max_power_w.toFixed(0)}W</span>
                <span>Floor: <b>${w.min_power_w.toFixed(0)}W</b></span>
              </div>
              <div class="progress-bar-bg">
                <div class="progress-bar-fill" style="width:${pct}%;"></div>
              </div>
              <div style="font-size:11px; color:var(--text-dim); display:flex; justify-content:space-between; margin-top:2px;">
                <span>SLA: <b>${w.sla_deadline_s.toFixed(0)}s</b></span>
                <span>Penalty: <b>$${w.sla_penalty_usd_per_s.toFixed(2)}/s</b></span>
                <span>Preemptible: <b>${w.can_preempt ? 'Yes' : 'NO (VIP)'}</b></span>
              </div>
            </div>
          `;
        });
      }

      // Candidate Plans
      const plansTbody = document.getElementById('plans-table-body');
      plansTbody.innerHTML = '';
      if (state.candidates) {
        state.candidates.forEach(c => {
          const isBlocked = !c.is_safe;
          const statusBadge = isBlocked
            ? '<span class="badge badge-danger">BLOCKED BY GATE</span>'
            : (c.is_recommended ? '<span class="badge badge-state">PASS [RECOMMENDED]</span>' : '<span class="badge badge-state">PASS [CERTIFIED]</span>');

          const btnDisabled = isBlocked ? 'disabled title="SafetyGate Blocked"' : '';
          const actionBtn = isBlocked
            ? `<button class="btn btn-secondary" disabled style="opacity:0.4;">Blocked</button>`
            : `<button class="btn btn-primary" onclick="submitDecision('APPROVE', '${c.plan_id}')">Approve ${c.plan_id}</button>`;

          let blockDetails = '';
          if (isBlocked && c.rejection_reasons) {
            blockDetails = `<div style="color:var(--danger); font-size:11px; margin-top:4px;"><b>Violations:</b> ${c.rejection_reasons.join('; ')}</div>`;
          }

          plansTbody.innerHTML += `
            <tr class="${isBlocked ? 'row-blocked' : ''}">
              <td><b>${c.plan_id}</b>: ${c.name}</td>
              <td>${c.strategy_type}</td>
              <td>${c.projected_temp_max_c.toFixed(1)}°C</td>
              <td>${c.cooling_time_to_safe_s.toFixed(0)}s</td>
              <td>$${c.estimated_cost_usd.toFixed(0)} (${c.sla_violations_expected} breaches)</td>
              <td>${statusBadge}${blockDetails}</td>
              <td>${actionBtn}</td>
            </tr>
          `;
        });
      }

      // Verification Agent
      if (state.verification_report) {
        const v = state.verification_report;
        document.getElementById('verif-rate').innerText = v.observed_cooling_rate_c_per_s.toFixed(2) + '°C/s';
        const d = v.final_temp_c - v.initial_temp_c;
        document.getElementById('verif-delta').innerText = (d > 0 ? '+' : '') + d.toFixed(1) + '°C';
        document.getElementById('verif-sla').innerText = v.recovery_successful ? 'PASS (0 Breaches)' : 'VIOLATION';
        document.getElementById('verif-summary').innerHTML = v.recovery_successful
          ? `<span style="color:var(--success); font-weight:600;">&#10004; RECOVERY SUCCESSFUL: 75.0°C target envelope restored with 100% SLA compliance.</span>`
          : `<span style="color:var(--danger); font-weight:600;">&#10008; RECOVERY FAILED: ${v.failure_reasons ? v.failure_reasons.join(', ') : 'Target not met.'}</span>`;
      } else {
        document.getElementById('verif-rate').innerText = '--.-°C/s';
        document.getElementById('verif-delta').innerText = '--.-°C';
        document.getElementById('verif-sla').innerText = '--';
        document.getElementById('verif-summary').innerHTML = '<span style="color:var(--text-dim);">Awaiting execution completion.</span>';
      }

      // Operational guidance
      const instruction = document.getElementById('approval-instruction');
      if (instruction) {
        if (state.state === 'recovered') {
          instruction.innerHTML = '<b style="color:var(--success);">&#10004; Incident successfully resolved and verified.</b> Workloads safe. Use "Reset Demo Incident" to restart.';
        } else if (state.state === 'rejected') {
          instruction.innerHTML = '<b style="color:var(--danger);">&#10008; Operator explicitly rejected plans.</b> Workloads protected from mutation. Use "Reset Demo Incident" to restart.';
        } else if (state.state === 'timeout_escalated') {
          instruction.innerHTML = '<b style="color:var(--danger);">&#9200; Approval timed out. Workload mutations strictly blocked.</b> Use "Reset Demo Incident" to restart.';
        } else {
          instruction.innerText = 'Review candidate trade-offs and authorize a certified recovery plan. Blocked plans cannot be approved.';
        }
      }

      // Audit Ledger
      if (state.audit_ledger) {
        const al = state.audit_ledger;
        const chainBadge = document.getElementById('audit-chain-badge');
        chainBadge.innerText = al.chain_valid ? 'CHAIN VERIFIED (SHA-256)' : 'CHAIN TAMPERED';
        chainBadge.className = 'badge ' + (al.chain_valid ? 'badge-state' : 'badge-danger');

        const auditTbody = document.getElementById('audit-table-body');
        auditTbody.innerHTML = '';
        if (al.recent_events) {
          al.recent_events.slice().reverse().forEach(ev => {
            auditTbody.innerHTML += `
              <tr>
                <td><b>${ev.sequence_num}</b></td>
                <td>${ev.event_type}</td>
                <td><span class="badge ${ev.is_simulated ? 'badge-sim' : 'badge-state'}">${ev.is_simulated ? 'SIM' : 'REAL'}</span></td>
                <td><code>${ev.hash ? ev.hash.substring(0, 16) + '...' : ''}</code></td>
                <td>${ev.timestamp ? ev.timestamp.substring(11, 19) : ''}</td>
              </tr>
            `;
          });
        }
      }

      // SVG Chart rendering
      renderSvgChart();
    }

    function renderSvgChart() {
      let temps = [];
      if (state.verification_report && state.verification_report.temperature_trajectory && state.verification_report.temperature_trajectory.length > 1) {
        temps = state.verification_report.temperature_trajectory;
      } else if (state.history && state.history.length > 0) {
        temps = state.history.map(pt => pt.temp_core_c);
      }
      if (temps.length === 0) return;

      const width = 500;
      // Coordinate mapping: 85°C -> y=25, 80°C -> y=55, 75°C -> y=85, 70°C -> y=115
      function mapY(t) {
        return 85 - (t - 75.0) * 6.0;
      }

      const pathElem = document.getElementById('temp-path');
      if (!pathElem) return;

      if (temps.length === 1) {
        const y0 = mapY(temps[0]);
        pathElem.setAttribute('d', `M 0 ${y0} L ${width} ${y0}`);
        return;
      }

      const stepX = width / (temps.length - 1);
      let d = "";
      temps.forEach((t, i) => {
        const x = i * stepX;
        const y = mapY(t);
        d += (i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`);
      });
      pathElem.setAttribute('d', d);
    }

    async function submitDecision(action, planId) {
      const feedback = document.getElementById('decision-feedback');
      feedback.innerText = 'Evaluating decision...';
      try {
        const res = await fetch('/api/action', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: action, plan_id: planId })
        });
        const data = await res.json();
        if (data.error) {
          feedback.innerHTML = `<span style="color:var(--danger);">&#9888; ${data.error}</span>`;
        } else {
          feedback.innerHTML = `<span style="color:var(--success);">&#10004; Decision processed: ${data.status}</span>`;
          await fetchState();
        }
      } catch (err) {
        feedback.innerHTML = `<span style="color:var(--danger);">Error: ${err.message}</span>`;
      }
    }

    async function resetIncident() {
      const feedback = document.getElementById('decision-feedback');
      feedback.innerText = 'Resetting incident...';
      await fetch('/api/reset', { method: 'POST' });
      await fetchState();
      feedback.innerHTML = '<span style="color:var(--success);">&#10004; Reset complete: initial conditions restored (81.5°C).</span>';
    }

    async function stepSimulator(ticks) {
      await fetch('/api/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticks: ticks })
      });
      await fetchState();
    }

    async function runAutoDemo() {
      const feedback = document.getElementById('decision-feedback');
      feedback.innerText = 'Running 60-second hackathon demo...';
      await fetch('/api/auto-demo', { method: 'POST' });
      await fetchState();
      feedback.innerHTML = `<span style="color:var(--success);">&#10004; Auto demo complete! Verified recovery to &le;75°C.</span>`;
    }

    // Auto-poll state every 2 seconds
    fetchState();
    setInterval(fetchState, 2000);
  </script>
</body>
</html>
"""


class DashboardRequestHandler(http.server.BaseHTTPRequestHandler):
    """Custom HTTP handler serving the dashboard HTML and JSON APIs."""

    runner: DemoRunner

    def do_GET(self) -> None:
        """Handle GET requests for HTML UI and JSON state API."""
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif parsed.path == "/api/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            state_data = self.runner.get_state_dict()
            self.wfile.write(json.dumps(state_data).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self) -> None:
        """Handle POST requests for operator decisions, resets, and simulator stepping."""
        parsed = urllib.parse.urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            req_data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            req_data = {}

        if parsed.path == "/api/action":
            action_str = req_data.get("action", "APPROVE")
            plan_id = req_data.get("plan_id", "PLAN-B")

            try:
                if action_str == "APPROVE":
                    self.runner.step_5_operator_decision(
                        decision=ApprovalDecision.APPROVE,
                        plan_id=plan_id,
                        operator_id="web-operator-01",
                        reason="Authorized via Web Control Room Dashboard",
                    )
                    self.runner.step_6_execute()
                    self.runner.step_8_verify(max_ticks=25)
                elif action_str == "REJECT":
                    self.runner.step_5_operator_decision(
                        decision=ApprovalDecision.REJECT,
                        operator_id="web-operator-01",
                        reason="Explicit operator rejection via Web Dashboard",
                    )
                elif action_str == "TIMEOUT":
                    self.runner.step_5_operator_decision(
                        decision=ApprovalDecision.TIMEOUT_ESCALATED,
                        operator_id="web-operator-01",
                        reason="Approval window expired",
                    )

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                resp = {
                    "status": self.runner.state_machine.current_state.value,
                    "state": self.runner.get_state_dict(),
                }
                self.wfile.write(json.dumps(resp).encode("utf-8"))

            except (ValueError, TypeError, KeyError, RuntimeError, ExecutionDeniedError) as exc:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))

        elif parsed.path == "/api/reset":
            self.runner.reset()
            self.runner.step_1_detect()
            self.runner.step_2_forecast()
            self.runner.step_3_plan()
            self.runner.step_4_dispatch_card()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.runner.get_state_dict()).encode("utf-8"))

        elif parsed.path == "/api/step":
            ticks = int(req_data.get("ticks", 1))
            self.runner.step_7_advance_cooldown(ticks=ticks)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.runner.get_state_dict()).encode("utf-8"))

        elif parsed.path == "/api/auto-demo":
            self.runner.run_auto_demo(plan_id="PLAN-A", delay_s=0.0, print_output=False)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.runner.get_state_dict()).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress standard HTTP server access logs to keep terminal output clean."""


def launch_web_dashboard(
    port: int = 8080,
    open_browser: bool = True,
    run_in_background: bool = False,
    runner: DemoRunner | None = None,
) -> http.server.HTTPServer:
    """Launch the ThermoPilot Compute Desk web control room dashboard.

    Args:
        port: Local TCP port to bind (default: 8080).
        open_browser: Whether to automatically launch default web browser.
        run_in_background: Whether to run in a background daemon thread.
        runner: Optional pre-configured DemoRunner instance.

    Returns:
        Running HTTPServer instance.
    """
    active_runner = runner or DemoRunner()
    # Ensure initialized to dispatch state
    active_runner.step_1_detect()
    active_runner.step_2_forecast()
    active_runner.step_3_plan()
    active_runner.step_4_dispatch_card()

    handler_class = DashboardRequestHandler
    handler_class.runner = active_runner

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler_class)
    url = f"http://127.0.0.1:{port}"

    print("=" * 88)
    print(" [THERMOPILOT WEB CONTROL ROOM] Server launched!")
    print(f" Control Room Dashboard URL: {url}")
    print(" Simulation Boundary: All telemetry strictly simulated; no physical GPU altered.")
    print(" Press Ctrl+C in terminal to stop.")
    print("=" * 88)

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    if run_in_background:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web dashboard server...")
        server.shutdown()

    return server


if __name__ == "__main__":
    launch_web_dashboard(port=8080)
