"""Discrete-time GPU thermal telemetry simulator.

Models two competing workloads sharing a 400W GPU node thermal envelope,
intake ambient heating anomalies, and configurable time-scaling for demonstration.

MODEL ASSUMPTIONS & LIMITATIONS:
This simulation is a synthetic physics-approximating model engineered for deterministic
algorithmic governance, safety gating, and presentation evaluation. It does NOT claim
to replace vendor hardware drivers (NVIDIA NVML / DCGM) or IPMI chassis controllers.
Key simplifying assumptions include:
1. Lumped Thermal Mass: The GPU silicon, heatsink, and thermal interface material are
   modeled as a single lumped thermal node with a constant thermal time constant
   (tau = 15.0s) and thermal resistance (R_th = 0.14 °C/W).
2. Linear Newton Cooling: Heat dissipation toward the ambient air intake scales linearly
   with the temperature gradient: dQ/dt = (T_core - T_amb) / R_th.
3. Node Workload Power Aggregation: Workloads Alpha and Beta share a single 400.0W total
   electrical power rail and dissipation path.
4. Empirical Junction & Fan Models: Memory/junction hotspot temperatures and fan RPMs
   are calculated via monotonic transfer functions rather than OEM-specific firmware tables.
5. Discrete Integration: First-order forward Euler integration is used with decoupled
   simulation time: T_{t+1} = T_t + ((P_t * R_th - (T_t - T_amb)) / tau) * (dt * time_scale).
"""

import random
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field, model_validator

from thermopilot.schemas import GPUTelemetry


class SimulatorConfig(BaseModel):
    """Configuration parameters for the GPU node thermal simulator."""

    power_budget_w: float = Field(
        default=400.0, ge=50.0, le=1000.0, description="Total node electrical power budget in Watts"
    )
    baseline_ambient_c: float = Field(
        default=22.0, description="Nominal rack intake ambient temperature in Celsius"
    )
    anomaly_ambient_c: float = Field(
        default=32.0, description="Intake ambient temperature during thermal anomaly"
    )
    thermal_resistance: float = Field(
        default=0.14, gt=0.0, description="Thermal resistance R_th in °C/W"
    )
    thermal_time_constant: float = Field(
        default=15.0, gt=0.0, description="Thermal time constant tau in seconds"
    )
    time_scale: float = Field(
        default=2.5, gt=0.0, description="Simulation acceleration factor for demo"
    )
    base_dt: float = Field(
        default=1.0, gt=0.0, description="Base integration timestep in seconds"
    )
    initial_temp_c: float | None = Field(
        default=71.2, description="Initial core temperature in Celsius (or steady-state if None)"
    )
    noise_std: float = Field(
        default=0.0, ge=0.0, description="Standard deviation of stochastic temperature noise"
    )
    seed: int | None = Field(
        default=42, description="Random seed for reproducible stochastic noise"
    )
    default_workloads: dict[str, float] = Field(
        default_factory=lambda: {"wl-alpha": 230.0, "wl-beta": 150.0},
        description="Initial workload power allocations in Watts",
    )

    @model_validator(mode="after")
    def validate_config(self) -> "SimulatorConfig":
        """Validate initial workload allocations and ODE numerical stability bounds."""
        for wid, p in self.default_workloads.items():
            if p < 0.0:
                raise ValueError(f"Workload power must be non-negative, got {p}W for '{wid}'")
        total_workload_power = sum(self.default_workloads.values())
        if total_workload_power > self.power_budget_w:
            raise ValueError(
                f"Initial workload allocations ({total_workload_power:.1f}W) exceed "
                f"configured node power budget ({self.power_budget_w:.1f}W)"
            )

        effective_dt = self.base_dt * self.time_scale
        max_stable_dt = 2.0 * self.thermal_time_constant
        if effective_dt >= max_stable_dt:
            raise ValueError(
                f"Effective integration timestep ({effective_dt:.2f}s = base_dt {self.base_dt}s * "
                f"time_scale {self.time_scale}) destabilizes the thermal equation. "
                f"Must be strictly less than 2 * tau ({max_stable_dt:.2f}s)."
            )
        return self


class TelemetrySimulator:
    """Discrete-time physics simulator emitting validated GPUTelemetry records."""

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        self.config = config or SimulatorConfig()
        self._rng = random.Random(self.config.seed)

        # Validate initial workload power allocations against node power budget
        for wid, p in self.config.default_workloads.items():
            if p < 0.0:
                raise ValueError(f"Workload power must be non-negative, got {p}W for '{wid}'")
        total_initial_power = sum(self.config.default_workloads.values())
        if total_initial_power > self.config.power_budget_w:
            raise ValueError(
                f"Initial workload allocations ({total_initial_power:.1f}W) exceed "
                f"configured node power budget ({self.config.power_budget_w:.1f}W)"
            )

        # Simulation clock completely decoupled from wall-clock time
        self._sim_time_s: float = 0.0
        self._tick_count: int = 0
        self._sim_datetime: datetime = datetime(2026, 10, 27, 10, 0, 0, tzinfo=timezone.utc)

        # Workload allocations (workload_id -> power_w)
        self._workload_powers: dict[str, float] = dict(self.config.default_workloads)

        # Ambient & anomaly state
        self._is_anomaly_active: bool = False
        self._anomaly_ambient_override: float | None = None
        self._scheduled_anomaly_start: int | None = None
        self._scheduled_anomaly_end: int | None = None

        # Core physical state
        if self.config.initial_temp_c is not None:
            self._core_temp_c: float = float(self.config.initial_temp_c)
        else:
            # Initialize to baseline steady state: T_amb + P_total * R_th
            self._core_temp_c = (
                self.config.baseline_ambient_c
                + self.effective_power_draw_w * self.config.thermal_resistance
            )

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def core_temp_c(self) -> float:
        """Current GPU core temperature in Celsius."""
        return self._core_temp_c

    @property
    def junction_temp_c(self) -> float:
        """Estimated hotspot / memory junction temperature in Celsius."""
        # Empirical linear offset proportional to power dissipation
        return round(self._core_temp_c + (0.02 * self.effective_power_draw_w) + 0.5, 2)

    @property
    def ambient_temp_c(self) -> float:
        """Current rack intake ambient temperature in Celsius."""
        if self.is_anomaly_active:
            return (
                self._anomaly_ambient_override
                if self._anomaly_ambient_override is not None
                else self.config.anomaly_ambient_c
            )
        return self.config.baseline_ambient_c

    @property
    def total_allocated_power_w(self) -> float:
        """Sum of all requested workload powers in Watts."""
        return sum(self._workload_powers.values())

    @property
    def effective_power_draw_w(self) -> float:
        """Actual power draw clamped to the hard node power budget."""
        return min(self.total_allocated_power_w, self.config.power_budget_w)

    @property
    def power_budget_w(self) -> float:
        """Node total power budget in Watts."""
        return self.config.power_budget_w

    @property
    def time_scale(self) -> float:
        """Simulation time scaling factor."""
        return self.config.time_scale

    @property
    def sim_time_s(self) -> float:
        """Total elapsed simulated seconds."""
        return self._sim_time_s

    @property
    def tick_count(self) -> int:
        """Number of simulation ticks elapsed."""
        return self._tick_count

    @property
    def is_anomaly_active(self) -> bool:
        """Whether the intake heating anomaly is currently active."""
        if self._is_anomaly_active:
            return True
        if self._scheduled_anomaly_start is not None:
            if self._scheduled_anomaly_end is not None:
                return (
                    self._scheduled_anomaly_start <= self._tick_count < self._scheduled_anomaly_end
                )
            return self._tick_count >= self._scheduled_anomaly_start
        return False

    @property
    def workload_powers(self) -> dict[str, float]:
        """Read-only copy of active workload power allocations."""
        return dict(self._workload_powers)

    # -----------------------------------------------------------------------
    # Workload & Power Budget Management
    # -----------------------------------------------------------------------

    def set_workload_power(
        self, workload_id: str, power_w: float, clamp: bool = True
    ) -> float:
        """Set allocated power for a specific workload.

        Args:
            workload_id: Unique workload identifier (e.g. 'wl-alpha').
            power_w: Desired power in Watts (must be >= 0.0).
            clamp: If True, clamps power so total allocation does not exceed
                   the node budget. If False, raises ValueError when exceeded.

        Returns:
            The actual power allocated to this workload in Watts.
        """
        if power_w < 0.0:
            raise ValueError(f"Workload power must be non-negative, got {power_w}W")

        other_power = sum(
            p for wid, p in self._workload_powers.items() if wid != workload_id
        )
        remaining_budget = max(0.0, self.config.power_budget_w - other_power)

        if clamp:
            allocated = min(power_w, remaining_budget)
        else:
            if other_power + power_w > self.config.power_budget_w:
                raise ValueError(
                    f"Total power ({other_power + power_w:.1f}W) exceeds "
                    f"power budget ({self.config.power_budget_w:.1f}W)"
                )
            allocated = power_w

        self._workload_powers[workload_id] = allocated
        return allocated

    def remove_workload(self, workload_id: str) -> None:
        """Remove a workload from the node (e.g. paused or migrated)."""
        self._workload_powers.pop(workload_id, None)

    def get_workload_power(self, workload_id: str) -> float:
        """Retrieve current power allocated to a workload (0.0 if inactive)."""
        return self._workload_powers.get(workload_id, 0.0)

    # -----------------------------------------------------------------------
    # Anomaly Controls
    # -----------------------------------------------------------------------

    def trigger_anomaly(self, target_ambient_c: float | None = None) -> None:
        """Deterministically trigger the intake heating anomaly."""
        self._is_anomaly_active = True
        self._anomaly_ambient_override = target_ambient_c

    def resolve_anomaly(self) -> None:
        """Deterministically restore ambient intake conditions to nominal baseline."""
        self._is_anomaly_active = False
        self._anomaly_ambient_override = None
        self._scheduled_anomaly_start = None
        self._scheduled_anomaly_end = None

    def schedule_anomaly(self, start_tick: int, end_tick: int | None = None) -> None:
        """Schedule an automatic anomaly window by simulation tick indices.

        Args:
            start_tick: Simulation tick index at which anomaly begins (inclusive, must be >= 0).
            end_tick: Simulation tick index at which anomaly ends (exclusive, must be > start_tick).
        """
        if not isinstance(start_tick, int) or isinstance(start_tick, bool):
            raise TypeError(f"start_tick must be an integer, got {type(start_tick).__name__}")
        if start_tick < 0:
            raise ValueError(f"start_tick must be non-negative, got {start_tick}")
        if end_tick is not None:
            if not isinstance(end_tick, int) or isinstance(end_tick, bool):
                raise TypeError(f"end_tick must be an integer, got {type(end_tick).__name__}")
            if end_tick <= start_tick:
                raise ValueError(
                    f"end_tick ({end_tick}) must be strictly greater than start_tick ({start_tick})"
                )
        self._scheduled_anomaly_start = start_tick
        self._scheduled_anomaly_end = end_tick

    # -----------------------------------------------------------------------
    # Step & Telemetry Generation
    # -----------------------------------------------------------------------

    def step(self, dt: float | None = None) -> GPUTelemetry:
        """Advance the simulation by one discrete timestep and emit telemetry.

        Args:
            dt: Base duration of the step in seconds. If None, uses config.base_dt.
                Effective physical delta is magnified by config.time_scale.

        Returns:
            A validated GPUTelemetry instance reflecting the new state.
        """
        # 1. Compute effective integration timestep
        base_step_s = dt if dt is not None else self.config.base_dt
        if base_step_s <= 0.0:
            raise ValueError(f"Integration timestep must be positive, got {base_step_s}s")

        effective_dt_s = base_step_s * self.config.time_scale
        max_stable_dt_s = 2.0 * self.config.thermal_time_constant
        if effective_dt_s >= max_stable_dt_s:
            raise ValueError(
                f"Effective timestep ({effective_dt_s:.2f}s) destabilizes forward Euler integration. "
                f"Must be strictly less than 2 * tau ({max_stable_dt_s:.2f}s)."
            )

        # 2. Integrate thermal ODE:
        #    dT = ((P * R_th - (T - T_amb)) / tau) * effective_dt
        p_eff = self.effective_power_draw_w
        r_th = self.config.thermal_resistance
        t_amb = self.ambient_temp_c
        tau = self.config.thermal_time_constant

        delta_temp = ((p_eff * r_th - (self._core_temp_c - t_amb)) / tau) * effective_dt_s

        # Apply stochastic noise if configured
        noise = 0.0
        if self.config.noise_std > 0.0:
            noise = self._rng.gauss(0.0, self.config.noise_std)

        self._core_temp_c += delta_temp + noise

        # 3. Advance decoupled simulation clock
        self._sim_time_s += effective_dt_s
        self._sim_datetime += timedelta(seconds=effective_dt_s)
        self._tick_count += 1

        # 4. Emit validated GPUTelemetry
        return self.current_telemetry()

    def current_telemetry(self) -> GPUTelemetry:
        """Generate a validated GPUTelemetry snapshot of the current state."""
        # Calculate fan speed percentage (monotonically ramps from 55% at 65°C to 100% at 85°C)
        t_core = self._core_temp_c
        if t_core <= 60.0:
            fan_pct = 50.0
        elif t_core >= 85.0:
            fan_pct = 100.0
        else:
            fan_pct = 50.0 + ((t_core - 60.0) / 25.0) * 50.0

        # Operating clock frequency (proportional to power allocation fraction)
        power_ratio = self.effective_power_draw_w / self.config.power_budget_w
        clock_freq = int(1200 + 645 * power_ratio)

        # Utilization (near 100% when active)
        utilization = min(100.0, max(0.0, 85.0 + 13.0 * power_ratio))

        return GPUTelemetry(
            timestamp=self._sim_datetime,
            gpu_id=0,
            temperature_core_c=round(self._core_temp_c, 2),
            temperature_junction_c=self.junction_temp_c,
            power_draw_w=round(self.effective_power_draw_w, 2),
            power_limit_w=round(self.config.power_budget_w, 2),
            fan_speed_pct=round(fan_pct, 1),
            clock_freq_mhz=clock_freq,
            utilization_pct=round(utilization, 1),
            ambient_temp_c=round(self.ambient_temp_c, 2),
        )

    def reset(self) -> None:
        """Reset the simulator to initial configured conditions."""
        self._rng = random.Random(self.config.seed)
        self._sim_time_s = 0.0
        self._tick_count = 0
        self._sim_datetime = datetime(2026, 10, 27, 10, 0, 0, tzinfo=timezone.utc)
        self._workload_powers = dict(self.config.default_workloads)
        self._is_anomaly_active = False
        self._anomaly_ambient_override = None
        self._scheduled_anomaly_start = None
        self._scheduled_anomaly_end = None

        if self.config.initial_temp_c is not None:
            self._core_temp_c = float(self.config.initial_temp_c)
        else:
            self._core_temp_c = (
                self.config.baseline_ambient_c
                + self.effective_power_draw_w * self.config.thermal_resistance
            )
