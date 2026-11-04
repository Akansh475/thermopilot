"""Thermal Forecast Agent for ThermoPilot Compute Desk.

Provides deterministic predictive perception by analyzing rolling GPU telemetry
to estimate Time-to-Throttle (t_TTT) before reaching the 85.0°C critical hardware trip.

MODEL ASSUMPTIONS & LIMITATIONS:
1. Linear & Quadratic Trend Extrapolation:
   This agent utilizes local linear regression (first-order derivative dT/dt) and
   velocity slope regression (second-order acceleration d^2T/dt^2) across a short
   rolling window of recent telemetry samples (default: 10 samples, ~10-25s simulated).
2. Non-Physical Simplification:
   The calculated forecast is an algorithmic early-warning tripwire, NOT a validated
   high-fidelity physical simulation. Real GPU thermodynamics follows exponential
   Newtonian cooling/heating toward a dynamic asymptotic steady-state (governed by
   differential equations dT/dt = (P * R_th - (T - T_amb)) / tau).
3. Horizon Invariance:
   Linear projection assumes instantaneous thermal rate (dT/dt) remains constant over
   the projection horizon. It does not account for fan curve acceleration, OEM firmware
   throttling steps, or changes in computational workload power.
4. Operational Purpose:
   Engineered strictly for deterministic, low-latency incident triggering to allow
   cluster governance, workload bidding, and human operator approval to occur 20-30 seconds
   ahead of reactive hardware safety trips.
"""

from collections import deque
from collections.abc import Sequence
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from thermopilot.schemas import GPUTelemetry


class ThermalTrend(str, Enum):
    """Directional trend of core temperature."""
    RISING = "RISING"
    STABLE = "STABLE"
    FALLING = "FALLING"


class ThermalRiskStatus(str, Enum):
    """Categorical thermal risk evaluation level."""
    NORMAL = "NORMAL"                 # Operating safely below warning threshold
    WARNING = "WARNING"               # Approaching warning threshold or rising trend detected
    CRITICAL_RISK = "CRITICAL_RISK"   # Rapid rise or imminent threshold breach (trigger incident)
    BREACHED = "BREACHED"             # Already at or above critical threshold (>= 85.0°C)
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"  # Not enough telemetry samples to calculate forecast


class ForecastConfig(BaseModel):
    """Configuration parameters for the thermal ForecastAgent."""
    model_config = ConfigDict(frozen=True)

    critical_threshold_c: float = Field(
        default=85.0,
        gt=0.0,
        description="Critical hardware thermal trip threshold in Celsius",
    )
    warning_threshold_c: float = Field(
        default=81.5,
        gt=0.0,
        description="Thermal warning and potential incident trigger temperature in Celsius",
    )
    incident_time_to_throttle_s: float = Field(
        default=30.0,
        gt=0.0,
        description="Maximum projected seconds to critical threshold for triggering an incident",
    )
    window_size: int = Field(
        default=10,
        ge=3,
        description="Maximum number of recent telemetry samples retained in the rolling buffer",
    )
    min_samples: int = Field(
        default=3,
        ge=2,
        description="Minimum telemetry samples required to compute rate and acceleration",
    )
    trend_rate_threshold_c_per_s: float = Field(
        default=0.02,
        ge=0.0,
        description="Minimum absolute temperature rate (°C/s) to classify as RISING or FALLING",
    )

    @model_validator(mode="after")
    def validate_thresholds(self) -> "ForecastConfig":
        if self.warning_threshold_c >= self.critical_threshold_c:
            raise ValueError(
                f"warning_threshold_c ({self.warning_threshold_c}°C) must be strictly "
                f"less than critical_threshold_c ({self.critical_threshold_c}°C)"
            )
        if self.min_samples > self.window_size:
            raise ValueError(
                f"min_samples ({self.min_samples}) cannot exceed window_size ({self.window_size})"
            )
        return self


class ForecastResult(BaseModel):
    """Structured deterministic forecast result emitted by ForecastAgent."""
    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp at which the forecast was computed",
    )
    gpu_id: int = Field(default=0, ge=0, description="GPU hardware identifier")
    current_temp_c: float | None = Field(
        default=None, description="Most recent core temperature in Celsius"
    )
    critical_threshold_c: float = Field(
        default=85.0, description="Target critical hardware trip threshold in Celsius"
    )
    trend: ThermalTrend = Field(
        ..., description="Current core temperature trend: RISING, STABLE, or FALLING"
    )
    rate_c_per_s: float | None = Field(
        default=None, description="First-order rate of temperature change (dT/dt) in °C/s"
    )
    acceleration_c_per_s2: float | None = Field(
        default=None, description="Second-order thermal acceleration (d^2T/dt^2) in °C/s^2"
    )
    time_to_threshold_s: float | None = Field(
        default=None,
        ge=0.0,
        description="Estimated seconds until reaching critical_threshold_c (None if not rising or stable)",
    )
    risk_status: ThermalRiskStatus = Field(
        ..., description="Thermal risk classification"
    )
    trigger_incident: bool = Field(
        default=False,
        description="True if thermal trajectory warrants triggering an automated recovery incident",
    )
    reason: str = Field(
        ..., description="Explainable deterministic rationale justifying the assessment"
    )
    sample_count: int = Field(
        default=0, ge=0, description="Number of telemetry samples ingested in the rolling window"
    )


class ForecastAgent:
    """Deterministic perception agent forecasting GPU thermal trajectory.

    Consumes GPUTelemetry samples, estimates first-order temperature derivative (dT/dt)
    and second-order acceleration (d^2T/dt^2), and determines whether thermal runaway
    warrants dispatching an automated recovery incident.
    """

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config = config or ForecastConfig()
        self._history: deque[GPUTelemetry] = deque(maxlen=self.config.window_size)

    @property
    def history(self) -> tuple[GPUTelemetry, ...]:
        """Read-only view of buffered telemetry samples in chronological order."""
        return tuple(self._history)

    @property
    def sample_count(self) -> int:
        """Current number of telemetry samples retained in the rolling buffer."""
        return len(self._history)

    def reset(self) -> None:
        """Clear the internal rolling telemetry buffer."""
        self._history.clear()

    def observe(self, telemetry: GPUTelemetry) -> ForecastResult:
        """Ingest a new telemetry sample into the rolling window and compute forecast.

        Args:
            telemetry: Validated GPUTelemetry instance.

        Returns:
            ForecastResult assessing current thermal trajectory and incident trigger status.
        """
        if not isinstance(telemetry, GPUTelemetry):
            raise TypeError(
                f"Expected GPUTelemetry instance, got {type(telemetry).__name__}"
            )

        if self._history and telemetry.timestamp < self._history[-1].timestamp:
            raise ValueError(
                f"Telemetry timestamp cannot regress: {telemetry.timestamp} is earlier "
                f"than latest buffered timestamp {self._history[-1].timestamp}"
            )

        self._history.append(telemetry)
        return self.forecast(self._history)

    def forecast(self, history: Sequence[GPUTelemetry]) -> ForecastResult:
        """Compute deterministic forecast across an arbitrary telemetry sequence.

        Args:
            history: Chronologically ordered sequence of GPUTelemetry records.

        Returns:
            ForecastResult assessing thermal trajectory and incident trigger status.
        """
        if history is None:
            raise ValueError("telemetry history cannot be None")

        if not isinstance(history, (list, tuple, deque)):
            raise TypeError(
                f"Expected sequence of GPUTelemetry, got {type(history).__name__}"
            )

        # Validate elements and timestamps
        for i, item in enumerate(history):
            if not isinstance(item, GPUTelemetry):
                raise TypeError(
                    f"Element at index {i} is not a GPUTelemetry instance: {type(item).__name__}"
                )
            if i > 0 and item.timestamp < history[i - 1].timestamp:
                raise ValueError(
                    f"Telemetry sequence must be monotonically non-decreasing in time; "
                    f"index {i} ({item.timestamp}) is earlier than index {i-1} ({history[i-1].timestamp})"
                )

        if len(history) < self.config.min_samples:
            latest = history[-1] if history else None
            return ForecastResult(
                timestamp=latest.timestamp if latest else datetime.now(timezone.utc),
                gpu_id=latest.gpu_id if latest else 0,
                current_temp_c=latest.temperature_core_c if latest else None,
                critical_threshold_c=self.config.critical_threshold_c,
                trend=ThermalTrend.STABLE,
                rate_c_per_s=None,
                acceleration_c_per_s2=None,
                time_to_threshold_s=None,
                risk_status=ThermalRiskStatus.INSUFFICIENT_DATA,
                trigger_incident=False,
                reason=(
                    f"Insufficient telemetry samples ({len(history)} provided, "
                    f"minimum {self.config.min_samples} required for reliable regression)."
                ),
                sample_count=len(history),
            )

        # Compute derivatives
        rate, accel = self._compute_derivatives(history)
        latest_sample = history[-1]
        current_temp = latest_sample.temperature_core_c
        t_crit = self.config.critical_threshold_c
        t_warn = self.config.warning_threshold_c
        threshold_rate = self.config.trend_rate_threshold_c_per_s

        # Determine directional trend
        if rate > threshold_rate:
            trend = ThermalTrend.RISING
        elif rate < -threshold_rate:
            trend = ThermalTrend.FALLING
        else:
            trend = ThermalTrend.STABLE

        # Compute Time-to-Throttle (t_TTT)
        if current_temp >= t_crit:
            time_to_threshold_s: float | None = 0.0
        elif trend == ThermalTrend.RISING and rate > 0.0:
            time_to_threshold_s = round((t_crit - current_temp) / rate, 2)
        else:
            time_to_threshold_s = None

        # Evaluate risk status and incident trigger
        if current_temp >= t_crit:
            risk_status = ThermalRiskStatus.BREACHED
            trigger_incident = True
            reason = (
                f"Critical hardware limit breached: Core temperature ({current_temp:.2f}°C) "
                f"exceeds critical trip threshold ({t_crit:.1f}°C)."
            )
        elif time_to_threshold_s is not None and (
            current_temp >= t_warn and time_to_threshold_s <= self.config.incident_time_to_throttle_s
        ):
            risk_status = ThermalRiskStatus.CRITICAL_RISK
            trigger_incident = True
            reason = (
                f"Critical thermal runaway imminent: Core temperature ({current_temp:.2f}°C) "
                f"projected to breach {t_crit:.1f}°C in {time_to_threshold_s:.1f}s "
                f"(rate: +{rate:.3f}°C/s, accel: {accel:+.4f}°C/s²)."
            )
        elif current_temp >= t_warn or (
            time_to_threshold_s is not None and time_to_threshold_s <= 60.0
        ):
            risk_status = ThermalRiskStatus.WARNING
            trigger_incident = False
            reason = (
                f"Elevated thermal warning: Core temperature ({current_temp:.2f}°C) "
                f"is approaching risk boundary (trend: {trend.value}, rate: {rate:+.3f}°C/s)."
            )
        else:
            risk_status = ThermalRiskStatus.NORMAL
            trigger_incident = False
            if trend == ThermalTrend.FALLING:
                reason = (
                    f"Thermal state nominal: Core temperature ({current_temp:.2f}°C) "
                    f"is actively cooling (rate: {rate:.3f}°C/s)."
                )
            else:
                reason = (
                    f"Thermal state nominal: Core temperature ({current_temp:.2f}°C) "
                    f"operating safely below warning threshold ({t_warn:.1f}°C, trend: {trend.value})."
                )

        return ForecastResult(
            timestamp=latest_sample.timestamp,
            gpu_id=latest_sample.gpu_id,
            current_temp_c=current_temp,
            critical_threshold_c=t_crit,
            trend=trend,
            rate_c_per_s=round(rate, 4),
            acceleration_c_per_s2=round(accel, 5),
            time_to_threshold_s=time_to_threshold_s,
            risk_status=risk_status,
            trigger_incident=trigger_incident,
            reason=reason,
            sample_count=len(history),
        )

    @staticmethod
    def _compute_derivatives(samples: Sequence[GPUTelemetry]) -> tuple[float, float]:
        """Calculate first-order derivative (dT/dt) and second-order acceleration (d^2T/dt^2)."""
        t0 = samples[0].timestamp
        times = [(s.timestamp - t0).total_seconds() for s in samples]
        temps = [s.temperature_core_c for s in samples]
        n = len(samples)

        total_duration = times[-1] - times[0]
        if total_duration <= 0.0:
            return 0.0, 0.0

        # Linear regression for rate (dT/dt)
        mean_t = sum(times) / n
        mean_temp = sum(temps) / n
        s_tt = sum((t - mean_t) ** 2 for t in times)
        s_ty = sum((t - mean_t) * (y - mean_temp) for t, y in zip(times, temps))

        rate = (s_ty / s_tt) if s_tt > 1e-9 else 0.0

        # Regression on consecutive interval velocities for acceleration (d^2T/dt^2)
        midpoints = []
        velocities = []
        for i in range(1, n):
            dt = times[i] - times[i - 1]
            if dt > 0.0:
                midpoints.append((times[i] + times[i - 1]) / 2.0)
                velocities.append((temps[i] - temps[i - 1]) / dt)

        m_count = len(midpoints)
        if m_count >= 2:
            mean_m = sum(midpoints) / m_count
            mean_v = sum(velocities) / m_count
            s_mm = sum((m - mean_m) ** 2 for m in midpoints)
            s_mv = sum((m - mean_m) * (v - mean_v) for m, v in zip(midpoints, velocities))
            accel = (s_mv / s_mm) if s_mm > 1e-9 else 0.0
        else:
            accel = 0.0

        return rate, accel
