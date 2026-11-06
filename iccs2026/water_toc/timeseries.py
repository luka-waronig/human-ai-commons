"""Descriptive per-realization diagnostics, not proofs of ABM attractors."""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import adfuller


def safe_adf_p(values, min_points=20):
    """Return NaN when the test is undefined or the supplied tail is too short.

    ADF tests a unit-root null; rejection is not evidence of a deterministic fixed
    point. A constant series is described separately, not assigned a fake p-value.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("ADF input must be a finite one-dimensional series")
    if len(values) < min_points or np.ptp(values) <= 1e-12:
        return float("nan")
    try:
        return float(adfuller(values, maxlag=1, regression="ct", autolag=None)[1])
    except ValueError:
        return float("nan")


def tail_diagnostics(values, variance_threshold=1e-3, trend_threshold=1e-3):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("tail must be a nonempty finite one-dimensional series")
    variance = float(np.var(values))
    slope = float(np.polyfit(np.arange(len(values)), values, 1)[0]) if len(values) > 1 else np.nan
    constant = bool(len(values) >= 20 and np.ptp(values) <= 1e-12)
    p_value = safe_adf_p(values)
    candidate = bool(
        len(values) >= 20 and variance < variance_threshold
        and abs(slope) < trend_threshold
        and (constant or (np.isfinite(p_value) and p_value < 0.05))
    )
    return dict(variance=variance, slope=slope, constant=constant,
                adf_p=p_value, stationary_low_variance=candidate)


def estimate_oscillation(times, values, min_points=50, min_cycles=4, min_period=3.0):
    """Fit b + A exp(-lambda*t) cos(2*pi*t/P + phase) to one realization.

    Offset and amplitude are estimated jointly. Positive lambda denotes decay;
    negative lambda denotes growth. Poor fits and unresolved cycles are flagged,
    not re-labelled as fixed points or limit cycles. A fit does not establish
    statistical significance; assess uncertainty across independent realizations.
    """
    t, y = np.asarray(times, float), np.asarray(values, float)
    if t.ndim != 1 or t.shape != y.shape or not np.isfinite(t).all() or not np.isfinite(y).all():
        raise ValueError("times and values must be matching finite 1D arrays")
    result = dict(status="insufficient_data", period_est=np.nan, damping_rate=np.nan,
                  baseline=np.nan, amplitude=np.nan,
                  r_squared=np.nan, n_peaks=0)
    if len(t) < min_points:
        return result
    steps = np.diff(t)
    if np.any(steps <= 0) or not np.allclose(steps, steps[0]):
        raise ValueError("oscillation estimation requires regularly spaced increasing times")
    spread = float(np.ptp(y))
    if spread <= 1e-10:
        return {**result, "status": "constant"}
    peaks, _ = find_peaks(y, prominence=spread * 0.1, distance=max(1, int(min_period / steps[0] / 2)))
    result["n_peaks"] = len(peaks)
    if len(peaks) < min_cycles:
        return {**result, "status": "too_few_cycles"}
    duration = t[-1] - t[0]
    u = (t - t[0]) / duration
    frequency = np.fft.rfftfreq(len(t), d=steps[0])
    eligible = (frequency >= min_cycles / duration) & (frequency <= 1 / min_period)
    if not eligible.any():
        return {**result, "status": "too_few_cycles"}
    spectrum = np.abs(np.fft.rfft(y - np.mean(y)))
    initial_frequency = frequency[np.flatnonzero(eligible)[np.argmax(spectrum[eligible])]]

    def signal(u, baseline, amplitude, decay, angular_cycles, phase):
        return baseline + amplitude * np.exp(-decay * u) * np.cos(angular_cycles * u + phase)

    def jacobian(u, baseline, amplitude, decay, angular_cycles, phase):
        exponential = np.exp(-decay * u)
        angle = angular_cycles * u + phase
        cosine, sine = np.cos(angle), np.sin(angle)
        return np.column_stack([np.ones_like(u), exponential * cosine,
            -amplitude * u * exponential * cosine,
            -amplitude * u * exponential * sine,
            -amplitude * exponential * sine])

    # Fit dimensionless decay over the observed window to avoid exp overflow.
    guess = [np.mean(y), spread / 2, 0.01, 2 * np.pi * initial_frequency * duration, 0.0]
    lower = [np.min(y) - spread, 0, -8, 2 * np.pi * min_cycles, -2 * np.pi]
    upper = [np.max(y) + spread, 2 * spread, 8, 2 * np.pi * duration / min_period, 2 * np.pi]
    try:
        coefficients, _ = curve_fit(signal, u, y, p0=guess, bounds=(lower, upper),
            jac=jacobian, x_scale=[1, 1, 1, 10, 1], maxfev=20000)
    except (RuntimeError, ValueError, FloatingPointError):
        return {**result, "status": "fit_failed"}
    prediction = signal(u, *coefficients)
    r_squared = float(1 - np.sum((y - prediction) ** 2) / np.sum((y - np.mean(y)) ** 2))
    if r_squared < 0.5:
        return {**result, "status": "poor_fit", "r_squared": r_squared}
    baseline, amplitude, decay, angular_cycles, _ = coefficients
    return dict(status="fit", period_est=float(2 * np.pi * duration / angular_cycles),
                damping_rate=float(decay / duration),
                baseline=float(baseline), amplitude=float(amplitude),
                r_squared=r_squared, n_peaks=len(peaks))
