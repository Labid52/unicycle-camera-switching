"""Direct simulation of asynchronous held camera-coordinate registers.

The theorem requires only a bounded orientation register |psi_hat|<=Psi; it does
not require a bound on the orientation update age.  This simulator therefore
updates theta_hat and psi_hat on independent clocks and maps every new psi_hat
sample into the baseline bounded chart.  The simulation is supporting evidence,
not part of the analytic certificate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from .model import BearingParams
from .implementation_errors import measured_feedback_rhs, reduce_orientation_register


@dataclass(frozen=True)
class SampledCase:
    h_theta: float
    h_psi: float
    n_theta: float = 0.0
    n_psi: float = 0.0
    d_bar: float = 0.0
    horizon: float = 2.0
    dt: float = 5.0e-4


def _reduce_true_psi(psi: float, p: BearingParams) -> float:
    """Apply the nominal orientation-chart coordinate reset to true psi."""
    return reduce_orientation_register(psi, p)


def _outward_score(theta: float, theta_hat: float, psi_hat: float,
                    p: BearingParams, d_bar: float) -> float:
    side = 1.0 if theta >= 0.0 else -1.0
    d = side * d_bar
    return side * measured_feedback_rhs(0.0, theta, theta_hat, psi_hat, p, d)[1]


def simulate(
    psi0: float,
    theta0: float,
    p: BearingParams,
    case: SampledCase,
    theta_phase: float = 0.0,
    psi_phase: float = 0.0,
) -> Dict[str, float | bool | int]:
    if min(case.h_theta, case.h_psi, case.dt) <= 0.0:
        raise ValueError("Periods and integration step must be positive")
    if min(case.n_theta, case.n_psi, case.d_bar) < 0.0:
        raise ValueError("Bounds must be nonnegative")

    psi = _reduce_true_psi(float(psi0), p)
    theta = float(theta0)
    theta_hat = theta
    psi_hat = reduce_orientation_register(psi, p)
    next_theta = max(0.0, float(theta_phase))
    next_psi = max(0.0, float(psi_phase))
    t = 0.0

    max_abs_theta = abs(theta)
    max_abs_e_theta = 0.0
    max_abs_psi_hat = abs(psi_hat)
    false_guard = False
    true_orientation_resets = 0
    theta_updates = 0
    psi_updates = 0

    def choose_theta_sample() -> float:
        candidates = (theta - case.n_theta, theta + case.n_theta)
        return max(candidates, key=lambda z: _outward_score(theta, z, psi_hat, p, case.d_bar))

    def choose_psi_sample() -> float:
        raw_candidates = (psi - case.n_psi, psi + case.n_psi)
        candidates = tuple(reduce_orientation_register(z, p) for z in raw_candidates)
        return max(candidates, key=lambda z: _outward_score(theta, theta_hat, z, p, case.d_bar))

    n_steps = int(math.ceil(case.horizon / case.dt))
    for _ in range(n_steps):
        while t + 1.0e-14 >= next_theta:
            theta_hat = choose_theta_sample()
            next_theta += case.h_theta
            theta_updates += 1
        while t + 1.0e-14 >= next_psi:
            psi_hat = choose_psi_sample()
            next_psi += case.h_psi
            psi_updates += 1

        side = 1.0 if theta >= 0.0 else -1.0
        d = side * case.d_bar

        def f(ps: float, th: float) -> np.ndarray:
            return measured_feedback_rhs(ps, th, theta_hat, psi_hat, p, d)

        y = np.array([psi, theta], dtype=float)
        k1 = f(y[0], y[1])
        k2 = f(*(y + 0.5 * case.dt * k1))
        k3 = f(*(y + 0.5 * case.dt * k2))
        k4 = f(*(y + case.dt * k3))
        y += (case.dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        psi_new, theta = float(y[0]), float(y[1])
        psi_reduced = _reduce_true_psi(psi_new, p)
        if abs(psi_reduced - psi_new) > 1.0e-12:
            true_orientation_resets += 1
        psi = psi_reduced
        t += case.dt

        max_abs_theta = max(max_abs_theta, abs(theta))
        max_abs_e_theta = max(max_abs_e_theta, abs(theta_hat - theta))
        max_abs_psi_hat = max(max_abs_psi_hat, abs(psi_hat))
        false_guard = false_guard or (abs(theta_hat) >= p.Theta)

        if abs(theta) > p.Theta + 0.2:
            break

    return {
        "core_preserved": bool(max_abs_theta <= math.pi/2.0 + 2.0e-5),
        "false_guard": bool(false_guard),
        "orientation_register_bounded": bool(max_abs_psi_hat <= p.Psi + 1.0e-12),
        "max_abs_theta": max_abs_theta,
        "max_abs_e_theta": max_abs_e_theta,
        "max_abs_psi_hat": max_abs_psi_hat,
        "true_orientation_resets": true_orientation_resets,
        "theta_updates": theta_updates,
        "psi_updates": psi_updates,
        "final_theta": theta,
        "final_psi": psi,
        "final_psi_hat": psi_hat,
    }


def grid_check(p: BearingParams, case: SampledCase) -> Dict[str, object]:
    theta0s = np.linspace(-math.pi/2.0, math.pi/2.0, 5)
    psi0s = np.linspace(-p.Psi, p.Psi, 5)
    phases = [
        (0.0, 0.0),
        (0.5*case.h_theta, 0.5*case.h_psi),
        (0.25*case.h_theta, 0.75*case.h_psi),
    ]
    worst = None
    n_runs = 0
    all_core = True
    any_guard = False
    all_registers = True
    for theta0 in theta0s:
        for psi0 in psi0s:
            for phase_theta, phase_psi in phases:
                out = simulate(float(psi0), float(theta0), p, case, phase_theta, phase_psi)
                n_runs += 1
                all_core = all_core and bool(out["core_preserved"])
                any_guard = any_guard or bool(out["false_guard"])
                all_registers = all_registers and bool(out["orientation_register_bounded"])
                if worst is None or float(out["max_abs_theta"]) > float(worst["max_abs_theta"]):
                    worst = {
                        "theta0": float(theta0), "psi0": float(psi0),
                        "phase_theta": phase_theta, "phase_psi": phase_psi, **out,
                    }
    return {
        "n_runs": n_runs,
        "all_core_preserved": all_core,
        "any_false_guard": any_guard,
        "all_orientation_registers_bounded": all_registers,
        "worst": worst,
    }
