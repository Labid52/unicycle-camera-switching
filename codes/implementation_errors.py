"""Implementation-level certificates for measured and sampled camera coordinates.

The controller is unchanged and is evaluated at

    theta_hat : held/measured active-camera bearing,
    psi_hat   : the controller's orientation-coordinate register.

The orientation register is represented in the same bounded hybrid chart used by
baseline controller, hence |psi_hat| <= Psi.  This is coordinate bookkeeping,
not a sensor-accuracy assumption: adding or subtracting 2*pi leaves the physical
orientation unchanged.

The plant evolves in true coordinates.  With the manuscript disturbance d=-d_omega,
which enters both psi_dot and theta_dot and cancels from w_dot, the exact flows are

    psi_dot   = -nu2(theta_hat, psi_hat) + d,
    theta_dot = -nu2(theta_hat, psi_hat)
                + k_rho*cos(theta_hat)*sin(theta) + d.

No camera frequency, noise distribution, orientation-update age, or hardware
performance value is assumed.  All numerical quantities are explicit inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, differential_evolution, minimize_scalar

from .model import BearingParams, g_func

HALF_PI = math.pi / 2.0
TWO_PI = 2.0 * math.pi


def g_prime(theta: float) -> float:
    """Derivative of g(theta)=sin(2theta)/(2theta), with g'(0)=0."""
    t = float(theta)
    if abs(t) < 1.0e-6:
        # g(t)=1-(2/3)t^2+(2/15)t^4+O(t^6)
        return -(4.0 / 3.0) * t + (8.0 / 15.0) * t**3
    return (2.0 * t * math.cos(2.0 * t) - math.sin(2.0 * t)) / (2.0 * t * t)


@lru_cache(maxsize=None)
def numerical_global_Lg(search_radius: float = 30.0) -> Dict[str, float]:
    """High-accuracy numerical evaluation of L_g=sup_R |g'| for reporting.

    The analytic statements retain L_g as the exact supremum.  The returned
    decimal is not used as a substitute for the definition in the proof.
    """
    result = differential_evolution(
        lambda x: -abs(g_prime(float(x[0]))),
        bounds=[(-search_radius, search_radius)],
        seed=117,
        tol=1.0e-13,
        polish=True,
        updating="immediate",
    )
    x = float(result.x[0])
    value = abs(g_prime(x))
    polish = minimize_scalar(
        lambda z: -abs(g_prime(float(z))), bounds=(0.0, 2.0), method="bounded",
        options={"xatol": 1.0e-14},
    )
    xp = float(polish.x)
    vp = abs(g_prime(xp))
    if vp > value:
        x, value = xp, vp
    return {"L_g": value, "argmax_abs": abs(x), "signed_argmax": x}


def W0(p: BearingParams) -> float:
    """Core-boundary mismatch constant Psi+pi/2 = 2*pi+delta."""
    return p.Psi + HALF_PI


def reduce_orientation_register(value: float, p: BearingParams) -> float:
    """Represent an orientation coordinate in the baseline bounded chart.

    This repeatedly applies the baseline +/-2*pi chart reset until the stored
    representative lies in [-Psi,Psi].  It changes only the coordinate
    representative, not the physical orientation.
    """
    x = float(value)
    while x >= p.Psi:
        x -= TWO_PI
    while x <= -p.Psi:
        x += TWO_PI
    return x


@dataclass(frozen=True)
class BarrierInputs:
    """Explicit inputs to the measured-coordinate barrier certificate."""
    e_theta: float
    d_bar: float = 0.0

    def validate(self) -> None:
        if self.e_theta < 0 or self.d_bar < 0:
            raise ValueError("Bearing-error and disturbance bounds must be nonnegative.")


@dataclass(frozen=True)
class BearingSamplingData:
    """Zero-order-held bearing data.

    h_theta is the maximum age of theta_hat and n_theta is its deterministic
    update-time error bound.  The orientation channel may update independently;
    its age does not enter the core-invariance/no-false-switch certificate because
    only the baseline chart property |psi_hat|<=Psi is used.
    """
    h_theta: float
    n_theta: float = 0.0

    def validate(self) -> None:
        if self.h_theta < 0 or self.n_theta < 0:
            raise ValueError("Sample age and update-time error must be nonnegative.")


def nu2_measured(theta_hat: float, psi_hat: float, p: BearingParams) -> float:
    return (
        p.k_rho * math.cos(theta_hat) * math.sin(theta_hat)
        + p.k_theta * theta_hat
        + p.k_psi * (theta_hat - psi_hat) * g_func(theta_hat)
    )


def measured_feedback_rhs(
    psi: float,
    theta: float,
    theta_hat: float,
    psi_hat: float,
    p: BearingParams,
    d: float = 0.0,
) -> np.ndarray:
    """Exact true [psi_dot, theta_dot] under unchanged measured feedback."""
    del psi  # true psi does not enter the held-feedback flow except through updates
    omega_nom = nu2_measured(theta_hat, psi_hat, p)
    dpsi = -omega_nom + float(d)
    dtheta = -omega_nom + p.k_rho * math.cos(theta_hat) * math.sin(theta) + float(d)
    return np.array([dpsi, dtheta], dtype=float)


def exact_theta_rhs_formula(
    theta: float,
    theta_hat: float,
    psi_hat: float,
    p: BearingParams,
    d: float = 0.0,
) -> float:
    """Exact measured-feedback bearing equation used in the manuscript patch."""
    return (
        -p.k_theta * theta_hat
        - p.k_psi * (theta_hat - psi_hat) * g_func(theta_hat)
        + p.k_rho * math.cos(theta_hat) * (math.sin(theta) - math.sin(theta_hat))
        + float(d)
    )


def analytic_error_budget(
    inputs: BarrierInputs,
    p: BearingParams,
    L_g: Optional[float] = None,
) -> Dict[str, float | bool]:
    """Sharpened sufficient core-invariance and no-false-switch certificate.

    B(E) = d_bar + k_theta E
           + k_psi (W0+E) L_g E
           + (k_rho/2) E^3.

    B(E)<=D*=k_theta*pi/2 certifies the true core |theta|<=pi/2.  E<delta
    separately prevents a guard computed from theta_hat from falsely activating.
    """
    inputs.validate()
    if L_g is None:
        L_g = numerical_global_Lg()["L_g"]
    E = inputs.e_theta
    B = (
        inputs.d_bar
        + p.k_theta * E
        + p.k_psi * (W0(p) + E) * L_g * E
        + 0.5 * p.k_rho * E**3
    )
    D_star = p.k_theta * HALF_PI
    return {
        "B": B,
        "D_star": D_star,
        "margin": D_star - B,
        "core_invariant": bool(B <= D_star + 1.0e-14),
        "false_switch_safe": bool(E < p.delta),
        "combined_safe": bool(B <= D_star + 1.0e-14 and E < p.delta),
        "L_g": float(L_g),
        "W0": W0(p),
    }


def certified_bearing_error_limit(
    p: BearingParams,
    d_bar: float = 0.0,
    L_g: Optional[float] = None,
    upper: float = math.pi,
) -> float:
    """Largest E satisfying the analytic barrier inequality alone."""
    if d_bar < 0:
        raise ValueError("d_bar must be nonnegative.")
    if L_g is None:
        L_g = numerical_global_Lg()["L_g"]
    D_star = p.k_theta * HALF_PI
    if d_bar >= D_star:
        return 0.0

    def residual(E: float) -> float:
        return float(analytic_error_budget(BarrierInputs(E, d_bar), p, L_g)["B"]) - D_star

    hi = min(float(upper), math.pi)
    while residual(hi) <= 0.0 and hi < 20.0:
        hi *= 1.5
    return float(brentq(residual, 0.0, hi, xtol=1.0e-13, rtol=1.0e-13))


def effective_bearing_error_limit(
    p: BearingParams,
    d_bar: float = 0.0,
    L_g: Optional[float] = None,
) -> Dict[str, float | str]:
    """Barrier limit, false-switch cap, and the binding combined limit."""
    barrier = certified_bearing_error_limit(p, d_bar=d_bar, L_g=L_g)
    effective = min(barrier, p.delta)
    binding = "barrier" if barrier < p.delta else "false-switch"
    return {
        "barrier_limit": barrier,
        "false_switch_cap": p.delta,
        "effective_limit": effective,
        "binding_constraint": binding,
    }


def exact_boundary_outward_rate_register(
    e_theta: float,
    side: int,
    p: BearingParams,
    d_bar: float = 0.0,
) -> Dict[str, float]:
    """Exact worst outward boundary rate for fixed e_theta.

    Maximization is analytical over the bounded controller register
    psi_hat in [-Psi,Psi] and over |d|<=d_bar.  The true psi and any e_psi are
    irrelevant to this barrier problem.
    """
    if side not in (-1, 1):
        raise ValueError("side must be -1 or +1")
    if d_bar < 0:
        raise ValueError("d_bar must be nonnegative")
    theta = side * HALF_PI
    z = theta + float(e_theta)
    # outward k_psi term = c*(z-psi_hat), c=-side*k_psi*g(z)
    c = -side * p.k_psi * g_func(z)
    psi_hat_star = -p.Psi if c >= 0.0 else p.Psi
    kpsi = c * (z - psi_hat_star)
    kr = side * p.k_rho * math.cos(z) * (side - math.sin(z))
    kth = -side * p.k_theta * z
    return {
        "outward_rate": kth + kpsi + kr + d_bar,
        "psi_hat_at_max": psi_hat_star,
        "side": float(side),
        "e_theta": float(e_theta),
    }


def exact_boundary_max_rate_register(
    e_theta_bar: float,
    p: BearingParams,
    d_bar: float = 0.0,
    n_grid: int = 4001,
) -> Dict[str, float]:
    """Numerical max of exact boundary rate over |e_theta|<=E and both sides."""
    if e_theta_bar < 0:
        raise ValueError("e_theta_bar must be nonnegative")
    best: Dict[str, float] | None = None
    for side in (-1, 1):
        if e_theta_bar == 0.0:
            cand = exact_boundary_outward_rate_register(0.0, side, p, d_bar)
        else:
            grid = np.linspace(-e_theta_bar, e_theta_bar, n_grid)
            vals = np.array([
                exact_boundary_outward_rate_register(float(e), side, p, d_bar)["outward_rate"]
                for e in grid
            ])
            idx = int(np.argmax(vals))
            e0 = float(grid[idx])
            cand = exact_boundary_outward_rate_register(e0, side, p, d_bar)
            lo = float(grid[max(0, idx - 1)])
            hi = float(grid[min(n_grid - 1, idx + 1)])
            if hi > lo:
                res = minimize_scalar(
                    lambda e: -exact_boundary_outward_rate_register(float(e), side, p, d_bar)["outward_rate"],
                    bounds=(lo, hi), method="bounded", options={"xatol": 1.0e-14},
                )
                polished = exact_boundary_outward_rate_register(float(res.x), side, p, d_bar)
                if polished["outward_rate"] > cand["outward_rate"]:
                    cand = polished
            for endpoint in (-e_theta_bar, e_theta_bar):
                endpoint_cand = exact_boundary_outward_rate_register(endpoint, side, p, d_bar)
                if endpoint_cand["outward_rate"] > cand["outward_rate"]:
                    cand = endpoint_cand
        if best is None or cand["outward_rate"] > best["outward_rate"]:
            best = cand
    assert best is not None
    return best


def numerical_boundary_threshold_register(
    p: BearingParams,
    d_bar: float = 0.0,
    upper: float = 1.4,
) -> Dict[str, float]:
    """Observed error threshold where exact boundary maximization first reaches 0.

    This is a numerical sensitivity reference, not a certificate.
    """
    def f(E: float) -> float:
        return exact_boundary_max_rate_register(E, p, d_bar=d_bar)["outward_rate"]

    if f(0.0) > 0.0:
        return {"threshold": 0.0, **exact_boundary_max_rate_register(0.0, p, d_bar)}
    hi = float(upper)
    while f(hi) <= 0.0 and hi < math.pi:
        hi *= 1.25
    threshold = float(brentq(f, 0.0, hi, xtol=2.0e-12, rtol=2.0e-12))
    at = exact_boundary_max_rate_register(threshold, p, d_bar=d_bar, n_grid=16001)
    return {"threshold": threshold, **at}


def bearing_rate_bound(E: float, p: BearingParams, d_bar: float = 0.0) -> float:
    """Bootstrap rate bound before a first core exit.

    For |theta|<=pi/2, |theta_hat-theta|<=E and |psi_hat|<=Psi,

      |theta_dot| <= A + B E,
      A = k_theta*pi/2 + k_psi*W0 + d_bar,
      B = k_theta + k_psi + k_rho.
    """
    if E < 0 or d_bar < 0:
        raise ValueError("E and d_bar must be nonnegative")
    A = p.k_theta * HALF_PI + p.k_psi * W0(p) + d_bar
    B = p.k_theta + p.k_psi + p.k_rho
    return A + B * E


def sampled_bearing_error_bound(
    data: BearingSamplingData,
    p: BearingParams,
    d_bar: float = 0.0,
) -> Dict[str, float | bool]:
    """Explicit ZOH error bound and its solvability condition.

      E <= n_theta + h_theta [A+B E]

    resolves to E=(n_theta+A h_theta)/(1-B h_theta) when B h_theta<1.
    """
    data.validate()
    if d_bar < 0:
        raise ValueError("d_bar must be nonnegative")
    A = p.k_theta * HALF_PI + p.k_psi * W0(p) + d_bar
    B = p.k_theta + p.k_psi + p.k_rho
    denom = 1.0 - B * data.h_theta
    if denom <= 0.0:
        return {
            "exists": False,
            "A": A,
            "B": B,
            "denominator": denom,
            "E_theta": math.inf,
        }
    E = (data.n_theta + A * data.h_theta) / denom
    return {
        "exists": True,
        "A": A,
        "B": B,
        "denominator": denom,
        "E_theta": E,
        "rate_bound": bearing_rate_bound(E, p, d_bar),
        "fixed_point_residual": E - data.n_theta - data.h_theta * bearing_rate_bound(E, p, d_bar),
    }


def sampled_certificate(
    data: BearingSamplingData,
    p: BearingParams,
    d_bar: float = 0.0,
    L_g: Optional[float] = None,
) -> Dict[str, float | bool | str]:
    out = sampled_bearing_error_bound(data, p, d_bar=d_bar)
    if not bool(out["exists"]):
        return {**out, "certified": False, "reason": "small-gain condition failed"}
    E = float(out["E_theta"])
    barrier = analytic_error_budget(BarrierInputs(E, d_bar), p, L_g=L_g)
    return {
        **out,
        **{f"barrier_{k}": v for k, v in barrier.items()},
        "certified": bool(barrier["combined_safe"]),
        "reason": "certified" if barrier["combined_safe"] else "barrier or false-switch condition failed",
    }


def max_admissible_bearing_age(
    p: BearingParams,
    n_theta: float = 0.0,
    d_bar: float = 0.0,
    L_g: Optional[float] = None,
) -> Dict[str, float | str]:
    """Supremal h_theta implied by the final closed-form certificate.

    If E_cap=min(E_barrier,delta), then
       h <= (E_cap-n)/(A+B E_cap).
    The strict no-false-switch inequality means this is a supremum when delta
    is binding, not an attained maximum.
    """
    if n_theta < 0 or d_bar < 0:
        raise ValueError("n_theta and d_bar must be nonnegative")
    limits = effective_bearing_error_limit(p, d_bar=d_bar, L_g=L_g)
    Ecap = float(limits["effective_limit"])
    A = p.k_theta * HALF_PI + p.k_psi * W0(p) + d_bar
    B = p.k_theta + p.k_psi + p.k_rho
    if n_theta >= Ecap:
        h = 0.0
    else:
        h = (Ecap - n_theta) / (A + B * Ecap)
    return {
        **limits,
        "A": A,
        "B": B,
        "n_theta": n_theta,
        "h_theta_sup": h,
        "rate_hz_infimum": math.inf if h <= 0.0 else 1.0 / h,
    }
