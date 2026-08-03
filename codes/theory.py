"""Analytical quantities for the hybrid camera-based unicycle manuscript.

The module implements the nominal reduced subsystem and its certified quantities:

* the recalled core invariance |theta| <= pi/2;
* the strict-FOV condition and theta_star envelope;
* the range-free Lyapunov ellipse;
* the exact persistent yaw-rate threshold D_star = k_theta*pi/2.

Implementation-level bearing-error and sampled-update certificates are kept in
implementation_errors.py.  This module does not claim a positive camera-handover
delay margin at the unchanged FOV boundary.
"""
from __future__ import annotations

import math
from typing import Optional

from .model import BearingParams, g_func


def phi_func(theta: float) -> float:
    """phi(theta) = theta/|g(theta)| = 2 theta^2 / |sin(2 theta)|.

    phi is strictly increasing on (0, pi/2) from 0 to +infinity, and decreasing
    from +infinity to phi(Theta) on (pi/2, Theta). This shape underlies Theorem B.
    """
    t = abs(float(theta))
    s = abs(math.sin(2.0 * t))
    if s < 1e-12:
        return math.inf
    return 2.0 * t * t / s


def _Phi_delta(p: BearingParams) -> float:
    """Phi_delta = min_{theta in [pi/2, Theta]} phi(theta), the binding value in
    Theorem B. phi decreases from +inf at pi/2 to an interior minimum at theta_m
    (the root of tan(2 theta)=theta in (pi/2, pi), theta_m ~= 2.1374) and then
    increases. Hence Phi_delta = phi(Theta) when Theta <= theta_m, else phi(theta_m).
    Using phi(Theta) alone (as in an earlier version) is UNSOUND for wide overlaps
    delta > theta_m - pi/2 ~= 0.567 rad.
    """
    import math
    half = math.pi / 2.0
    theta_m = 2.137314  # root of tan(2 theta) = theta on (pi/2, pi)
    Th = p.Theta
    if Th <= theta_m:
        return phi_func(Th)
    return phi_func(theta_m)


def confinement_condition_C(p: BearingParams) -> dict:
    """Evaluate condition (C) of Theorem B. Returns the two sides and a flag.

    (C) holds  <=>  c = k_psi W / k_theta  <  Phi_delta = min_{[pi/2,Theta]} phi.
    """
    c = p.k_psi * p.W / p.k_theta
    Phi = _Phi_delta(p)
    return {"c": c, "Phi_delta": Phi, "phi_Theta": phi_func(p.Theta),
            "holds": bool(c < Phi)}


def theta_star(p: BearingParams) -> Optional[float]:
    """Certified confinement bound theta* of Theorem B (None if (C) fails).

    theta* is the unique root of phi(theta) = c on (0, pi/2). The bearing then
    satisfies |theta(t)| <= max(|theta0|, theta*) < Theta for all t.
    """
    cond = confinement_condition_C(p)
    if not cond["holds"]:
        return None
    c = cond["c"]
    # phi strictly increasing 0 -> +inf on (0, pi/2): unique root by bisection.
    lo, hi = 1e-5, math.pi / 2.0 - 1e-7
    flo = phi_func(lo) - c
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = phi_func(mid) - c
        if flo * fm <= 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def core_barrier(p: BearingParams) -> dict:
    """Theorem A data: the gain-independent core barrier and the overlap band.

    The barrier sits at |theta| = pi/2 for every gain choice; the FOV boundary is
    Theta = pi/2 + delta; the overlap band is [pi/2 - delta, pi/2 + delta].
    """
    return {
        "theta_barrier": math.pi / 2.0,
        "Theta": p.Theta,
        "overlap_inner": math.pi / 2.0 - p.delta,
        "overlap_outer": math.pi / 2.0 + p.delta,
        "rate_at_barrier": -p.k_theta * (math.pi / 2.0),  # theta_dot at +pi/2, any w
    }


def disturbance_budget(p: BearingParams) -> float:
    """Exact core-confinement disturbance budget D* = k_theta * pi/2 (Theorem C)."""
    return p.k_theta * math.pi / 2.0


def visibility_margin(p: BearingParams) -> Optional[float]:
    """Angular margin from the certified reach bound to the FOV edge, Theta - theta*.

    Returns None when (C) fails (the bearing may reach the edge, but only inside
    the overlap band, so handoff remains graceful -- Theorem A).
    """
    ts = theta_star(p)
    return None if ts is None else (p.Theta - ts)


def lyapunov_V(theta: float, w: float, p: BearingParams) -> float:
    """Lyapunov function for the bearing subsystem (Lemma, trajectory certificate):
        V(theta, w) = 1/2 theta^2 + (k_psi/(2 k_rho)) w^2,
    which satisfies Vdot = -k_theta theta^2 <= 0 along (theta, w) trajectories.
    """
    return 0.5 * theta * theta + (p.k_psi / (2.0 * p.k_rho)) * w * w


def lyapunov_Vdot(theta: float, w: float, p: BearingParams) -> float:
    """Analytic Vdot = grad V . f. Equals -k_theta theta^2 (cross terms cancel)."""
    import math
    g = g_func(theta)
    th_dot = -p.k_theta * theta + p.k_psi * w * g
    w_dot = -(p.k_rho / 2.0) * math.sin(2.0 * theta)
    return theta * th_dot + (p.k_psi / p.k_rho) * w * w_dot


def lyapunov_bound(theta0: float, w0: float, p: BearingParams) -> float:
    """Trajectory bound from V non-increasing: |theta(t)| <= this value for all t."""
    import math
    return math.sqrt(theta0 * theta0 + (p.k_psi / p.k_rho) * w0 * w0)


def lyapunov_safe(theta0: float, w0: float, p: BearingParams) -> bool:
    """True if (theta0, w0) lies in the Lyapunov safe ellipse
        theta0^2 + (k_psi/k_rho) w0^2 < Theta^2,
    which guarantees |theta(t)| < Theta for all t (full-FOV confinement), for ANY
    positive gains -- with no gain-geometry condition (cf. confinement_condition_C).
    """
    return bool(theta0 * theta0 + (p.k_psi / p.k_rho) * w0 * w0 < p.Theta ** 2)


def orientation_reset(theta: float, w: float, p: BearingParams):
    """Baseline orientation chart reset in (theta, w) coordinates:
        psi^+ = psi - 2*pi*sgn(psi),   theta^+ = theta,     psi = w + theta.
    Returns (theta, w^+). Active on the guard {|psi| >= Psi}, Psi = Theta + pi.
    """
    import math
    psi = w + theta
    w_plus = w - 2.0 * math.pi * (1.0 if psi >= 0.0 else -1.0)
    return theta, w_plus


def lyapunov_reset_change(theta: float, w: float, p: BearingParams) -> float:
    """V^+ - V across the orientation reset. By the Proposition,
        V^+ - V = -(k_psi/k_rho) * 2*pi * (Theta - theta*sgn(psi))
                <= -(k_psi/k_rho) * 2*pi * (Theta - |theta|)  <= 0   for |theta| <= Theta,
    so the reset never increases V on the field of view. (Uses Psi = Theta + pi.)
    """
    _, w_plus = orientation_reset(theta, w, p)
    return lyapunov_V(theta, w_plus, p) - lyapunov_V(theta, w, p)
