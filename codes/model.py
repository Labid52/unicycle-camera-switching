"""Verified closed-loop model of the hybrid camera-based unicycle.

All equations below are taken *verbatim* (up to the notation rename
(r, beta, alpha) -> (rho, psi, theta)) from the nominal stabilizer of
Ballaben et al., Automatica 183 (2026) 112502, eqs. (9), (15)-(18). The
reproduction of these equations against the source paper and against the
authors' released code was checked term by term.

Camera coordinates (active chart): eta = [rho, psi, theta, sigma], where
    rho   >= 0  : range to the visual reference,
    psi         : orientation error ( = beta in the source ),
    theta       : bearing of the reference in the active camera ( = alpha ),
    sigma       : active camera index (+1 forward, -1 backward).

Closed-loop flow with the nominal feedback substituted (k_rho, k_theta, k_psi>0):
    rho_dot   = -k_rho * rho * cos(theta)^2
    psi_dot   = -(k_rho/2) sin(2 theta) - k_theta theta - k_psi (theta - psi) g(theta)
    theta_dot =                          - k_theta theta - k_psi (theta - psi) g(theta)
with
    g(theta) = cos(theta) sin(theta)/theta = sin(2 theta)/(2 theta),  g(0):=1.

Key structural fact used throughout: the (psi, theta) bearing dynamics are
*independent of rho*. Introducing w := psi - theta gives the planar autonomous
"bearing subsystem"
    theta_dot = -k_theta theta + k_psi w g(theta)
    w_dot     = -(k_rho/2) sin(2 theta).
The range merely decays, rho_dot = -k_rho rho cos^2(theta), once theta(t) is known.

The field of view is |theta| <= Theta := pi/2 + delta, 0 < delta < pi/2; the
camera-overlap band (both cameras see the reference) is pi/2 - delta <= |theta| <= pi/2 + delta.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def g_func(theta: float) -> float:
    """g(theta) = sin(2 theta)/(2 theta), with the removable value g(0)=1.

    Note g(pi/2) = sin(pi)/pi = 0 exactly; this single zero is what produces the
    gain-independent visibility barrier in theory.core_barrier.
    """
    t = float(theta)
    if abs(t) < 1e-9:
        return 1.0 - (2.0 * t * t) / 3.0  # Taylor of sin(2t)/(2t)
    return math.sin(2.0 * t) / (2.0 * t)


@dataclass(frozen=True)
class BearingParams:
    """Feedback gains and camera geometry."""
    k_rho: float = 2.0
    k_theta: float = 1.0
    k_psi: float = 1.0
    delta: float = math.pi / 10.0

    @property
    def Theta(self) -> float:
        return math.pi / 2.0 + self.delta

    @property
    def W(self) -> float:
        """A-priori bound on |w| = |psi - theta| over the source paper's domain
        psi in [-3pi/2 - delta, 3pi/2 + delta], |theta| <= Theta:
            |w| <= Theta + (3pi/2 + delta) = 2 pi + 2 delta.
        """
        return 2.0 * math.pi + 2.0 * self.delta

    @property
    def Psi(self) -> float:
        """Orientation-error chart bound, |psi| <= Psi = 3*pi/2 + delta. Note the
        structural identity Psi = Theta + pi, which makes the orientation reset
        non-increasing for the bearing Lyapunov function (see theory/certify)."""
        return 3.0 * math.pi / 2.0 + self.delta


def bearing_rhs(state: np.ndarray, p: BearingParams) -> np.ndarray:
    """RHS of the nominal bearing subsystem (theta, w)."""
    theta, w = float(state[0]), float(state[1])
    g = g_func(theta)
    dtheta = -p.k_theta * theta + p.k_psi * w * g
    dw = -(p.k_rho / 2.0) * math.sin(2.0 * theta)
    return np.array([dtheta, dw], dtype=float)


def bearing_rhs_perturbed(state: np.ndarray, p: BearingParams,
                          d_theta: float = 0.0) -> np.ndarray:
    """Bearing subsystem with an additive lumped bearing-rate disturbance d_theta.

    d_theta is an additive disturbance in the already reduced bearing equation.
    It represents the yaw-rate channel analyzed in the manuscript after the sign
    change d=-d_omega.  Measurement error is not modeled here; the exact
    measured-feedback dynamics are implemented in implementation_errors.py.
    """
    base = bearing_rhs(state, p)
    base[0] += float(d_theta)
    return base


def closed_loop_rhs(eta: np.ndarray, p: BearingParams,
                    d_theta: float = 0.0) -> np.ndarray:
    """Full closed-loop flow [rho_dot, psi_dot, theta_dot] (sigma constant in flow)."""
    rho, psi, theta = float(eta[0]), float(eta[1]), float(eta[2])
    g = g_func(theta)
    drho = -p.k_rho * rho * math.cos(theta) ** 2
    dtheta = -p.k_theta * theta - p.k_psi * (theta - psi) * g + float(d_theta)
    dpsi = -(p.k_rho / 2.0) * math.sin(2.0 * theta) - p.k_theta * theta \
           - p.k_psi * (theta - psi) * g
    return np.array([drho, dpsi, dtheta], dtype=float)


def simulate_bearing(theta0: float, w0: float, p: BearingParams,
                     d_theta: float = 0.0, T: float = 14.0, dt: float = 0.002):
    """Integrate the bearing subsystem; return (t, theta[], w[])."""
    n = int(round(T / dt))
    th = np.empty(n + 1)
    ww = np.empty(n + 1)
    th[0], ww[0] = theta0, w0
    s = np.array([theta0, w0], dtype=float)
    for k in range(n):
        s = s + dt * bearing_rhs_perturbed(s, p, d_theta=d_theta)
        th[k + 1], ww[k + 1] = s[0], s[1]
        if abs(s[0]) > 4.0:  # diverged past any FOV; freeze for plotting
            th[k + 1:] = s[0]
            ww[k + 1:] = s[1]
            break
    t = np.arange(n + 1) * dt
    return t, th, ww


def simulate_closed_loop(eta0: np.ndarray, p: BearingParams, d_theta: float = 0.0,
                         T: float = 14.0, dt: float = 0.002):
    """Integrate the full closed loop [rho, psi, theta]; return (t, eta_array)."""
    n = int(round(T / dt))
    out = np.empty((n + 1, 3))
    out[0] = eta0
    s = np.array(eta0, dtype=float)
    for k in range(n):
        s = s + dt * closed_loop_rhs(s, p, d_theta=d_theta)
        if s[0] < 0.0:
            s[0] = 0.0
        out[k + 1] = s
    t = np.arange(n + 1) * dt
    return t, out


def simulate_bearing_hybrid(theta0, w0, p, T=30.0, dt=0.002, rk4=True):
    """Hybrid bearing simulation in (theta, w): continuous flow plus the
    orientation chart reset of the baseline,
        psi^+ = psi - 2*pi*sgn(psi),   theta^+ = theta,   active on {|psi| >= Psi},
    where psi = w + theta. In (theta, w) this reads w^+ = w - 2*pi*sgn(w+theta),
    theta unchanged. Returns (t, theta[], w[], info) where info records the number
    of orientation resets, whether the camera-switch guard {|theta|>=Theta} was
    reached, and the worst per-event increase of V = 1/2 theta^2 + (k_psi/2k_rho) w^2
    (<= 0 confirms the hybrid Lyapunov certificate).
    """
    import numpy as np
    Theta = p.Theta; Psi = p.Psi
    n = int(round(T / dt))
    th = np.empty(n + 1); ww = np.empty(n + 1)
    th[0], ww[0] = theta0, w0
    state = np.array([theta0, w0], dtype=float)

    def Vfun(t, w):
        return 0.5 * t * t + (p.k_psi / (2.0 * p.k_rho)) * w * w

    def f(st):
        t, w = st
        g = g_func(t)
        return np.array([-p.k_theta * t + p.k_psi * w * g,
                         -(p.k_rho / 2.0) * math.sin(2.0 * t)])

    n_reset = 0; cam_switch = False; worst_event = -1e18
    for k in range(n):
        if rk4:
            k1 = f(state); k2 = f(state + dt / 2 * k1)
            k3 = f(state + dt / 2 * k2); k4 = f(state + dt * k3)
            state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            state = state + dt * f(state)
        t, w = float(state[0]), float(state[1])
        if abs(t) >= Theta:
            cam_switch = True
        psi = w + t
        if abs(psi) >= Psi:
            Vb = Vfun(t, w)
            w = w - 2.0 * math.pi * (1.0 if psi >= 0 else -1.0)
            state[1] = w
            worst_event = max(worst_event, Vfun(t, w) - Vb)
            n_reset += 1
        th[k + 1], ww[k + 1] = state[0], state[1]
    t_arr = np.arange(n + 1) * dt
    info = {"n_reset": n_reset, "camera_switch": cam_switch,
            "worst_event_dV": worst_event}
    return t_arr, th, ww, info
