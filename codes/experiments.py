"""Deterministic numerical experiments used by the manuscript figures.

The nominal figures use the original fixed-step forward-Euler protocol.  The
persistent-disturbance sweep applies a constant disturbance on each trajectory,
aligned with the sign of its initial bearing.  Additional implementation-error,
sampled-update, and W-conservatism audits are in implementation_experiments.py.
"""
from __future__ import annotations

import math
import numpy as np

from .model import BearingParams, simulate_bearing, g_func
from .theory import theta_star, disturbance_budget, confinement_condition_C


# ----------------------------------------------------------------------------
# Fig. 1: phase portrait and core-invariance boundary
# ----------------------------------------------------------------------------
def phase_portrait_data(p: BearingParams, n_traj: int = 24, seed: int = 7):
    """Return a vector field grid and a bundle of confined trajectories."""
    th_grid = np.linspace(-p.Theta, p.Theta, 27)
    w_grid = np.linspace(-p.W, p.W, 21)
    TH, WW = np.meshgrid(th_grid, w_grid)
    dTH = -p.k_theta * TH + p.k_psi * WW * np.vectorize(g_func)(TH)
    dWW = -(p.k_rho / 2.0) * np.sin(2.0 * TH)

    rng = np.random.default_rng(seed)
    trajs = []
    for _ in range(n_traj):
        th0 = rng.uniform(-math.pi / 2.0, math.pi / 2.0)   # start in core (in-view)
        w0 = rng.uniform(-p.W, p.W)
        t, th, w = simulate_bearing(th0, w0, p, T=12.0, dt=0.003)
        trajs.append((t, th, w))
    return {"TH": TH, "WW": WW, "dTH": dTH, "dWW": dWW, "trajs": trajs}


# ----------------------------------------------------------------------------
# Fig. 2: strict-FOV certificate versus sampled numerical reach
# ----------------------------------------------------------------------------
def _swing_ceiling(p: BearingParams, n_w: int = 23, T: float = 11.0, dt: float = 0.004):
    """Genuine outward reach: sup over low-bearing ICs of max|theta|."""
    best = 0.0
    for th0 in np.linspace(0.0, 1.0, 11):
        for w0 in np.linspace(-p.W, p.W, n_w):
            _, th, _ = simulate_bearing(th0, w0, p, T=T, dt=dt)
            best = max(best, float(np.max(np.abs(th))))
    return best


def bound_vs_reach_sweep(k_rho=2.0, k_theta=1.0, delta=math.pi / 10.0,
                         ratios=None):
    """Sweep k_psi/k_theta; return certified theta* and brute swing ceiling."""
    if ratios is None:
        ratios = np.linspace(0.2, 4.0, 10)
    rows = []
    for r in ratios:
        p = BearingParams(k_rho, k_theta, r * k_theta, delta)
        ts = theta_star(p)
        sc = _swing_ceiling(p)
        rows.append({"ratio": float(r), "theta_star": ts, "swing_ceiling": sc,
                     "Theta": p.Theta, "C_holds": confinement_condition_C(p)["holds"]})
    return rows


def bound_vs_delta_sweep(k_rho=2.0, k_theta=1.0, k_psi=1.0, deltas=None):
    if deltas is None:
        deltas = np.linspace(math.pi / 18.0, math.pi / 2.2, 9)
    rows = []
    for d in deltas:
        p = BearingParams(k_rho, k_theta, k_psi, d)
        ts = theta_star(p)
        sc = _swing_ceiling(p)
        rows.append({"delta": float(d), "theta_star": ts, "swing_ceiling": sc,
                     "Theta": p.Theta})
    return rows


# ----------------------------------------------------------------------------
# Fig. 3: exact persistent yaw-disturbance threshold
# ----------------------------------------------------------------------------
def disturbance_budget_sweep(p: BearingParams, n_ic: int = 60,
                             D_over_Dstar=None, seed: int = 2026):
    """Vectorized reproduction of the persistent-disturbance sweep.

    For each trajectory the disturbance is constant and aligned with the sign of
    its initial bearing.  The routine preserves the original grid, horizon, and
    forward-Euler step while integrating each disturbance case in batch.
    """
    Dstar = disturbance_budget(p)
    if D_over_Dstar is None:
        D_over_Dstar = np.linspace(0.0, 1.6, 25)
    rng = np.random.default_rng(seed)
    half = math.pi / 2.0
    core_theta = rng.uniform(-half, half, n_ic)
    core_w = rng.uniform(-p.W, p.W, n_ic)
    extra_n = n_ic // 2
    signs = np.where(rng.uniform(-1.0, 1.0, extra_n) >= 0.0, 1.0, -1.0)
    extra_theta = signs * rng.uniform(half, p.Theta, extra_n)
    extra_w = rng.uniform(-p.W, p.W, extra_n)
    view_theta0 = np.concatenate([core_theta, extra_theta])
    view_w0 = np.concatenate([core_w, extra_w])

    def batch_rate(theta0: np.ndarray, w0: np.ndarray, D: float, limit: float) -> float:
        theta = theta0.astype(float).copy()
        w = w0.astype(float).copy()
        best = np.abs(theta).copy()
        d = D * np.where(theta0 >= 0.0, 1.0, -1.0)
        dt = 0.004
        for _ in range(int(round(12.0 / dt))):
            g = np.ones_like(theta)
            mask = np.abs(theta) >= 1.0e-8
            g[mask] = np.sin(2.0 * theta[mask]) / (2.0 * theta[mask])
            g[~mask] = 1.0 - (2.0 / 3.0) * theta[~mask] ** 2
            dtheta = -p.k_theta * theta + p.k_psi * w * g + d
            dw = -(p.k_rho / 2.0) * np.sin(2.0 * theta)
            theta += dt * dtheta
            w += dt * dw
            best = np.maximum(best, np.abs(theta))
        return float(np.mean(best <= limit + 1.0e-4))

    rows = []
    for ratio in D_over_Dstar:
        D = float(ratio) * Dstar
        rows.append({
            "D_over_Dstar": float(ratio),
            "D": D,
            "core_rate": batch_rate(core_theta, core_w, D, half),
            "fov_rate": batch_rate(view_theta0, view_w0, D, p.Theta),
        })
    return {"Dstar": Dstar, "rows": rows}


# ----------------------------------------------------------------------------
# Fig. 5 : trajectory-based safe set in the (theta0, w0) plane (Lyapunov Lemma)
# ----------------------------------------------------------------------------
def lyapunov_safe_set_data(p: BearingParams, n_theta: int = 61, n_w: int = 61,
                           T: float = 16.0, dt: float = 0.004):
    """Vectorized safe-set sweep on the original deterministic grid."""
    Th = p.Theta
    th_grid = np.linspace(-Th, Th, n_theta)
    w_grid = np.linspace(-p.W, p.W, n_w)
    TH0, W0 = np.meshgrid(th_grid, w_grid, indexing="xy")
    theta = TH0.ravel().astype(float)
    w = W0.ravel().astype(float)
    best = np.abs(theta).copy()
    for _ in range(int(round(T / dt))):
        g = np.ones_like(theta)
        mask = np.abs(theta) >= 1.0e-8
        g[mask] = np.sin(2.0 * theta[mask]) / (2.0 * theta[mask])
        g[~mask] = 1.0 - (2.0 / 3.0) * theta[~mask] ** 2
        dtheta = -p.k_theta * theta + p.k_psi * w * g
        dw = -(p.k_rho / 2.0) * np.sin(2.0 * theta)
        theta += dt * dtheta
        w += dt * dw
        best = np.maximum(best, np.abs(theta))
    safe = (best < Th - 1.0e-3).reshape(n_w, n_theta)
    return {
        "theta_grid": th_grid,
        "w_grid": w_grid,
        "safe": safe,
        "Theta": Th,
        "W": p.W,
        "half": math.pi / 2.0,
        "ratio": p.k_psi / p.k_rho,
        "condC": confinement_condition_C(p)["holds"],
    }

