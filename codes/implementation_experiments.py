"""Numerical tables and figure data for the final merged revision.

The analytic certificate is proved separately.  Numerical boundary maximization
and direct sampled simulations are used only to quantify conservatism and check
implementation behavior; they are never presented as certificates.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

from .model import BearingParams
from . import experiments as original_experiments
from .theory import theta_star
from .sampled_simulation import SampledCase, grid_check
from .implementation_errors import (
    BarrierInputs,
    BearingSamplingData,
    analytic_error_budget,
    certified_bearing_error_limit,
    effective_bearing_error_limit,
    max_admissible_bearing_age,
    numerical_boundary_threshold_register,
    numerical_global_Lg,
    sampled_certificate,
)


def nominal_grid_reach(
    p: BearingParams,
    theta_values: np.ndarray,
    w_values: np.ndarray,
    horizon: float = 11.0,
    dt: float = 0.001,
) -> Dict[str, float]:
    """Vectorized fixed-step RK4 reach on an explicitly specified grid."""
    TH0, WG0 = np.meshgrid(theta_values, w_values, indexing="ij")
    theta = TH0.ravel().astype(float)
    w = WG0.ravel().astype(float)
    best = np.abs(theta).copy()
    n_steps = int(round(horizon / dt))

    def rhs(th: np.ndarray, ww: np.ndarray):
        gg = np.ones_like(th)
        mask = np.abs(th) >= 1.0e-8
        gg[mask] = np.sin(2.0*th[mask])/(2.0*th[mask])
        gg[~mask] = 1.0 - (2.0/3.0)*th[~mask]**2
        return (
            -p.k_theta*th + p.k_psi*ww*gg,
            -(p.k_rho/2.0)*np.sin(2.0*th),
        )

    for _ in range(n_steps):
        k1t, k1w = rhs(theta, w)
        k2t, k2w = rhs(theta+0.5*dt*k1t, w+0.5*dt*k1w)
        k3t, k3w = rhs(theta+0.5*dt*k2t, w+0.5*dt*k2w)
        k4t, k4w = rhs(theta+dt*k3t, w+dt*k3w)
        theta += (dt/6.0)*(k1t+2.0*k2t+2.0*k3t+k4t)
        w += (dt/6.0)*(k1w+2.0*k2w+2.0*k3w+k4w)
        best = np.maximum(best, np.abs(theta))

    idx = int(np.argmax(best))
    ts = theta_star(p)
    return {
        "n_initial_conditions": int(best.size),
        "horizon": horizon,
        "integrator": "vectorized fixed-step RK4",
        "dt": dt,
        "theta_star": ts,
        "observed_max_reach": float(best[idx]),
        "worst_theta0": float(TH0.ravel()[idx]),
        "worst_w0": float(WG0.ravel()[idx]),
    }


def _batched_rk4_endpoint_audit(
    cases: list[tuple[str, BearingParams]],
    n_w: int,
    dt: float,
    horizon: float = 11.0,
) -> list[dict[str, float]]:
    """Integrate all conservatism cases in one vectorized fixed-step RK4 loop."""
    theta_parts = []
    w_parts = []
    kr_parts = []
    kt_parts = []
    kp_parts = []
    case_parts = []
    w0_parts = []
    for case_id, (_, params) in enumerate(cases):
        w0 = np.linspace(-params.W, params.W, n_w)
        theta_parts.append(np.zeros_like(w0))
        w_parts.append(w0.copy())
        w0_parts.append(w0.copy())
        kr_parts.append(np.full_like(w0, params.k_rho))
        kt_parts.append(np.full_like(w0, params.k_theta))
        kp_parts.append(np.full_like(w0, params.k_psi))
        case_parts.append(np.full(w0.shape, case_id, dtype=int))

    theta = np.concatenate(theta_parts)
    w = np.concatenate(w_parts)
    w0_all = np.concatenate(w0_parts)
    kr = np.concatenate(kr_parts)
    kt = np.concatenate(kt_parts)
    kp = np.concatenate(kp_parts)
    case_id = np.concatenate(case_parts)
    best = np.abs(theta).copy()

    def rhs(th: np.ndarray, ww: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gg = np.ones_like(th)
        mask = np.abs(th) >= 1.0e-8
        gg[mask] = np.sin(2.0 * th[mask]) / (2.0 * th[mask])
        gg[~mask] = 1.0 - (2.0 / 3.0) * th[~mask] ** 2
        return -kt * th + kp * ww * gg, -(kr / 2.0) * np.sin(2.0 * th)

    for _ in range(int(round(horizon / dt))):
        k1t, k1w = rhs(theta, w)
        k2t, k2w = rhs(theta + 0.5 * dt * k1t, w + 0.5 * dt * k1w)
        k3t, k3w = rhs(theta + 0.5 * dt * k2t, w + 0.5 * dt * k2w)
        k4t, k4w = rhs(theta + dt * k3t, w + dt * k3w)
        theta += (dt / 6.0) * (k1t + 2.0 * k2t + 2.0 * k3t + k4t)
        w += (dt / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w)
        best = np.maximum(best, np.abs(theta))

    results: list[dict[str, float]] = []
    for cid, (_, params) in enumerate(cases):
        idxs = np.flatnonzero(case_id == cid)
        local = idxs[int(np.argmax(best[idxs]))]
        results.append({
            "theta_star": float(theta_star(params)),
            "observed_max_reach": float(best[local]),
            "worst_w0": float(w0_all[local]),
        })
    return results


def conservatism_table() -> List[Dict[str, float | str]]:
    """Finite-horizon audit of conservatism introduced by the global W bound.

    The audit fixes theta0=0 and spans the full admissible w0 interval, thereby
    isolating the mismatch-driven bearing excursion.  The reported run uses 81
    w0 values and dt=2.5e-4 s; independent checks use 41 values and dt=5e-4 s.
    """
    cases = [
        ("(2,1,1), delta=pi/10", BearingParams(2.0,1.0,1.0,math.pi/10.0)),
        ("(2,1,0.3), delta=pi/10", BearingParams(2.0,1.0,0.3,math.pi/10.0)),
        ("(1,2,1), delta=pi/10", BearingParams(1.0,2.0,1.0,math.pi/10.0)),
        ("(1,2,1), delta=pi/4", BearingParams(1.0,2.0,1.0,math.pi/4.0)),
    ]
    coarse = _batched_rk4_endpoint_audit(cases, n_w=41, dt=0.0005)
    fine81 = _batched_rk4_endpoint_audit(cases, n_w=81, dt=0.00025)

    # The 81-point grid contains the 41-point grid exactly.  Re-run only the
    # 41-point subset at the fine time step for the separate grid-resolution check.
    fine41 = _batched_rk4_endpoint_audit(cases, n_w=41, dt=0.00025)

    rows: List[Dict[str, float | str]] = []
    for (name, _), c, f41, f81 in zip(cases, coarse, fine41, fine81):
        ts = float(f81["theta_star"])
        reach = float(f81["observed_max_reach"])
        rows.append({
            "case": name,
            "theta_star_rad": ts,
            "observed_w_driven_reach_rad": reach,
            "gap_rad": ts - reach,
            "relative_gap_percent": 100.0 * (ts - reach) / ts,
            "theta0_rad": 0.0,
            "worst_w0": float(f81["worst_w0"]),
            "w0_grid_reported": "81 points on [-W,W]",
            "w0_grid_check": "41 points on [-W,W]",
            "horizon_s": 11.0,
            "dt_reported_s": 0.00025,
            "dt_halving_difference_rad": abs(float(c["observed_max_reach"]) - float(f41["observed_max_reach"])),
            "w_grid_doubling_difference_rad": abs(float(f41["observed_max_reach"]) - reach),
            "interpretation": "finite-horizon grid audit with theta0=0; not an exact reachable-set gap",
        })
    return rows

def implementation_threshold_table() -> List[Dict[str, float | str]]:
    """Final register-based barrier table; no e_psi box is used."""
    Lg = numerical_global_Lg()["L_g"]
    cases = [
        (2.0,1.0,1.0),
        (2.0,1.0,0.3),
        (1.0,2.0,1.0),
        (1.0,2.0,0.3),
    ]
    rows: List[Dict[str, float | str]] = []
    for kr, kt, kp in cases:
        p = BearingParams(kr,kt,kp,math.pi/10.0)
        limits = effective_bearing_error_limit(p, d_bar=0.0, L_g=Lg)
        observed = numerical_boundary_threshold_register(p, d_bar=0.0)
        age = max_admissible_bearing_age(p, n_theta=0.0, d_bar=0.0, L_g=Lg)
        barrier = float(limits["barrier_limit"])
        effective = float(limits["effective_limit"])
        rows.append({
            "gains": f"({kr:g},{kt:g},{kp:g})",
            "delta_deg": 18.0,
            "barrier_limit_deg": math.degrees(barrier),
            "false_switch_cap_deg": 18.0,
            "effective_certified_deg": math.degrees(effective),
            "binding_constraint": str(limits["binding_constraint"]),
            "observed_boundary_threshold_deg": math.degrees(float(observed["threshold"])),
            "observed_minus_barrier_deg": math.degrees(float(observed["threshold"])-barrier),
            "certified_fraction_of_observed": barrier/float(observed["threshold"]),
            "worst_e_theta_deg": math.degrees(float(observed["e_theta"])),
            "worst_side": int(observed["side"]),
            "worst_psi_hat_rad": float(observed["psi_hat_at_max"]),
            "h_theta_sup_ms_zero_update_error": 1000.0*float(age["h_theta_sup"]),
            "rate_infimum_hz_zero_update_error": float(age["rate_hz_infimum"]),
            "d_bar": 0.0,
        })
    return rows


def disturbance_tradeoff_table() -> List[Dict[str, float | str]]:
    """Combined d and bearing-error sensitivity for the baseline gains."""
    p = BearingParams(2.0,1.0,1.0,math.pi/10.0)
    Lg = numerical_global_Lg()["L_g"]
    Dstar = p.k_theta*math.pi/2.0
    rows = []
    for ratio in (0.0,0.25,0.50,0.75):
        dbar = ratio*Dstar
        limits = effective_bearing_error_limit(p,d_bar=dbar,L_g=Lg)
        age = max_admissible_bearing_age(p,n_theta=0.0,d_bar=dbar,L_g=Lg)
        rows.append({
            "d_over_Dstar": ratio,
            "d_bar": dbar,
            "barrier_limit_deg": math.degrees(float(limits["barrier_limit"])),
            "effective_limit_deg": math.degrees(float(limits["effective_limit"])),
            "binding_constraint": str(limits["binding_constraint"]),
            "h_theta_sup_ms": 1000.0*float(age["h_theta_sup"]),
        })
    return rows


def sampled_sensitivity_table() -> List[Dict[str, float | bool | str]]:
    """Analytic bearing-age cases plus independently chosen orientation ages.

    h_psi and n_psi are reported only to document asynchrony in the supporting
    simulations; they do not enter the theorem.
    """
    p = BearingParams(2.0,1.0,1.0,math.pi/10.0)
    Lg = numerical_global_Lg()["L_g"]
    cases = [
        (0.005,0.010,0.0,0.0),
        (0.010,0.100,0.0,math.radians(5.0)),
        (0.020,0.500,0.0,math.radians(20.0)),
        (0.010,0.020,math.radians(0.25),math.radians(0.5)),
    ]
    rows: List[Dict[str, float | bool | str]] = []
    for ht,hp,nt,np_ in cases:
        cert = sampled_certificate(BearingSamplingData(ht,nt),p,d_bar=0.0,L_g=Lg)
        rows.append({
            "h_theta_s": ht,
            "h_psi_s_simulation_only": hp,
            "n_theta_deg": math.degrees(nt),
            "n_psi_deg_simulation_only": math.degrees(np_),
            "fixed_point_exists": bool(cert["exists"]),
            "E_theta_deg": math.degrees(float(cert["E_theta"])),
            "barrier_margin": float(cert.get("barrier_margin", math.nan)),
            "false_switch_safe": bool(cert.get("barrier_false_switch_safe",False)),
            "certified": bool(cert["certified"]),
            "orientation_age_in_theorem": "not used",
        })
    return rows


def sampled_dynamic_checks() -> List[Dict[str, object]]:
    p = BearingParams(2.0,1.0,1.0,math.pi/10.0)
    cases = [
        ("moderate_asynchrony", SampledCase(0.010,0.100,0.0,math.radians(5.0))),
        ("large_orientation_age", SampledCase(0.020,0.500,0.0,math.radians(20.0))),
        ("bounded_bearing_update_error", SampledCase(
            0.010,0.020,math.radians(0.25),math.radians(0.5))),
    ]
    rows = []
    for name, case in cases:
        out = grid_check(p,case)
        rows.append({"name":name,"case":case.__dict__,**out})
    return rows


def original_protocol_euler_reach(p: BearingParams) -> Dict[str, float]:
    """Vectorized reproduction of experiments._swing_ceiling.

    This uses exactly the original deterministic grid and forward-Euler update:
    theta0 in {0,0.1,...,1.0}, 23 w0 values in [-W,W], T=11 s, dt=0.004 s.
    """
    theta_values = np.linspace(0.0,1.0,11)
    w_values = np.linspace(-p.W,p.W,23)
    TH0, WG0 = np.meshgrid(theta_values,w_values,indexing="ij")
    theta = TH0.ravel().astype(float)
    w = WG0.ravel().astype(float)
    best = np.abs(theta).copy()
    dt=0.004
    n_steps=int(round(11.0/dt))
    for _ in range(n_steps):
        gg=np.ones_like(theta)
        mask=np.abs(theta)>=1.0e-8
        gg[mask]=np.sin(2.0*theta[mask])/(2.0*theta[mask])
        gg[~mask]=1.0-(2.0/3.0)*theta[~mask]**2
        dtheta=-p.k_theta*theta+p.k_psi*w*gg
        dw=-(p.k_rho/2.0)*np.sin(2.0*theta)
        theta=theta+dt*dtheta
        w=w+dt*dw
        best=np.maximum(best,np.abs(theta))
    idx=int(np.argmax(best))
    return {
        "observed_max_reach":float(best[idx]),
        "worst_theta0":float(TH0.ravel()[idx]),
        "worst_w0":float(WG0.ravel()[idx]),
        "n_initial_conditions":int(best.size),
        "integrator":"forward Euler",
        "dt":dt,
        "horizon":11.0,
    }


def revised_fig2_ratio_data() -> List[Dict[str, float | bool]]:
    """Corrected Fig. 2(a) data using the original project protocol.

    The original grid includes theta0 up to 1 rad.  The theorem must therefore
    be compared with max{max_grid |theta0|, theta_star}, not theta_star alone.
    """
    rows = []
    for ratio in np.linspace(0.2,4.0,10):
        p=BearingParams(2.0,1.0,float(ratio),math.pi/10.0)
        ts=theta_star(p)
        reach=original_protocol_euler_reach(p)
        envelope=math.nan if ts is None else max(1.0,float(ts))
        rows.append({
            "k_psi_over_k_theta":float(ratio),
            "theta_star":math.nan if ts is None else float(ts),
            "initial_bearing_grid_max":1.0,
            "theorem_envelope_on_grid":envelope,
            "observed_reach":float(reach["observed_max_reach"]),
            "Theta":p.Theta,
            "condition_C_holds":ts is not None,
            "worst_theta0":float(reach["worst_theta0"]),
            "worst_w0":float(reach["worst_w0"]),
            "protocol":"original forward-Euler grid: theta0=0:0.1:1.0, 23 w0 values",
        })
    return rows


def revised_fig2_delta_data() -> List[Dict[str, float | bool]]:
    """Corrected Fig. 2(b) data using the original project protocol."""
    rows=[]
    for delta in np.linspace(math.pi/18.0,math.pi/2.2,9):
        p=BearingParams(2.0,1.0,1.0,float(delta))
        ts=theta_star(p)
        reach=original_protocol_euler_reach(p)
        envelope=math.nan if ts is None else max(1.0,float(ts))
        rows.append({
            "delta_rad":float(delta),
            "delta_deg":math.degrees(float(delta)),
            "theta_star":math.nan if ts is None else float(ts),
            "initial_bearing_grid_max":1.0,
            "theorem_envelope_on_grid":envelope,
            "observed_reach":float(reach["observed_max_reach"]),
            "Theta":p.Theta,
            "condition_C_holds":ts is not None,
            "worst_theta0":float(reach["worst_theta0"]),
            "worst_w0":float(reach["worst_w0"]),
            "protocol":"original forward-Euler grid: theta0=0:0.1:1.0, 23 w0 values",
        })
    return rows


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_revised_fig2(path: Path, ratio_rows: List[Dict[str,float|bool]],
                        delta_rows: List[Dict[str,float|bool]]) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(7.0,3.05))

    x = np.array([float(r["k_psi_over_k_theta"]) for r in ratio_rows])
    observed = np.array([float(r["observed_reach"]) for r in ratio_rows])
    envelope = np.array([float(r["theorem_envelope_on_grid"]) for r in ratio_rows])
    edge = np.array([float(r["Theta"]) for r in ratio_rows])
    axes[0].plot(x,observed,marker="s",linestyle="--",label="maximum sampled reach")
    axes[0].plot(x,envelope,marker="o",label=r"theorem envelope $\max\{|\theta_0|,\theta^\star\}$")
    axes[0].plot(x,edge,linestyle="-.",label=r"FOV edge $\Theta$")
    axes[0].axhline(math.pi/2.0,linestyle=":",label=r"core boundary $\pi/2$")
    axes[0].set_xlabel(r"$k_\psi/k_\theta$")
    axes[0].set_ylabel("bearing magnitude (rad)")
    axes[0].set_title("(a)")
    axes[0].grid(True,alpha=0.3)

    xd = np.array([float(r["delta_deg"]) for r in delta_rows])
    observedd = np.array([float(r["observed_reach"]) for r in delta_rows])
    enveloped = np.array([float(r["theorem_envelope_on_grid"]) for r in delta_rows])
    edged = np.array([float(r["Theta"]) for r in delta_rows])
    axes[1].plot(xd,observedd,marker="s",linestyle="--",label="maximum sampled reach")
    axes[1].plot(xd,enveloped,marker="o",label=r"theorem envelope $\max\{|\theta_0|,\theta^\star\}$")
    axes[1].plot(xd,edged,linestyle="-.",label=r"FOV edge $\Theta$")
    axes[1].axhline(math.pi/2.0,linestyle=":",label=r"core boundary $\pi/2$")
    axes[1].set_xlabel(r"overlap $\delta$ (deg)")
    axes[1].set_title("(b)")
    axes[1].grid(True,alpha=0.3)

    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=2,frameon=False,
               bbox_to_anchor=(0.5,1.04))
    fig.tight_layout(rect=(0,0,1,0.84))
    fig.savefig(path.with_suffix(".png"),dpi=300,bbox_inches="tight")
    fig.savefig(path.with_suffix(".eps"),format="eps",bbox_inches="tight")
    plt.close(fig)


def run_all(outdir: str | Path) -> Dict[str,object]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    (outdir/"figures").mkdir(exist_ok=True)
    data = {
        "implementation_thresholds": implementation_threshold_table(),
        "disturbance_tradeoff": disturbance_tradeoff_table(),
        "sampled_sensitivity": sampled_sensitivity_table(),
        "sampled_dynamic_checks": sampled_dynamic_checks(),
        "w_bound_conservatism": conservatism_table(),
        "fig2_full_grid_ratio": revised_fig2_ratio_data(),
        "fig2_full_grid_delta": revised_fig2_delta_data(),
    }
    (outdir/"implementation_experiments.json").write_text(json.dumps(data,indent=2),encoding="utf-8")
    for name, rows in data.items():
        if isinstance(rows,list) and rows and isinstance(rows[0],dict):
            _write_csv(outdir/f"{name}.csv",rows)
    _write_revised_fig2(outdir/"figures"/"fig2_certified_bound_final",
                        data["fig2_full_grid_ratio"],data["fig2_full_grid_delta"])
    return data


if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="results")
    args=ap.parse_args()
    print(json.dumps(run_all(args.out),indent=2))
