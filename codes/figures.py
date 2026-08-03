
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .model import BearingParams
from . import experiments as ex

plt.rcParams.update({
    "figure.dpi": 200, "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "legend.frameon": False, "legend.fontsize": 8.3,
    "ps.fonttype": 42, "pdf.fonttype": 42,
})
C_EDGE = "#b00020"; C_BAR = "#1565c0"; C_OVL = "#fde7b0"
C_TRAJ = "#37474f"; C_CERT = "#6a1b9a"; C_BRUTE = "#2e7d32"


def _save(fig, stem: Path):
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".eps"), bbox_inches="tight", format="eps")
    plt.close(fig)


def _csv(stem: Path, header, rows):
    with open(stem.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def fig_phase_portrait(p: BearingParams, stem: Path):
    d = ex.phase_portrait_data(p); half = math.pi / 2.0
    fig, (axp, axt) = plt.subplots(1, 2, figsize=(7.2, 3.3))
    axp.streamplot(d["TH"], d["WW"], d["dTH"], d["dWW"], density=1.0,
                   color="0.78", linewidth=0.6, arrowsize=0.7)
    axp.axvspan(half - p.delta, half + p.delta, color=C_OVL, lw=0, zorder=0)
    axp.axvspan(-half - p.delta, -half + p.delta, color=C_OVL, lw=0, zorder=0)
    for (t, th, w) in d["trajs"]:
        axp.plot(th, w, color=C_TRAJ, lw=0.8)
    for x in (half, -half):
        axp.axvline(x, color=C_BAR, lw=1.4, ls="--")
    for x in (p.Theta, -p.Theta):
        axp.axvline(x, color=C_EDGE, lw=1.4)
    axp.set_xlim(-p.Theta - 0.05, p.Theta + 0.05)
    axp.set_xlabel(r"bearing $\theta$ [rad]"); axp.set_ylabel(r"$w=\psi-\theta$ [rad]")
    axp.set_title("(a) bearing phase flow")
    for (t, th, w) in d["trajs"]:
        axt.plot(t, th, color=C_TRAJ, lw=0.8)
    axt.axhspan(half - p.delta, half + p.delta, color=C_OVL, lw=0, zorder=0)
    axt.axhspan(-half - p.delta, -half + p.delta, color=C_OVL, lw=0, zorder=0)
    axt.axhline(half, color=C_BAR, lw=1.4, ls="--", label=r"$\pm\pi/2$ barrier")
    axt.axhline(-half, color=C_BAR, lw=1.4, ls="--")
    axt.axhline(p.Theta, color=C_EDGE, lw=1.4, label=r"FOV edge $\pm\Theta$")
    axt.axhline(-p.Theta, color=C_EDGE, lw=1.4)
    axt.plot([], [], color=C_TRAJ, lw=1.0, label=r"$\theta(t)$, in-view ICs")
    axt.add_patch(plt.Rectangle((0, 0), 0, 0, color=C_OVL, label="overlap band"))
    axt.set_xlabel("time [s]"); axt.set_ylabel(r"$\theta(t)$ [rad]")
    axt.set_title("(b) in-view trajectories stay in core")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    h, l = axt.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), fontsize=8.2)
    _save(fig, stem)
    rows = []
    for i, (t, th, w) in enumerate(d["trajs"][:8]):
        for k in range(0, len(t), 20):
            rows.append([i, f"{t[k]:.4f}", f"{th[k]:.5f}", f"{w[k]:.5f}"])
    _csv(stem, ["traj_id", "t", "theta", "w"], rows)


def fig_bound_vs_reach(stem: Path, delta=math.pi / 10.0):
    rr = ex.bound_vs_reach_sweep(delta=delta); dd = ex.bound_vs_delta_sweep()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.3))
    ratios = [r["ratio"] for r in rr]
    ts = [r["theta_star"] if r["theta_star"] is not None else np.nan for r in rr]
    sc = [r["swing_ceiling"] for r in rr]; Th = rr[0]["Theta"]
    a1.plot(ratios, ts, "o-", color=C_CERT, ms=3.5, label=r"certified $\theta^\star$")
    a1.plot(ratios, sc, "s--", color=C_BRUTE, ms=3.0, label="maximum sampled reach")
    a1.axhline(Th, color=C_EDGE, lw=1.4, label=r"FOV edge $\Theta$")
    a1.axhline(math.pi / 2.0, color=C_BAR, lw=1.2, ls=":", label=r"$\pi/2$ barrier")
    a1.set_xlabel(r"gain ratio $k_\psi/k_\theta$"); a1.set_ylabel("max bearing reach [rad]")
    a1.set_title(r"(a) sweep $k_\psi/k_\theta$, $\delta=\pi/10$")
    dvals = [r["delta"] for r in dd]
    ts2 = [r["theta_star"] if r["theta_star"] is not None else np.nan for r in dd]
    sc2 = [r["swing_ceiling"] for r in dd]; Th2 = [r["Theta"] for r in dd]
    a2.plot(dvals, ts2, "o-", color=C_CERT, ms=3.5)
    a2.plot(dvals, sc2, "s--", color=C_BRUTE, ms=3.0)
    a2.plot(dvals, Th2, color=C_EDGE, lw=1.4)
    a2.axhline(math.pi / 2.0, color=C_BAR, lw=1.2, ls=":")
    a2.set_xlabel(r"overlap half-angle $\delta$ [rad]"); a2.set_ylabel("max bearing reach [rad]")
    a2.set_title(r"(b) sweep $\delta$, $k_\psi=k_\theta$")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), fontsize=8.2)
    _save(fig, stem)
    rows = [[f"{r['ratio']:.3f}", r['theta_star'], f"{r['swing_ceiling']:.4f}",
             f"{r['Theta']:.4f}", r['C_holds']] for r in rr]
    _csv(stem, ["k_psi_over_k_theta", "theta_star", "swing_ceiling", "Theta", "C_holds"], rows)


def fig_disturbance_budget(p: BearingParams, stem: Path):
    res = ex.disturbance_budget_sweep(p); rows = res["rows"]
    x = [r["D_over_Dstar"] for r in rows]; core = [r["core_rate"] for r in rows]
    fov = [r["fov_rate"] for r in rows]
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    ax.plot(x, core, "o-", color=C_BAR, ms=3.2, label=r"core confinement $|\theta|\leq\pi/2$")
    ax.plot(x, fov, "s-", color=C_BRUTE, ms=3.0, label=r"FOV visibility $|\theta|\leq\Theta$")
    ax.axvline(1.0, color=C_EDGE, lw=1.5, ls="--", label=r"$D^\star=k_\theta\pi/2$")
    ax.set_xlabel(r"disturbance amplitude $\sup|d|\,/\,D^\star$")
    ax.set_ylabel("preservation rate over sampled ICs"); ax.set_ylim(-0.03, 1.05)
    ax.set_title("Core robust-invariance threshold")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.02), fontsize=8.2)
    _save(fig, stem)
    _csv(stem, ["D_over_Dstar", "core_rate", "fov_rate"],
         [[f"{r['D_over_Dstar']:.3f}", f"{r['core_rate']:.3f}", f"{r['fov_rate']:.3f}"] for r in rows])


def fig_lyapunov_safe_set(p: BearingParams, stem: Path):
    """(theta0,w0) plane: brute-force safe set, core strip, Lyapunov ellipse."""
    import numpy as np
    d = ex.lyapunov_safe_set_data(p)
    th = d["theta_grid"]; w = d["w_grid"]; half = d["half"]; Th = d["Theta"]
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    # brute-force true safe set (filled)
    ax.contourf(th, w, d["safe"].astype(float), levels=[0.5, 1.5],
                colors=["#c8e6c9"], alpha=1.0)
    ax.contour(th, w, d["safe"].astype(float), levels=[0.5],
               colors=["#2e7d32"], linewidths=1.0)
    # core strip |theta0|<=pi/2
    ax.axvspan(-half, half, color="#bbdefb", alpha=0.45, lw=0, zorder=0)
    for x in (half, -half):
        ax.axvline(x, color=C_BAR, lw=1.3, ls="--")
    # Lyapunov ellipse theta0^2 + (k_psi/k_rho) w0^2 = Theta^2
    tt = np.linspace(-Th, Th, 400)
    val = (Th**2 - tt**2) / d["ratio"]
    val = np.clip(val, 0, None)
    we = np.sqrt(val)
    ax.plot(tt, we, color=C_CERT, lw=1.8)
    ax.plot(tt, -we, color=C_CERT, lw=1.8)
    # FOV edges
    for x in (Th, -Th):
        ax.axvline(x, color=C_EDGE, lw=1.3)
    ax.set_xlim(-Th - 0.03, Th + 0.03); ax.set_ylim(-d["W"], d["W"])
    ax.set_xlabel(r"initial bearing $\theta_0$ [rad]")
    ax.set_ylabel(r"initial $w_0=\psi_0-\theta_0$ [rad]")
    cc = "holds" if d["condC"] else "fails"
    ax.set_title(rf"Safe initial conditions ($k_\psi/k_\theta$ large, (C) {cc})")
    # proxy legend handles
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    handles = [mpatches.Patch(fc="#c8e6c9", ec="#2e7d32", label="sampled safe set"),
               mpatches.Patch(fc="#bbdefb", ec=C_BAR, label=r"core $|\theta_0|\leq\pi/2$"),
               Line2D([0],[0], color=C_CERT, lw=1.8, label=r"Lyapunov ellipse (Lemma)"),
               Line2D([0],[0], color=C_EDGE, lw=1.3, label=r"FOV edge $\pm\Theta$")]
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), fontsize=7.8)
    _save(fig, stem)
    # csv: safe-set mask
    rows = []
    for i, w0 in enumerate(d["w_grid"]):
        for j, th0 in enumerate(d["theta_grid"]):
            rows.append([f"{th0:.4f}", f"{w0:.4f}", int(d["safe"][i, j])])
    _csv(stem, ["theta0", "w0", "safe"], rows)



def make_all(outdir: Path, p: BearingParams | None = None):
    """Generate the nominal manuscript figures other than revised Fig. 2.

    Revised Fig. 2 is generated by implementation_experiments.run_all(), which
    uses the theorem-correct grid-level envelope.
    """
    p = p or BearingParams(2.0, 1.0, 1.0, math.pi / 10.0)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stems = {
        "phase": outdir / "fig1_bearing_barrier",
        "budget": outdir / "fig3_core_confinement_budget",
    }
    fig_phase_portrait(p, stems["phase"])
    fig_disturbance_budget(p, stems["budget"])
    p_fail = BearingParams(5.0, 1.0, 5.0, math.pi / 10.0)
    stems["lyap"] = outdir / "fig5_lyapunov_safe_set"
    fig_lyapunov_safe_set(p_fail, stems["lyap"])
    return {key: stem.with_suffix(".eps") for key, stem in stems.items()}
