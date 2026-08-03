"""Produce the certified-quantities table (the paper's results table).

For each gain set we report: condition (C) status, certified confinement bound
theta*, the visibility margin Theta - theta*, and the exact disturbance budget
D* = k_theta*pi/2. All are closed-form (theory.py) and independently
certified (certify.py).
"""
from __future__ import annotations

import math
from pathlib import Path

from .model import BearingParams
from .theory import (theta_star, confinement_condition_C, disturbance_budget,
                     visibility_margin)

GAIN_SETS = [
    ("baseline",           BearingParams(2.0, 1.0, 1.0, math.pi / 10.0)),
    ("reduced k_psi",      BearingParams(2.0, 1.0, 0.3, math.pi / 10.0)),
    ("increased k_theta",  BearingParams(1.0, 2.0, 1.0, math.pi / 10.0)),
    ("wide overlap d=pi/4", BearingParams(1.0, 2.0, 1.0, math.pi / 4.0)),
]


def build_table():
    rows = []
    for name, p in GAIN_SETS:
        c = confinement_condition_C(p)
        ts = theta_star(p)
        m = visibility_margin(p)
        rows.append({
            "name": name,
            "k_rho": p.k_rho, "k_theta": p.k_theta, "k_psi": p.k_psi,
            "delta": p.delta, "Theta": p.Theta,
            "C_holds": c["holds"], "c": c["c"], "phi_Theta": c["phi_Theta"],
            "theta_star": ts, "margin": m,
            "D_star": disturbance_budget(p),
        })
    return rows


def write_table(path: Path):
    rows = build_table()
    lines = []
    lines.append("Certified-visibility quantities (closed form; see certify.py for checks)")
    lines.append("=" * 74)
    header = f"{'gain set':22s} {'(C)':>4s} {'theta*':>8s} {'Theta':>7s} {'margin':>7s} {'D*[rad/s]':>10s}"
    lines.append(header)
    lines.append("-" * 74)
    for r in rows:
        ts = f"{r['theta_star']:.3f}" if r['theta_star'] is not None else "  --"
        mg = f"{r['margin']:.3f}" if r['margin'] is not None else "  --"
        lines.append(f"{r['name']:22s} {('Y' if r['C_holds'] else 'n'):>4s} "
                     f"{ts:>8s} {r['Theta']:7.3f} {mg:>7s} {r['D_star']:10.3f}")
    lines.append("-" * 74)
    lines.append("(C): strict-FOV condition (Y implies that the certified flow does not reach the camera guard).")
    lines.append("theta*: certified bearing reach bound; margin = Theta - theta* (angular safety to FOV edge).")
    lines.append("D* = k_theta*pi/2: exact persistent yaw-disturbance threshold preserving core invariance.")
    txt = "\n".join(lines)
    Path(path).write_text(txt, encoding="utf-8")
    return txt
