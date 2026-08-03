"""Deterministic numerical checks of the manuscript certificates.

Run ``python -m codes.certify`` (or ``certify.run_all()``). The checks compare
closed-form quantities in ``theory.py`` with finite numerical sweeps of the
nonlinear dynamics in ``model.py``. They support the analytical proofs but do
not replace them or constitute exact reachable-set computations.
"""
from __future__ import annotations

import math
import numpy as np

from .model import BearingParams, simulate_bearing
from .theory import (theta_star, confinement_condition_C, core_barrier,
                     disturbance_budget)


def _max_reach(p, theta0, w0, d_theta=0.0, T=8.0, dt=0.004):
    _, th, _ = simulate_bearing(theta0, w0, p, d_theta=d_theta, T=T, dt=dt)
    return float(np.max(np.abs(th)))


def certify_theorem_A(p: BearingParams, n_theta=25, n_w=41, verbose=True) -> dict:
    """Core C0={|theta|<=pi/2} is forward invariant for ALL admissible orientations.

    Check: every IC with |theta0| <= pi/2 has max|theta(t)| <= pi/2 + tol.
    """
    half = math.pi / 2.0
    worst = 0.0
    for th0 in np.linspace(-half, half, n_theta):
        for w0 in np.linspace(-p.W, p.W, n_w):
            worst = max(worst, _max_reach(p, th0, w0))
    cb = core_barrier(p)
    ok = worst <= half + 5e-3
    if verbose:
        print(f"  [A] gains(kr={p.k_rho},kth={p.k_theta},kpsi={p.k_psi}) d={p.delta:.3f}: "
              f"max|theta| from core ICs = {worst:.4f}  (barrier=pi/2={half:.4f}, "
              f"rate@barrier={cb['rate_at_barrier']:.3f})  -> {'PASS' if ok else 'FAIL'}")
    assert ok, "Core-invariance check failed: a core IC escaped {|theta|<=pi/2}."
    return {"max_reach_core": worst, "barrier": half, "pass": ok}


def certify_theorem_B(p: BearingParams, n_theta=30, n_w=41, verbose=True) -> dict:
    """Certified bound: max|theta(t)| <= max(|theta0|, theta*) for every interior IC.

    Also reports the genuine swing ceiling from low-bearing starts vs theta*.
    """
    cond = confinement_condition_C(p)
    ts = theta_star(p)
    if ts is None:
        if verbose:
            print(f"  [B] (C) fails (c={cond['c']:.3f} >= phi(Theta)={cond['phi_Theta']:.3f}); "
                  f"no confinement bound claimed (handoff still graceful by Thm A).")
        return {"condition_C": False, "theta_star": None}
    Th = p.Theta
    max_excess = -9.0
    swing_ceiling = 0.0
    for th0 in np.linspace(-(Th - 1e-3), Th - 1e-3, n_theta):
        for w0 in np.linspace(-p.W, p.W, n_w):
            r = _max_reach(p, th0, w0)
            max_excess = max(max_excess, r - max(abs(th0), ts))
            if abs(th0) <= 0.3:
                swing_ceiling = max(swing_ceiling, r)
    ok = (max_excess <= 5e-3) and (ts < Th) and (swing_ceiling <= ts + 5e-3)
    if verbose:
        print(f"  [B] gains(kth={p.k_theta},kpsi={p.k_psi}) d={p.delta:.3f}: "
              f"theta*={ts:.4f} (Theta={Th:.4f}, margin={Th-ts:.4f}); "
              f"max[reach-env]={max_excess:+.4f}; swing ceiling={swing_ceiling:.4f} "
              f"-> {'PASS' if ok else 'FAIL'}")
    assert ok, "Strict-FOV envelope check failed: reach exceeded certified envelope."
    return {"condition_C": True, "theta_star": ts, "margin": Th - ts,
            "max_excess": max_excess, "swing_ceiling": swing_ceiling, "pass": ok}


def certify_theorem_C(p: BearingParams, n_theta=18, n_w=29, verbose=True) -> dict:
    """Exact core-confinement budget D* = k_theta*pi/2 (both sides).

    Lower side: with sup|d| = 0.97 D* no core IC crosses pi/2.
    Upper side: with sup|d| = 1.03 D* some core IC crosses pi/2.
    Also brackets the empirical cross-threshold and checks it equals D*.
    """
    Dstar = disturbance_budget(p)
    half = math.pi / 2.0

    def crosses(D):
        for th0 in np.linspace(0.0, half - 0.02, n_theta):
            for w0 in np.linspace(-p.W, p.W, n_w):
                _, th, _ = simulate_bearing(th0, w0, p, d_theta=D, T=8.0, dt=0.005)
                if np.max(np.abs(th)) > half + 1e-4:
                    return True
        return False

    below = not crosses(0.97 * Dstar)
    above = crosses(1.03 * Dstar)
    # bisection bracket of the true threshold
    lo, hi = 0.0, 3.5 * Dstar / (math.pi / 2.0)  # generous
    lo, hi = 0.0, 2.0 * Dstar
    for _ in range(12):
        m = 0.5 * (lo + hi)
        if crosses(m):
            hi = m
        else:
            lo = m
    thr = 0.5 * (lo + hi)
    ratio = thr / Dstar
    ok = below and above and (0.95 <= ratio <= 1.05)
    if verbose:
        print(f"  [C] gains(kth={p.k_theta}) d={p.delta:.3f}: D*=k_th*pi/2={Dstar:.4f}; "
              f"empirical cross-threshold={thr:.4f} (ratio={ratio:.3f}); "
              f"safe@0.97D*={below}, crosses@1.03D*={above} -> {'PASS' if ok else 'FAIL'}")
    assert ok, "Disturbance-threshold check failed: budget does not match k_theta*pi/2."
    return {"D_star": Dstar, "empirical_threshold": thr, "ratio": ratio, "pass": ok}


def run_all(verbose=True) -> bool:
    gain_sets = [
        BearingParams(2.0, 1.0, 1.0, math.pi / 10.0),   # baseline
        BearingParams(1.0, 2.0, 1.0, math.pi / 10.0),   # increased bearing gain
        BearingParams(2.0, 1.0, 0.3, math.pi / 10.0),   # reduced mismatch gain
        BearingParams(1.0, 2.0, 1.0, math.pi / 4.0),    # larger overlap
        BearingParams(2.0, 1.0, 1.0, math.pi / 6.0),
    ]
    print("=" * 78)
    print("SELF-CERTIFICATION OF CERTIFIED-VISIBILITY THEOREMS (brute force vs theory)")
    print("=" * 78)
    print("\nCore-invariance boundary {|theta|<=pi/2}:")
    for p in gain_sets:
        certify_theorem_A(p, verbose=verbose)
    print("\nStrict-FOV envelope theta* under condition (C):")
    for p in gain_sets:
        certify_theorem_B(p, verbose=verbose)
    print("\nExact persistent disturbance threshold D*=k_theta*pi/2:")
    for p in gain_sets:
        certify_theorem_C(p, verbose=verbose)
    print("\nLemma  (Lyapunov certificate Vdot=-k_theta*theta^2, ellipse confinement):")
    for p in gain_sets:
        certify_lyapunov(p, verbose=verbose)
    print("\nProposition  (orientation-reset compatibility + HYBRID invariance):")
    for p in [BearingParams(8.0, 1.0, 1.0, math.pi / 10.0),
              BearingParams(5.0, 1.0, 1.0, math.pi / 4.0)]:
        certify_reset_compatibility(p, verbose=verbose)
    print("\nALL CHECKS PASSED.")
    return True



def certify_lyapunov(p: BearingParams, n_pts=20000, n_theta=21, n_w=31,
                     verbose=True) -> dict:
    """Verify the Lyapunov certificate (Lemma):
      (a) Vdot = -k_theta theta^2 to machine precision at random points;
      (b) every IC inside the safe ellipse stays in the FOV and respects the bound
          |theta(t)| <= sqrt(theta0^2 + (k_psi/k_rho) w0^2).
    """
    import numpy as np
    from .theory import lyapunov_Vdot, lyapunov_safe, lyapunov_bound
    rng = np.random.default_rng(0)
    # (a) identity check
    max_id = 0.0
    for _ in range(n_pts):
        th = rng.uniform(-3.0, 3.0); w = rng.uniform(-9.0, 9.0)
        max_id = max(max_id, abs(lyapunov_Vdot(th, w, p) + p.k_theta * th * th))
    # (b) ellipse confinement + bound
    Th = p.Theta; inside_ok = True; bound_ok = True; n_inside = 0
    for th0 in np.linspace(-Th + 1e-3, Th - 1e-3, n_theta):
        for w0 in np.linspace(-p.W, p.W, n_w):
            if lyapunov_safe(th0, w0, p):
                n_inside += 1
                _, th, _ = simulate_bearing(th0, w0, p, T=14.0, dt=0.003)
                mx = float(np.max(np.abs(th)))
                if mx >= Th:
                    inside_ok = False
                if mx > lyapunov_bound(th0, w0, p) + 1e-2:
                    bound_ok = False
    ok = (max_id < 1e-10) and inside_ok and bound_ok
    if verbose:
        print(f"  [Lyap] gains(kr={p.k_rho},kth={p.k_theta},kpsi={p.k_psi}) d={p.delta:.3f}: "
              f"max|Vdot+k_th th^2|={max_id:.1e}; {n_inside} ellipse ICs, all stay in FOV={inside_ok}, "
              f"bound holds={bound_ok} -> {'PASS' if ok else 'FAIL'}")
    assert ok, "Lyapunov certificate violated."
    return {"max_identity_err": max_id, "ellipse_confinement": inside_ok,
            "bound_holds": bound_ok, "pass": ok}


def certify_reset_compatibility(p: BearingParams, n_theta=41, n_psi=12,
                                n_ic=200, verbose=True) -> dict:
    """Verify the Proposition (reset compatibility) and HYBRID invariance:
      (a) V^+ <= V at the orientation reset for all |theta| <= Theta and all
          psi in the jump set {|psi| >= Psi}  (machine-precision identity);
      (b) initial conditions inside the Lyapunov ellipse stay in the field of view
          under the FULL hybrid dynamics (flow + orientation resets actually
          firing): V never increases across resets, no camera switch occurs, and
          |theta(t)| < Theta throughout.
    """
    import numpy as np
    from .model import simulate_bearing_hybrid
    from .theory import lyapunov_reset_change, lyapunov_safe
    Th = p.Theta; Psi = p.Psi

    # (a) reset non-increase on the whole jump set
    worst_drop = -1e18
    for theta in np.linspace(-Th, Th, n_theta):
        for psi in np.concatenate([np.linspace(Psi, Psi + 1.5, n_psi),
                                    np.linspace(-Psi - 1.5, -Psi, n_psi)]):
            w = psi - theta
            worst_drop = max(worst_drop, lyapunov_reset_change(theta, w, p))
    reset_ok = worst_drop <= 1e-9

    # (b) hybrid invariance of the ellipse (resets allowed to fire)
    rng = np.random.default_rng(3)
    n_in = 0; total_resets = 0; any_cam = False
    worst_event = -1e18; thmax = 0.0; inside_ok = True
    for _ in range(n_ic):
        th0 = rng.uniform(-Th, Th); w0 = rng.uniform(-3.0 * Th, 3.0 * Th)
        if lyapunov_safe(th0, w0, p):
            n_in += 1
            _, th, _, info = simulate_bearing_hybrid(th0, w0, p, T=30.0, dt=0.003)
            total_resets += info["n_reset"]
            any_cam = any_cam or info["camera_switch"]
            worst_event = max(worst_event, info["worst_event_dV"])
            mx = float(np.max(np.abs(th))); thmax = max(thmax, mx)
            if mx >= Th:
                inside_ok = False
    hybrid_ok = (worst_event <= 1e-9) and (not any_cam) and inside_ok
    ok = reset_ok and hybrid_ok
    if verbose:
        print(f"  [Reset] gains(kr={p.k_rho},kth={p.k_theta},kpsi={p.k_psi}) d={p.delta:.3f}: "
              f"max(V^+ - V) at reset={worst_drop:.1e} (<=0); "
              f"hybrid: {n_in} ellipse ICs, resets fired={total_resets}, cam switch={any_cam}, "
              f"worst event dV={worst_event:.1e}, max|theta|={thmax:.4f}<Theta={Th:.4f} -> "
              f"{'PASS' if ok else 'FAIL'}")
    assert ok, "Reset-compatibility / hybrid invariance violated."
    return {"reset_nonincrease": reset_ok, "hybrid_invariant": hybrid_ok,
            "resets_fired": total_resets, "max_reset_dV": worst_drop,
            "worst_event_dV": worst_event, "pass": ok}


def quick_certify(verbose=True):
    """Small deterministic smoke test for installation and equation consistency.

    The full grids used for the archived results remain available through
    run_all().  This reduced test is intended to finish quickly on a laptop.
    """
    p = BearingParams(2.0, 1.0, 1.0, math.pi / 10.0)
    print("QUICK CERTIFICATION (reduced deterministic grids):")
    certify_theorem_A(p, n_theta=7, n_w=9, verbose=verbose)
    certify_theorem_B(p, n_theta=7, n_w=9, verbose=verbose)
    certify_theorem_C(p, n_theta=5, n_w=7, verbose=verbose)
    certify_lyapunov(p, n_pts=800, n_theta=7, n_w=9, verbose=verbose)
    certify_reset_compatibility(
        BearingParams(8.0, 1.0, 1.0, math.pi / 10.0),
        n_theta=13, n_psi=5, n_ic=20, verbose=verbose,
    )
    print("QUICK CHECKS PASSED.")
    return True


if __name__ == "__main__":
    import sys
    if "--quick" in sys.argv:
        quick_certify()
    else:
        run_all()
