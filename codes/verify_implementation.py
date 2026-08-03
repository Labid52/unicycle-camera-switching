"""Builder-side symbolic and numerical verification of the final merged result."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import sympy as sp

from .model import BearingParams, g_func
from .implementation_errors import (
    BarrierInputs,
    BearingSamplingData,
    analytic_error_budget,
    exact_boundary_max_rate_register,
    exact_theta_rhs_formula,
    measured_feedback_rhs,
    numerical_boundary_threshold_register,
    numerical_global_Lg,
    sampled_bearing_error_bound,
)


def symbolic_identity_check() -> Dict[str, object]:
    kr,kt,kp,theta,thh,psh,d=sp.symbols('kr kt kp theta thh psh d', real=True)
    g=sp.Function('g')
    nu2=kr*sp.cos(thh)*sp.sin(thh)+kt*thh+kp*(thh-psh)*g(thh)
    from_kin=-nu2+kr*sp.cos(thh)*sp.sin(theta)+d
    claimed=(-kt*thh-kp*(thh-psh)*g(thh)
             +kr*sp.cos(thh)*(sp.sin(theta)-sp.sin(thh))+d)
    residual=sp.simplify(from_kin-claimed)
    return {"pass":bool(residual==0),"residual":str(residual)}


def direct_formula_random_check(n:int=25000,seed:int=117)->Dict[str,object]:
    rng=np.random.default_rng(seed)
    max_err=0.0
    for _ in range(n):
        p=BearingParams(float(rng.uniform(.1,6)),float(rng.uniform(.1,6)),
                        float(rng.uniform(.1,6)),float(rng.uniform(.03,1.3)))
        psi=float(rng.uniform(-p.Psi,p.Psi))
        theta=float(rng.uniform(-math.pi/2,math.pi/2))
        thh=float(rng.uniform(-p.Theta,p.Theta))
        psh=float(rng.uniform(-p.Psi,p.Psi))
        d=float(rng.uniform(-1,1))
        direct=float(measured_feedback_rhs(psi,theta,thh,psh,p,d)[1])
        formula=exact_theta_rhs_formula(theta,thh,psh,p,d)
        max_err=max(max_err,abs(direct-formula))
    return {"pass":bool(max_err<2e-12),"max_abs_error":max_err,"n":n}


def barrier_inequality_random_check(n:int=75000,seed:int=2026)->Dict[str,object]:
    """Sample the final admissible variables directly: e_theta, psi_hat and d."""
    rng=np.random.default_rng(seed)
    Lg=numerical_global_Lg()["L_g"]
    worst=-math.inf
    worst_data=None
    for _ in range(n):
        p=BearingParams(float(rng.uniform(.1,6)),float(rng.uniform(.1,6)),
                        float(rng.uniform(.1,6)),float(rng.uniform(.03,1.3)))
        E=float(rng.uniform(0,min(.8,p.delta+.4)))
        D=float(rng.uniform(0,1.2))
        side=int(rng.choice([-1,1]))
        theta=side*math.pi/2
        eth=float(rng.uniform(-E,E))
        thh=theta+eth
        psh=float(rng.uniform(-p.Psi,p.Psi))
        d=float(rng.uniform(-D,D))
        outward=side*exact_theta_rhs_formula(theta,thh,psh,p,d)
        budget=analytic_error_budget(BarrierInputs(E,D),p,L_g=Lg)
        upper=-p.k_theta*math.pi/2+float(budget["B"])
        violation=outward-upper
        if violation>worst:
            worst=violation
            worst_data={"E":E,"D":D,"side":side,"theta_hat":thh,"psi_hat":psh}
    return {"pass":bool(worst<=3e-10),"worst_violation":worst,"worst_case":worst_data,"n":n}


def orientation_register_independence_check()->Dict[str,object]:
    p=BearingParams(2,1,1,math.pi/10)
    max_dev=0.0
    expected=-p.k_theta*math.pi/2
    for side in (-1,1):
        theta=side*math.pi/2
        for psh in np.linspace(-p.Psi,p.Psi,2001):
            outward=side*exact_theta_rhs_formula(theta,theta,float(psh),p,0.0)
            max_dev=max(max_dev,abs(outward-expected))
    return {"pass":bool(max_dev<2e-12),"max_abs_deviation":max_dev}


def cubic_kr_boundary_check()->Dict[str,object]:
    ratios=[]
    for e in np.geomspace(1e-5,.4,200):
        exact=abs(math.sin(e))*(1-math.cos(e))
        upper=.5*e**3
        ratios.append(exact/upper)
    mx=max(ratios)
    return {"pass":bool(mx<=1+1e-6),"max_exact_to_bound_ratio":mx}


def fixed_point_random_check(n:int=20000,seed:int=305)->Dict[str,object]:
    rng=np.random.default_rng(seed)
    max_res=0.0
    checked=0
    for _ in range(n):
        p=BearingParams(float(rng.uniform(.1,6)),float(rng.uniform(.1,6)),
                        float(rng.uniform(.1,6)),float(rng.uniform(.03,1.3)))
        B=p.k_theta+p.k_psi+p.k_rho
        h=float(rng.uniform(0,.98/B))
        nt=float(rng.uniform(0,.2))
        D=float(rng.uniform(0,1.0))
        out=sampled_bearing_error_bound(BearingSamplingData(h,nt),p,D)
        if out["exists"]:
            checked+=1
            max_res=max(max_res,abs(float(out["fixed_point_residual"])))
    return {"pass":bool(max_res<2e-11),"max_abs_residual":max_res,"n":checked}


def threshold_resolution_check()->List[Dict[str,float]]:
    p=BearingParams(2,1,1,math.pi/10)
    rows=[]
    for n in (1001,4001,16001):
        # Root finder internally uses its default; evaluate final root with requested grid.
        root=numerical_boundary_threshold_register(p,0.0)["threshold"]
        at=exact_boundary_max_rate_register(float(root),p,0.0,n_grid=n)
        rows.append({"n_grid":n,"threshold":float(root),"outward_at_threshold":float(at["outward_rate"]),
                     "e_theta_at_max":float(at["e_theta"]),"psi_hat_at_max":float(at["psi_hat_at_max"])})
    return rows


def run_all(outdir:str|Path)->Dict[str,object]:
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    result={
        "symbolic_identity":symbolic_identity_check(),
        "random_formula":direct_formula_random_check(),
        "barrier_inequality":barrier_inequality_random_check(),
        "orientation_register_independence":orientation_register_independence_check(),
        "cubic_kr":cubic_kr_boundary_check(),
        "fixed_point":fixed_point_random_check(),
        "L_g":numerical_global_Lg(),
        "threshold_resolution":threshold_resolution_check(),
    }
    keys=("symbolic_identity","random_formula","barrier_inequality",
          "orientation_register_independence","cubic_kr","fixed_point")
    result["all_core_checks_passed"]=all(bool(result[k]["pass"]) for k in keys)
    (outdir/"builder_verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    if not result["all_core_checks_passed"]:
        raise AssertionError("Builder verification failed")
    return result


if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument("--out",default="results")
    args=ap.parse_args();print(json.dumps(run_all(args.out),indent=2))
