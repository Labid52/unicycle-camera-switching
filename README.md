# Hybrid Camera Unicycle: reproducible code for the bearing-invariance paper

This repository reproduces the numerical calculations and implementation-level
certificates used in the revised manuscript **“Bearing Invariance and Robust
Disturbance Thresholds for Hybrid Camera-Based Unicycle Stabilization.”**

The controller and hybrid rules are not redesigned. The added modules analyze
bounded bearing error, zero-order-held bearing updates, asynchronous orientation
register updates, the measured guard, and the conservatism introduced by the
global mismatch bound `W`.

## Project map

| Path | Purpose |
|---|---|
| `codes/model.py` | nominal reduced and full closed-loop models; forward-Euler and hybrid integrators |
| `codes/theory.py` | nominal core, strict-FOV, Lyapunov, and disturbance quantities |
| `codes/certify.py` | numerical checks of the nominal analytical results |
| `codes/experiments.py` | original numerical protocols for Figs. 1, 3, and 5 |
| `codes/figures.py` | generation of Figs. 1, 3, and 5 |
| `codes/implementation_errors.py` | exact measured-feedback dynamics, error budget, and sampled-bearing bound |
| `codes/sampled_simulation.py` | direct zero-order-held and asynchronous-coordinate simulations |
| `codes/implementation_experiments.py` | revised Fig. 2, measured-error table, sampled checks, and `theta0=0` conservatism audit |
| `codes/verify_implementation.py` | symbolic and numerical verification of the new equations and bounds |
| `run_project.py` | integrated reproduction entry point |
| `reference_results/` | precomputed outputs included for comparison |
| `manuscript/` | full blue and clean LaTeX sources plus revised figure assets |

## Environment

Designed for Python 3.10 or later; tested in the build environment with Python 3.13.5.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run commands from the repository root, the directory that contains
`run_project.py` and the `codes/` package.

## Run the complete project

Run the following stages as separate commands from the repository root. Each
command starts a fresh Python process, which keeps plotting, numerical sweeps,
and symbolic auditing isolated.

```bash
python run_project.py implementation --out results
python run_project.py nominal --quick --out results
python run_project.py audit --out results
python run_project.py collect --out results
```

The commands above reproduce the revised implementation tables, corrected
Fig. 2, reduced nominal certification check, Figs. 1, 3, and 5, and the
symbolic/numerical audit. To run the full nominal certification grids, replace
the second command with:

```bash
python run_project.py nominal --out results
```

The full nominal certification is more expensive. The Lyapunov safe-set figure
uses the deterministic 61 x 61 initial-condition grid specified in the
manuscript. The underlying modules can also be run directly:

```bash
python -m codes.implementation_experiments --out results/implementation
python -m codes.run_all --quick --out results/nominal
python -m codes.verify_implementation --out results/audit
```

## Output correspondence

`results/figures/fig2_certified_bound_final.{png,eps}` is the revised Fig. 2.
It retains the original forward-Euler initial-condition grid and compares the
sampled reach with the theorem's grid-level envelope
`max(max_grid |theta0|, theta_star)`.

`results/implementation/w_bound_conservatism.csv` uses `theta0=0` and 81 values of
`w0 in [-W,W]`, thereby isolating the mismatch-driven excursion affected by the
substitution `|w(t)| <= W`.

`results/implementation/implementation_thresholds.csv` contains the analytical
bearing-error limit, the overlap-limited effective bound, deterministic
boundary-maximization reference, and sampled-bearing age supremum.

`results/audit/builder_verification.json` records the symbolic identity,
random direct-formula comparison, barrier-inequality checks, orientation-register
independence check, cubic `k_rho` estimate, and fixed-point residual check.

## Scope

The implementation-level result certifies true-core invariance and prevention
of a false measured guard under the stated deterministic bounds. It does not
claim full sampled-data convergence for arbitrarily stale orientation data, nor
a positive switching-delay margin after the unchanged FOV guard is reached.
