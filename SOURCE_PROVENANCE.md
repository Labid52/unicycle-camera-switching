# Source provenance

The numerical implementation in this repository is based on the simulation code
used for the analyses reported in **“Bearing Invariance and Robust Disturbance
Thresholds for Hybrid Camera-Based Unicycle Stabilization.”**

The nominal simulation protocols follow those described in the paper. Dedicated
modules implement the measured-coordinate analysis, sampled-bearing analysis,
strict-FOV sweep, implementation-threshold calculations, and the `theta0=0`
conservatism study.

The controller and hybrid update laws implemented here are the same laws
analyzed in the paper; the repository does not introduce a different controller
or switching rule.

Precomputed outputs in `reference_results/` are provided as known-good
references for checking a fresh reproduction run.
