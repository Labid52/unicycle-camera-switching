# Validation record

The repository was checked using four independent invocations from the
repository root:

```bash
python run_project.py implementation --out stage_test
python run_project.py nominal --quick --out stage_test
python run_project.py audit --out stage_test
python run_project.py collect --out stage_test
```

Observed runtimes in the tested environment were approximately 24 s, 11 s, 4 s,
and less than 1 s, respectively. Runtime will vary with hardware and software
environment.

The implementation audit reported:

- exact symbolic measured-feedback identity residual: `0`;
- maximum direct-formula discrepancy over 25,000 random cases: `7.11e-15`;
- no violation in 75,000 random barrier checks;
- orientation-register-only boundary deviation: `2.22e-16`;
- fixed-point residual below `3e-14`;
- all nominal reduced-grid checks passed.

These checks validate the symbolic identities, numerical formulas, barrier
conditions, sampled-coordinate calculations, and reduced-grid numerical
protocols implemented in the repository.
