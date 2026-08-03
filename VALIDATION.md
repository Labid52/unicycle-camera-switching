# Validation record

The final package was checked in four independent invocations from the repository root:

```bash
python run_project.py implementation --out stage_test
python run_project.py nominal --quick --out stage_test
python run_project.py audit --out stage_test
python run_project.py collect --out stage_test
```

Observed runtimes in the build environment were approximately 24 s, 11 s, 4 s,
and less than 1 s, respectively. The implementation audit reported:

- exact symbolic measured-feedback identity residual: `0`;
- maximum direct-formula discrepancy over 25,000 random cases: `7.11e-15`;
- no violation in 75,000 random barrier checks;
- orientation-register-only boundary deviation: `2.22e-16`;
- fixed-point residual below `3e-14`;
- all nominal reduced-grid checks passed.

The full blue LaTeX source was syntax-compiled with a local IEEEtran-based
compatibility wrapper because `ieeeconf.cls` and the actual `Ref.bib` were not
part of the supplied files. The source compiled without LaTeX errors or
overfull boxes. Final pagination must be checked with the author's actual
L-CSS class and bibliography.
