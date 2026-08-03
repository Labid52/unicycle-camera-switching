#!/usr/bin/env python3
"""Run one reproducibility stage per invocation.

Separate invocations keep the numerical, plotting, and symbolic-audit stages
isolated and make failures easier to diagnose.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def collect(out: Path) -> None:
    figures_out = out / "figures"
    figures_out.mkdir(parents=True, exist_ok=True)
    for stage in (out / "nominal" / "figures", out / "implementation" / "figures"):
        if stage.exists():
            for src in stage.iterdir():
                if src.is_file():
                    shutil.copy2(src, figures_out / src.name)

    audit_file = out / "audit" / "builder_verification.json"
    impl_file = out / "implementation" / "implementation_experiments.json"
    if not audit_file.exists() or not impl_file.exists():
        raise FileNotFoundError("Run the implementation and audit stages before collect.")
    verification = json.loads(audit_file.read_text(encoding="utf-8"))
    implementation = json.loads(impl_file.read_text(encoding="utf-8"))
    manifest = {
        "implementation_sections": sorted(implementation.keys()),
        "implementation_verification_passed": bool(verification["all_core_checks_passed"]),
        "manuscript_figure_directory": str(figures_out.resolve()),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("implementation", "nominal", "audit", "collect"))
    parser.add_argument("--out", default="results", help="root output directory")
    parser.add_argument("--quick", action="store_true", help="reduced nominal certification grid")
    parser.add_argument("--no-figures", action="store_true", help="skip nominal figures")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.stage == "implementation":
        from codes.implementation_experiments import run_all
        run_all(out / "implementation")
    elif args.stage == "nominal":
        from codes import certify, figures, report
        from codes.model import BearingParams
        nominal = out / "nominal"
        (nominal / "figures").mkdir(parents=True, exist_ok=True)
        (nominal / "tables").mkdir(parents=True, exist_ok=True)
        certify.quick_certify() if args.quick else certify.run_all()
        if not args.no_figures:
            figures.make_all(nominal / "figures", BearingParams())
        report.write_table(nominal / "tables" / "certified_quantities.txt")
    elif args.stage == "audit":
        from codes.verify_implementation import run_all
        run_all(out / "audit")
    else:
        collect(out)


if __name__ == "__main__":
    main()
