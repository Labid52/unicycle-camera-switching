#!/usr/bin/env python3
"""Run nominal certification, generate manuscript figures, and write the table."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from .model import BearingParams
from . import certify, figures, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results", help="output directory")
    parser.add_argument("--quick", action="store_true", help="use the reduced certification grid")
    parser.add_argument("--no-figures", action="store_true", help="skip nominal figure generation")
    args = parser.parse_args()

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    if args.quick:
        certify.quick_certify()
    else:
        certify.run_all()

    if not args.no_figures:
        p = BearingParams(2.0, 1.0, 1.0, math.pi / 10.0)
        figures.make_all(out / "figures", p)

    report.write_table(out / "tables" / "certified_quantities.txt")
    print(f"Nominal outputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
