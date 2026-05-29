"""Aggregate ``runs/rq1_*.json`` into the Table III row values.

Usage::

    python parse_results.py /runs/rq1_*.json --out table_iii.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate RQ1 JSONs into Table III.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for path in args.inputs:
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        rows.append({
            "system": payload["system"],
            "mIoU": payload["macro"]["mIoU"],
            "ECE": payload["macro"]["ECE"],
        })

    rows.sort(key=lambda r: r["system"])
    with args.out.open("w", encoding="utf-8") as fh:
        fh.write("% Auto-generated from rq1_*.json by parse_results.py\n")
        fh.write("\\begin{tabular}{lcc}\n")
        fh.write("\\toprule\nSystem & mIoU & ECE \\\\\n\\midrule\n")
        for r in rows:
            fh.write(f"{r['system']} & {r['mIoU']:.1f} & {r['ECE']:.2f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
