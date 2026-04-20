"""Aggregate outputs/reports/*.json into a Markdown table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "outputs" / "reports")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    for p in sorted(args.reports_dir.glob("*.json")):
        with open(p) as f:
            r = json.load(f)
        rows.append((r.get("experiment", p.stem), r.get("dataset", "?"),
                     r.get("num_images", 0), r.get("mean_iou", 0) * 100,
                     r.get("pixel_acc", 0) * 100,
                     r.get("mean_class_acc", 0) * 100))
    lines = ["| experiment | dataset | N | mIoU | pixAcc | meanAcc |",
             "|---|---|---:|---:|---:|---:|"]
    for name, ds, n, m, pa, ma in rows:
        lines.append(f"| {name} | {ds} | {n} | {m:.2f} | {pa:.2f} | {ma:.2f} |")
    out = "\n".join(lines)
    print(out)
    if args.out:
        args.out.write_text(out + "\n")


if __name__ == "__main__":
    main()
