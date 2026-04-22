"""Export the repo's NYUv2 layout for supervised RGB-D baselines."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.supervised.export import (  # noqa: E402
    DFORMER_EXPORT_DIRNAME,
    ESANET_EXPORT_DIRNAME,
    export_nyuv2_for_backend,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("dformerv2", "esanet"), required=True)
    ap.add_argument("--dataset", type=str, default="nyuv2_40")
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ds_cfg = OmegaConf.load(REPO / "configs" / "dataset" / f"{args.dataset}.yaml")
    source_root = REPO / ds_cfg.root
    splits_file = source_root / ds_cfg.splits_file

    default_dir = DFORMER_EXPORT_DIRNAME if args.backend == "dformerv2" else ESANET_EXPORT_DIRNAME
    out_root = args.out_root or (REPO / "data" / "nyuv2_supervised" / default_dir)
    if args.backend == "dformerv2":
        out_root = out_root / "NYUDepthv2" if out_root.name != "NYUDepthv2" else out_root

    summary = export_nyuv2_for_backend(
        source_root=source_root,
        splits_file=splits_file,
        out_root=out_root,
        backend=args.backend,
        force=args.force,
    )
    print(
        f"exported backend={summary.backend} root={summary.root} "
        f"train={summary.train_count} test={summary.test_count}"
    )


if __name__ == "__main__":
    main()
