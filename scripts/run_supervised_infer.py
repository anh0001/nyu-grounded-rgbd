"""Run a supervised RGB-D backend and score it with the repo evaluator."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.datasets.nyuv2 import NYUv2Dataset  # noqa: E402
from src.eval.metrics import ConfusionAccumulator  # noqa: E402
from src.supervised.export import export_nyuv2_for_backend  # noqa: E402
from src.supervised.runtime import (  # noqa: E402
    ensure_backend_repo,
    ensure_checkpoint,
    ensure_python_dependencies,
    load_supervised_experiment,
    write_report,
)


HELPER_BY_BACKEND = {
    "dformerv2": REPO / "scripts" / "_run_dformerv2_backend.py",
    "esanet": REPO / "scripts" / "_run_esanet_backend.py",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--force-export", action="store_true")
    args = ap.parse_args()

    cfg_path, cfg, ds_cfg = load_supervised_experiment(args.config)
    sup_cfg = OmegaConf.select(cfg, "supervised")
    if sup_cfg is None:
        raise ValueError(f"{cfg_path} is not a supervised experiment config")

    exp_name = str(cfg.name)
    out_dir = args.out or (REPO / "outputs" / "predictions" / exp_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_root = REPO / ds_cfg.root
    splits_file = ds_root / ds_cfg.splits_file
    export_root = REPO / Path(str(sup_cfg.export_root))
    export_summary = export_nyuv2_for_backend(
        source_root=ds_root,
        splits_file=splits_file,
        out_root=export_root,
        backend=str(sup_cfg.backend),
        force=args.force_export,
    )
    if export_summary.train_count != 795 or export_summary.test_count != 654:
        raise RuntimeError(
            f"unexpected exported split sizes train={export_summary.train_count} test={export_summary.test_count}"
        )

    ensure_python_dependencies(OmegaConf.to_container(sup_cfg.get("deps", []), resolve=True))
    repo_dir = ensure_backend_repo(str(sup_cfg.repo_url), REPO / Path(str(sup_cfg.repo_dir)))
    checkpoint_path = ensure_checkpoint(
        str(sup_cfg.checkpoint_url),
        REPO / Path(str(sup_cfg.checkpoint_path)),
        member=OmegaConf.select(sup_cfg, "checkpoint_member", default=None),
    )

    helper = HELPER_BY_BACKEND[str(sup_cfg.backend)]
    helper_args = [
        sys.executable,
        str(helper),
        "--repo-dir",
        str(repo_dir),
        "--export-root",
        str(export_root),
        "--checkpoint",
        str(checkpoint_path),
        "--out-dir",
        str(out_dir),
        "--device",
        args.device,
    ]
    if args.limit is not None:
        helper_args.extend(["--limit", str(args.limit)])
    if str(sup_cfg.backend) == "dformerv2":
        helper_args.extend(["--config-module", str(sup_cfg.config_module)])
    elif str(sup_cfg.backend) == "esanet":
        helper_args.extend(["--height", str(int(sup_cfg.height)), "--width", str(int(sup_cfg.width))])

    start = time.time()
    subprocess.run(helper_args, check=True, cwd=REPO)
    wall_time_s = time.time() - start

    ds = NYUv2Dataset(
        root=ds_root,
        split=str(cfg.split),
        protocol=ds_cfg.protocol,
        splits_file=ds_cfg.splits_file,
    )
    if args.limit is not None:
        ds.ids = ds.ids[: args.limit]

    acc = ConfusionAccumulator(
        num_classes=ds_cfg.num_classes,
        ignore_index=ds_cfg.ignore_index,
        class_names=ds.class_names(),
    )
    for i in tqdm(range(len(ds)), desc="eval", unit="img"):
        sample = ds[i]
        pred_path = out_dir / f"{sample.idx:04d}.png"
        if not pred_path.exists():
            raise FileNotFoundError(f"missing prediction: {pred_path}")

        pred = np.array(Image.open(pred_path))
        if pred.ndim != 2:
            raise ValueError(f"prediction must be single-channel: {pred_path}")
        if pred.min() < 0 or pred.max() > 40:
            raise ValueError(f"prediction values out of range 0..40: {pred_path}")
        acc.update(pred, sample.label)

    metrics = acc.result()
    report_path = REPO / "outputs" / "reports" / f"{exp_name}.json"
    write_report(
        report_path,
        {
            "experiment": exp_name,
            "dataset": cfg.dataset,
            "num_images": len(ds),
            "mean_iou": metrics.mean_iou,
            "pixel_acc": metrics.pixel_acc,
            "mean_class_acc": metrics.mean_class_acc,
            "per_class_iou": {
                ds.class_names()[i + 1]: float(v)
                for i, v in enumerate(metrics.per_class_iou.tolist())
            },
            "wall_time_s": wall_time_s,
            "backend": str(sup_cfg.backend),
        },
    )
    print(metrics.table())
    print(f"saved predictions={out_dir} report={report_path}")


if __name__ == "__main__":
    main()
