"""Render paper-ready qualitative segmentation figures from saved predictions."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.datasets.nyuv2 import NYUv2Dataset  # noqa: E402


@dataclass(frozen=True)
class MethodSpec:
    label: str
    pred_dir: Path


def parse_method(spec: str) -> MethodSpec:
    if "=" in spec:
        label, path_str = spec.split("=", 1)
    else:
        path_str = spec
        label = Path(path_str).name
    pred_dir = Path(path_str)
    if not pred_dir.is_absolute():
        pred_dir = REPO / pred_dir
    return MethodSpec(label=label.strip(), pred_dir=pred_dir)


def build_palette(num_classes: int) -> np.ndarray:
    tab20 = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:, :3]
    tab20b = plt.get_cmap("tab20b")(np.linspace(0, 1, 20))[:, :3]
    colors = np.concatenate([tab20, tab20b], axis=0)
    palette = np.zeros((num_classes + 1, 3), dtype=np.uint8)
    palette[1:] = np.round(colors[:num_classes] * 255.0).astype(np.uint8)
    return palette


def colorize_labels(label_map: np.ndarray, palette: np.ndarray) -> np.ndarray:
    return palette[label_map]


def colorize_depth(depth: np.ndarray, valid_depth: np.ndarray) -> np.ndarray:
    depth_vis = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not np.any(valid_depth):
        return depth_vis
    valid_vals = depth[valid_depth]
    lo, hi = np.percentile(valid_vals, [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    depth_norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    depth_rgb = plt.get_cmap("magma")(1.0 - depth_norm)[..., :3]
    depth_vis[valid_depth] = np.round(depth_rgb[valid_depth] * 255.0).astype(np.uint8)
    return depth_vis


def present_class_miou(pred: np.ndarray, gt: np.ndarray, ignore_index: int) -> float:
    valid = gt != ignore_index
    if not np.any(valid):
        return 0.0
    gt_classes = np.unique(gt[valid])
    ious: list[float] = []
    for class_id in gt_classes.tolist():
        pred_mask = pred == class_id
        gt_mask = gt == class_id
        union = np.logical_and(valid, np.logical_or(pred_mask, gt_mask)).sum()
        if union == 0:
            continue
        inter = np.logical_and(valid, np.logical_and(pred_mask, gt_mask)).sum()
        ious.append(float(inter) / float(union))
    return float(np.mean(ious)) if ious else 0.0


def load_pred(pred_dir: Path, idx: int) -> np.ndarray:
    return np.array(Image.open(pred_dir / f"{idx:04d}.png"), dtype=np.uint8)


def resolve_out(path: str | None, default_stem: str) -> Path:
    if path is None:
        return REPO / "outputs" / "figures" / f"{default_stem}.png"
    out = Path(path)
    if not out.is_absolute():
        out = REPO / out
    return out


def select_examples(
    ds: NYUv2Dataset,
    methods: list[MethodSpec],
    ignore_index: int,
    main_label: str,
    baseline_label: str | None,
    num_samples: int,
) -> tuple[list[int], dict[int, dict[str, float]]]:
    method_map = {method.label: method for method in methods}
    ranked: list[tuple[float, int]] = []
    metrics_by_idx: dict[int, dict[str, float]] = {}
    for sample in ds:
        idx = sample.idx
        row_metrics: dict[str, float] = {}
        missing = False
        for method in methods:
            pred_path = method.pred_dir / f"{idx:04d}.png"
            if not pred_path.exists():
                missing = True
                break
            pred = load_pred(method_map[method.label].pred_dir, idx)
            row_metrics[method.label] = present_class_miou(pred, sample.label, ignore_index)
        if missing:
            continue
        main_score = row_metrics[main_label]
        baseline_score = row_metrics[baseline_label] if baseline_label else 0.0
        delta = main_score - baseline_score
        row_metrics["_main"] = main_score
        row_metrics["_delta"] = delta
        row_metrics["_nclasses"] = float(len(np.unique(sample.label[sample.label != ignore_index])))
        complexity_bonus = min(row_metrics["_nclasses"] / 10.0, 1.0)
        rank_score = main_score + 0.75 * delta + 0.05 * complexity_bonus
        ranked.append((rank_score, idx))
        metrics_by_idx[idx] = row_metrics

    ranked.sort(reverse=True)
    if not ranked:
        return [], metrics_by_idx

    pool = ranked[: max(num_samples * 8, num_samples)]
    chosen: list[int] = []
    used: set[int] = set()
    for pos in np.linspace(0, len(pool) - 1, num=num_samples):
        _, idx = pool[int(round(pos))]
        if idx not in used:
            chosen.append(idx)
            used.add(idx)
    if len(chosen) < num_samples:
        for _, idx in pool:
            if idx in used:
                continue
            chosen.append(idx)
            used.add(idx)
            if len(chosen) == num_samples:
                break
    return chosen, metrics_by_idx


def save_row_strip(
    out_dir: Path,
    idx: int,
    panels: list[np.ndarray],
    labels: list[str],
    palette: np.ndarray,
) -> None:
    del palette
    fig, axes = plt.subplots(1, len(panels), figsize=(2.5 * len(panels), 2.1), squeeze=False)
    for col, (panel, label) in enumerate(zip(panels, labels)):
        ax = axes[0, col]
        ax.imshow(panel)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(out_dir / f"{idx:04d}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help="Prediction directories as label=path. Example: baseline=outputs/predictions/week1_baseline",
    )
    ap.add_argument("--dataset", type=str, default="nyuv2_40")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--main", type=str, default=None, help="Method label to optimize when auto-selecting")
    ap.add_argument("--baseline", type=str, default=None, help="Method label used for improvement scoring")
    ap.add_argument("--ids", type=int, nargs="*", default=None, help="Explicit NYUv2 frame ids to render")
    ap.add_argument("--num-samples", type=int, default=6)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--save-rows", action="store_true")
    ap.add_argument(
        "--annotate",
        choices=["none", "id", "scores"],
        default="scores",
        help="Overlay per-row text on the RGB column",
    )
    args = ap.parse_args()

    methods = [parse_method(spec) for spec in args.methods]
    method_labels = [method.label for method in methods]
    main_label = args.main or method_labels[-1]
    if main_label not in method_labels:
        raise ValueError(f"--main '{main_label}' not in methods {method_labels}")
    if args.baseline and args.baseline not in method_labels:
        raise ValueError(f"--baseline '{args.baseline}' not in methods {method_labels}")

    ds_cfg = OmegaConf.load(REPO / "configs" / "dataset" / f"{args.dataset}.yaml")
    ds = NYUv2Dataset(
        root=REPO / ds_cfg.root,
        split=args.split,
        protocol=ds_cfg.protocol,
        splits_file=ds_cfg.splits_file,
    )
    sample_by_idx = {sample.idx: sample for sample in ds}
    palette = build_palette(int(ds_cfg.num_classes))

    if args.ids:
        chosen_ids = args.ids
        metrics_by_idx: dict[int, dict[str, float]] = {}
        for idx in chosen_ids:
            sample = sample_by_idx[idx]
            row_metrics = {}
            for method in methods:
                pred = load_pred(method.pred_dir, idx)
                row_metrics[method.label] = present_class_miou(pred, sample.label, int(ds_cfg.ignore_index))
            metrics_by_idx[idx] = row_metrics
    else:
        chosen_ids, metrics_by_idx = select_examples(
            ds=ds,
            methods=methods,
            ignore_index=int(ds_cfg.ignore_index),
            main_label=main_label,
            baseline_label=args.baseline,
            num_samples=args.num_samples,
        )

    if not chosen_ids:
        raise RuntimeError("No matching predictions found for the requested methods.")

    out_path = resolve_out(args.out, default_stem=f"qualitative_{main_label}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row_dir = out_path.parent / f"{out_path.stem}_rows"
    if args.save_rows:
        row_dir.mkdir(parents=True, exist_ok=True)

    column_labels = ["RGB", "Depth", "GT", *method_labels]
    num_cols = len(column_labels)
    fig_w = max(2.35 * num_cols, 8.5)
    fig_h = max(1.95 * len(chosen_ids), 4.0)
    fig, axes = plt.subplots(len(chosen_ids), num_cols, figsize=(fig_w, fig_h), squeeze=False)

    metadata: dict[str, object] = {
        "dataset": args.dataset,
        "split": args.split,
        "methods": [{ "label": m.label, "pred_dir": str(m.pred_dir.relative_to(REPO)) if m.pred_dir.is_relative_to(REPO) else str(m.pred_dir)} for m in methods],
        "main": main_label,
        "baseline": args.baseline,
        "selected_ids": chosen_ids,
        "per_image_metrics": {},
    }

    for col, title in enumerate(column_labels):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    for row, idx in enumerate(chosen_ids):
        sample = sample_by_idx[idx]
        rgb = sample.rgb
        depth_vis = colorize_depth(sample.depth, sample.valid_depth)
        gt_vis = colorize_labels(sample.label, palette)
        panels = [rgb, depth_vis, gt_vis]
        row_metrics = metrics_by_idx[idx]
        metric_parts = [f"{label} {row_metrics[label]:.2f}" for label in method_labels if label in row_metrics]
        row_text = f"#{idx:04d}"
        if metric_parts:
            row_text += " | " + " | ".join(metric_parts)
        metadata["per_image_metrics"][f"{idx:04d}"] = row_metrics

        if args.annotate != "none":
            text = f"#{idx:04d}" if args.annotate == "id" else row_text
            axes[row, 0].text(
                0.02,
                0.98,
                text,
                transform=axes[row, 0].transAxes,
                va="top",
                ha="left",
                fontsize=9,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.72, "pad": 2.0, "edgecolor": "none"},
            )

        for method in methods:
            pred = load_pred(method.pred_dir, idx)
            panels.append(colorize_labels(pred, palette))

        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel)
            ax.axis("off")

        if args.save_rows:
            save_row_strip(row_dir, idx, panels, column_labels, palette)

    fig.tight_layout(pad=0.15, w_pad=0.1, h_pad=0.25)
    fig.savefig(out_path, dpi=260, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=260, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"saved figure: {out_path}")
    print(f"saved figure: {out_path.with_suffix('.pdf')}")
    print(f"saved metadata: {meta_path}")
    if args.save_rows:
        print(f"saved row strips: {row_dir}")


if __name__ == "__main__":
    main()
