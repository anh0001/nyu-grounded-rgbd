"""Run zero-shot inference on a split and save semantic maps + reports."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.datasets.nyuv2 import NYUv2Dataset  # noqa: E402
from src.eval.metrics import ConfusionAccumulator  # noqa: E402
from src.models.gdino import GroundingDINO  # noqa: E402
from src.models.sam_wrapper import SAMWrapper  # noqa: E402
from src.pipeline.depth_features import compute_features  # noqa: E402
from src.pipeline.depth_logit_fusion import (  # noqa: E402
    apply_depth_logit_fusion,
    logits_to_labels,
)
from src.pipeline.dense_ov_fusion import run_dense_ov_fusion  # noqa: E402
from src.pipeline.proposals import build_proposals  # noqa: E402
from src.pipeline.region_fill import fill_residual_slic  # noqa: E402
from src.pipeline.semantic_fusion import rasterize  # noqa: E402
from src.prompts.alias_bank import build_chunks  # noqa: E402


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("run_infer")


def load_cfg(path: Path):
    cfg = OmegaConf.load(path)
    # merge referenced sub-configs
    ds_cfg = OmegaConf.load(REPO / "configs" / "dataset" / f"{cfg.dataset}.yaml")
    mdl_cfg = OmegaConf.load(REPO / "configs" / "model" / f"{cfg.model}.yaml")
    pl_cfg = OmegaConf.load(REPO / "configs" / "pipeline" / f"{cfg.pipeline}.yaml")
    return cfg, ds_cfg, mdl_cfg, pl_cfg


def resize_frame(rgb, depth, valid, scale: float):
    if abs(scale - 1.0) < 1e-6:
        return rgb, depth, valid
    import cv2
    import numpy as np

    height, width = rgb.shape[:2]
    new_width = max(int(round(width * scale)), 32)
    new_height = max(int(round(height * scale)), 32)
    rgb_s = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    depth_s = cv2.resize(depth, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    valid_u8 = valid.astype(np.uint8) * 255
    valid_s = cv2.resize(valid_u8, (new_width, new_height), interpolation=cv2.INTER_NEAREST) > 0
    return rgb_s, depth_s.astype(np.float32), valid_s.astype(np.bool_)


def main() -> None:
    logger = setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True,
                    help="experiment config name (no .yaml) or path")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = REPO / "configs" / "experiment" / f"{args.config}.yaml"
    cfg, ds_cfg, mdl_cfg, pl_cfg = load_cfg(cfg_path)

    exp_name = cfg.name
    out_dir = args.out or (REPO / "outputs" / "predictions" / exp_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = REPO / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    ds = NYUv2Dataset(root=REPO / ds_cfg.root, split=cfg.split, protocol=ds_cfg.protocol,
                      splits_file=ds_cfg.splits_file)
    logger.info("loaded config=%s experiment=%s split=%s device=%s", cfg_path, exp_name, cfg.split, args.device)
    logger.info("dataset=%s protocol=%s images=%d output_dir=%s", ds_cfg.name, ds_cfg.protocol, len(ds), out_dir)

    pipeline_type = str(OmegaConf.select(pl_cfg, "type", default="proposal_heuristic"))
    dense_models = {}
    dense_cfg = OmegaConf.select(pl_cfg, "dense_ov", default=None)
    if pipeline_type == "dense_ov_rgbd":
        from src.datasets.nyuv2_meta import load_nyu40_bank, load_nyu40_descriptions
        from src.models.dense_clip import DenseCLIPEncoder
        from src.models.dense_dino import DenseDINOEncoder
        from src.pipeline.sam_regions import SAMRegionGenerator

        bank_file = OmegaConf.select(pl_cfg, "prompts.bank_file", default="nyu40_aliases_v2.json")
        desc_file = str(OmegaConf.select(dense_cfg, "prompt_desc_file", default="nyu40_descriptions.json"))
        bank = load_nyu40_bank(bank_file)
        descriptions = load_nyu40_descriptions(desc_file)
        class_entries = [
            {
                "class_id": cls["id"],
                "name": cls["name"],
                "aliases": cls.get("aliases", []),
                "description": descriptions.get(int(cls["id"]), ""),
            }
            for cls in bank["classes"]
        ]

        clip_encoder = DenseCLIPEncoder(
            model_id=str(mdl_cfg.dense_clip.model_id),
            input_size=int(mdl_cfg.dense_clip.input_size),
            device=args.device,
        )
        dino_encoder = DenseDINOEncoder(
            model_id=str(mdl_cfg.dense_dino.model_id),
            input_size=int(mdl_cfg.dense_dino.input_size),
            output_dims=int(mdl_cfg.dense_dino.output_dims),
            device=args.device,
        )
        sam_generator = SAMRegionGenerator(
            model_id=str(mdl_cfg.sam.model_id),
            device=args.device,
            points_per_batch=int(mdl_cfg.sam.points_per_batch),
            pred_iou_thresh=float(mdl_cfg.sam.pred_iou_thresh),
            stability_score_thresh=float(mdl_cfg.sam.stability_score_thresh),
            min_mask_area=int(mdl_cfg.sam.min_mask_area),
            max_regions=int(mdl_cfg.sam.max_regions),
        )
        text_bank = clip_encoder.build_text_bank(class_entries)
        dense_models = {
            "clip_encoder": clip_encoder,
            "dino_encoder": dino_encoder,
            "sam_generator": sam_generator,
            "text_bank": text_bank,
        }
        logger.info(
            "models ready dense_clip=%s dense_dino=%s sam=%s pipeline=%s",
            mdl_cfg.dense_clip.model_id,
            mdl_cfg.dense_dino.model_id,
            mdl_cfg.sam.model_id,
            cfg.pipeline,
        )
    else:
        gdino = GroundingDINO(
            model_id=mdl_cfg.gdino.model_id,
            device=args.device,
            box_threshold=float(mdl_cfg.gdino.box_threshold),
            text_threshold=float(mdl_cfg.gdino.text_threshold),
        )
        sam = SAMWrapper(
            backend=mdl_cfg.sam.backend,
            checkpoint=mdl_cfg.sam.checkpoint,
            model_type=mdl_cfg.sam.model_type,
            device=args.device,
        )
        logger.info(
            "models ready gdino=%s sam=%s/%s pipeline=%s",
            mdl_cfg.gdino.model_id,
            mdl_cfg.sam.backend,
            mdl_cfg.sam.model_type,
            cfg.pipeline,
        )

        bank_file = OmegaConf.select(pl_cfg, "prompts.bank_file", default=None)
        chunks = build_chunks(
            protocol="nyu40" if ds_cfg.num_classes == 40 else "nyu13",
            bank_file=bank_file,
        )

        clip_reranker = None
        clip_cfg = OmegaConf.select(pl_cfg, "clip_rerank", default=None)
        if clip_cfg is not None and bool(OmegaConf.select(clip_cfg, "enabled", default=False)):
            from src.datasets.nyuv2_meta import NYU40_NAMES
            from src.models.clip_reranker import SigLIPReranker

            exclude_ids = OmegaConf.select(clip_cfg, "exclude_class_ids", default=None)
            exclude_set = set(int(i) for i in exclude_ids) if exclude_ids else set()
            class_ids = [i for i in range(1, 41) if i not in exclude_set]
            class_names = [NYU40_NAMES[i] for i in class_ids]
            clip_reranker = SigLIPReranker(
                model_id=str(OmegaConf.select(clip_cfg, "model_id",
                                              default="google/siglip-base-patch16-224")),
                device=args.device,
                class_names=class_names,
                class_ids=class_ids,
            )
            logger.info("clip reranker ready model=%s", clip_cfg.get("model_id", "siglip-base"))
        w_clip = float(OmegaConf.select(pl_cfg, "clip_rerank.w_clip", default=0.0))
        clip_reassign_margin = float(
            OmegaConf.select(pl_cfg, "clip_rerank.reassign_margin", default=0.0)
        )
        clip_reassign_min_top = float(
            OmegaConf.select(pl_cfg, "clip_rerank.reassign_min_top", default=0.0)
        )
        per_chunk_thresholds_cfg = OmegaConf.select(pl_cfg, "prompts.per_chunk_thresholds", default=None)
        per_chunk_thresholds: dict[str, tuple[float, float]] | None = None
        if per_chunk_thresholds_cfg is not None:
            per_chunk_thresholds = {
                str(k): (float(v[0]), float(v[1])) for k, v in dict(per_chunk_thresholds_cfg).items()
            }
        clip_bg_fill = str(OmegaConf.select(pl_cfg, "clip_rerank.background_fill", default="mean"))
        clip_bg_alpha = float(OmegaConf.select(pl_cfg, "clip_rerank.bg_alpha", default=0.5))
        clip_small_ctx = float(OmegaConf.select(pl_cfg, "clip_rerank.small_mask_context", default=0.40))
        clip_small_area = float(OmegaConf.select(pl_cfg, "clip_rerank.small_mask_area_frac", default=0.02))

        dense_fill_encoder = None
        dense_fill_bank = None
        dense_fill_min_logit = 0.0
        dense_fill_cfg = OmegaConf.select(pl_cfg, "dense_fill", default=None)
        if dense_fill_cfg is not None and bool(OmegaConf.select(dense_fill_cfg, "enabled", default=False)):
            from src.datasets.nyuv2_meta import load_nyu40_bank, load_nyu40_descriptions
            from src.models.dense_clip import DenseCLIPEncoder

            bank_file_df = OmegaConf.select(pl_cfg, "prompts.bank_file", default="nyu40_aliases_v2.json")
            desc_file_df = str(OmegaConf.select(dense_fill_cfg, "prompt_desc_file", default="nyu40_descriptions.json"))
            df_bank = load_nyu40_bank(bank_file_df)
            df_descs = load_nyu40_descriptions(desc_file_df)
            df_entries = [
                {
                    "class_id": cls["id"],
                    "name": cls["name"],
                    "aliases": cls.get("aliases", []),
                    "description": df_descs.get(int(cls["id"]), ""),
                }
                for cls in df_bank["classes"]
            ]
            dense_fill_encoder = DenseCLIPEncoder(
                model_id=str(OmegaConf.select(dense_fill_cfg, "model_id", default="openai/clip-vit-large-patch14")),
                input_size=int(OmegaConf.select(dense_fill_cfg, "input_size", default=336)),
                device=args.device,
            )
            dense_fill_bank = dense_fill_encoder.build_text_bank(df_entries)
            dense_fill_min_logit = float(OmegaConf.select(dense_fill_cfg, "min_logit", default=0.0))
            logger.info("dense_fill encoder ready model=%s", dense_fill_cfg.get("model_id", "clip-L"))

        dense_seed_encoder = None
        dense_seed_bank = None
        dense_seed_cfg_obj = None
        dense_seed_cfg = OmegaConf.select(pl_cfg, "dense_seed", default=None)
        if dense_seed_cfg is not None and bool(OmegaConf.select(dense_seed_cfg, "enabled", default=False)):
            from src.datasets.nyuv2_meta import load_nyu40_bank, load_nyu40_descriptions
            from src.models.dense_clip import DenseCLIPEncoder
            from src.pipeline.dense_seed import DenseSeedConfig

            if dense_fill_encoder is not None and dense_fill_bank is not None:
                dense_seed_encoder = dense_fill_encoder
                dense_seed_bank = dense_fill_bank
            else:
                bank_file_ds = OmegaConf.select(pl_cfg, "prompts.bank_file", default="nyu40_aliases_v2.json")
                desc_file_ds = str(OmegaConf.select(dense_seed_cfg, "prompt_desc_file", default="nyu40_descriptions.json"))
                ds_bank = load_nyu40_bank(bank_file_ds)
                ds_descs = load_nyu40_descriptions(desc_file_ds)
                ds_entries = [
                    {
                        "class_id": cls["id"],
                        "name": cls["name"],
                        "aliases": cls.get("aliases", []),
                        "description": ds_descs.get(int(cls["id"]), ""),
                    }
                    for cls in ds_bank["classes"]
                ]
                dense_seed_encoder = DenseCLIPEncoder(
                    model_id=str(OmegaConf.select(dense_seed_cfg, "model_id", default="openai/clip-vit-large-patch14")),
                    input_size=int(OmegaConf.select(dense_seed_cfg, "input_size", default=336)),
                    device=args.device,
                )
                dense_seed_bank = dense_seed_encoder.build_text_bank(ds_entries)
            dense_seed_cfg_obj = DenseSeedConfig(
                min_logit=float(OmegaConf.select(dense_seed_cfg, "min_logit", default=10.0)),
                min_component_area=int(OmegaConf.select(dense_seed_cfg, "min_component_area", default=400)),
                max_seeds_per_class=int(OmegaConf.select(dense_seed_cfg, "max_seeds_per_class", default=2)),
                sam_score_floor=float(OmegaConf.select(dense_seed_cfg, "sam_score_floor", default=0.6)),
            )
            logger.info("dense_seed enabled min_logit=%.2f max_per_class=%d",
                        dense_seed_cfg_obj.min_logit, dense_seed_cfg_obj.max_seeds_per_class)

    acc = ConfusionAccumulator(
        num_classes=ds_cfg.num_classes,
        ignore_index=ds_cfg.ignore_index,
        class_names=ds.class_names(),
    )

    limit = args.limit if args.limit is not None else cfg.limit
    ids = list(range(len(ds)))
    if limit:
        ids = ids[:int(limit)]
    logger.info("starting inference on %d image(s)", len(ids))

    if pipeline_type == "dense_ov_rgbd":
        scales = [float(s) for s in OmegaConf.select(dense_cfg, "tta.scales", default=[1.0])]
        use_tta = bool(OmegaConf.select(dense_cfg, "tta.enabled", default=False))
        use_hflip = bool(OmegaConf.select(dense_cfg, "tta.hflip", default=False))

        def _infer_frame(rgb, depth, valid):
            import cv2
            import numpy as np

            votes: list[np.ndarray] = []
            eval_scales = scales if use_tta else [1.0]
            flip_options = [False, True] if (use_tta and use_hflip) else [False]
            for scale in eval_scales:
                rgb_s, depth_s, valid_s = resize_frame(rgb, depth, valid, scale)
                for do_flip in flip_options:
                    if do_flip:
                        rgb_i = np.ascontiguousarray(rgb_s[:, ::-1])
                        depth_i = np.ascontiguousarray(depth_s[:, ::-1])
                        valid_i = np.ascontiguousarray(valid_s[:, ::-1])
                    else:
                        rgb_i, depth_i, valid_i = rgb_s, depth_s, valid_s

                    feat_local = compute_features(depth_i, valid=valid_i)
                    logits = run_dense_ov_fusion(
                        rgb=rgb_i,
                        clip_encoder=dense_models["clip_encoder"],
                        dino_encoder=dense_models["dino_encoder"],
                        sam_generator=dense_models["sam_generator"],
                        text_bank=dense_models["text_bank"],
                        tiles_y=int(OmegaConf.select(dense_cfg, "tiles_y", default=2)),
                        tiles_x=int(OmegaConf.select(dense_cfg, "tiles_x", default=2)),
                        overlap=float(OmegaConf.select(dense_cfg, "overlap", default=0.25)),
                        clip_logit_weight=float(
                            OmegaConf.select(dense_cfg, "clip_logit_weight", default=0.65)
                        ),
                        region_logit_weight=float(
                            OmegaConf.select(dense_cfg, "region_logit_weight", default=0.35)
                        ),
                        min_region_area=int(
                            OmegaConf.select(dense_cfg, "min_region_area", default=256)
                        ),
                    )
                    logits = apply_depth_logit_fusion(
                        logits=logits,
                        feat=feat_local,
                        wall_boost=float(OmegaConf.select(pl_cfg, "depth_fusion.wall_boost", default=0.35)),
                        floor_boost=float(OmegaConf.select(pl_cfg, "depth_fusion.floor_boost", default=0.9)),
                        ceiling_boost=float(OmegaConf.select(pl_cfg, "depth_fusion.ceiling_boost", default=0.55)),
                        contradict_penalty=float(
                            OmegaConf.select(pl_cfg, "depth_fusion.contradict_penalty", default=0.2)
                        ),
                        floor_thr=float(OmegaConf.select(pl_cfg, "depth_fusion.floor_thr", default=0.8)),
                        ceiling_thr=float(
                            OmegaConf.select(pl_cfg, "depth_fusion.ceiling_thr", default=-0.8)
                        ),
                        wall_thr=float(OmegaConf.select(pl_cfg, "depth_fusion.wall_thr", default=0.3)),
                    )
                    if do_flip:
                        logits = logits[:, :, ::-1]
                    if logits.shape[1:] != rgb.shape[:2]:
                        logits = np.stack(
                            [
                                cv2.resize(
                                    logits[cid],
                                    (rgb.shape[1], rgb.shape[0]),
                                    interpolation=cv2.INTER_LINEAR,
                                )
                                for cid in range(logits.shape[0])
                            ],
                            axis=0,
                        ).astype(np.float32)
                    votes.append(logits)
            mean_logits = np.mean(np.stack(votes, axis=0), axis=0)
            return logits_to_labels(mean_logits)
    else:
        flip_tta = bool(OmegaConf.select(pl_cfg, "fusion.flip_tta", default=False))
        cc_min_area = int(OmegaConf.select(pl_cfg, "fusion.cc_min_area", default=0))
        use_ransac_planes = bool(OmegaConf.select(pl_cfg, "fusion.ransac_planes", default=False))

        def _infer_frame(rgb, depth, valid):
            feat_local = compute_features(depth, valid=valid)
            cands_local = build_proposals(
                rgb=rgb, depth_feat=feat_local, gdino=gdino, sam=sam,
                chunks=chunks, image_id=None,
                box_threshold=float(mdl_cfg.gdino.box_threshold),
                text_threshold=float(mdl_cfg.gdino.text_threshold),
                box_iou_same=float(pl_cfg.box_nms.iou_same),
                box_iou_cross=float(pl_cfg.box_nms.iou_cross),
                mask_iou_same=float(pl_cfg.mask_nms.iou_same),
                mask_iou_cross=float(pl_cfg.mask_nms.iou_cross),
                per_chunk_thresholds=per_chunk_thresholds,
                clip_reranker=clip_reranker,
                w_clip=w_clip,
                clip_reassign_margin=clip_reassign_margin,
                clip_reassign_min_top=clip_reassign_min_top,
                clip_background_fill=clip_bg_fill,
                clip_bg_alpha=clip_bg_alpha,
                clip_small_mask_context=clip_small_ctx,
                clip_small_mask_area_frac=clip_small_area,
            )
            dense_logits_local = None
            dense_cids_local = None
            if dense_fill_encoder is not None and dense_fill_bank is not None:
                dense_logits_local = dense_fill_encoder.encode_logits(rgb, dense_fill_bank)
                dense_cids_local = list(dense_fill_bank.class_ids)

            if dense_seed_encoder is not None and dense_seed_bank is not None:
                import numpy as _np

                from src.pipeline.dense_seed import seed_candidates_from_dense

                if dense_seed_bank is dense_fill_bank and dense_logits_local is not None:
                    seed_logits = dense_logits_local
                    seed_cids = dense_cids_local
                else:
                    seed_logits = dense_seed_encoder.encode_logits(rgb, dense_seed_bank)
                    seed_cids = list(dense_seed_bank.class_ids)
                existing_cids = {int(c.class_id) for c in cands_local if c.score > 0.35}
                H, W = rgb.shape[:2]
                excl = _np.zeros((H, W), dtype=bool)
                for c in cands_local:
                    if c.score > 0.4 and c.mask.shape == (H, W):
                        excl |= c.mask
                extra = seed_candidates_from_dense(
                    rgb=rgb,
                    dense_logits=seed_logits,
                    dense_class_ids=seed_cids,
                    sam=sam,
                    cfg=dense_seed_cfg_obj,
                    existing_class_ids=existing_cids,
                    exclusion_mask=excl,
                )
                if extra:
                    cands_local = cands_local + extra
            sem_local = rasterize(
                cands_local, feat_local, num_classes=40,
                fill_structural_by_geometry=bool(pl_cfg.fusion.fill_structural_by_geometry),
                cc_min_area=cc_min_area,
                use_ransac_planes=use_ransac_planes,
                dense_logits=dense_logits_local,
                dense_class_ids=dense_cids_local,
                dense_min_logit=dense_fill_min_logit,
            )
            if bool(pl_cfg.fusion.residual_slic):
                sem_local = fill_residual_slic(sem_local, rgb, feat_local, cands_local)
            return sem_local

    t0 = time.time()
    for i in tqdm(ids, desc="infer", unit="img", dynamic_ncols=True):
        sample = ds[i]
        sem40 = _infer_frame(sample.rgb, sample.depth, sample.valid_depth)
        if pipeline_type != "dense_ov_rgbd" and flip_tta:
            import numpy as _np

            rgb_f = _np.ascontiguousarray(sample.rgb[:, ::-1])
            depth_f = _np.ascontiguousarray(sample.depth[:, ::-1])
            valid_f = (
                _np.ascontiguousarray(sample.valid_depth[:, ::-1])
                if sample.valid_depth is not None else None
            )
            sem_f = _infer_frame(rgb_f, depth_f, valid_f)[:, ::-1]
            # Majority vote between orig and flipped; tie → keep orig.
            disagree = sem40 != sem_f
            # For disagreement pixels, prefer the non-background/non-fallback class.
            from src.pipeline.semantic_fusion import (  # noqa: WPS433
                FALLBACK_PROP,
                FALLBACK_STUFF,
            )
            fb = {FALLBACK_PROP, FALLBACK_STUFF, 39}
            prefer_f = disagree & _np.isin(sem40, list(fb)) & ~_np.isin(sem_f, list(fb))
            sem40 = sem40.copy()
            sem40[prefer_f] = sem_f[prefer_f]
        if ds_cfg.num_classes == 13:
            from src.pipeline.semantic_fusion import _map_40_to_13
            pred = _map_40_to_13(sem40)
        else:
            pred = sem40
        Image.fromarray(pred).save(out_dir / f"{sample.idx:04d}.png")
        acc.update(pred, sample.label)

    metrics = acc.result()
    report = {
        "experiment": exp_name,
        "dataset": ds_cfg.name,
        "num_images": len(ids),
        "mean_iou": metrics.mean_iou,
        "pixel_acc": metrics.pixel_acc,
        "mean_class_acc": metrics.mean_class_acc,
        "per_class_iou": {
            (metrics.class_names[i + 1] if i + 1 < len(metrics.class_names) else f"c{i+1}"):
                float(metrics.per_class_iou[i])
            for i in range(metrics.num_classes)
        },
        "wall_time_s": time.time() - t0,
    }
    with open(report_dir / f"{exp_name}.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info(
        "finished inference images=%d wall_time_s=%.2f report=%s",
        len(ids),
        report["wall_time_s"],
        report_dir / f"{exp_name}.json",
    )
    print(metrics.table())
    print(f"\nreport -> {report_dir / f'{exp_name}.json'}")


if __name__ == "__main__":
    main()
