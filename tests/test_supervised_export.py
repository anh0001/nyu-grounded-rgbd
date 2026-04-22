from pathlib import Path

import numpy as np
from PIL import Image

from src.supervised.export import export_nyuv2_for_backend


def _build_fake_source(root: Path) -> Path:
    src = root / "data" / "nyuv2"
    (src / "rgb").mkdir(parents=True)
    (src / "depth").mkdir(parents=True)
    (src / "labels40").mkdir(parents=True)
    (src / "splits").mkdir(parents=True)

    for idx in (1, 2):
        stem = f"{idx:04d}"
        Image.fromarray(np.full((4, 5, 3), idx, dtype=np.uint8), mode="RGB").save(src / "rgb" / f"{stem}.png")
        Image.fromarray(np.full((4, 5), idx, dtype=np.uint8), mode="L").save(src / "labels40" / f"{stem}.png")
        np.save(src / "depth" / f"{stem}.npy", np.full((4, 5), idx / 10.0, dtype=np.float32))

    (src / "splits" / "gupta_795_654.json").write_text('{"train":[1],"test":[2]}')
    return src


def test_export_dformerv2_layout(tmp_path: Path) -> None:
    src = _build_fake_source(tmp_path)
    out_root = tmp_path / "export" / "NYUDepthv2"
    summary = export_nyuv2_for_backend(
        source_root=src,
        splits_file=src / "splits" / "gupta_795_654.json",
        out_root=out_root,
        backend="dformerv2",
    )

    assert summary.train_count == 1
    assert summary.test_count == 1
    assert (out_root / "RGB" / "0001.png").exists()
    assert (out_root / "Depth" / "0001.png").exists()
    assert (out_root / "Label" / "0002.png").exists()
    assert (out_root / "train.txt").read_text().strip() == "RGB/0001.png"
    assert (out_root / "test.txt").read_text().strip() == "RGB/0002.png"


def test_export_esanet_layout(tmp_path: Path) -> None:
    src = _build_fake_source(tmp_path)
    out_root = tmp_path / "export_esanet"
    summary = export_nyuv2_for_backend(
        source_root=src,
        splits_file=src / "splits" / "gupta_795_654.json",
        out_root=out_root,
        backend="esanet",
    )

    assert summary.train_count == 1
    assert summary.test_count == 1
    assert (out_root / "train" / "rgb" / "0001.png").exists()
    assert (out_root / "train" / "depth" / "0001.png").exists()
    assert (out_root / "test" / "labels_40" / "0002.png").exists()
    assert (out_root / "train.txt").read_text().strip() == "0001"
    assert (out_root / "test.txt").read_text().strip() == "0002"
    depth = np.array(Image.open(out_root / "train" / "depth" / "0001.png"))
    assert depth.dtype == np.uint16
