from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_nyuv2.py"
SPEC = spec_from_file_location("prepare_nyuv2", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
prepare_nyuv2 = module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_nyuv2)


def test_resolve_meta_mat_accepts_upstream_alias(tmp_path: Path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    alias = meta / "class13Mapping.mat"
    alias.write_bytes(b"not-empty")

    resolved = prepare_nyuv2.resolve_meta_mat(
        meta / "classMapping13.mat",
        "NYU13 class mapping",
        alt_names=("class13Mapping.mat",),
    )

    assert resolved == alias


def test_resolve_meta_mat_rejects_empty_file(tmp_path: Path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    empty = meta / "classMapping13.mat"
    empty.write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="empty/truncated NYU13 class mapping"):
        prepare_nyuv2.resolve_meta_mat(empty, "NYU13 class mapping")


def test_build_lut13_accepts_raw_894_mapping() -> None:
    lut, space = prepare_nyuv2.build_lut13(list(range(1, 895)))

    assert space == "raw"
    assert lut.shape == (895,)
    assert lut[0] == 0
    assert lut[894] == 894


def test_build_lut13_accepts_nyu40_mapping() -> None:
    lut, space = prepare_nyuv2.build_lut13(list(range(1, 41)))

    assert space == "nyu40"
    assert lut.shape == (41,)
    assert lut[0] == 0
    assert lut[40] == 40
