"""Common runtime helpers for supervised benchmark backends."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_supervised_experiment(config_arg: str):
    cfg_path = Path(config_arg)
    if not cfg_path.exists():
        cfg_path = REPO_ROOT / "configs" / "experiment" / f"{config_arg}.yaml"
    cfg = OmegaConf.load(cfg_path)
    ds_cfg = OmegaConf.load(REPO_ROOT / "configs" / "dataset" / f"{cfg.dataset}.yaml")
    return cfg_path, cfg, ds_cfg


def ensure_backend_repo(repo_url: str, repo_dir: Path) -> Path:
    repo_dir = Path(repo_dir)
    if repo_dir.exists():
        return repo_dir
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
        check=True,
        cwd=REPO_ROOT,
    )
    return repo_dir


def ensure_python_dependencies(deps: list[dict[str, str]] | None) -> None:
    if not deps:
        return
    missing = [dep["pip"] for dep in deps if importlib.util.find_spec(dep["module"]) is None]
    if not missing:
        return
    subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True, cwd=REPO_ROOT)


def ensure_checkpoint(url: str, path: Path, member: str | None = None) -> Path:
    path = Path(path)
    resolved = _resolved_checkpoint_path(path, member)
    if resolved.exists():
        return resolved
    if path.exists():
        if member is not None:
            return _extract_tar_member(path, member)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if "drive.google.com" in url:
        _download_gdrive(url, path)
    else:
        _download_http(url, path)
    if member is not None:
        return _extract_tar_member(path, member)
    return path


def write_report(report_path: Path, payload: dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(payload, f, indent=2)


def _download_http(url: str, path: Path) -> None:
    with urllib.request.urlopen(url) as response, open(path, "wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _download_gdrive(url: str, path: Path) -> None:
    try:
        import gdown
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "gdown is required to download Google Drive checkpoints; install it or use a local checkpoint path"
        ) from exc
    gdown.download(url=url, output=str(path), quiet=False)


def _resolved_checkpoint_path(path: Path, member: str | None) -> Path:
    if member is None:
        return path
    return path.parent / member


def _extract_tar_member(archive_path: Path, member: str) -> Path:
    target = archive_path.parent / member
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extract(member, path=archive_path.parent)
    return target
