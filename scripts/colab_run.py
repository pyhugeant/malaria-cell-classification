#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict

import yaml


def run_cmd(cmd: list[str], cwd: str | None = None) -> str:
    out = subprocess.check_output(cmd, cwd=cwd).decode("utf-8", errors="replace").strip()
    return out


def sanitize(s: str) -> str:
    s = s.strip()
    s = s.replace(" ", "_").replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

# define shutdown runtime
# def maybe_shutdown_colab_runtime() -> None:
#     try:
#         from google.colab import runtime  # type: ignore
#         print("Training finished. Unassigning Colab runtime...")
#         runtime.unassign()
#     except Exception as e:
#         print(f"Could not unassign Colab runtime automatically: {e}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive_root", required=True, help="Drive root, e.g. /content/drive/MyDrive/mitosis_detection")
    ap.add_argument("--data_csv", required=True, help="CSV path on Drive")
    ap.add_argument("--raw_tif_dir", required=True, help="raw_tif directory on Drive")
    ap.add_argument("--checkout", default="", help="Git ref to checkout (branch/tag/commit). Optional.")
    ap.add_argument("--project_name", default="mitosis_meta_ana", help="Run grouping folder name under runs/")
    ap.add_argument("--run_prefix", default="train", help="Prefix for run_name")
    ap.add_argument("--note", default="", help="Optional short note for run_name")
    ap.add_argument("--config_base", default="configs/base.yaml", help="Base config path in repo")
    ap.add_argument("--config_out", default="configs/colab.yaml", help="Generated config path (should be gitignored)")
    ap.add_argument("--no_install", action="store_true", help="Skip pip install -r requirements.txt")
    ap.add_argument("--dry_run", action="store_true", help="Only create config/meta, do not start training")
    ap.add_argument("--shutdown_runtime", action="store_true", help="After successful training, unassign the Colab runtime")
    ap.add_argument("--resume", default="", help="Checkpoint path to resume from")
    ap.add_argument("--override_lr", default="", help="Override lr after resume, e.g. 3e-5")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    # Optional: checkout a specific ref for reproducibility
    if args.checkout:
        subprocess.run(["git", "fetch", "origin"], check=False)
        subprocess.run(["git", "checkout", args.checkout], check=True)

    # Install deps (optional)
    if not args.no_install:
        subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)

    # Git metadata
    git_commit = run_cmd(["git", "rev-parse", "HEAD"])
    git_branch = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    git_remote = run_cmd(["git", "remote", "get-url", "origin"])

    # Build standardized output_dir on Drive: runs/<project_name>/YYYYMMDD
    today = dt.datetime.now().strftime("%Y%m%d")
    drive_root = Path(args.drive_root)
    output_dir = drive_root / "runs" / sanitize(args.project_name) / today

    # Standardized run name
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    ref_part = sanitize(args.checkout if args.checkout else git_branch)
    note_part = sanitize(args.note) if args.note else ""
    sha8 = git_commit[:8]

    parts = [sanitize(args.run_prefix), ts, ref_part, sha8]
    if note_part:
        parts.append(note_part)
    run_name = "_".join([p for p in parts if p])

    run_dir = output_dir / run_name
    meta_dir = run_dir / "meta"
    out_subdir = run_dir / "outputs"

    ensure_dir(meta_dir)
    ensure_dir(out_subdir)

    # Load base config and write a generated colab config
    cfg = load_yaml(Path(args.config_base))
    cfg["data"]["csv_path"] = str(Path(args.data_csv))
    cfg["data"]["raw_tif_dir"] = str(Path(args.raw_tif_dir))
    cfg["project"]["output_dir"] = str(output_dir)
    cfg["logging"]["run_name"] = run_name

    save_yaml(Path(args.config_out), cfg)

    # Write run metadata files
    (meta_dir / "git_commit.txt").write_text(git_commit + "\n", encoding="utf-8")
    (meta_dir / "git_branch.txt").write_text(git_branch + "\n", encoding="utf-8")
    (meta_dir / "git_remote.txt").write_text(git_remote + "\n", encoding="utf-8")
    (meta_dir / "run_name.txt").write_text(run_name + "\n", encoding="utf-8")
    (meta_dir / "created_local_time.txt").write_text(ts + "\n", encoding="utf-8")
    (meta_dir / "config_used.yaml").write_text(Path(args.config_out).read_text(encoding="utf-8"), encoding="utf-8")

    try:
        freeze = run_cmd(["pip", "freeze"])
        (meta_dir / "pip_freeze.txt").write_text(freeze + "\n", encoding="utf-8")
    except Exception:
        (meta_dir / "pip_freeze.txt").write_text("pip freeze failed\n", encoding="utf-8")

    cmd_parts = ["python", "train.py", "--config", args.config_out]

    if args.resume:
        cmd_parts += ["--resume", args.resume]

    if args.override_lr:
        cmd_parts += ["--override_lr", str(args.override_lr)]

    cmdline = " ".join(cmd_parts)
    (meta_dir / "command.txt").write_text(cmdline + "\n", encoding="utf-8")

    print("repo_root:", str(repo_root))
    print("git_branch:", git_branch)
    print("git_commit:", git_commit)
    print("run_dir:", str(run_dir))
    print("config_out:", args.config_out)
    print("train_cmd:", cmdline)

    if args.dry_run:
        print("dry_run: skipping training")
        return

    # Start training
    subprocess.run(cmd_parts, check=True)

    if args.shutdown_runtime:
        # maybe_shutdown_colab_runtime()
        print("Training finished successfully.")
        print("Please run `from google.colab import runtime; runtime.unassign()` in a notebook cell to release GPU.")

if __name__ == "__main__":
    main()