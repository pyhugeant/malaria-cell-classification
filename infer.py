from __future__ import annotations

import argparse
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_yaml, resolve_device
from src.data.split import group_split_by_fov
from src.datasets.malaria_dataset import EventSequenceDataset
from src.models.model import MitosisNet
from src.inference.postprocess import heatmap_peak_xy, mask_centroid
from src.eval.metrics import l2_error_xy, summarize_errors

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--max_batches", default=50, type=int)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = resolve_device(str(cfg["project"]["device"]))

    df = pd.read_csv(str(cfg["data"]["csv_path"]))
    fov_col = str(cfg["data"]["csv_columns"]["fov_id"])
    split_cfg = cfg["data"]["split"]
    split = group_split_by_fov(
        df=df,
        fov_col=fov_col,
        seed=int(cfg["project"]["seed"]),
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
    )
    if args.split == "train":
        fovs = split.train_fovs
    elif args.split == "test":
        fovs = split.test_fovs
    else:
        fovs = split.val_fovs

    ds = EventSequenceDataset(df, cfg, fovs, train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    model = MitosisNet(
        base_channels=int(cfg["model"]["encoder"]["base_channels"]),
        temporal_type=str(cfg["model"]["temporal"]["type"]),
        hidden_channels=int(cfg["model"]["temporal"]["hidden_channels"]),
    ).to(device)
    model.eval()

    # Load latest checkpoint if provided
    # For simplicity, this script runs with random weights unless you load a checkpoint manually.
    # You can add: --ckpt path/to/epoch_xxx.pt

    meta_errors = []
    ana_errors = []

    x_col = str(cfg["data"]["csv_columns"]["x"])
    y_col = str(cfg["data"]["csv_columns"]["y"])
    swap_xy = bool(cfg["dataset"].get("swap_xy", False))

    for b, (frames, meta_t, ana_t, phase) in enumerate(tqdm(loader, desc="infer")):
        if b >= int(args.max_batches):
            break
        frames = frames.to(device)
        with torch.no_grad():
            meta_logits, ana_logits = model(frames)
        meta_prob = torch.sigmoid(meta_logits)[0].cpu().numpy()  # [T,1,H,W]
        ana_prob = torch.sigmoid(ana_logits)[0].cpu().numpy()

        # Use target peak as GT point in patch coordinates (approx)
        # Note: For precise GT, use the original CSV (full-image coords) and track crop offsets.
        # This skeleton reports errors in patch coordinates only.
        meta_gt = meta_t[0].numpy()  # [T,1,H,W]
        ana_gt = ana_t[0].numpy()

        for t in range(meta_prob.shape[0]):
            px, py, _ = heatmap_peak_xy(meta_prob[t, 0])
            gx, gy, _ = heatmap_peak_xy(meta_gt[t, 0])
            meta_errors.append(l2_error_xy((px, py), (gx, gy)))

            c = mask_centroid(ana_prob[t, 0], thr=float(cfg["inference"]["ana"]["mask_threshold"]), min_area=int(cfg["inference"]["ana"]["centroid_min_area"]))
            if c is not None:
                gx2, gy2, _ = heatmap_peak_xy(ana_gt[t, 0])
                ana_errors.append(l2_error_xy(c, (gx2, gy2)))

    radii = [float(x) for x in cfg["eval"]["pck_radii_px"]]
    meta_res = summarize_errors(meta_errors, radii)
    ana_res = summarize_errors(ana_errors, radii)

    print("meta_mean_error_px", meta_res.mean_error)
    print("meta_pck", meta_res.pck)
    print("ana_mean_error_px", ana_res.mean_error)
    print("ana_pck", ana_res.pck)

if __name__ == "__main__":
    main()
