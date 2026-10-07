#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DLCV HW1 – Problem 1: Classification (Setting C)
Usage:
    python3 src/inference.py <csv_path> <img_dir> <out_csv> --ckpt ckpt/settingC_best.pth
"""

import os, sys, types, csv, glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models

# =========================================================
# NumPy 2.x compatibility shim for numpy._core.* paths (pandas / pickle)
# =========================================================
if "numpy._core" not in sys.modules:
    _core = types.ModuleType("numpy._core")
    submods = {}
    if hasattr(np.core, "multiarray"):        submods["multiarray"] = np.core.multiarray
    if hasattr(np.core, "umath"):             submods["umath"] = np.core.umath
    if hasattr(np.core, "numeric"):           submods["numeric"] = np.core.numeric
    if hasattr(np.core, "_multiarray_umath"): submods["_multiarray_umath"] = np.core._multiarray_umath
    sys.modules["numpy._core"] = _core
    np._core = _core
    for name, obj in submods.items():
        sys.modules[f"numpy._core.{name}"] = obj
        setattr(_core, name, obj)

# =========================================================
# constants
# =========================================================
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

# =========================================================
# build and load the model (Setting C)
# =========================================================
def build_resnet50_classifier(num_classes: int = 65) -> nn.Module:
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def load_cls_model(ckpt_path: str, num_classes: int = 65, device: torch.device | str = "cpu") -> nn.Module:
    model = build_resnet50_classifier(num_classes=num_classes)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # accept {'state_dict': ...} or a plain state_dict; strict=False for compatibility
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model

# =========================================================
# image preprocessing (no extra resizing)
# =========================================================
def preprocess_pil(pil_img):
    # ensure RGB
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    W, H = pil_img.size
    C = len(pil_img.getbands())

    # convert PIL bytes to a torch.Tensor directly (without numpy)
    t = torch.frombuffer(pil_img.tobytes(), dtype=torch.uint8)
    t = t.view(H, W, C).permute(2, 0, 1).to(torch.float32).div_(255.0)

    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t = (t - mean) / std
    return t


# =========================================================
# read the CSV with pandas, falling back to the csv module
#   returns [(id, filename), ...] with the input ids unchanged
# =========================================================
def read_id_and_filename_list(csv_path: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    try:
        import pandas as pd  # imported after the shim for numpy 2.x compatibility
        df = pd.read_csv(csv_path, encoding="utf-8-sig", sep=",", engine="python")
        # use column names if present, otherwise column positions
        if all(c in df.columns for c in ["id", "filename"]):
            ids = df["id"].tolist()
            fns = df["filename"].astype(str).tolist()
        else:
            # column 0 is the id, column 1 is the filename
            ids = df.iloc[:, 0].tolist()
            fns = df.iloc[:, 1].astype(str).tolist()
        for _id, _fn in zip(ids, fns):
            rows.append((int(_id), _fn))
        if len(rows) == 0:
            raise RuntimeError("Parsed zero rows via pandas.")
        return rows
    except Exception as e:
        print(f"[WARN] pandas read_csv failed, fallback to csv module. reason: {e}")
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)  # skip the header
            for r in reader:
                if len(r) >= 2:
                    try:
                        rid = int(r[0])
                    except Exception:
                        # skip rows whose first column is not an integer
                        continue
                    rows.append((rid, str(r[1]).strip()))
        if len(rows) == 0:
            raise RuntimeError(f"Failed to parse rows from {csv_path}")
        return rows

# =========================================================
# main inference loop: one image at a time (no DataLoader workers)
#   outputs use the ids from the input CSV
# =========================================================
@torch.no_grad()
def infer_cls(model: nn.Module, csv_path: str, img_dir: str, out_csv: str, device: torch.device):
    pairs = read_id_and_filename_list(csv_path)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "filename", "label"])

        for _id, fname in pairs:
            img_path = os.path.join(img_dir, fname)
            # warn and skip missing files
            if not os.path.isfile(img_path):
                print(f"[WARN] file not found: {img_path} (skip)")
                continue
            pil_img = Image.open(img_path).convert("RGB")
            x = preprocess_pil(pil_img).unsqueeze(0).to(device)
            out = model(x)
            pred = out.argmax(1).item()
            writer.writerow([_id, fname, int(pred)])

    print(f"Saved CSV: {out_csv}")

# =========================================================
# Main
# =========================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="DLCV HW1 – P1 Inference (Setting C)")
    parser.add_argument("csv_path", type=str, help="path to val/test csv (with columns: id,filename,...)")
    parser.add_argument("img_dir",  type=str, help="path to image folder")
    parser.add_argument("out_csv",  type=str, help="path to output csv")
    parser.add_argument("--ckpt",   type=str, required=True, help="path to checkpoint (e.g., ckpt/settingC_best.pth)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("=== Running Problem 1 (Classification) ===")
    print("CSV path:", args.csv_path)
    print("Image folder:", args.img_dir)
    print("Output CSV:", args.out_csv)
    print("Checkpoint:", args.ckpt)

    model = load_cls_model(args.ckpt, num_classes=65, device=device)
    infer_cls(model, args.csv_path, args.img_dir, args.out_csv, device)

if __name__ == "__main__":
    main()
