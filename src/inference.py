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
import torchvision.transforms.functional as TF
from torchvision import models

# =========================================================
# NumPy 2.x 相容補丁（修復舊路徑 numpy._core.*，避免奇怪的 pandas / pickle 兼容問題）
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
# 常數（與你可跑版本一致）
# =========================================================
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

# =========================================================
# 模型構建 / 載入（與 Setting C 一致）
# =========================================================
def build_resnet50_classifier(num_classes: int = 65) -> nn.Module:
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def load_cls_model(ckpt_path: str, num_classes: int = 65, device: torch.device | str = "cpu") -> nn.Module:
    model = build_resnet50_classifier(num_classes=num_classes)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # 同時支援 {'state_dict': ...} 或純 state_dict；strict=False 提升相容性
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model

# =========================================================
# 影像前處理（維持你原本能跑的邏輯，不額外 resize）
# =========================================================
def preprocess_pil(pil_img: Image.Image) -> torch.Tensor:
    x = TF.to_tensor(pil_img)
    x = TF.normalize(x, mean=MEAN, std=STD)
    return x

# =========================================================
# 穩健讀 CSV：優先用 pandas（指定編碼/engine）；失敗則改用 csv 標準庫
#   - 回傳：[(id, filename), ...]，保持輸入的 id（與你已測通版本一致）
# =========================================================
def read_id_and_filename_list(csv_path: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    try:
        import pandas as pd  # 放在 shim 之後 import，避免 numpy 2.x 相容性雷
        df = pd.read_csv(csv_path, encoding="utf-8-sig", sep=",", engine="python")
        # 優先依欄名存取；若沒有欄名就 fall back 到位置
        if all(c in df.columns for c in ["id", "filename"]):
            ids = df["id"].tolist()
            fns = df["filename"].astype(str).tolist()
        else:
            # 位置：第 0 欄視為 id、第 1 欄視為 filename
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
            header = next(reader, None)  # 丟表頭
            for r in reader:
                if len(r) >= 2:
                    try:
                        rid = int(r[0])
                    except Exception:
                        # 若第一欄不是整數，嘗試跳過
                        continue
                    rows.append((rid, str(r[1]).strip()))
        if len(rows) == 0:
            raise RuntimeError(f"Failed to parse rows from {csv_path}")
        return rows

# =========================================================
# 推論主流程：逐張處理（避免 DataLoader/多進程小雷）
#   - 依輸入 CSV 的 id 原樣輸出（與你的可跑版一致）
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
            # 容錯：若檔案不存在，直接跳過或報警（這裡選擇報警並跳過）
            if not os.path.isfile(img_path):
                print(f"[WARN] file not found: {img_path} (skip)")
                continue
            pil_img = Image.open(img_path).convert("RGB")
            x = preprocess_pil(pil_img).unsqueeze(0).to(device)
            out = model(x)
            pred = out.argmax(1).item()
            writer.writerow([_id, fname, int(pred)])

    print(f"✅ Saved CSV: {out_csv}")

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
