#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DLCV HW1 Inference
- Problem 1: Classification (Setting C)
    usage: python3 src/inference.py test.csv img_dir output.csv --ckpt setting_c.pth
- Problem 2: Segmentation (DeepLabV3, 7 classes)
    usage: python3 src/inference.py img_dir out_dir --ckpt Model_B.pth
"""

import os, sys, glob, types, csv
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import models
from torchvision.models.segmentation import deeplabv3_resnet101

# -------------------------------
# NumPy 2.x → 1.24 相容補丁 (pickle 中用到 numpy._core.*)
# -------------------------------
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

# -------------------------------
# 常數
# -------------------------------
NUM_CLASSES_P2 = 7
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

IDX2RGB = {
    0: (0,   0,   0  ),  # background
    1: (255, 255, 255),
    2: (0,   0,   255),
    3: (0,   255, 0  ),
    4: (255, 0,   255),
    5: (255, 255, 0  ),
    6: (0,   255, 255),
}

# -------------------------------
# 分割模型 (Problem 2)
# -------------------------------
def build_deeplab_resnet101_infer(num_classes=NUM_CLASSES_P2, aux_loss=True):
    model = deeplabv3_resnet101(weights=None, aux_loss=aux_loss)
    in_ch = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
    if aux_loss and getattr(model, "aux_classifier", None) is not None:
        in_ch_aux = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = nn.Conv2d(in_ch_aux, num_classes, kernel_size=1)
    return model

def load_seg_model(ckpt_path: str):
    model = build_deeplab_resnet101_infer(num_classes=NUM_CLASSES_P2, aux_loss=True)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

# -------------------------------
# 分類模型 (Problem 1)
# -------------------------------
def build_resnet50_classifier(num_classes=65):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def load_cls_model(ckpt_path: str, num_classes=65):
    model = build_resnet50_classifier(num_classes=num_classes)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

# -------------------------------
# 前處理
# -------------------------------
def preprocess_pil(pil_img: Image.Image) -> torch.Tensor:
    x = TF.to_tensor(pil_img)
    x = TF.normalize(x, mean=MEAN, std=STD)
    return x

# -------------------------------
# Segmentation 推論
# -------------------------------
@torch.no_grad()
def infer_seg(model, img_path, device):
    pil_img = Image.open(img_path).convert("RGB")
    H, W = pil_img.height, pil_img.width
    x = preprocess_pil(pil_img).unsqueeze(0).to(device)
    out = model(x)["out"]
    pred = out.argmax(1)[0].cpu().numpy().astype(np.uint8)
    if pred.shape != (H, W):
        pred = np.array(Image.fromarray(pred, mode="L").resize((W, H), Image.NEAREST), dtype=np.uint8)
    return pred

def save_index_mask_as_rgb(index_mask: np.ndarray, save_path: str):
    h, w = index_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, rgb_tuple in IDX2RGB.items():
        rgb[index_mask == idx] = rgb_tuple
    Image.fromarray(rgb).save(save_path)

# -------------------------------
# Classification 推論
# -------------------------------
@torch.no_grad()
def infer_cls(model, csv_path, img_dir, out_csv, device):
    import pandas as pd
    df = pd.read_csv(csv_path)
    results = []
    for idx, row in df.iterrows():
        fname = row["filename"]
        img_path = os.path.join(img_dir, fname)
        pil_img = Image.open(img_path).convert("RGB")
        x = preprocess_pil(pil_img).unsqueeze(0).to(device)
        out = model(x)
        pred = out.argmax(1).item()
        results.append([row["id"], fname, pred])
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "filename", "label"])
        writer.writerows(results)
    print(f"Saved CSV: {out_csv}")

# -------------------------------
# Main
# -------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("arg1", type=str, help="Problem1: test.csv | Problem2: img_dir")
    parser.add_argument("arg2", type=str, help="Problem1: img_dir | Problem2: out_dir")
    parser.add_argument("arg3", type=str, nargs="?", help="Problem1: output.csv | Problem2: (unused)")
    parser.add_argument("--ckpt", type=str, required=True, help="path to checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Problem 1: 三個位置參數
    if args.arg3 is not None:
        print("Running Problem 1 (Classification)")
        model = load_cls_model(args.ckpt).to(device)
        infer_cls(model, args.arg1, args.arg2, args.arg3, device)

    # Problem 2: 兩個位置參數
    else:
        print("Running Problem 2 (Segmentation)")
        os.makedirs(args.arg2, exist_ok=True)
        model = load_seg_model(args.ckpt).to(device)
        img_paths = sorted(glob.glob(os.path.join(args.arg1, "*_sat.jpg")))
        if len(img_paths) == 0:
            print(f"[WARN] 找不到 *_sat.jpg：{args.arg1}")
            sys.exit(0)
        for ip in img_paths:
            name = os.path.basename(ip).replace("_sat.jpg", "_mask.png")
            save_path = os.path.join(args.arg2, name)
            pred = infer_seg(model, ip, device)
            save_index_mask_as_rgb(pred, save_path)
            print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()

