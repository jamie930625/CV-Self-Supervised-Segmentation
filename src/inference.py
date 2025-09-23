#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DLCV HW1 - Problem 2 Inference
- DeepLabV3(ResNet101) with aux head (aux_loss=True), 7 classes (含背景0)
- 與你的訓練腳本結構完全對齊（主頭/aux頭最後一層皆改為 num_classes=7）
- 自動處理 NumPy 2.x→1.24 的 pickle 相容 (numpy._core*)
- 輸入: <img_dir> 內所有 *_sat.jpg
- 輸出: <out_dir>/xxxx_mask.png (RGB 配色與訓練一致)
"""

import os, sys, glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import deeplabv3_resnet101

# -------------------------------
# NumPy 2.x → 1.24 相容補丁 (pickle 中用到 numpy._core.*)
# -------------------------------
import types
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
# 常數與配色（與訓練一致）
# -------------------------------
NUM_CLASSES = 7
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

# index -> RGB
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
# 建模：與訓練腳本完全對齊 (aux_loss=True)
# -------------------------------
def build_deeplab_resnet101_infer(num_classes=NUM_CLASSES, aux_loss=True):
    model = deeplabv3_resnet101(weights=None, aux_loss=aux_loss)
    # 主頭改成 num_classes
    in_ch = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
    # aux 頭也改成 num_classes
    if aux_loss and getattr(model, "aux_classifier", None) is not None:
        in_ch_aux = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = nn.Conv2d(in_ch_aux, num_classes, kernel_size=1)
    return model

def load_model(ckpt_path: str):
    model = build_deeplab_resnet101_infer(num_classes=NUM_CLASSES, aux_loss=True)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        # 若有少/多鍵（例如不同版權重），放寬保證可跑通
        print(f"[load_model] strict=True 失敗，改 strict=False：{e}")
        model.load_state_dict(state, strict=False)
    model.eval()
    return model

# -------------------------------
# 推論與存檔
# -------------------------------
@torch.no_grad()
def infer_tensor(model, x: torch.Tensor, device: torch.device, out_size_hw):
    x = x.to(device, non_blocking=True).unsqueeze(0)  # [1,3,H,W]
    out = model(x)["out"]  # [1,C,H,W]
    pred = out.argmax(1)[0].to("cpu").numpy().astype(np.uint8)  # [H,W]
    H, W = out_size_hw
    if pred.shape != (H, W):
        # 安全起見：若輸出大小不一致，補最近鄰縮放回輸入大小
        pred = np.array(Image.fromarray(pred, mode="L").resize((W, H), resample=Image.NEAREST), dtype=np.uint8)
    return pred

def save_index_mask_as_rgb(index_mask: np.ndarray, save_path: str):
    h, w = index_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, rgb_tuple in IDX2RGB.items():
        rgb[index_mask == idx] = rgb_tuple
    Image.fromarray(rgb).save(save_path)

def preprocess_pil(pil_img: Image.Image) -> torch.Tensor:
    x = TF.to_tensor(pil_img)
    x = TF.normalize(x, mean=MEAN, std=STD)
    return x

# -------------------------------
# 入口（符合作業規範：兩個位置參數 + 可選 --ckpt）
# -------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("img_dir", type=str, help="testing images directory (包含 *_sat.jpg)")
    parser.add_argument("out_dir", type=str, help="output directory (輸出 *_mask.png)")
    parser.add_argument("--ckpt", type=str, default="ckpts/best_mIoU.pth", help="path to checkpoint (.pth)")
    args = parser.parse_args()

    # 作業規定限制的是 bash 指令不可改環境；在 Python 內建立輸出資料夾是允許的
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 讀取模型
    print("Loading checkpoint:", args.ckpt)
    model = load_model(args.ckpt).to(device)

    # 掃描測試影像
    img_paths = sorted(glob.glob(os.path.join(args.img_dir, "*_sat.jpg")))
    if len(img_paths) == 0:
        print(f"[WARN] 找不到 *_sat.jpg：{args.img_dir}")
        sys.exit(0)

    for ip in img_paths:
        name = os.path.basename(ip).replace("_sat.jpg", "_mask.png")
        save_path = os.path.join(args.out_dir, name)

        pil_img = Image.open(ip).convert("RGB")
        H, W = pil_img.height, pil_img.width
        x = preprocess_pil(pil_img)
        pred = infer_tensor(model, x, device, out_size_hw=(H, W))
        save_index_mask_as_rgb(pred, save_path)

        print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()

