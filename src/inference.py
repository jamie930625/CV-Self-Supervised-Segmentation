import argparse
import os
import torch
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import pandas as pd
import numpy as np


def load_model(ckpt_path, problem):
    """
    載入模型並讀取 state_dict
    problem = 1 → ResNet50 (分類, 65 類)
    problem = 2 → DeepLabV3+ ResNet101 (分割, 7 類)
    """
    if problem == 1:
        model = resnet50(weights=None, num_classes=65)  # Office-Home 有 65 類
    else:
        model = deeplabv3_resnet101(weights=None, num_classes=7)  # Segmentation 有 7 類

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 如果是 state_dict 格式
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
    else:
        # 萬一是 torch.save(model)
        model = ckpt

    model.eval()
    return model


# -------------------
# Problem 1: 分類
# -------------------
def inference_p1(model, csv_path, img_dir, output_csv):
    df = pd.read_csv(csv_path)
    results = []

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, row["filename"])
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)
            pred = torch.argmax(logits, dim=1).item()

        results.append([row["id"], row["filename"], pred])

    pd.DataFrame(results, columns=["id", "filename", "label"]).to_csv(output_csv, index=False)


# -------------------
# Problem 2: 分割
# -------------------
def inference_p2(model, img_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    for fname in os.listdir(img_dir):
        if not fname.endswith("_sat.jpg"):
            continue

        img_path = os.path.join(img_dir, fname)
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)["out"]  # DeepLabv3 輸出 dict
            pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

        out_path = os.path.join(out_dir, fname.replace("_sat.jpg", "_mask.png"))
        Image.fromarray(pred.astype(np.uint8)).save(out_path)


# -------------------
# Main
# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=int, required=True, help="1 = classification, 2 = segmentation")
    parser.add_argument("--csv", type=str, help="Problem 1: input csv")
    parser.add_argument("--img_dir", type=str, required=True, help="image directory")
    parser.add_argument("--output", type=str, help="Problem 1: output csv")
    parser.add_argument("--out_dir", type=str, help="Problem 2: output directory")
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoint path")
    args = parser.parse_args()

    model = load_model(args.ckpt, args.problem)

    if args.problem == 1:
        assert args.csv is not None and args.output is not None
        inference_p1(model, args.csv, args.img_dir, args.output)
    else:
        assert args.out_dir is not None
        inference_p2(model, args.img_dir, args.out_dir)

