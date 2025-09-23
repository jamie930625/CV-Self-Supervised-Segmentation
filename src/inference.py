#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet101

# ---------------------------
# Problem 1: Dataset (OfficeHome)
# ---------------------------
class OfficeHomeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 1])  # filename
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.data.iloc[idx, 0], self.data.iloc[idx, 1]  # id, filename


# ---------------------------
# Problem 2: Dataset (Segmentation)
# ---------------------------
class SegDataset(Dataset):
    def __init__(self, root, transform=None):
        self.img_paths = sorted([os.path.join(root, n) for n in os.listdir(root) if n.endswith('_sat.jpg')])
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(self.img_paths[idx])


# ---------------------------
# Load Model
# ---------------------------
def load_model(ckpt_path, problem):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if problem == "p1":
        model = models.resnet50()
        in_dim = model.fc.in_features
        model.fc = nn.Linear(in_dim, 65)
        model.load_state_dict(ckpt)
    elif problem == "p2":
        model = deeplabv3_resnet101(weights=None, aux_loss=False, num_classes=7)
        model.load_state_dict(ckpt['state_dict'])
    else:
        raise ValueError("problem must be p1 or p2")

    model.eval()
    return model


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, required=True, choices=["p1", "p2"])
    parser.add_argument("--csv", type=str, help="csv file for problem 1")
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, args.problem).to(device)

    if args.problem == "p1":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        dataset = OfficeHomeDataset(args.csv, args.img_dir, transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        outputs = []
        with torch.no_grad():
            for imgs, ids, filenames in loader:
                imgs = imgs.to(device)
                preds = model(imgs).argmax(1).cpu().numpy()
                for i, f, p in zip(ids, filenames, preds):
                    outputs.append([i, f, p])

        df = pd.DataFrame(outputs, columns=["id", "filename", "label"])
        df.to_csv(args.out, index=False)
        print(f"✅ Saved classification results to {args.out}")

    elif args.problem == "p2":
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        dataset = SegDataset(args.img_dir, transform)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        os.makedirs(args.out, exist_ok=True)

        with torch.no_grad():
            for imgs, fnames in loader:
                imgs = imgs.to(device)
                out = model(imgs)["out"]
                preds = out.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

                base = fnames[0].replace("_sat.jpg", "_mask.png")
                save_path = os.path.join(args.out, base)
                Image.fromarray(preds).save(save_path)

        print(f"✅ Saved segmentation masks to {args.out}")

