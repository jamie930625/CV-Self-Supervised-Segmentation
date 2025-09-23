#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DLCV HW1 Train Script
包含 Problem 1 (Set C 分類, ResNet50) 與 Problem 2 (Model B 分割, DeepLabV3+ ResNet101)
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms import functional as TF
from torchvision.models.segmentation import deeplabv3_resnet101
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm
import random
import time
from dataclasses import dataclass


# ============================================================
# Problem 1: OfficeHome Dataset for Classification
# ============================================================
class OfficeHomeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 1])  # filename
        label = int(self.data.iloc[idx, 2])  # label
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def train_problem1(train_csv, val_csv, train_dir, val_dir, save_path="setC_best_finetuned.pth"):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = OfficeHomeDataset(train_csv, train_dir, transform=train_transform)
    val_dataset = OfficeHomeDataset(val_csv, val_dir, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = models.resnet50(weights=None)
    in_dim = model.fc.in_features
    model.fc = nn.Linear(in_dim, 65)  # Office-Home: 65 classes
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc")]
    fc_params = model.fc.parameters()

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-5},
        {"params": fc_params, "lr": 1e-4}
    ], weight_decay=1e-4)

    num_epochs = 10
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"[P1 Epoch {epoch+1}/{num_epochs}]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total
        train_loss = total_loss / total

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total

        print(f"[P1 Epoch {epoch+1}] Train Loss={train_loss:.4f} Train Acc={train_acc:.4f} Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best P1 model at epoch {epoch+1} (Val Acc={best_val_acc:.4f})")


# ============================================================
# Problem 2: Semantic Segmentation (DeepLabV3+ ResNet101)
# ============================================================
class SegDataset(Dataset):
    COLOR_MAP = {
        (0, 0, 0): 0,
        (255, 255, 255): 1,
        (0, 0, 255): 2,
        (0, 255, 0): 3,
        (255, 0, 255): 4,
        (255, 255, 0): 5,
        (0, 255, 255): 6,
    }

    def __init__(self, root, crop_size=512, scale_min=0.5, scale_max=2.0,
                 is_train=True, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
        self.img_paths, self.mask_paths = [], []
        names = sorted([n for n in os.listdir(root) if n.endswith('_sat.jpg')])
        for n in names:
            ip = os.path.join(root, n)
            bn = n.replace('_sat.jpg', '')
            mp = os.path.join(root, bn + '_mask.png')
            if os.path.isfile(mp):
                self.img_paths.append(ip)
                self.mask_paths.append(mp)
        self.is_train = is_train
        self.crop_size = crop_size
        self.scale_min, self.scale_max = scale_min, scale_max
        self.mean, self.std = mean, std
        self.num_classes = len(self.COLOR_MAP)

    def _rgb_to_index(self, mask_rgb):
        h, w, _ = mask_rgb.shape
        mask = np.zeros((h, w), dtype=np.int64)
        for rgb, idx in self.COLOR_MAP.items():
            mask[(mask_rgb == rgb).all(axis=-1)] = idx
        return mask

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('RGB')

        if self.is_train:
            scale = random.uniform(self.scale_min, self.scale_max)
            new_w, new_h = int(img.width*scale), int(img.height*scale)
            img = img.resize((new_w,new_h), Image.BILINEAR)
            mask = mask.resize((new_w,new_h), Image.NEAREST)
            i,j,h,w = transforms.RandomCrop.get_params(img, (self.crop_size,self.crop_size))
            img = TF.crop(img, i,j,h,w)
            mask = TF.crop(mask, i,j,h,w)
            if random.random()<0.5:
                img = TF.hflip(img); mask = TF.hflip(mask)

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, mean=self.mean, std=self.std)
        mask_np = np.array(mask)
        mask_idx = self._rgb_to_index(mask_np)
        mask_t = torch.from_numpy(mask_idx).long()
        return img_t, mask_t


@dataclass
class Args:
    train_root: str
    val_root: str
    save_dir: str
    epochs: int = 200
    batch_size: int = 8
    lr: float = 0.004
    num_workers: int = 4
    ignore_index: int = 0


def train_problem2(args: Args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    train_set = SegDataset(args.train_root, is_train=True)
    val_set = SegDataset(args.val_root, is_train=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model = deeplabv3_resnet101(weights=None, num_classes=train_set.num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)

    best_miou = -1.0
    for epoch in range(1, args.epochs+1):
        model.train()
        for imgs, masks in tqdm(train_loader, desc=f"[P2 Epoch {epoch}/{args.epochs}]"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            out = model(imgs)["out"]
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()

        # 簡單 validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                out = model(imgs)["out"]
                pred = out.argmax(1)
                correct += (pred == masks).sum().item()
                total += masks.numel()
        acc = correct / total
        print(f"[P2 Epoch {epoch}] Val Pixel Acc: {acc:.4f}")

        # 存最好模型
        if acc > best_miou:
            best_miou = acc
            torch.save({'epoch':epoch, 'state_dict':model.state_dict(), 'best_mIoU':best_miou},
                       os.path.join(args.save_dir, "best_mIoU.pth"))
            print(f"✅ Saved best P2 model at epoch {epoch} (Pixel Acc={acc:.4f})")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=int, required=True, help="1=classification, 2=segmentation")
    parser.add_argument("--train_root", type=str, required=False)
    parser.add_argument("--val_root", type=str, required=False)
    parser.add_argument("--save_dir", type=str, required=False, default="ckpt")
    args_in = parser.parse_args()

    if args_in.problem == 1:
        # 範例用法: python3 train.py --problem 1
        train_csv = "data_2025/p1_data/office/train.csv"
        val_csv   = "data_2025/p1_data/office/val.csv"
        train_dir = "data_2025/p1_data/office/train"
        val_dir   = "data_2025/p1_data/office/val"
        train_problem1(train_csv, val_csv, train_dir, val_dir)
    else:
        # 範例用法: python3 train.py --problem 2 --train_root ... --val_root ...
        args = Args(train_root=args_in.train_root, val_root=args_in.val_root, save_dir=args_in.save_dir)
        train_problem2(args)

