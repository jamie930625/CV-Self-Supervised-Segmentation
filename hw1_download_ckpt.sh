#!/bin/bash
# ============================================================
# DLCV HW1 – Download checkpoints for Problem 1 (Setting C) & Problem 2 (Model B)
# ============================================================

set -e
mkdir -p ckpt

echo "🔽 Downloading Setting C checkpoint..."
wget -O ckpt/settingC.pth "https://www.dropbox.com/scl/fi/59ri4t1vdhmwk7szbqx6d/settingC.pth?rlkey=fm600hpreflzejgmtrrjf7g4t&st=8gjqzcuw&dl=1"

echo "🔽 Downloading Model B checkpoint..."
wget -O ckpt/ModelB.pth "https://www.dropbox.com/scl/fi/vgwfk1p3d0apb4ksa7kr9/ModelB.pth?rlkey=o559my9rv5f8fpyfl5ta005wg&st=7s47u12o&dl=1"

echo "✅ All checkpoints downloaded successfully into ./ckpt/"
ls -lh ckpt

