#!/bin/bash
# ============================================================
# DLCV HW1 – Checkpoint Downloader
# Author: B11103049 林立潔
# ============================================================

set -euo pipefail

# 建立 ckpt 資料夾
mkdir -p ckpt

# =============================
# Problem 1: Setting C checkpoint
# =============================
echo "🔽 Downloading checkpoint for Problem 1 (Setting C)..."
wget -O ckpt/settingC.pth "https://www.dropbox.com/scl/fi/oxq9efxc10z8ifbe3iqng/settingC_best.pth?rlkey=i5x6mhaozt97sl7fdezaowd64&st=juzv7lj2&dl=1"

# =============================
# Problem 2: Model B checkpoint
# =============================
echo "🔽 Downloading checkpoint for Problem 2 (Model B)..."
wget -O ckpt/ModelB.pth "https://www.dropbox.com/scl/fi/vgwfk1p3d0apb4ksa7kr9/ModelB.pth?rlkey=o559my9rv5f8fpyfl5ta005wg&st=2hrd8t0z&dl=1"

echo "✅ All checkpoints downloaded successfully!"
