#!/bin/bash
# ============================================================
# DLCV HW1 – Download script for DINO pretrained checkpoint (Setting C)
# ============================================================

set -euo pipefail
mkdir -p ckpt

echo "🔽 Downloading DINO checkpoint from Dropbox..."
wget -O ckpt/DINO_checkpoint.pth "https://www.dropbox.com/scl/fi/cdrp3wlw4r7cc3ew1f2bn/DINO_checkpoint.pth?rlkey=3mx0gakkbw3vta2mj8vcsmm6i&st=m4xkz3jo&dl=1"
echo "✅ Checkpoint successfully downloaded to ckpt/DINO_checkpoint.pth"
