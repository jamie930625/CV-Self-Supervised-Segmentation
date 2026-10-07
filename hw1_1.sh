#!/bin/bash
set -e
echo "=== Problem 1: Inference (Setting C) ==="
echo "CSV path: $1"
echo "Image folder: $2"
echo "Output CSV: $3"
mkdir -p "$(dirname "$3")"

if [ -f ckpt/settingC_best.pth ]; then
  CKPT="ckpt/settingC_best.pth"
elif [ -f ckpt/settingC.pth ]; then
  CKPT="ckpt/settingC.pth"
else
  echo "Cannot find checkpoint in ckpt/. Expected settingC_best.pth or settingC.pth"; exit 1
fi
echo "Using checkpoint: $CKPT"
python3 src/inference.py "$1" "$2" "$3" --ckpt "$CKPT"
echo "Inference completed successfully!"
