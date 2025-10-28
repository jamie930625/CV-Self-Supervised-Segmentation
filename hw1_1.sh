#!/bin/bash
set -e

# $1: path to images csv  (e.g., .../office/test.csv)
# $2: path to image folder (e.g., .../office/test/)
# $3: path to output csv   (e.g., output_p1/test_pred.csv)

echo "=== Problem 1: Inference (Setting C) ==="
echo "CSV path: $1"
echo "Image folder: $2"
echo "Output CSV: $3"

# 允許用環境變數覆寫 ckpt 路徑（預設為下載腳本放置的位置）
CKPT_PATH="${CKPT_PATH:-ckpt/settingC_best.pth}"

# 確保輸出資料夾存在
mkdir -p "$(dirname "$3")"

python3 src/inference.py "$1" "$2" "$3" --ckpt "$CKPT_PATH"

echo "✅ Inference completed successfully!"
