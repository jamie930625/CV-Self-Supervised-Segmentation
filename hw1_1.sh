#!/bin/bash
# ============================================================
# DLCV HW1 – Problem 1 (Setting C) Inference Script
# Author: B11103049 林立潔
# ============================================================

set -e

# $1: path to CSV file (e.g., hw1_hiddendata/p1_data/office/test.csv)
# $2: path to image folder (e.g., hw1_hiddendata/p1_data/office/test/)
# $3: path to output CSV file (e.g., output_p1/test_pred.csv)

echo "=== Problem 1: Inference (Setting C) ==="
echo "CSV path: $1"
echo "Image folder: $2"
echo "Output CSV: $3"

# Run inference
python3 src/inference.py "$1" "$2" "$3"

echo "✅ Inference completed successfully!"
