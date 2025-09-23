#!/bin/bash
# Problem 2 Inference
# $1: img_dir, $2: output_dir

python3 src/inference.py $1 $2 --ckpt Model_B.pth

