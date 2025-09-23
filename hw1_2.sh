
#!/bin/bash
# Problem 2
# $1: img_dir, $2: output_dir

python3 src/inference.py \
  --problem p2 \
  --img_dir $1 \
  --out $2 \
  --ckpt Model_B.pth

