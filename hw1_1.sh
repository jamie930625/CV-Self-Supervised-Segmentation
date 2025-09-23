
#!/bin/bash
# Problem 1
# $1: csv, $2: img_dir, $3: output csv

python3 src/inference.py \
  --problem p1 \
  --csv $1 \
  --img_dir $2 \
  --out $3 \
  --ckpt setting_c.pth

