
#!/bin/bash
# Usage: bash hw1_2.sh <img_dir> <out_dir>
python3 src/inference.py --problem 2 --img_dir $1 --out_dir $2 --ckpt Model_B.pth

