
#!/bin/bash
# Usage: bash hw1_1.sh <csv_path> <img_dir> <output_csv>
python3 src/inference.py --problem 1 --csv $1 --img_dir $2 --output $3 --ckpt setting_c.pth

