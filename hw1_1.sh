
#!/bin/bash
# Problem 1 Inference
# $1: path to images csv file (e.g., test.csv)
# $2: path to folder containing images
# $3: path to output csv file (predicted labels)

python3 src/inference.py $1 $2 $3 --ckpt setting_c.pth

