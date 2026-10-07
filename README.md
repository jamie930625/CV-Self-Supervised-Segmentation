# Self-Supervised Pre-training and Semantic Segmentation

Course project for NTU Deep Learning for Computer Vision (Fall 2025), HW1. The full report is in `hw1_b11103049.pdf`.

## Part 1: Self-supervised pre-training with DINO

I implemented **DINO** (self-distillation with no labels) and pre-trained a ResNet-50 backbone from scratch on Mini-ImageNet, then fine-tuned it for image classification on Office-Home.

- Multi-crop augmentation (2 global and 4 local crops), color jitter, Gaussian blur, and solarization; cosine learning-rate schedule with warm-up.
- t-SNE visualization of the learned features at the first and last epochs.

**Office-Home classification accuracy**

| Setting | Backbone initialization | Fine-tuning | Accuracy |
|---|---|---|---|
| A | random | full | 0.31 |
| B | DINO on ImageNet-1k (provided weights) | full | 0.82 |
| C | **my DINO pre-training on Mini-ImageNet** | full | **0.73** |
| D | DINO on ImageNet-1k (provided weights) | classifier only | 0.79 |
| E | **my DINO pre-training on Mini-ImageNet** | classifier only | 0.69 |

Self-supervised pre-training lifts accuracy from 0.31 (random) to 0.73 without any labels, and full fine-tuning adds 3 to 4 points over a frozen backbone.

## Part 2: Semantic segmentation of satellite images

Seven-class land-cover segmentation (urban, agriculture, water, and others) on aerial images.

| Model | mIoU |
|---|---|
| U-Net, implemented from scratch (baseline) | 0.53 |
| **DeepLabV3+ with ResNet-101** | **0.76** |

- Ablation: removing U-Net skip connections lowers mIoU, since high-resolution spatial detail is lost.
- Applied the **Segment Anything Model (SAM)** zero-shot to validation images and compared it with the supervised models.

## Files

```text
src/train.py           # training for both problems
src/inference.py       # inference and CSV / mask output
hw1_1.sh, hw1_2.sh     # inference entry points
hw1_download_ckpt.sh   # downloads the trained checkpoints
mean_iou_evaluate.py   # mIoU evaluation (provided with the assignment)
viz_mask.py            # mask visualization
```

## Environment

```bash
conda create -n cv_env python=3.8
conda activate cv_env
pip install torch==2.4.0 torchvision==0.19.0
pip install -r requirements.txt
```
