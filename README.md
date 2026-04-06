# Self-Supervised Representation Learning & Semantic Segmentation

This repository contains two main computer vision projects focusing on feature representation learning without labels and dense pixel-level prediction. 

## Part 1: Self-Supervised Pre-training (DINO)
In this section, I implemented **DINO** (Self-Distillation with No Labels) to pre-train a ResNet50 backbone on the Mini-ImageNet dataset. The learned visual representations were then evaluated on the Office-Home dataset for downstream image classification.

### Key Highlights:
- **Architecture**: ResNet50 backbone pre-trained entirely from scratch without supervision.
- **Evaluation**: Fine-tuned a classifier on the frozen backbone and achieved robust classification accuracy.
- **Visualization**: Leveraged t-SNE to visualize the clustering capability of the learned representations in high-dimensional space.

## Part 2: Semantic Segmentation on Satellite Imagery
The second part focuses on segmenting geographic regions (e.g., urban, agriculture, water) from aerial satellite images. 

### Key Highlights:
- **Baseline Model**: Developed a standard **U-Net** architecture from scratch.
- **Improved Architecture**: Implemented an advanced CNN-based segmentation model (e.g., DeepLab / FCN) to handle multi-scale context, significantly improving the Mean Intersection over Union (mIoU).
- **Foundation Model Application**: Utilized Meta's **Segment Anything Model (SAM)** to perform zero-shot segmentation and compared the performance with fully supervised models.

## Environment Setup
To reproduce the environment and run the inference scripts:
```bash
conda create -n cv_env python=3.8
conda activate cv_env
pip install torch==2.4.0 torchvision==0.19.0
pip install -r requirements.txt
