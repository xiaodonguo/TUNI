# TUNI: Real-time RGB-T Semantic Segmentation with Unified Multi-Modal Feature Extraction and Cross-Modal Feature Fusion
## Introduction
This repository contains the code for TUNI, which has been submitted to ICRA 2026. The current repository includes the model files, evaluation files, pre-trained weights, and prediction images to facilitate the review process.

✨**2025-9-5**✨ : Upload model files, evaluation files, pre-trained weights, prediction images. 

## Method 
![picture1](./fig/fig2.png)
The CM-SSM consists of two image encoders to extract the features of RGB and thermal images, four CM-SSA moudules to perform RGB-T feature fusion in four stages, and an MLP decoder to predict the semantic segmentation maps.

![picture2](./fig/fig3.png)
The CM-SS2D consists of three steps: 1) cross-modal selective scanning, 2) cross-modal state space modeling and 3) scan merging.

## Reqiurements
Python==3.9  
Pytorch==2.0.1  
Cuda==11.8  
mamba-ssm==1.0.1  
selective-scan==0.0.1  
mmcv==2.2.0  

| Models |Backbone| Dataset  | mIoU | Weights|
|------|------|------------|------|--------------|
| CM-SSM|EfficientVit-B1    | [CART](https://github.com/aerorobotics/caltech-aerial-rgbt-dataset)      | 75.1   | [pth](https://github.com/xiaodonguo/CMSSM/releases/download/v1.0.1/CART.pth)     |
| CM-SSM|EfficeintVit-B1   | [PST900](https://github.com/ShreyasSkandanS/pst900_thermal_rgb)     | 85.9    | [pth](https://github.com/xiaodonguo/CMSSM/releases/download/v1.0.1/PST900.pth)     |
| CM-SSM|ConvNeXtV2-A    | [SUS](https://github.com/xiaodonguo/SUS_dataset)      | 82.5   | [pth](https://github.com/xiaodonguo/CMSSM/releases/download/v1.0.1/SUS.pth)     |
| CM-SSM|ConvNeXtV2-A   | [FMB](https://github.com/JinyuanLiu-CV/SegMiF)     | 60.7    | [pth](https://github.com/xiaodonguo/CMSSM/releases/download/v1.0.1/FMB.pth)     |
# Concat
If any questions, please contact 3120245534@bit.edu.cn.
