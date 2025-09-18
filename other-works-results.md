# Results From Other Works

This .md file contains several results from other papers that were (possibly) used to compare to BALF in our paper.
Some of the results were extracted from plots, and, since we appreciate transparency, we provide the raw data in the repository. We did our best to extract precise results.
Note that this is not an exhaustive list including every result from the mentioned papers.
All results are with no fine-tuning, unless otherwise noted.

## [Going Beyond Neural Network Feature Similarity: The Network Feature Complexity and Its Interpretation Using Category Theory (ICLR 2024)](https://openreview.net/pdf?id=4bSQ3lsfEV)

### Results

#### ResNet-50 (V1 ckpt.) on Imagenet - percentage of parameters petained vs. accuracy (extracted from plot)

Gathered from plot:
| Percentage of Parameters Retained | Accuracy (%)         |
|-----------------------------------|----------------------|
| 100.00                            | 76.13                |
| 94.35                             | 74.77                |
| 83.88                             | 69.20                |
| 79.58                             | 65.68                |
| 64.02                             | 27.16                |
| 59.29                             | 21.70                |

Transformed:
| $\Delta$ Params (%) | $\Delta$ Acc (pp) |
|---------------------|-------------------|
| 0.00                | 0.00              |
| -5.65               | -1.36             |
| -16.12              | -6.94             |
| -20.42              | -10.45            |
| -35.98              | -48.97            |
| -40.71              | -54.43            |

### Notes

The method is called "IFM" in the paper. FLOPs were not reported for this model.



## [TVSprune - Pruning non-discriminative filters via total variation separability of intermediate representations without fine tuning](https://openreview.net/pdf?id=sZI1Oj9KBKy)

### Results

These were available in tables in the paper.

| Model      | Dataset   | $\Delta$ Params (%) | IterTVSPrune $\Delta$ Acc (pp) | CHIP $\Delta$ Acc (pp) |
|------------|-----------|---------------------|-------------------------------|------------------------|
| ResNet-50  | ImageNet  | -4.76               | -3.02                         | -3.41                  |
| ResNet-50  | ImageNet  | -9.98               | -10.21                        | -10.08                 |
| ResNet-50  | ImageNet  | -24.65              | -31.30                        | -34.20                 |
| ResNet-56  | CIFAR-10  | -3.5                | -1.47                         | -1.36                  |
| ResNet-56  | CIFAR-10  | -7.6                | -4.82                         | -5.56                  |
| ResNet-56  | CIFAR-10  | -12.3               | -9.86                         | -10.22                 |
| ResNet-20  | CIFAR-10  | -2.4                | -1.4                          | -                      |
| ResNet-20  | CIFAR-10  | -4.8                | -4.7                          | -                      |
| ResNet-20  | CIFAR-10  | -7.1                | -9.41                         | -                      |

### Notes
Their main method is called IterTVSPrune. They also report results for CHIP in a fair setting (no fine-tuning), but the results are not better than theirs. FLOPs were not reported.


## [DFPC: Data Flow Driven Pruning of Coupled Channels Without Data (ICLR 2024)](https://openreview.net/pdf?id=mhnHqRqcjYU)

### Results


#### ResNet-50 on ImageNet - percentage of parameters petained vs. accuracy (extracted from plot)

Gathered from plot:
| Params Retained (%) | Top-1 Acc (%) |
|---------------------|---------------|
| 100.00              | 76.13         |
| 94.34               | 70.35         |
| 89.08               | 62.25         |
| 86.34               | 51.93         |
| 80.87               | 36.34         |
| 75.24               | 21.31         |

Transformed:
| $\Delta$ Params (%) | $\Delta$ Acc (pp) |
|---------------------|-------------------|
| 0.00                | 0.00              |
| -5.66               | -5.78             |
| -10.92              | -13.88            |
| -13.66              | -24.20            |
| -19.13              | -39.79            |
| -24.76              | -54.82            |




## [GTP-ViT: Efficient Vision Transformers via Graph-based Token Propagation](https://openaccess.thecvf.com/content/WACV2024/papers/Xu_GTP-ViT_Efficient_Vision_Transformers_via_Graph-Based_Token_Propagation_WACV_2024_paper.pdf)

### Results

#### DeiT-B on ImageNet - Top-1 accuracy vs. GMACs (extracted from plot)

Table copied from paper:

| Method     | #Param | 15.3 GMACs | 13.1 GMACs | 11.6 GMACs | 9.8 GMACs | 8.8 GMACs |
|------------|--------|------------|------------|------------|-----------|-----------|
| DyViT | 89.5M  | 79.9       | 77.7       | 75.5       | 69.5      | 50.2      |
| EViT  | 86.6M  | 81.7       | 81.3       | 80.5       | 78.7      | 75.1      |
| Evo-ViT | 86.6M | 81.5       | 80.9       | 79.1       | 78.5      | 60.6      |
| Tri-Level | 86.6M | 64.6     | 64.6       | 64.6       | 64.6      | 64.6      |
| ToMe   | 86.6M  | 81.6       | 81.2       | 80.6       | 79.5      | 77.8      |
| ATS   | 86.6M  | 81.8       | 81.7       | 81.4       | 80.5      | 79.6      |
| GTP | 86.6M | 81.8 | 81.5 | 80.9 | 80.0 | 78.3 |

Transformed (only for GTP):
| GMACs | $\Delta$ FLOPs (%) | Top-1 Acc (%) | $\Delta$ Top-1 (pp) |
|-------|--------------------|---------------|---------------------|
| 17.6 | 0.0                | 81.8          | 0.0                 |
| 15.3 | -13.07             | 81.8          | 0.0                 |
| 13.1 | -25.57             | 81.5          | -0.3                |
| 11.6 | -34.09             | 80.9          | -0.9                |
| 9.8  | -44.32             | 80.0          | -1.8                |
| 8.8  | -50.00             | 78.3          | -3.5                |


### Notes
Token merging does not reduce the number of parameters, so all methods except DyViT have the same number of parameters. Reduction in GMACs is proportional to reduction in FLOPs.
Base accuracy of DeiT-B is 81.8%.


## Compressing Neural Networks: Towards Determining the Optimal Layer-wise Decomposition (https://openreview.net/pdf?id=BvJkwMhyInm)


### Results on ResNet-20, CIFAR-10

Gathered from plot in paper.

| $\Delta$ Params (%)         | $\Delta$  Acc (%)                |
|---------------------|--------------------------|
| -9.88                | 0.09                     |
| -17.44               | -0.40                    |
| -23.71               | -1.50                    |
| -30.05               | -3.37                    |
| -35.74               | -5.24                    |
| -40.87               | -7.42                    |
| -45.26               | -10.52                   |
| -50.00               | -25.22                   |
| -53.91               | -31.73                   |
| -57.50               | -43.96                   |
| -61.00               | -45.67                   |
| -64.18               | -54.95                   |
| -66.95               | -56.66                   |
| -69.15               | -62.70                   |
| -71.84               | -66.88                   |

#### Notes
The method is called ALDS. FLOPs reductions were not reported.

## [Feature Variance Ratio-Guided Channel Pruning for Deep Convolutional Network Acceleration Results](https://openaccess.thecvf.com/content/ACCV2020/papers/He_Feature_Variance_Ratio-Guided_Channel_Pruning_for_Deep_Convolutional_Network_Acceleration_ACCV_2020_paper.pdf?utm_source=chatgpt.com)

ResNet-20, CIFAR-10.

| $\Delta$ FLOPs (%)           | Accuracy (%)           |
|-----------------------|-----------------------|
| -0.19                 | 92.50                 |
| -5.04                  | 92.50                 |
| -12.29                 | 91.25                 |
| -19.84                 | 90.00                 |
| -29.51                 | 86.67                 |
| -39.38                 | 80.14                 |
| -49.35                 | 67.64                 |
| -54.00                 | 60.42                 |
| -59.81                 | 54.17                 |
| -70.56                 | 37.92                 |
| -79.86                 | 25.28                 |


Transformed data: FLOPs delta and acc delta

| Δ FLOPs (%)           | Δ Accuracy (%)        |
|-----------------------|----------------------|
| -0.19                 | 0.00                 |
| -5.04                  | 0.00                 |
| -12.29                 | -1.25                |
| -19.84                 | -2.50                |
| -29.51                 | -5.83                |
| -39.38                 | -12.36               |
| -49.35                 | -24.86               |
| -54.00                 | -32.08               |
| -59.81                 | -38.33               |
| -70.56                 | -54.58               |
| -79.86                 | -67.22               |


#### Notes
Method is called SFVR.

## [SVD-NAS: Coupling Low-Rank Approximation and Neural Architecture Search](https://openaccess.thecvf.com/content/WACV2023/papers/Yu_SVD-NAS_Coupling_Low-Rank_Approximation_and_Neural_Architecture_Search_WACV_2023_paper.pdf)

| Model            | Method    | $\Delta$ FLOPs (%) | $\Delta$ Params (%) | $\Delta$ Top-1 (pp) |
|------------------|-----------|--------------------|---------------------|---------------------|
| ResNet-18        | SVD-NAS   | -58.60             | -68.05              | -13.35              |
| ResNet-18        | LR-S2     | -56.49             | -57.91              | -38.13              |
| ResNet-18        | ALDS      | -42.31             | -65.14              | -18.70              |
| ResNet-18        | F-Group   | -42.31             | -10.66              | -69.34              |
| MobileNetV2      | SVD-NAS   | -12.54             | -9.00               | -15.09              |
| MobileNetV2      | LR-S2     | -3.81              | -6.24               | -17.46              |
| MobileNetV2      | ALDS      | -2.62              | -37.61              | -16.95              |
| EfficientNet-B0  | SVD-NAS   | -22.17             | -16.41              | -10.11              |
| EfficientNet-B0  | LR-S2     | -18.73             | -14.56              | -22.08              |
| EfficientNet-B0  | ALDS      | -7.65              | -10.02              | -16.88              |


## [Training-Free Restoration of Pruned Neural Networks](https://arxiv.org/pdf/2502.08474)

### Results on MobileNet-V2 (Base acc. 71.88) - L2-norm (their method only (marked as "ours"))

| $\Delta$ filters (%)  | Accuracy | $\Delta$ Accuracy |
|---------------|----------|------------|
| -5            | 67.46    | -4.42      |
| -10           | 53.41    | -18.47     |

### Notes
We call the method L2+REST in our paper. Moreover, the reported delta is reduction on number of filters, not FLOPs or parameters, but we take it to be approximately the number of parameters.


## [Dense Vision Transformer Compression with Few Samples](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Dense_Vision_Transformer_Compression_with_Few_Samples_CVPR_2024_paper.pdf)

Called in the paper
| Model  | Method                    | $\Delta$ FLOPs (%) | Top-1 Acc (%) | $\Delta$ Top-1 (pp)|
|--------|---------------------------|----------------|---------------|---------|
| DeiT-B | Original                  | -              | 81.80         | -       |
| DeiT-B | PRACTISE (impl. by DC-ViT)| -16.6          | 79.30         | -2.50   |
| DeiT-B | DC-ViT               | -16.6          | 81.26         | -0.54   |

### Notes

Called DC-ViT.