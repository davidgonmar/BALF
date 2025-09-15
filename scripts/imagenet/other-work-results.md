
Unless noted, methods do not use fine-tuning

## GOING BEYOND NEURAL NETWORK FEATURE SIMILARITY: THE NETWORK FEATURE COMPLEXITY AND ITS INTERPRETATION USING CATEGORY THEORY (ICLR 2024)

https://openreview.net/pdf?id=4bSQ3lsfEV
method requires no data both before and after pruning (paper says that)

### ResNet 50 (V1 ckpt) parameters retained vs. accuracy
99,59386281588448; 76,39836289222373
94,31407942238268; 74,7612551159618
92,52707581227438; 74,10641200545702
83,75451263537906; 69,08594815825376
79,53068592057762; 65,81173260572987
64,01624548736461; 27,17598908594816
59,305054151624546; 21,609822646657577
56,29963898916968; 15,934515688949531
52,238267148014444; 12,551159618008185
41,75992779783394; 2,1828103683492515
24,945848375451263; 0,10914051841747607

Seems like a method that does not use expensive search.

The method is called "IFM" in the paper.

## TVSPRUNE - PRUNING NON-DISCRIMINATIVE FILTERS VIA TOTAL VARIATION SEPARABILITY OF INTERMEDIATE REPRESENTATIONS WITHOUT FINE TUNING

https://openreview.net/pdf?id=sZI1Oj9KBKy

Results are without fine-tuning, method is called ITERTVSPRUNE. They also provide results about "CHIP (in the original paper, CHIP used finetuning, but they re-evaluate it without finetuning here).

| Model     | Dataset  | Param. Sparsity | ITERTVSPRUNE | CHIP   |
|-----------|----------|-----------------|--------------|--------|
| ResNet50  | ImageNet | 4.76%           | -3.02%       | -3.41% |
|           |          | 9.98%           | -10.21%      | -10.08% |
|           |          | 24.65%          | -31.3%       | -34.2% |
| ResNet56  | CIFAR10  | 3.5%            | -1.47%       | -1.36% |
|           |          | 7.6%            | -4.82%       | -5.56% |
|           |          | 12.3%           | -9.86%       | -10.22% |
| ResNet20  | CIFAR10  | 2.4%            | 1.4%         | -      |
|           |          | 4.8%            | 4.7%         | -      |
|           |          | 7.1%            | 9.41%        | -      |

Requires access to dataset distribution, which in practice means access to data. But they split the test dataset for that.

They also have VGG baselines.


## DFPC: DATA FLOW DRIVEN PRUNING OF COUPLED CHANNELS WITHOUT DATA

https://openreview.net/pdf?id=mhnHqRqcjYU

For ResNet50, params and compression

0,8908221685401607; 62,250556644956205
0,9433589836529868; 70,35107467935225
0,8634159876986531; 51,926070345513864
0,808655406941796; 36,34127232489605
0,7523667260271197; 21,30827886955484

Does not require data.



# VIT RESULTS FROM https://openaccess.thecvf.com/content/WACV2024/papers/Xu_GTP-ViT_Efficient_Vision_Transformers_via_Graph-Based_Token_Propagation_WACV_2024_paper.pdf

| Method     | #Param | 15.3 GMACs | 13.1 GMACs | 11.6 GMACs | 9.8 GMACs | 8.8 GMACs |
|------------|--------|------------|------------|------------|-----------|-----------|
| DyViT [32] | 89.5M  | 79.9       | 77.7       | 75.5       | 69.5      | 50.2      |
| EViT [24]  | 86.6M  | 81.7       | 81.3       | 80.5       | 78.7      | 75.1      |
| Evo-ViT [40] | 86.6M | 81.5       | 80.9       | 79.1       | 78.5      | 60.6      |
| Tri-Level [21] | 86.6M | 64.6     | 64.6       | 64.6       | 64.6      | 64.6      |
| ToMe [1]   | 86.6M  | 81.6       | 81.2       | 80.6       | 79.5      | 77.8      |
| ATS [17]   | 86.6M  | 81.8       | 81.7       | 81.4       | 80.5      | 79.6      |
| **GTP (ours)** | **86.6M** | **81.8** | **81.5** | **80.9** | **80.0** | **78.3** |



# ALDS results (https://openreview.net/pdf?id=BvJkwMhyInm)

On ResNet-20, CIFAR-10
These are params percentage elimination and accuracy delta

9,88220420963669; 0,08514264361274648
17,44383484143143; -0,3982636209021866
23,705486077885613; -1,4968761240889261
30,049486105824126; -3,3688967427176344
35,74307735827311; -5,239241050069307
40,86798000998802; -7,41739801566651
45,262430022734975; -10,52150742291586
49,997799841448895; -25,224433633788834
53,90905312859054; -31,729359544322705
57,5028549676437; -43,955116765557385
61,00110706390588; -45,665163807042596
64,18442217899512; -54,95171874290624
66,95096440283157; -56,65987993420478
69,1542660375843; -62,69648638171702
71,84285978703863; -66,87867347900945

# SVPR Results (https://openaccess.thecvf.com/content/ACCV2020/papers/He_Feature_Variance_Ratio-Guided_Channel_Pruning_for_Deep_Convolutional_Network_Acceleration_ACCV_2020_paper.pdf?utm_source=chatgpt.com)

ResNet-20, CIFAR-10, no FT. These are FLOPs percentage elimination and abs accuracy.
-0,1863045347087926; 92,5
5,036132602621958; 92,5
12,290726413066839; 91,25
19,835455620030086; 90
39,3807758435418; 80,13888888888889
54,003331184182244; 60,416666666666664
59,81208360197721; 54,166666666666664
70,56280894046851; 37,91666666666666
79,8593649258543; 25,27777777777777
29,50985923060392; 86,66666666666666
49,35418009886094; 67,63888888888889

Transformed data: FLOPs delta and acc delta
-0,1863045347087926; 0
5,036132602621958; 0
12,290726413066839; -1,25
19,835455620030086; -2,5
39,3807758435418; -12,361111111111114
54,003331184182244; -32,083333333333336
59,81208360197721; -38,333333333333336
70,56280894046851; -54,583333333333343
79,8593649258543; -67,222222222222229
29,50985923060392; -5,833333333333343
49,35418009886094; -24,861111111111114


# SVD-NAS

| Model         | Method        | Δ FLOPs (%) | Δ Params (%) | Δ Top-1 (pp) | Δ Top-5 (pp) |
|---------------|---------------|-------------|--------------|--------------|--------------|
| ResNet-18     | SVD-NAS       | -58.60      | -68.05       | -13.35       | -9.14        |
| ResNet-18     | ALDS [18]     | -42.31      | -65.14       | -18.70       | -13.38       |
| ResNet-18     | LR-S2 [8]     | -56.49      | -57.91       | -38.13       | -33.93       |
| ResNet-18     | F-Group [21]  | -42.31      | -10.66       | -69.34       | -87.63       |
| MobileNetV2   | SVD-NAS       | -12.54      | -9.00        | -15.09       | -7.79        |
| MobileNetV2   | ALDS [18]     | -2.62       | -37.61       | -16.95       | -10.91       |
| MobileNetV2   | LR-S2 [8]     | -3.81       | -6.24        | -17.46       | -10.34       |
| EfficientNet-B0 | SVD-NAS     | -22.17      | -16.41       | -10.11       | -5.49        |
| EfficientNet-B0 | ALDS [18]   | -7.65       | -10.02       | -16.88       | -9.96        |
| EfficientNet-B0 | LR-S2 [8]   | -18.73      | -14.56       | -22.08       | -14.15       |


# Training-Free Restoration of Pruned Neural Networks

### MobileNet-V2 (Acc. 71.88) — L2-norm (Ours only)

| Pruning Ratio | Accuracy | Δ Accuracy |
|---------------|----------|------------|
| 5%            | 67.46    | -4.42      |
| 10%           | 53.41    | -18.47     |