
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

For ResNet50

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