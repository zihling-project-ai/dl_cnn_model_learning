# 架構圖核對與來源

這次檢查十張圖。舊圖多為本專案學習版，不能當作原論文架構；Inception 舊圖尚有過期的六個一般區塊配置。本輪只調整文件／圖示，不修改模型或訓練。

| 模型 | 圖示基準 | 舊圖與本輪修正重點 |
|---|---|---|
| LeNet-5 | [LeCun et al. (1998) · Figure 2 / Section II-B](https://princetonuniversity.github.io/NEU-PSY-502/_static/pdf/Class%204/LeCunn1998.pdf) | 恢復單通道32×32、可訓練 subsampling、C3部分連接、C5卷積與RBF輸出。 |
| AlexNet：原論文基準 | [Krizhevsky et al. (2012) · Figure 2 / Section 3.5](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) | 隱藏FC改回4096／4096、1000類；恢復LRN與分組連接的說明。 |
| AlexNet：LRN 參考基準 | [Krizhevsky et al. (2012) · Figure 2 / Section 3](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) | 同原論文AlexNet基準；LRN並非本專案新發明，另列bias等差異。 |
| VGG16 · Configuration D | [Simonyan & Zisserman · Table 1](https://arxiv.org/abs/1409.1556) | 分類器恢復4096／4096／1000，採configuration D。 |
| VGG16＋BN：原 VGG 基準 | [VGG Table 1；BN 為另加的變體](https://arxiv.org/abs/1409.1556) | 使用原VGG16基準，明示原論文無BN；本專案FC寬度亦不同。 |
| GoogLeNet · Inception v1 | [Szegedy et al. · Table 1 / Figures 2–3](https://arxiv.org/abs/1409.4842) | 補前段LRN與1000類，標記4a、4d輔助輸出及訓練用途。 |
| Inception v2 · TF-Slim 參考 | [明確採官方 inception_v2.py；非所有 v2 同名版本](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v2.py) | 改成TF-Slim八個一般模組＋兩個Reduction，前段可分離卷積；不再沿用舊六模組圖。 |
| ResNet-50 · Bottleneck | [He et al. (2015) · Table 1 / Figure 5](https://arxiv.org/abs/1512.03385) | 標示原始224輸入與1000類，保留3／4／6／3及Bottleneck；邊界取樣另列差異。 |
| MobileNet v1 · α=1 | [Howard et al. (2017) · Table 1](https://arxiv.org/abs/1704.04861) | 依Table1的13組DW/PW及ReLU文字；另說明實作ReLU6、Dropout與六類。 |
| EfficientNet-B0 | [Tan & Le (2019) · Table 1；官方 MBConv](https://arxiv.org/abs/1905.11946) | 依Table1逐段列kernel、expand、repeat、stride；恢復224輸入及1000類。 |

以上為自行重繪的階段概覽，不是原論文圖片的複製，也不宣稱已重現原訓練。核心實作差異見 [差異表](model-differences.md)。
