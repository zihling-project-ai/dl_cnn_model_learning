# 模型特色與論文架構差異

這裡練習的是各模型的核心設計。部分實作使用較小的分類器或簡化的層配置，保留模型特色不代表逐層等同原論文。下表只比較架構與層的計算方式；作業系統、資料集、Optimizer、學習率與訓練輪數不同，不列為架構差異。

分類輸出依任務改為六類，另列為任務適配；縮小隱藏全連接層、刪除層或改變通道連接，仍會明確標示。

| 模型／設定名稱 | 保留的模型特色 | 本專案與論文／參考實作的主要差異 | 實作定位 |
| --- | --- | --- | --- |
| [LeNet](../models/lenet.py) · `lenet` | 小型卷積、平均池化、Tanh、全連接分類器。 | 原版 C3 為部分通道連接，目前使用一般 Conv2D；原版 subsampling 含可訓練縮放與偏置，目前為固定平均池化；C5 改用 Flatten＋Dense(120)，在目前輸入尺寸下不等同原卷積；RBF 輸出改為 Softmax。[論文](https://princetonuniversity.github.io/NEU-PSY-502/_static/pdf/Class%204/LeCunn1998.pdf) | 簡化 LeNet-5，保留作為入門基準。 |
| [AlexNet](../models/alexnet.py) · `alexnet` | 大卷積核、ReLU、重疊最大池化、Dropout。 | 隱藏全連接層為 512／512，原版為 4096／4096；原版第 2、4、5 層的分組連接改為一般卷積；省略 LRN。[論文](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) | 簡化 AlexNet，無 LRN 的對照版本。 |
| [AlexNet＋LRN](../models/alexnet_lrn.py) · `alexnet_lrn` | AlexNet 的卷積與 Dropout，加上局部反應正規化 LRN。 | 同樣縮小為 512／512，且未使用原版分組連接。LRN 的 bias=1 與論文 k=2 不同，這一項屬層內超參數差異。[論文](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) | 含 LRN 的簡化 AlexNet，縮小分類器。 |
| [VGG16](../models/vgg16.py) · `vgg16` | 五個卷積區塊，連續堆疊 3×3 卷積；13 層卷積＋3 層全連接。 | 卷積配置對應原版 configuration D；兩個隱藏全連接層縮為 256／256，原版為 4096／4096。[論文 Table 1](https://arxiv.org/html/1409.1556v6) | VGG16 主幹，縮小分類器。 |
| [VGG16＋BN](../models/vgg16_bn.py) · `vgg16_bn` | VGG 配置加上 Conv→BN→ReLU。 | 隱藏全連接層為 512／512。BN 是原 VGG 的擴充設計，並非 2014 VGG 論文原版。[VGG 論文](https://arxiv.org/html/1409.1556v6) | VGG16-BN 變體，縮小分類器。 |
| [GoogLeNet](../models/googlenet.py) · `googlenet` | Inception v1 的多分支、1×1 降維、分支串接、全域平均池化；可開關兩個輔助分類器。 | 主要 Inception 配置大致對應，但前段省略原版兩個 LRN 層；池化的空間尺寸隨輸入尺寸調整。[論文](https://arxiv.org/pdf/1409.4842) | GoogLeNet／Inception v1 風格實作，省略 LRN。 |
| [Inception v2](../models/inception.py) · `inception` | 多分支、1×1 降維、兩個 3×3、BN、分支串接，以及多分支降採樣。 | 對齊 TF-Slim 的可分離卷積前段、八個一般區塊與兩個 Reduction，共十個模組；保留本專案 GAP、六類輸出、BN／初始化與 Dropout 強度，不宣稱逐點重現原訓練。[參考實作](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v2.py) | Inception v2／BN-Inception 拓樸的 Keras 學習實作。 |
| [ResNet50](../models/resnet.py) · `resnet50` | 1×1→3×3→1×1 Bottleneck、殘差相加、投影捷徑，區塊數 3／4／6／3。 | 主幹配置對應 ResNet50；第一層使用 TensorFlow same 補邊，與原作者固定 pad=3 的邊界取樣可能不同，不宣稱逐點數值等價。[論文](https://arxiv.org/pdf/1512.03385)、[原作者實作](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-50-deploy.prototxt) | 保留 ResNet50 核心架構。 |
| [MobileNet v1](../models/mobilenet_v1.py) · `mobilenet_v1` | Depthwise＋Pointwise、13 組通道與步幅配置、ReLU6、全域平均池化。 | 主幹配置對應 v1；ReLU6 亦見於官方實作，不是架構錯誤。目前 Dropout=0.2 與官方預設強度不同，主要屬正則化設定差異。[論文](https://arxiv.org/pdf/1704.04861)、[官方實作](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) | 保留 MobileNet v1 核心架構。 |
| [EfficientNet-B0](../models/efficientnet.py) · `efficientnet_b0` | MBConv、SE、條件式殘差，以及 B0 的階段通道、重複數與卷積尺寸。 | SE 縮減寬度依區塊輸入通道計算；殘差支路使用逐樣本遮罩，丟棄率依 block_idx／total_blocks 增加。保留本專案 BN 與初始化設定，不要求原論文環境或訓練方法。[參考實作](https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py) | 對齊 B0 核心配置與 SE／支路丟棄方式。 |

