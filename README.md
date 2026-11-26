# CNN 模型學習實驗室

## 一、專案核心目標

用六類動物影像實作經典 CNN，透過互動介面比較模型、學習率與優化器，練習解讀曲線和分類指標。模型保留各自核心特色，部分結構依學習任務簡化。

目前為分階段整理版本：程式與部分文件已上傳；設定、依賴、資料與最終實驗結果仍待補齊。

## 二、CNN 基礎名詞介紹

| 名詞 | 簡單理解 |
|---|---|
| 卷積／Kernel | 用一組可學習的權重掃過局部區域，提取特徵。 |
| 特徵圖／Channel | 卷積產生的特徵表示；不同通道可學到不同模式。 |
| Stride／Padding | 控制移動步幅與邊界補值，影響輸出尺寸。 |
| Pooling／GAP | 彙整空間資訊；GAP 將每個通道取全域平均。 |
| ReLU／Swish | 加入非線性，讓模型能表達複雜關係。 |
| BN | 用訓練批次的統計量調整特徵；推論通常使用移動統計量。 |
| 殘差連接 | 將捷徑輸入與支路輸出相加，幫助深層網路學習。 |
| Depthwise／Pointwise | 分別處理各通道的空間特徵，再用 1×1 卷積混合通道。 |
| SE | 學習各通道的權重，調整特徵的重要程度。 |
| Epoch／Batch／學習率 | 看完一次訓練集／每次更新使用的圖片數／權重更新的步伐。 |

## 三、本次模型

LeNet、AlexNet、AlexNet＋LRN、VGG16、VGG16＋BN、GoogLeNet、Inception v2、ResNet50、MobileNet v1、EfficientNet-B0。

- CNN Lab 的「模型介紹」頁籤整合架構圖與論文差異，沿用相同介面風格；啟動方式見 [網站說明](webs/README.md)。
- [模型介紹離線 HTML](docs/models.html)：下載專案後可直接用瀏覽器閱讀；GitHub 檔案頁會顯示原始碼。
- [模型特色與論文架構差異](docs/model-differences.md)：各實作保留與調整的部分；[架構圖核對與來源](docs/architecture-review.md)。

## 四、實驗方法

固定資料切分、160×160 RGB、Batch 32 與 Seed 42；每模型比較三種學習率與四種優化器，每組訓練 10 輪。使用最佳驗證權重觀察 Loss、Accuracy、F1、Recall 與混淆矩陣，測試集留到最後。

詳見 [實驗方法與目前限制](docs/experiment-method.md) · [本機網站說明](webs/README.md)。介面提供訓練設定、曲線、分類表現與實驗比較；目前沒有公開部署網站。

**訓練曲線**：觀察訓練與驗證 Accuracy 隨輪次的變化。

<img src="docs/images/cnn-lab-curves.jpg" alt="GoogLeNet 訓練與驗證 Accuracy 曲線" width="1000">

**實驗比較**：依模型、優化器與學習率篩選已完成的結果；圖中為 LeNet 篩選示例。

<img src="docs/images/cnn-lab-comparison.jpg" alt="LeNet 多組參數實驗比較與篩選" width="1000">

**分類表現**：混淆矩陣與六類 Precision、Recall、F1。

<img src="docs/images/cnn-lab-classification.jpg" alt="CNN Lab 訓練設定、混淆矩陣與六類分類指標" width="1000">

*實際介面截圖，呈現個別驗證結果，並非最終測試成績或模型排名。*

## 五、資料夾內容

| 檔案／資料夾 | 分工 |
|---|---|
| [main_train.py](main_train.py) | 訓練入口，讀設定、建立模型、保存最佳權重與紀錄。 |
| [main_test.py](main_test.py) | 評估入口，載入權重並計算分類指標。 |
| [models/](models/) | 十種 CNN 實作與統一模型入口。 |
| [tools/](tools/) | 資料、標記、設定與網站後端工具。 |
| [tests/](tests/) | 資料安全、架構及介面流程檢查；完整執行仍需補齊環境與設定。 |
| [webs/](webs/) | CNN Lab 前端與操作說明。 |
| [docs/](docs/) | 模型圖解、架構差異、實驗方法與截圖。 |
| [data_animal_pictures/](data_animal_pictures/) | 目前只有匿名資料統計，沒有照片。 |

| 工具 | 用途 |
|---|---|
| [data_safety.py](tools/data_safety.py) | 檢查分類資料操作，預覽並避免覆寫。 |
| [dataloader.py](tools/dataloader.py) | 讀取圖片、檢查類別、套用訓練增強。 |
| [labeling.py](tools/labeling.py)、[labeling_fast.py](tools/labeling_fast.py) | 使用現成模型輔助標記；結果仍需人工抽查。 |
| [main_fix_filenames.py](tools/main_fix_filenames.py) | 預覽與安全重新命名圖片。 |
| [utils.py](tools/utils.py) | 讀取設定、建立優化器、保存實驗資訊。 |
| [web_app.py](tools/web_app.py)、[web_worker.py](tools/web_worker.py) | 提供本機介面、派工、進度與驗證報告。 |
| [cuda_check.py](tools/cuda_check.py)、[run_model_checks.py](tools/run_model_checks.py) | 檢查裝置或模型執行狀況。 |
