# CNN Lab：共用操作說明與各環境啟動方式

CNN Lab 將模型介紹、訓練曲線、分類表現與實驗比較放在同一個本機介面。GitHub 本身不會執行這個 Python 網站。

> 專案提供環境清單與設定範例，不提供照片或權重。先看 [安裝與驗證](../docs/setup.md)。本機後端目前以 macOS 為驗證平台；原生 Windows 不支援，Linux／WSL2 與 Docker 尚待實測。

## 展示模式與本機模式

- 根目錄 `index.html` 是無權重展示入口，讀取 `results/showcase.json`；可以切模型、參數、觀察節點與混淆矩陣。使用者只需瀏覽器，部署端只需要靜態檔案。
- 靜態網站的「用這組設定在本機訓練」會顯示啟動指引，不會嘗試使用訪客GPU或保存權重。根目錄若直接以 file:// 開啟可能無法讀取JSON；請使用一般靜態網站伺服器。
- 啟動本機 `tools.web_app` 後，同一介面能切換「專案展示結果／這台電腦的實驗」。本機環境與分類圖片備齊後，可用所選設定從零訓練，結果保存在自己的電腦。未備齊時按鈕會停用。
- 靜態展示不含原始照片，混淆矩陣可查看數值，不提供誤判圖片。
- 目前展示仍是快照；未完成、失敗與最後測試集成績不包入。更新方式見 [展示數值說明](../results/README.md)。

## 共用操作

| 頁籤 | 用途 |
|---|---|
| 優化器教學 | SGD、RMSprop、Adam、AdamW 的原理、設定與公平比較；`/#optimizers` 可直接開啟。 |
| 模型介紹 | 十種模型架構圖、核心特色與論文差異；不需要權重。 |
| 訓練曲線 | 查看同一組訓練前 1／5／10 輪的 Loss 或 Accuracy。 |
| 分類表現 | 最佳權重的驗證混淆矩陣與各類別 Precision、Recall、F1。 |
| 實驗比較 | 篩選模型、優化器、學習率，列出完整完成 10 輪的結果。 |

1. 先準備同樣六類的 `data_animal_pictures/train`、`val`，兩組資料不可重複；分類順序由訓練資料決定並檢查一致性。照片未隨 repo 發布。
2. 選模型、學習率與優化器，按「開始訓練」並確認。沒有舊權重也能從零開始；開網頁本身不建立訓練，但若已有 queued 排程，服務會自動接續。
3. 每組固定 Batch 32、10 輪；1／5／10 是觀察與保存節點。介面只評估驗證集，測試集留到最後。
4. 取消會取消同一整批待辦，當前工作於批次間停止；已寫入檔案保留。關閉瀏覽器不會停止背景訓練。

詳細條件及計分、耗時限制見 [實驗方法](../docs/experiment-method.md)。截圖是個別驗證結果，不是最終排名。

## Mac：本機 Python＋Apple Metal

這是目前開發與訓練使用的方式，Metal 需要適用的 TensorFlow 與 tensorflow-metal 環境。提供 `requirements.txt` 與選用的 `requirements-metal.txt`。建立環境不會自動完成 Lab 的 GPU 派工驗證；首版操作流程先使用 CPU。

完整環境準備後，在專案根目錄執行：

```sh
python -B -m tools.web_app
```

開啟 http://127.0.0.1:8765/ ，「模型介紹」可直接用 http://127.0.0.1:8765/#models 。新安裝不帶本機 GPU 驗證紀錄，預設由啟動服務的 Python 環境執行。開發環境另支援 `.venv-macos-metal` 派工；請勿複製他人的裝置驗證紀錄或自行把驗證開關設為 true。是否使用 GPU 以裝置檢查與工作紀錄為準。

## NVIDIA：Docker（尚待實測）

目標是 Linux 或 Windows WSL2＋相容 NVIDIA GPU。需相容驅動、Docker Compose 與容器 GPU 支援。**Windows／Linux NVIDIA 建置與完整訓練流程尚未實測**；Docker 檔案仍在本機準備稿，尚未加入本 repo。

待 Docker 包裝與相關設定完成審核上傳後，預定執行：

```sh
docker compose up --build
```

預定開啟 http://127.0.0.1:8766/ ，模型介紹同樣位於 `/#models`。Docker 只改啟動方式，頁籤與參數操作相同。Mac Metal 不使用這份 NVIDIA 容器。

## 結果放哪裡、怎麼停止

| 環境 | 紀錄與權重 |
|---|---|
| Mac 本機 | `logs/macos_ui/`、`weights/macos_ui/` |
| Docker 準備稿 | 外部掛載的 `docker-results/logs/`、`docker-results/weights/`；容器內沿用 `macos_ui` 子目錄名稱，環境標示 CUDA。 |

Docker 預定以 `docker compose down` 停止。停止容器、關機或結束訓練程序會失去記憶體中的當前狀態；已保存權重與紀錄仍在，但尚不提供完整 optimizer／亂數狀態的斷點續訓。

## 不啟動服務也能閱讀模型

下載完整專案，直接用瀏覽器開啟 [docs/models.html](../docs/models.html)，保留旁邊的 `images/` 與 `webs/style.css`。這是模型介紹的離線入口，Windows、Mac、Linux 都不需要 Python 即可閱讀；訓練功能仍需服務與環境。
