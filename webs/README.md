# CNN Lab：共用操作說明與各環境啟動方式

CNN Lab 將模型介紹、訓練曲線、分類表現與實驗比較放在同一個本機介面。GitHub 本身不會執行這個 Python 網站。

> 目前 repo 是分階段上傳版本，尚未包含待審核的設定、完整依賴與照片。以下啟動方式是完整檔案備齊後的操作；Docker 尚待實測，不能視為已驗證的安裝流程。

## 共用操作

| 頁籤 | 用途 |
|---|---|
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

這是目前開發與訓練使用的方式，Metal 需要適用的 TensorFlow 與 tensorflow-metal 環境。環境檔及裝置驗證程序仍待整理上傳；只有建立 venv 不會自動啟用 GPU。

完整環境準備後，在專案根目錄執行：

```sh
.venv-macos/bin/python -B -m tools.web_app
```

開啟 http://127.0.0.1:8765/ ，「模型介紹」可直接用 http://127.0.0.1:8765/#models 。目前設計由 CPU 環境提供介面、已驗證的訓練工作派至 `.venv-macos-metal`；是否使用 GPU 以每組顯示的執行環境為準。

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
