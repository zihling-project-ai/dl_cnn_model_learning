# NVIDIA Docker（待硬體實測）

> [!WARNING]
> 已提供建置檔與靜態檢查，尚未在 NVIDIA 電腦上完成映像建置、服務啟動或模型訓練實測。不能宣稱 Docker 已解決 Inception 異常。此容器不支援 Mac Metal。

## 需要什麼

- Linux x86-64＋相容 NVIDIA GPU，或 Windows WSL2 的 NVIDIA GPU 環境。
- 相容 NVIDIA 驅動、Docker Engine／Desktop、Docker Compose v2，及容器 GPU 支援。
- 自己有權使用、已分好 train／val／test 和六類子資料夾的圖片。參考 [README](../README.md#五資料夾內容)。

容器基於 `tensorflow/tensorflow:2.15.0-gpu`。`platform: linux/amd64` 指定映像架構，不會把 Mac GPU 變成 NVIDIA GPU。環境需求參考 [TensorFlow 安裝說明](https://www.tensorflow.org/install/pip)、[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。這些文件以各自現行版本為主，這份固定 2.15 設定仍須實測。

## 從專案根目錄啟動

先將自己的圖片放在 `data_animal_pictures/`，再執行：

```sh
docker compose config
docker compose up --build
```

預定開啟 http://127.0.0.1:8766/ 。首次使用空的輸出資料夾時，只開介面、不訓練；確認開始後才訓練。若重用的 logs 有 queued 排程，服務會自動接續，啟動前先確認工作狀態。

啟動工具會檢查 CUDA 裝置及小矩陣計算；沒有可用 GPU 就停止，不會靜默改成 CPU。**裝置檢查通過不等於十種模型都已驗證。** runtime 的 dispatch_models 僅代表允許派工，不是通過模型測試的清單。

## 資料怎麼保存

| 主機 | 容器內 | 用途 |
|---|---|---|
| data_animal_pictures/ | /app/data_animal_pictures/ | 唯讀照片，不打包進映像 |
| docker-results/logs/ | /app/logs/ | 紀錄與本機 runtime |
| docker-results/weights/ | /app/weights/ | 新訓練權重 |

程式使用 `tools/`、`webs/`；不依賴開發者的 Mac 虛擬環境。Compose 將介面只綁在本機 127.0.0.1:8766；不要直接把訓練 API 當成公開網路服務。

停止服務：

```sh
docker compose down
```

已寫入主機的結果保留；記憶體中的當前訓練狀態不保留。目前沒有完整 optimizer／亂數狀態的斷點續訓。容器設定為不自動重啟。

## 尚待驗證的清單

- 實際拉取基礎映像與安裝固定依賴。
- NVIDIA GPU 偵測、介面連線、資料唯讀與輸出保存。
- 用少量自己的圖片完成訓練與驗證，核對數值範圍。
- Inception 與 CPU 固定輸入對照；不以「沒報錯」代替正確性檢查。
- Windows WSL2 的檔案權限、掛載與停止／重啟行為。
