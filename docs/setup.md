# 安裝與驗證

## 只看展示結果：不需要 TensorFlow

在專案根目錄執行：

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Windows 可使用 `py -m http.server 8000 --bind 127.0.0.1`。開啟 http://127.0.0.1:8000/ ，可篩選模型與參數、查看曲線和驗證混淆矩陣。這是已保存的數值快照；不會啟動訓練，也不會使用訪客 GPU。直接雙擊 HTML 可能無法讀取 JSON。

## 自己訓練：Python 3.11＋Keras 2

本專案使用 TensorFlow 2.15／Keras 2.15 API，不適合直接換成 Keras 3。首版以 macOS Apple Silicon 驗證；Linux／WSL2 尚待實測。後端使用 Unix 的 `fcntl`，**原生 Windows 暫不支援本機 Lab**；展示模式不受影響。

在專案根目錄建立環境（macOS／Linux）：

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt` 固定主要套件版本，不是完整的跨平台鎖檔；硬體與間接依賴仍可能不同。請使用 Python 3.11，不要使用系統最新 Python 直接安裝。此流程不下載訓練照片。

1. 按 [README 第五節](../README.md#五資料夾內容) 放好自己的六類照片。
2. 命令列使用 `config.yaml`；建議複製為 `config_user.yaml` 再修改。執行 `python main_train.py --config config_user.yaml` 才會開始訓練。
3. 網頁使用 `config_lab.yaml`，執行 `python -B -m tools.web_app`，開啟 http://127.0.0.1:8765/ 。資料未備妥時不能開始訓練；新副本沒有待辦工作，啟動服務本身不訓練。
4. 權重與紀錄留在自己的 `weights/`、`logs/`，不會自動上傳；停止服務按 Ctrl+C。若已開始訓練，先在介面取消工作，關閉頁面不會停止背景工作。

目前有一張圖片的路徑就能讀圖，不代表資料品質通過檢查；請自行確保 train／val／test 沒有重複或同源洩漏。

## 選用功能

- 輔助標記：`python -m pip install -r requirements-labeling.txt`。第一次實際使用標記工具會下載外部預訓練模型，已分類照片不需要此功能。
- Apple Metal：`python -m pip install -r requirements-metal.txt`，只適用相容 Mac。可執行 `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"` 檢查裝置。這份安裝清單不代表所有模型 GPU 訓練均通過驗證；Lab 另有固定環境派工機制，新安裝不附個人 runtime 紀錄。優先以 CPU 完成啟動檢查，再驗證 GPU。
- NVIDIA Docker：尚待實測，已提供 [Docker 設定與說明](../docker/README.md)；不要將上述 Mac 外掛裝進 NVIDIA 容器。

平台參考：[TensorFlow 安裝文件](https://www.tensorflow.org/install/pip)、[Apple Metal 外掛文件](https://developer.apple.com/metal/tensorflow-plugin/)。官方文件可能以新版為主，本專案仍使用上方固定版本。

## 不訓練的檢查

```sh
python main_train.py --help
python main_test.py --help
python -m tools.web_app --help
python -m unittest discover -s tests -p test_showcase.py -v
```

完整測試需要安裝 `requirements-labeling.txt`，再執行 `python -B -m unittest discover -s tests -v`。測試使用暫存資料與模擬工作，不執行正式十輪訓練。展示快照、CPU／GPU標示、暫停耗時及尚未完成的實驗限制見 [實驗方法](experiment-method.md)。

驗證結果另見 [發布檢查紀錄](release-check.md)。
