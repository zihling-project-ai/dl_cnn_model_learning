# 優化器學習筆記：權重如何一步一步更新

## 先把三件事分開

- **Loss（損失）**：衡量目前預測與正確標籤的差距；越小通常表示模型對這批資料的預測更符合答案。
- **Gradient（梯度）**：Loss 對每個可訓練參數的變化率，指出局部往哪個方向移動會讓 Loss 上升。
- **Optimizer（優化器）**：利用梯度與過去的更新紀錄，決定怎麼更新權重。學習率控制更新尺度，不是分類的判斷門檻。

```text
照片＋標籤 → 預測 → 計算 Loss → 反向傳播求梯度
                                      ↓
                            優化器更新權重 → 下一批
```

調整優化器不會改變照片的正確標籤，也不會把貓改成狗；它改變模型如何學到分類規則。相同學習率用在不同優化器上，實際更新量未必相同。

## SGD：沿著當下梯度往下走

最簡單的更新（沒有 momentum、weight decay）：

```text
新權重 = 舊權重 − 學習率 × 梯度
```

假設一個權重是 1.0，梯度為 2.0：

| 學習率 | 一次更新後 |
|---|---|
| 0.001 | 0.998 |
| 0.0001 | 0.9998 |
| 0.00001 | 0.99998 |

這只是單一權重、單一步驟的示意，不能直接推算模型準確率。步伐太小可能十輪還沒學好；太大可能越過較好的區域或不穩定。

**Momentum（動量）**會保留過去更新方向的一部分，像推球時保留慣性；連續一致的方向能累積，遇到方向改變也不會立即完全忘掉過去。因此本專案的 SGD 不一定是最單純的 SGD。

## RMSprop：依各參數最近的梯度大小調整尺度

它累積梯度平方的移動平均，利用其平方根縮放當前梯度。近期梯度較大的參數，更新受到較多縮放；較小者受到的縮放不同。

- `rho`：控制平方梯度平均保留多少歷史。
- `epsilon`：放在分母中的小數值，幫助數值穩定；並非永遠可以忽略。
- 本專案也使用設定中的 momentum。

自適應縮放不代表學習率可以隨便設；仍需看訓練與驗證曲線。

## Adam：同時記方向與大小

Adam 維護梯度的移動平均與平方梯度的移動平均，並做早期的偏差修正，再據此更新各參數。

白話理解：一份紀錄估計最近常往哪裡走，另一份估計梯度的尺度。它常是實驗起點，但不能因此宣稱一定比 SGD 好，或十輪結果就代表模型最終能力。

## AdamW：Adam 加上分開處理的權重衰減

AdamW 在自適應梯度更新之外，額外讓權重往零縮小，稱為 **decoupled weight decay（解耦權重衰減）**。兩者分開處理，與直接把 L2 懲罰加入 Adam 的 Loss 不完全相同。

權重衰減有助限制權重成長，但不保證解決過擬合；太強也可能妨礙學習。此處談的是優化器的衰減，不等於模型內所有可能的正則化設定。

## 這份程式實際用了什麼

建立優化器的入口是 [tools/utils.py](../tools/utils.py) 的 `build_optimizer()`。以下對應已固定的 TensorFlow／Keras 2.15 環境：

| 優化器 | 由 YAML 控制 | 其他使用的值 |
|---|---|---|
| SGD | learning_rate、momentum、weight_decay | nesterov=False |
| RMSprop | learning_rate、momentum | rho=0.9、epsilon=0.001、centered=False |
| Adam | learning_rate | beta_1=0.9、beta_2=0.999、epsilon=0.0000001；程式不套用 YAML 的 weight_decay |
| AdamW | learning_rate、weight_decay | beta_1=0.9、beta_2=0.999、epsilon=0.0000001；不使用 YAML 的 momentum |

`momentum` 是 SGD／RMSprop 的設定，不會覆寫 Adam／AdamW 的 beta_1。Adam 雖然有梯度平均，也不是讀取同一個 momentum 欄位。

- `config.yaml` 是命令列入門範例，weight_decay=0，學習率=0.0001。
- `config_lab.yaml` 是介面基礎，momentum=0.9、weight_decay=0.0005；介面再覆寫模型、學習率與優化器。
- **目前 Lab 只有 SGD／AdamW 套用 weight_decay；Adam／RMSprop 不套用，非零時會提示。** 這是本專案的選擇，不表示 Keras 的其他優化器不能支援衰減。

## 如何公平比較與閱讀結果

1. 比學習率：固定模型、資料、優化器、batch、seed 與訓練預算，只改 learning_rate。
2. 比優化器：固定模型與資料，記錄 learning_rate、momentum、epsilon、weight_decay；不能只看優化器名稱。
3. 本專案比較的是「目前這四組優化器配方」。不同優化器的附加設定不同，**不能把分數差完全歸因於演算法本身**。
4. 若要專門研究演算法差別，可另設 weight_decay=0 的控制實驗；若要比較各方法可達到的表現，則給各方法相同調參預算，允許不同最佳學習率。
5. 看多輪 train／validation Loss、Accuracy、Macro F1 與逐類 Recall，低分也保留。高 Accuracy 但少數類別 Recall 很低，仍可能分類不均。
6. 最佳權重依十輪內的驗證 Accuracy 選出，並非依最高 F1 選出；不能各拿不同輪的最高指標拼成一筆結果。測試集留到選參數完成後，避免反覆使用測試成績挑參數。

目前只跑固定十輪、單一 seed 的參數格點；不能把這份結果當成所有優化器或模型的普遍排名。

## 官方參考

- [Keras 2：SGD](https://keras.io/2/api/optimizers/sgd/)
- [Keras 2：RMSprop](https://keras.io/2/api/optimizers/rmsprop/)
- [Keras 2：Adam](https://keras.io/2/api/optimizers/adam/)
- [Keras 2：AdamW](https://keras.io/2/api/optimizers/adamw/)

網站的「優化器教學」是這份筆記的簡明版，展示數值是歷史實測；上面的單一步驟數字僅為概念示例。
