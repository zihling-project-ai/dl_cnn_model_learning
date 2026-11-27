# 展示數值

`showcase.json` 是進行中實驗的快照，只收錄完整10輪且有驗證報告的結果。包含歷史曲線、最佳權重的驗證評估、六類統計；不含權重、照片、本機路徑、原始實驗代號或逐圖預測。最終測試集未評分。

十輪總覽與比較表的 Accuracy／Loss／F1 取同一次重新評估；1／5輪與每輪曲線取歷史紀錄，不宣稱是重新評估的結果。權重依歷史驗證 Accuracy 選出。歷史耗時可能包含暫停／休眠，不作速度排名。CPU與GPU結果保留標示，實驗仍待收尾。

更新展示（在有訓練紀錄的電腦操作）：

```sh
python3 tools/export_showcase.py --source "/path/to/training-project" --output results/showcase.json
python3 tools/build_showcase.py
```

匯出只讀已完成紀錄，不訓練、不載入模型。靜態展示在一般HTTP靜態網站即可執行，不需要TensorFlow。直接雙擊 index.html 可能被瀏覽器的本機檔案讀取限制阻擋，請用靜態伺服器或網站託管。
