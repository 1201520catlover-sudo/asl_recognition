# ASL 手語辨識與即時打字系統

### 本專案利用 Google MediaPipe 進行手部關節偵測，並結合 歐幾里得距離 (Euclidean Distance) 演算法，實現美國手語 (ASL) A-Z 字母的即時辨識與文字輸入功能。

- ### collect_data.py
  資料採樣工具：用於錄製使用者個人化的 A-Z 手勢角度特徵，並存入 CSV。

- ### hand_data.csv
  特徵資料庫：儲存 A-Z 每個字母的五維角度特徵數據。

- ### asl_recognition_final.py
  辨識主程式：具備即時辨識與「打字機」輸入介面，支援刪除與重置功能。

- ### asl_check.py
  正確率測試工具：透過人工標記（按鍵輸入）與系統預測即時比對，自動生成統計報告。

- ### result_a~z.txt
  實驗報告：由 asl_check.py 生成，詳細記錄每個字母的測試次數與正確率。
