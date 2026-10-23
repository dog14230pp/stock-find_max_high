# 高點研究（台指期 tick）

用 290 段 tick 資料（frame），研究「能不能在 frame 最高點那一分鐘發出訊號」。
目前的主要因子是 **type1_ratio = 外盤成交量 / 全部成交量**。

---

## 1. 換電腦後快速接上

需要：Windows、Python 3.11、VS Code（Jupyter 擴充套件）、Microsoft Edge（報告截圖用，可省略）。

```powershell
# 1. 把專案資料夾放到新電腦（例如 D:\stock），進到資料夾
cd D:\stock

# 2. 重建 Python 環境（.venv 不要從舊電腦複製，要重建）
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 3. 確認能跑：產生策略標記 → 產生報告（會自動打開瀏覽器）
python strategies/type1.py
python lib/backtest_report.py marks/type1
```

看到 `Hit 101 / 290` 就代表資料和程式都正常。

notebook：用 VS Code 打開 `notebooks/讀取20261017高點研究.ipynb`，kernel 選 `.venv`，從第一個 cell 開始跑。

### Claude Code 設定（在每台電腦的使用者資料夾，不會跟著專案走）
舊電腦的這兩個檔要另外複製到新電腦的 `C:\Users\<帳號>\.claude\`：

| 檔案 | 作用 |
|---|---|
| `settings.json` | 禁止 Claude 執行刪除指令（rm、Remove-Item 等 deny 規則） |
| `CLAUDE.md` | 規則：只在專案資料夾內動作、暫存檔放 `.tmp/`、不刪檔 |

Claude Code 對這個專案的記憶在 `C:\Users\<帳號>\.claude\projects\d--stock\`；專案路徑不同（例如不是 `D:\stock`）就不會自動接上，可以不管。

---

## 2. 資料夾結構

```
D:\stock\
├─ data\20261017高點研究\     原始資料（只讀）
│   └─ df_dict_ticks.pkl.gz     290 個 frame 的 tick
├─ lib\                         共用程式
│   ├─ backtest_report.py       讀標記 CSV → Hit/Miss 報告
│   ├─ tick_plot.py             畫 K 線（可指定時間區間、標最高點）
│   ├─ minute_profile.py        每分 K 統計、內外盤、分布圖
│   ├─ type1_segments.py        type1 區間切分
│   └─ tick_utils.py            依時間切出 tick
├─ strategies\type1.py          策略 → 產生 marks\type1\
├─ marks\<策略>\df_XXX.csv      策略輸出的標記點
├─ reports\<策略>\index.html    Hit/Miss 報告
├─ analysis\<日期_主題>\        每次獨立分析：程式 + 結果放一起
├─ notebooks\                   探索用 notebook
└─ .tmp\                        暫存（截圖等），可整個刪
```

---

## 3. 資料

- `df_dict_ticks.pkl.gz` 讀進來是 dict，共 290 個 frame。
  - key：`(開始時間, 結束時間)`，是標記的研究區間
  - value：該段的 tick DataFrame，實際資料會比 key 的開始時間早一些
- tick 欄位：`ts, close, bid_price, ask_price, volume, bid_volume, ask_volume, tick_type`
- `tick_type`：1 = 外盤（主動買）、2 = 內盤（主動賣）、0 = 無法判定（由資料反推，與 Shioaji 編碼一致）
- frame 編號 `df_XXX` = dict 的順序（`list(df_dict.values())[i]`），不是時間順序

```python
import sys; sys.path.insert(0, 'lib')
from backtest_report import load_frames
df_list, key_list = load_frames()
```

---

## 4. 策略評估流程（主要工作流）

1. 寫策略：對每個 frame 算出想標記的時間點，存成 CSV
2. 跑報告：`python lib/backtest_report.py marks/<策略名>`
3. 看報告：`reports/<策略名>/index.html`（自動打開）

**標記 CSV 格式**
- 位置：`marks/<策略名>/df_000.csv`、`df_001.csv` …（沒標記的 frame 可以不放）
- 內容只有一欄 `time`，可寫到秒，會對齊到所在的 1 分 K：
  ```
  time
  2021-05-13 09:44:00
  ```
- 寫檔可用 `save_marks(times, df_idx, 'marks/<策略名>')`，範例見 `strategies/type1.py`

**Hit 的定義**：frame 最高價（第一次出現）所在的那一分鐘，有被標記。

**目前結果**：`type1`（外盤比例 > 0.45 起算、連續兩分鐘 < 0.45 結束，標記區間內最後一次 > 0.5 的分鐘）→ Hit 101 / 290。

---

## 5. 分析紀錄（analysis\）

| 資料夾 | 內容 |
|---|---|
| `2026-10-17_每分K統計` | df 0 的每分鐘價格分布、統計、每分鐘疊加圖 |
| `2026-10-18_type1區間與最高點` | 全部 frame 的 type1 區間、標記與最高點是否重疊 |
| `2026-10-21_type1_ratio` | 秒級 / 分級 type1_ratio、最高點資訊、高 ratio 秒數報告 |

`2026-10-21_type1_ratio` 的內容：
- `make.py` → `sec/df_XXX.csv`（每秒）、`min/df_XXX.csv`（每分）、`frame_max.csv`（每個 frame 的最高點）
- `sec_ratio_count.py` → `sec_ratio_count.html` 報告：
  - 每分鐘 type1_ratio ≥ 0.7 / 0.8 / 0.9 的秒數與成交量、≥ 0.9 成交量佔比
  - 每個 frame 的「每分鐘成交量 vs type1_ratio」相關係數
  - 梯度 / 速度（類 NN momentum）：梯度 = 這分鐘 ratio − 上一分鐘；速度 = β × 上一分鐘速度 + (1 − β) × 梯度，起點 = 0；β 在檔案開頭 `BETA`（目前 0.1）
  - 最高價所在分鐘的列會標色
- 重跑：先 `python analysis/2026-10-21_type1_ratio/make.py`，再 `python analysis/2026-10-21_type1_ratio/sec_ratio_count.py`

---

## 6. 目前已知的觀察

- 每秒的 type1_ratio 很吵：≥ 0.9 的秒有 95% 是剛好 = 1，其中過半那一秒只有 1 筆成交。
- 每分鐘成交量和 type1_ratio 幾乎沒有相關（per frame 的 Pearson 中位數約 0.05）。
- type1 策略命中 101 / 290，隨機標同樣多個點的期望約 27 次；但標記點本來就偏向上漲段，這個比較只能說明「比亂標好」。

---

## 7. 不是這套流程的舊檔（保留未動）

`stock_analysis.py`、`uptrend_analysis.py`、`goog_prices.csv`、`qqq_prices.csv`、`v261017.txt`、`results\`、`讀取資料.ipynb`、`minute_profile\Method_1.txt`、`reports\a\`、`analysis\sec_ratio_count.html`
