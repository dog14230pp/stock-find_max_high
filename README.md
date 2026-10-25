# 高點研究（台指期 tick）

用 290 段 tick 資料（frame），研究「能不能在 frame 最高點那一分鐘發出訊號」。
目前的主要因子是 **type1_ratio = 外盤成交量 / 全部成交量**。

---

## 1. 換電腦後快速接上

需要：Windows、Python 3.11～3.13（python.org 安裝版，含 py launcher）、Git、VS Code（Jupyter 擴充套件）。

- **程式碼、設定、規則** → GitHub（git）。包含 `CLAUDE.md`（給 Claude 的規則）、`.claude/settings.json`（禁止 Claude 刪檔），換電腦不用另外複製任何設定。
- **原始資料和分析輸出** → Google Drive 的 `stock_drive_YYYYMMDD.zip`（`data\`、`marks\`、`reports\`，以及 `analysis\` 裡的 csv / png / html）。
- 專案放哪個資料夾都可以，程式的路徑都是相對專案根目錄。**用 VS Code 打開專案根目錄（有這份 README 的那層）再開 Claude Code**，它才會自動讀到 `CLAUDE.md`。

### 第一次在新電腦

```powershell
git clone https://github.com/dog14230pp/stock-find_max_high.git
cd stock-find_max_high
# 把 Google Drive 最新的 stock_drive_*.zip 解壓縮到這個資料夾
powershell -ExecutionPolicy Bypass -File tools\setup.ps1     # 建 .venv、裝套件、檢查資料
```

看到 `packages OK` 和 `data OK: analysis frames = 242` 就可以開始。想再確認一次結果：
`.venv\Scripts\python.exe strategies/type1.py`，再 `.venv\Scripts\python.exe lib/backtest_report.py marks/type1 --all`，應該看到 `Hit 101 / 290`（不加 `--all` 是 242 個 frame：`Hit 84 / 242`）。

notebook：用 VS Code 打開 `notebooks/讀取20261017高點研究.ipynb`，kernel 選 `.venv`。

### 兩台電腦之間來回搬

**離開這台電腦前**
```powershell
git add -A; git commit -m "說明"; git push       # 程式碼、設定、README → GitHub
.venv\Scripts\python.exe tools/pack_drive.py     # 資料與輸出 → _transfer\stock_drive_YYYYMMDD.zip，上傳 Google Drive
```

**到另一台電腦**
```powershell
git pull                                          # 先拿最新程式碼
# 下載最新的 zip，解壓縮到專案資料夾，同名檔案選「取代」（zip 只有資料和輸出，不會蓋掉程式）
powershell -ExecutionPolicy Bypass -File tools\setup.ps1     # requirements.txt 有變就會更新套件
```

注意：
- 開工前先 `git pull`，收工後 `git push`。兩台都改了同一個檔案又沒同步，會產生衝突。
- 解壓縮只會新增或覆蓋，不會刪除。另一台電腦刪掉的檔案，這台還會留著。
- 如果只改了程式、沒動資料，可以不用 zip，直接重跑程式產生輸出。
- 論文 PDF 不進 git，要用時照 `papers/README.md` 重新下載。

### 套件版本

pandas 固定 **2.3.3**（requirements.txt）。其中一台電腦的 Windows「Smart App Control」會擋 pandas 3 的 DLL（錯誤訊息：`DLL load failed … 應用程式控制原則已封鎖此檔案`），所以兩台統一用 2.3.3。
已確認：用 2.3.3 重跑 `2026-10-21_type1_ratio`，583 個 CSV 和報告跟原本 pandas 3 的輸出完全一樣。

### Claude Code

- 規則在專案的 `CLAUDE.md`、禁止刪檔在專案的 `.claude/settings.json`，都跟著 git 走。
- Claude 的個人記憶（`C:\Users\<帳號>\.claude\projects\…`）只存在單一電腦，不要依賴它；重要的進度和結論都寫進 README 第 5 節。

---

## 2. 資料夾結構

```
D:\stock\
├─ data\20261017高點研究\     原始資料（只讀）
│   └─ df_dict_ticks.pkl.gz     290 個 frame 的 tick
├─ lib\                         共用程式
│   ├─ backtest_report.py       讀標記 CSV → Hit/Miss 報告
│   ├─ start_backtest.py        起點回測核心（Hit、找到的點、虧損 / 獲利），策略比較用
│   ├─ tick_plot.py             畫 K 線（可指定時間區間、標最高點）
│   ├─ minute_profile.py        每分 K 統計、內外盤、分布圖
│   ├─ type1_segments.py        type1 區間切分
│   └─ tick_utils.py            依時間切出 tick
├─ strategies\type1.py          策略 → 產生 marks\type1\
├─ strategies\registry.py       策略清單（S1、S2…），策略比較用
├─ strategies\compare.py        策略比較 → reports\compare\
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

### 分析用的 frame：242 個（2026-10-24 起）

**最高點落在前 5 根 1 分 K 的 frame 不分析**：290 個排除 48 個，剩 242 個。之後所有分析都從這 242 個挑。

```python
from frame_filter import keep_frames
frames = keep_frames()          # 242 個 df_idx（門檻在 lib/frame_filter.py 的 MIN_BAR_POS = 5）
```

- 「第幾根」從 tick 資料的第一分鐘起算（0 起算，只數有成交的分鐘），不是從 frame_start 起算。
- 最高點 = 整段 tick（含 frame_start 前約 15 分鐘的資料）中最高價第一次出現的那筆，與 `backtest_report.bars_of` 相同。
- 排除名單與說明：`analysis/2026-10-24_max_early_frames/`（`excluded.csv`）。
- **待確認**：被排除的 48 個中有 47 個最高點落在 frame_start 之前（資料緩衝區）。如果之後決定最高點要在 frame_start ~ frame_end 裡找，242 這個數字會變。
- 2026-10-24 之前的分析沒有用這個過濾：`2026-10-23_max_minute_sec_ratio`、`2026-10-23_ratio_divergence` 排除前 10 根（剩 238）；其他是全部 290 個。

---

## 4. 策略評估流程（主要工作流）

### 正式策略

新策略都拿來跟正式策略比；要換掉正式策略時更新這一節。
評估原則（2026-10-25）：Hit 越多越好、虧損越少越好、多找的點越少越好；Hit 10%～20% 是初步門檻。

#### 策略 1 = S5（2026-10-25 定案）

**規則**：每根 1 分 K 收盤時檢查，5 個條件同時成立就標記這根（= 認為是高點，下一根 open 賣出）。
只用這根和之前的資料。

| # | 條件 | 參數 |
|---|---|---|
| 1 | 成交量 > 1.5 × 前 5 根（t-5 ~ t-1）平均成交量 | `n=5, mult=1.5` |
| 2 | type1_ratio（外盤量 / 全部量）≥ 0.5 | `ratio_min=0.5` |
| 3 | 創新高：這根 high ≥ 前 60 根的最高價（前面不足 60 根就用現有的） | `new_high=60` |
| 4 | 上影線（high − max(open, close)）≥ 整根長度（high − low）的 15% | `wick_min=0.15` |
| 5 | 上影線 ≥ 0.8 × 實體（\|close − open\|）；十字線一定符合 | `wick_body_min=0.8` |

**成績**（`reports/compare/summary.csv`，2026-10-25）

| | 242（分析用） | 全部 290 | 48（被排除） |
|---|---|---|---|
| Hit | **35（14.5%）** | 35（12.1%） | 0 |
| 找到的點 / 多找的點 | 203 / 168 | 205 / 170 | 2 / 2 |
| 點在最高點前 / 後 | 167 / 1 | 167 / 3 | 0 / 2 |
| 沒有任何點的 frame | 121 | 167 | 46 |
| 沒出手的起點 | 56% | 60% | 93% |
| 虧損 中位 / p90 / 最大（點） | 33 / 61 / 173 | 33 / 61 / 173 | 25 / 25 / 77 |
| 獲利 中位 / p10（點） | 201 / 58 | 201 / 58 | 128 / 58 |

虧損 = s ~ 標記之間最高價 − 標記下一根 open；獲利 = 標記下一根 open − s ~ 標記之間最低價（定義見 `lib/start_backtest.py`）。

**程式**

| 檔案 | 用途 |
|---|---|
| `strategies/registry.py` | 策略定義：`STRATEGIES['S5']`，條件由 `vol_rise()` 組合 |
| `strategies/type1_vol.py` | 條件 1、2 的 `vol_mask()`（`vol_rise()` 會呼叫） |
| `lib/start_backtest.py` | 每分鐘資料 `minute_bars()`、起點回測 `run()`、統計 `summary()`、報告 HTML、K 線圖 `plot_frame()` |
| `strategies/compare.py` | 執行入口：跑回測、寫報告、更新比較總表 |
| `lib/frame_filter.py` | 242 / 48 分組（`keep_frames()`） |
| `lib/backtest_report.py` | 讀資料 `load_frames()`、最高點 `bars_of()` |

重跑：`.venv\Scripts\python.exe strategies\compare.py S5`（約 2 分鐘，含 290 張圖）

**輸出**（不進 git，靠 `tools/pack_drive.py` 打包）

| 位置 | 內容 |
|---|---|
| `reports/compare/S5/index.html` | 完整報告：Hit 與找到的點、虧損與獲利（含極端案例算式）、依起點距離分組、每個 frame 的表、290 張圖 |
| `reports/compare/S5/plots/df_XXX.png` | 每個 frame 的圖：K 線（★ 找到的點、◆ 最高點、▼→● 最大虧損的買賣點）＋成交量（藍 = 標記那根） |
| `reports/compare/S5/detail.csv` | 每個 frame × 每個起點：賣點、最壞買點、最好買點的時間與價格，虧損 / 獲利 |
| `reports/compare/S5/frames.csv` | 每個 frame：Hit、最高點、找到的點與時間 |
| `reports/compare/index.html`、`summary.csv` | 所有策略的比較總表 |

**怎麼來的**：S1（量 + 連漲）→ `analysis/2026-10-25_s1_early_vs_hit` 發現 Hit 都是新高、上影線較長 → S2 / S3 →
`analysis/2026-10-25_wick_body` 看上影線 / 實體 → S4（≥ 0.9）、S5（≥ 0.8，比 S4 多 3 個 Hit、只多 13 個點）。

**已知限制**
- 條件 3～5 是從同一批資料看出來再回測的，有過度配適風險，數字可能偏樂觀；最好用新的 frame 驗證。
- 每個 frame 只有約 70 根，「前 60 根新高」在這裡幾乎等於「資料開始以來新高」，實盤（前面有完整歷史）結果可能不同。
- 一半的 frame 完全不出手（121 / 242）；虧損中位數 33 點，大約是一根上影線的長度。

### 策略比較（2026-10-24 起的主要流程）

1. 在 `strategies/registry.py` 的 `STRATEGIES` 加一筆：編號、名稱、說明、標記函式 `mask(mb)`（只能用 t 與之前的資料）
2. 跑：`python strategies/compare.py S2`（不給編號 = 全部重跑；約 1 分鐘，`--no-plots` 不畫圖只要幾秒）
3. 看：`reports/compare/index.html`（比較總表，可切換 全部 290 / 242 / 48），點編號看該策略的完整報告；
   報告最後有 290 張 K 線圖（★ 找到的點、◆ 最高點、灰底 frame 區間），可篩 Hit / Miss / 沒找到點

比較的指標（全部 290 frames 都記錄，分 242 / 48）：Hit、找到的點、多找的點（= 找到的點 − Hit）、
虧損（s ~ 標記之間最高價 − 標記下一根 open）、獲利（標記下一根 open − s ~ 標記之間最低價）、沒出手比例。定義見 `lib/start_backtest.py`。
（summary.csv 另有隨機期望 Hit 與倍數，報表不顯示。）

| 編號 | 規則 | Hit（242） | 找到的點（242 / 290） | 虧損 中位 / 最大（290） |
|---|---|---|---|---|
| S1 | 成交量 > 2 × 前 5 根平均、ratio ≥ 0.5、close 連漲 3 次（= `type1_vol5_x2_up3`） | 24（9.9%） | 285 / 297 | 13 / 331 |
| S2 | 成交量 > 2 × 前 5 根平均、ratio ≥ 0.5、創前 60 根新高、上影線 ≥ 15% | 37（15.3%） | 236 / 236 | 28 / 173 |
| S3 | 同 S2，成交量 > 1.5 倍 | 66（27.3%） | 442 / 444 | 23 / 173 |
| S4 | S3 + 上影線 ≥ 0.9 × 實體 | 32（13.2%） | 190 | 34 / 173 |
| **S5 = 正式策略 1** | S3 + 上影線 ≥ 0.8 × 實體 | 35（14.5%） | 203 / 205 | 33 / 173 |
| S6 | S5，但量的基準改成前 5 根的 softmax 加權平均（溫度 1），倍數 1.2 | 50（20.7%） | 264 / 265 | 33 / 173 |
| S7 | S6，但量的條件換成 corr(Open, Volume, 10) ≥ 0.2 且 corr(Close, Volume, 10) ≥ 0.2 | 53（21.9%） | 308 / 311 | 30 / 223 |
| S8_alpha | S6 + corr(Close, Volume, 10) ≥ 0.2（原本的 S8，含 S5 的條件 2～5） | 40（16.5%） | 184 / 185 | 32 / 173 |
| S8 | 純粹：softmax 量 × 1.2 + corr(Close, Volume, 10) ≥ 0.2（不帶其他條件） | 86（35.5%） | 1415 / 1531 | 16 / 597 |
| S9 | softmax 量 × 1.2 + 創前 60 根新高 + 上影線 ≥ 整根 15% + corr(Close, Volume, 10) ≥ 0.2（= S8_alpha 拿掉條件 2、5） | 75（31.0%） | 457 / 459 | 23 / 246 |

- 2026-10-25 改名：原本的 S8 → S8_alpha；S8 改成純粹版。S7、S8_alpha 帶了 S5 的條件 2～5，是 Claude 自行加的，不是使用者指定。
- 新方向（使用者）：先求 Hit 高（目標 > 30%，最好 > 40%），再加條件刪點。S8 純粹版當起點候選。
- corr(Close − Open, Volume) ≥ 0.2 取代 corr(Close)：softmax 量 + 它 Hit 66 / 1302 點，比 S8 差；三個一起 56 / 989。
- S8_alpha 拿掉條件（242 frames，2026-10-26）：拿掉 4 → 完全不變（被 5 涵蓋）；拿掉 5 → Hit 66 / 395 點；
  拿掉 2 → 45 / 220；**拿掉 2、5（剩量 + 新高 + 上影線 ≥ 15% + corr）→ Hit 75（31.0%）/ 457 點、準度 16.4%、虧損 23 / 57 / 246**。
- S8_alpha vs 正式策略 1（S5）：Hit 多 5 個、點少 19 個、虧損差不多，三個目標都比較好（2026-10-25）。
- 穩定度：242 frames 中 219 個在 2026 年，不能按年拆；依日期切前後半（各 121 個，切在 2026-06-12）：
  S8_alpha 前半 Hit 17 / 83 點（20.5%）、後半 23 / 101（22.8%），兩半都是準度最高；S5 14.6% / 19.6%、S6 17.1% / 20.7%。
  參數是用全部資料挑的，這不是真正的樣本外驗證。
- S6 / S7 搭配（242 frames）：任一標記 Hit 71 / 423 點（16.8%）、兩者都標記 32 / 149（21.5%）；兩邊各有約 20 個對方抓不到的 Hit。
  S6 或「只看 Close ≥ 0.2」：78 / 475（16.4%）。

- 價量相關 corr 取代量（其他條件同 S6，242 frames，2026-10-25）：
  不看量 Hit 90 / 700 點（12.9%）；Open ≥ 0.3：45 / 273（16.5%）；Open ≥ 0.2：53 / 336（15.8%）；
  **Close ≥ 0.2：68 / 395（17.2%）**；Open 且 Close ≥ 0.2（S7）：53 / 308（17.2%）；S6（量放大）：50 / 264（18.9%）。
- Close 版比 Open 版好；加了 Close 之後 Open 版只是在刪點、沒多帶 Hit。corr 加在 S6 上面（量 + Open ≥ 0.3）：30 / 141（21.3%）。

- softmax 平均 = 權重 softmax(量 / 前 5 根平均 / 溫度) 的加權平均，介於平均和最大值之間（`registry.softmax_base`）。
- 同樣 Hit 數下，softmax 比平均少標約 1 成的點（例：softmax×1.2 Hit 50 / 264 點 vs 平均×1.3 Hit 51 / 300 點）；
  門檻很嚴時（平均×1.7、×2）兩者差不多（2026-10-25，242 frames，其他條件同 S5）。

- S2 比 S1 Hit 多、點還更少；S3 Hit 最多但點也最多（2026-10-25）。
- 新高 + 上影線的條件是從 S1 同一批資料看出來的（`analysis/2026-10-25_s1_early_vs_hit`），有過度配適風險。
- 「前 60 根新高」在這份資料裡幾乎等於「資料開始以來新高」（每個 frame 約 70 根），實盤效果可能不同。

### 單純 Hit / Miss（舊流程）

1. 寫策略：對每個 frame 算出想標記的時間點，存成 CSV
2. 跑報告：`python lib/backtest_report.py marks/<策略名>`（預設只評估 242 個 frame；`--all` = 全部 290 個）
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

**目前結果**（初步目標：242 個 frame 中抓到 10%～20% 的最高點，約 24～48 個；
評估原則（2026-10-25）：**Hit 越多越好、虧損越少越好、多找的點越少越好**）

策略只能用當下以前的資料（不可偷看未來）。倍數 = Hit / 隨機標同樣多點的期望 Hit。

| 策略 | 規則 | Hit（242 frames） | 標記數 | 隨機期望 Hit | 倍數 |
|---|---|---|---|---|---|
| `type1_rise3` | 連續 3 根 1 分 K 的 type1_ratio 嚴格遞增，且第三根 > 0.5，標記第三根 | 28 | 2453 | 33.9 | 0.8 |
| `type1_rise5` | 同上，連續 5 根，第五根 > 0.49（`--n 5 --above 0.49`） | 4 | 178 | 2.6 | 1.5 |
| `type1_vol5` | 這根成交量 > 前 5 根（t-5 ~ t-1）平均，且這根 type1_ratio >= 0.5（`strategies/type1_vol.py`） | 130 | 3404 | 46.5 | 2.8 |
| `type1_vol5_x1.5` | 同上，成交量 > 1.5 倍平均（`--mult 1.5`） | 77 | 1409 | 18.6 | 4.1 |
| `type1_vol5_x2` | 同上，成交量 > 2 倍平均（`--mult 2`） | 44 | 654 | 8.4 | 5.3 |
| `type1_vol5_x3` | 同上，成交量 > 3 倍平均（`--mult 3`） | 17 | 197 | 2.6 | 6.7 |
| `type1_vol5_x2_r55` | 成交量 > 2 倍平均，且 type1_ratio >= 0.55（`--mult 2 --above 0.55`） | 36 | 574 | 7.3 | 5.0 |
| `type1_vol5_up3` | `type1_vol5` 再加 close 連漲 3 次（c[t-3] < c[t-2] < c[t-1] < c[t]，`--rise 3`） | 56 | 1173 | 16.0 | 3.5 |
| `type1_vol5_x1.5_up2` | 成交量 > 1.5 倍平均、ratio ≥ 0.5、close 連漲 2 次（`--mult 1.5 --rise 2`） | 50 | 915 | 12.1 | 4.1 |
| `type1_vol5_x1.5_up3` | 同上，連漲 3 次 | 37 | 575 | 7.6 | 4.9 |
| `type1_vol5_x2_up2` | 成交量 > 2 倍平均、ratio ≥ 0.5、close 連漲 2 次（`--mult 2 --rise 2`） | 32 | 452 | 5.8 | 5.5 |
| `type1_vol5_x2_up3` | 同上，連漲 3 次 | 24 | 285 | 3.6 | 6.6 |
| `type1_vol5_x2_r55_up3` | 成交量 > 2 倍平均、ratio ≥ 0.55、close 連漲 3 次 | 21 | 262 | 3.3 | 6.3 |

- 隨機期望 = 每個 frame 的「標記數 / 1 分 K 根數」加總。
- type1_ratio 連續遞增（rise）跟隨機差不多；「量放大 + 外盤過半」（vol）有效，量的門檻越嚴倍數越高；3 倍時 127 個 frame 沒有標記，太少。ratio 門檻從 0.5 拉到 0.55 沒有幫助（2026-10-24）。
- 加 close 連漲（`--rise`）倍數都會提高，但 Hit 大約砍半；同樣 Hit 水準下，不見得比單純提高量的倍數好（例如 x1.5_up3：37 / 4.9 vs x2：44 / 5.3）。
- 舊的 `type1`（Hit 84 / 242、全部 290 個是 101）**不可行，不再比較**：標記的是「區間內最後一次 > 0.5 的分鐘」，要等區間結束才知道，用到了未來資料。

---

## 5. 分析紀錄（analysis\）

| 資料夾 | 內容 |
|---|---|
| `2026-10-17_每分K統計` | df 0 的每分鐘價格分布、統計、每分鐘疊加圖 |
| `2026-10-18_type1區間與最高點` | 全部 frame 的 type1 區間、標記與最高點是否重疊 |
| `2026-10-21_type1_ratio` | 秒級 / 分級 type1_ratio、最高點資訊、高 ratio 秒數報告 |
| `2026-10-23_ratio_vs_price` | 大雜燴：sec_ratio_count ＋ 每分鐘 ratio vs 漲跌（同分鐘、下一分鐘）的相關係數、依 ratio 分組 |
| `2026-10-23_max_minute_sec_ratio` | 最高點那一分鐘內各秒 ratio 的分布；`--plot N` 畫單一 frame |
| `2026-10-23_sec_ratio_dist` | 每分鐘各秒 ratio 的分布，依「這分鐘 / 下一分鐘漲跌」分組比較（`all` = 全部 frame） |
| `2026-10-23_ratio_divergence` | ratio 與漲幅的背離 z（只用過去資料回歸）、最高點前後事件研究 |
| `2026-10-24_max_early_frames` | 最高點在前 5 根 K 的 48 個 frame（排除名單） |
| `2026-10-24_vol_x2_plots` | `type1_vol` 策略的 242 張圖（K 線＋成交量門檻線＋ratio），參數同策略，輸出到 `<策略名>/index.html`，可篩 Hit / Miss / 沒標記。已畫：`type1_vol5_x2_r55`、`type1_vol5_x2_up3` |
| `2026-10-25_s1_early_vs_hit` | S1 找到的 285 個點（242 frames）分 Hit / 太早（前 1～10 根）/ 更早 / 之後，比較標記當下的特徵：上影線、是否創新高、已漲幅度、量… 結論見下方 |
| `2026-10-25_max_bar_shape` | 最高點那根的形狀 vs type1_ratio > 0.5（242 frames）：大紅棒（影線 < 實體一半）28 根，100% ratio > 0.5；一般紅棒 109 根，89%；綠棒或十字 105 根，46%。各組 frame 編號在 `max_bars.csv` |
| `2026-10-25_buy_then_sell` | 「前買後賣」檢查（242 frames）：每分鐘把有成交的秒切成前後兩段（各至少 3 秒），分數 =（前段外盤 + 後段內盤）÷ 總量，取最好的切點。加在 S8 上：門檻 0.6 → Hit 78 / 1126 點（S8 86 / 1415）；Hit 與其他點的分數中位數 0.69 vs 0.67，分得不太開 |
| `2026-10-25_flow_momentum` | 買賣壓動量（類 ML momentum，242 frames）：推力 F =（外盤 − 內盤量）/ 前 5 根平均量、累積 v = EMA(F)、價格加速度。最高點那根的 v（β=0.9）在 frame 內平均第 87 百分位、「買壓 > 0 但價格減速」是一般 K 棒的 2.2 倍；但在 S8 的點裡分不出 Hit，取代量或加在 S6 上都沒贏 S8 → 資訊已被「新高 + 上影線」涵蓋，不採用。
純動量策略（只用 F / v / 加速度）回測：準度 3～8%（S8 21.7%），例 v ≥ 0.2 且 B：Hit 35 / 453 點；不採用 |
| `2026-10-25_alpha_check` | 文獻因子快速檢查（242 frames）：Alpha#12、Alpha#6（−corr(Open, Volume, 10)）、Amihud ILLIQ。只有 Alpha#6 明顯：最高點那根在 frame 內平均落在第 33 百分位（50 = 沒資訊），42% 的最高點落在最低 20%（一般 K 棒 17%）；S6 沒抓到的 frame 也一樣有效（34）。三個都不能幫 S6 刪點 |
| `2026-10-25_log_view` | （不採用）290 frames 的圖：左邊原值、右邊 ln(OHLC) 與 ln(成交量)，★ = 正式策略 1。價格取 log 形狀幾乎不變（frame 內只波動 1～2%）；成交量取 log 後爆量被壓平 |
| `2026-10-25_wick_body` | S1～S3 標記那根的上影線 / 實體（242 frames）：Hit vs 非 Hit。比例 < 0.25 的標記幾乎不會是 Hit（3～6%），≥ 2 最常是 Hit（約 20%），中間差別不大 |
| `2026-10-24_start_backtest` | 起點回測（290 frames，242 / 48 分開）：每根 1 分 K 當起點 s，看 s 之後的標記數、Hit（s ≤ 最高點才計）、虧損（s ~ 標記之間最高價 − 標記下一根 open）。策略 `type1_vol5_x2_up3` |

S1 Hit vs 太早的點（2026-10-25，242 frames，285 個點）：
- 「創資料開始以來新高」：Hit 100%（必然）、太早 83%、更早 60%。只留創新高的點 → Hit 24 不變，點從 285 降到 199。
  注意：資料開始時間是 frame 標好後往前推的，實盤要改成固定窗口（例如前 60 根新高）。
- 上影線：Hit 中位數 32%、太早 16%（最高點那根本來就容易「衝高後拉回」）。創新高 + 上影線 ≥ 15% → Hit 21、點剩 123。
- 已漲幅度（離資料開始最低點）：Hit 和太早差不多（382 vs 380 點），只能刷掉「更早」的點。
- 過濾只能減少雜訊、不會增加 Hit；要提高 Hit 得先放寬基本條件，再用這些條件過濾。

起點回測的結論（2026-10-24，`type1_vol5_x2_up3`）：
Hit 與找到的點跟起點無關（標記是 sliding window），看整個 frame：242 frames Hit 24（9.9%）、找到 285 個點（256 個在最高點之前、5 個在之後）。
每根 1 分 K 當起點 s（15075 個），賣點 = s 之後第一個標記的下一根 open：46% 的起點沒出手；
虧損（s ~ 標記之間最高價 − 賣點）中位數 13 點、p90 51、最大 239（df 134）；
獲利（賣點 − s ~ 標記之間最低價）中位數 197 點、p10 75，起點在最高點前 1～10 根時中位數 112。
虧損的意義：只看「找高點（賣）」這一方，在沒抓到最高點時最糟會虧多少；買點錯誤是「找低點」那一方的責任，不在這裡處理，所以這個策略本來就不需要停損。例：最大虧損 331 點（df 45，被排除組）是買在最高點後大跌，策略在下跌時本來就不會出手，反彈時才賣。

type1_ratio 目前的結論（2026-10-24）：與同一分鐘漲跌高度相關（機械性），但對下一分鐘漲跌、辨認最高點都沒有預測力；
還沒測的：大單（單筆口數大）的 ratio、5 分鐘尺度。

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
