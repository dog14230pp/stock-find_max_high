# 高點研究專案：給 Claude 的規則

這個檔案跟著 git 走，是跨電腦的唯一依據；不要依賴個人記憶或某台電腦的 `~/.claude` 設定。
先讀 README.md（換電腦流程、資料、分析紀錄都在裡面）。

## 基本規則

- 只在專案資料夾內動作；暫存檔放 `.tmp/`（不進 git）。
- 不刪檔：`.claude/settings.json`（進 git）有 deny 規則；要清掉的檔案請使用者自己刪。
- 路徑一律從 `Path(__file__)` 推算，不寫死 `D:\...`（兩台電腦的專案位置可能不同）。
- 用專案的 `.venv`（`tools/setup.ps1` 建立）。套件版本照 requirements.txt；pandas 固定 2.3.3，因為其中一台的 Windows Smart App Control 會擋 pandas 3 的 DLL。

## 分析用的 frame

- **新的分析一律只用 242 個 frame**：最高點落在前 5 根 1 分 K 的 48 個 frame 排除。
  ```python
  sys.path.insert(0, str(ROOT / 'lib'))
  from frame_filter import keep_frames
  frames = keep_frames()
  ```
  不要自己重寫過濾條件；門檻改 `lib/frame_filter.py` 的 `MIN_BAR_POS`。
- 策略只能用當下以前的資料，不可用未來資訊（例如要等區間結束才知道的標記）。舊的 `type1` 策略就是因此不再當比較基準。
- 正式策略記在 README 第 4 節「正式策略」（目前策略 1 = S5）。新策略放進 `strategies/registry.py`、用 `strategies/compare.py` 跑，跟正式策略比。
- 報告和輸出要寫明用了幾個 frame（例如「242 frames」），讓結果可以對照。
- 例外：使用者明確要求看全部 290 個、或要看被排除的 frame 時。
- 待確認：最高點目前在整段 tick（含 frame_start 前的緩衝）中找。使用者還沒決定是否改成只在 frame_start ~ frame_end 內找；改了的話 242 會變，相關分析要重跑。

## 跟使用者溝通

- 設計或修改策略前，先確認要用哪些條件；不要自行帶入其他策略的條件（曾把 S5 的條件偷帶進 S7、S8）。
- 回覆放慢、一次講一件事；使用者常跟不上太快、太多的內容。
- 每次實驗前照 `.claude/skills/experiment-plan/SKILL.md`：一步一步、用圖解說明，使用者說 OK 才實作。

## 慣例

- 每次分析開新資料夾 `analysis/<日期_主題>/`，程式和輸出放一起；輸出路徑用 `Path(__file__).parent`。
- 程式（.py）進 git；輸出（csv / png / html）不進 git，靠 `tools/pack_drive.py` 打包到 Google Drive。
- 新分析完成後，在 README.md 第 5 節的表格補一列，並把重要結論寫進去（換電腦後靠 README 接上進度）。
