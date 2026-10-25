"""策略 type1：對每個 frame 算出標記時間，寫成 marks/type1/df_XXX.csv。

執行（在專案根目錄）：
    python strategies/type1.py
    python lib/backtest_report.py marks/type1
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))

from backtest_report import load_frames, save_marks  # noqa: E402
from type1_segments import find_type1_segments       # noqa: E402


def type1_last_above(df, threshold=0.45, mark_threshold=0.5):
    """外盤比例 > threshold 起算、連續兩分鐘 < threshold 結束；
    標記區間內最後一次外盤比例 > mark_threshold 的那分鐘。"""
    seg, _ = find_type1_segments(df, threshold=threshold, mark_threshold=mark_threshold)
    return seg['mark'].dropna().tolist() if len(seg) else []


if __name__ == '__main__':
    out = ROOT / 'marks' / 'type1'
    df_list, _ = load_frames()
    for i, df in enumerate(df_list):
        save_marks(type1_last_above(df), i, out)
    print(f'寫好 {len(df_list)} 個 CSV 到 {out}')
