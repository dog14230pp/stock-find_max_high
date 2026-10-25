"""策略 type1_rise：連續 N 根 1 分 K 的 type1_ratio 嚴格遞增，且最後一根 > 門檻 → 標記最後一根。

type1_ratio = 外盤成交量 / 全部成交量（分母含 type 0，與 analysis/2026-10-21_type1_ratio 相同）。
「連續 N 根」= 相鄰 N 根有成交的 1 分 K（與 min/ 的列一致）。
只對 frame_filter.keep_frames() 的 242 個 frame 寫 CSV。

執行（在專案根目錄）：
    python strategies/type1_rise3.py                              3 根、> 0.5  → marks/type1_rise3
    python strategies/type1_rise3.py --n 5 --above 0.49           5 根、> 0.49 → marks/type1_rise5
    python lib/backtest_report.py marks/type1_rise3
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))

from backtest_report import load_frames, save_marks  # noqa: E402
from frame_filter import keep_frames, max_bar_table  # noqa: E402

N_RISE = 3          # 連續幾根嚴格遞增
LAST_ABOVE = 0.5    # 最後一根要大於多少


def minute_ratio(df):
    t = df[['ts', 'volume', 'tick_type']].copy()
    t['ts'] = pd.to_datetime(t['ts'])
    t['v1'] = t['volume'].where(t['tick_type'] == 1, 0)
    g = t.groupby(t['ts'].dt.floor('1min'))
    return g['v1'].sum() / g['volume'].sum()


def rise3(df, n=N_RISE, last_above=LAST_ABOVE):
    r = minute_ratio(df)
    up = r.diff() > 0
    ok = up.rolling(n - 1).sum().eq(n - 1) & (r > last_above)
    return r.index[ok].tolist()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='連續 N 根 type1_ratio 嚴格遞增且最後一根 > 門檻')
    ap.add_argument('--n', type=int, default=N_RISE, help=f'連續幾根（預設 {N_RISE}）')
    ap.add_argument('--above', type=float, default=LAST_ABOVE, help=f'最後一根門檻（預設 {LAST_ABOVE}）')
    ap.add_argument('--name', help='輸出 marks/<name>（預設 type1_rise<n>）')
    a = ap.parse_args()
    out = ROOT / 'marks' / (a.name or f'type1_rise{a.n}')
    df_list, key_list = load_frames()
    frames = keep_frames(table=max_bar_table(df_list, key_list))
    n_marks = 0
    for i in frames:
        marks = rise3(df_list[i], a.n, a.above)
        n_marks += len(marks)
        save_marks(marks, i, out)
    print(f'n={a.n}, above={a.above}：寫好 {len(frames)} frames 的 CSV 到 {out}，共 {n_marks} 個標記')
