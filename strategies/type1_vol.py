"""策略 type1_vol：這根 1 分 K 的成交量 > 倍數 × 前 N 根（t-N ~ t-1，不含自己）的平均成交量，且這根 type1_ratio >= 門檻
（可選：且 close 連漲 RISE 次，即 c[t-RISE] < … < c[t-1] < c[t]）→ 標記這根。

type1_ratio = 外盤成交量 / 全部成交量（分母含 type 0，與 analysis/2026-10-21_type1_ratio 相同）。
「前 N 根」= 前面相鄰 N 根有成交的 1 分 K；不足 N 根的開頭幾根不標。
只對 frame_filter.keep_frames() 的 242 個 frame 寫 CSV。

執行（在專案根目錄）：
    python strategies/type1_vol.py                        N = 5、>= 0.5 → marks/type1_vol5
    python strategies/type1_vol.py --n 10 --above 0.55    → marks/type1_vol10
    python strategies/type1_vol.py --mult 2               成交量 > 2 倍平均 → marks/type1_vol5_x2
    python strategies/type1_vol.py --mult 2 --rise 3      再加 close 連漲 3 次 → marks/type1_vol5_x2_up3
    python lib/backtest_report.py marks/type1_vol5
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))

from backtest_report import load_frames, save_marks  # noqa: E402
from frame_filter import keep_frames, max_bar_table  # noqa: E402

N_WINDOW = 5        # 前幾根算平均成交量
RATIO_MIN = 0.5     # 這根 type1_ratio 至少多少
VOL_MULT = 1.0      # 成交量要大於平均的幾倍
RISE = 0            # close 要連漲幾次（0 = 不看）


def minute_table(df):
    t = df[['ts', 'close', 'volume', 'tick_type']].copy()
    t['ts'] = pd.to_datetime(t['ts'])
    t['v1'] = t['volume'].where(t['tick_type'] == 1, 0)
    t = t.sort_values('ts', kind='stable')
    g = t.groupby(t['ts'].dt.floor('1min'))
    m = g[['volume', 'v1']].sum()
    m['close'] = g['close'].last()
    m['ratio'] = m['v1'] / m['volume']
    return m


def vol_mask(m, n=N_WINDOW, ratio_min=RATIO_MIN, mult=VOL_MULT, rise=RISE):
    """m = minute_table()；回傳 (每分鐘是否標記, 成交量門檻 = mult × 前 n 根平均)。"""
    thr = mult * m['volume'].rolling(n).mean().shift(1)     # t-n ~ t-1
    ok = (m['volume'] > thr) & (m['ratio'] >= ratio_min)
    if rise:
        ok &= (m['close'].diff() > 0).rolling(rise).sum().eq(rise)
    return ok, thr


def vol_above_mean(df, n=N_WINDOW, ratio_min=RATIO_MIN, mult=VOL_MULT, rise=RISE):
    m = minute_table(df)
    ok, _ = vol_mask(m, n, ratio_min, mult, rise)
    return m.index[ok].tolist()


def default_name(n, ratio_min, mult, rise):
    return (f'type1_vol{n}' + (f'_x{mult:g}' if mult != VOL_MULT else '')
            + (f'_r{round(ratio_min * 100)}' if ratio_min != RATIO_MIN else '') + (f'_up{rise}' if rise else ''))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='成交量 > 前 N 根平均，且 type1_ratio >= 門檻')
    ap.add_argument('--n', type=int, default=N_WINDOW, help=f'前幾根算平均（預設 {N_WINDOW}）')
    ap.add_argument('--above', type=float, default=RATIO_MIN, help=f'type1_ratio 門檻（預設 {RATIO_MIN}）')
    ap.add_argument('--mult', type=float, default=VOL_MULT, help=f'成交量 > 幾倍平均（預設 {VOL_MULT:g}）')
    ap.add_argument('--rise', type=int, default=RISE, help=f'close 連漲幾次（預設 {RISE} = 不看）')
    ap.add_argument('--name', help='輸出 marks/<name>（預設 type1_vol<n>，再依參數加 _x<mult>、_r<ratio>、_up<rise>）')
    a = ap.parse_args()
    out = ROOT / 'marks' / (a.name or default_name(a.n, a.above, a.mult, a.rise))
    df_list, key_list = load_frames()
    frames = keep_frames(table=max_bar_table(df_list, key_list))
    n_marks = 0
    for i in frames:
        marks = vol_above_mean(df_list[i], a.n, a.above, a.mult, a.rise)
        n_marks += len(marks)
        save_marks(marks, i, out)
    print(f'n={a.n}, above={a.above}, mult={a.mult:g}, rise={a.rise}：寫好 {len(frames)} frames 的 CSV 到 {out}，共 {n_marks} 個標記')
