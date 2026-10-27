"""S8_alpha 標記的那一分鐘，再檢查 |右邊斜率| > |左邊斜率|（242 frames）。

斜率定義同 analysis/2026-10-26_minute_v_slope/minute_v_slope.py：
    用這一分鐘的最高價切左右（出現多次就合併，中間不看）、% = (價 − 最高價) ÷ 最高價 × 100、單位 % / 秒；
    任一邊不足 N 筆（或時間差 = 0）→ 算不出斜率 → 不通過（不標記）。
規則：S8_alpha 標記 且 |右邊斜率| > |左邊斜率| → 標記
    兩個版本：迴歸斜率（reg）、頭尾斜率（end）
回測用 lib/start_backtest.py（與 strategies/compare.py 相同的 Hit / 點 / 虧損定義）。
mask 只拿得到 1 分 K，看不到 tick，所以沒有放進 registry，在這裡逐 frame 帶入斜率。

輸出（與本檔同資料夾）：summary.csv、marks.csv（S8_alpha 每個點的斜率與是否通過）
執行（在專案根目錄）：python analysis/2026-10-26_s8a_right_slope/s8a_right_slope.py [--n 3]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
sys.path.insert(0, str(ROOT / 'strategies'))
sys.path.insert(0, str(ROOT / 'analysis' / '2026-10-26_minute_v_slope'))

import start_backtest as sb  # noqa: E402
from backtest_report import load_frames  # noqa: E402
from frame_filter import keep_frames, max_bar_table  # noqa: E402
from minute_v_slope import N_MIN, minute_slopes  # noqa: E402
from registry import STRATEGIES  # noqa: E402

BASE = 'S8_alpha'
VARIANTS = {'reg': ('left_reg', 'right_reg'), 'end': ('left_end', 'right_end')}


def steep_right(sl, kind):
    """|右邊斜率| > |左邊斜率|；算不出斜率 = False。"""
    left, right = VARIANTS[kind]
    return (sl[right].abs() > sl[left].abs()).fillna(False)


def run(df_list, keep, slopes, kind):
    base = STRATEGIES[BASE]['mask']
    rows, infos = [], []
    for i, df in enumerate(df_list):
        if kind is None:
            mask_fn = base
        else:
            cond = steep_right(slopes[i], kind)
            mask_fn = lambda mb, c=cond: base(mb).fillna(False) & c.reindex(mb.index, fill_value=False)  # noqa: E731
        r, info = sb.run_frame(i, df, mask_fn)
        rows += r
        infos.append(info)
    detail, frames = pd.DataFrame(rows), pd.DataFrame(infos)
    for d in (detail, frames):
        d.insert(1, 'excluded', ~d['df_idx'].isin(keep))
    return sb.summary(detail, frames).loc['keep']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=N_MIN, help='每一邊至少幾筆成交')
    args = ap.parse_args()

    df_list, key_list = load_frames()
    keep = set(keep_frames(table=max_bar_table(df_list, key_list)))
    slopes = {i: minute_slopes(df, args.n).set_index('minute') for i, df in enumerate(df_list)}

    # S8_alpha 每個點（242 frames）的斜率
    mk = []
    for i in sorted(keep):
        mb, max_time, _ = sb.minute_bars(df_list[i])
        m = STRATEGIES[BASE]['mask'](mb).fillna(False)
        sl = slopes[i].reindex(mb.index[m.to_numpy(dtype=bool)])
        sl.insert(0, 'df_idx', i)
        sl['hit'] = sl.index == max_time.floor('1min')
        for k in VARIANTS:
            sl[f'pass_{k}'] = steep_right(sl, k)
        mk.append(sl.reset_index(names='minute'))
    mk = pd.concat(mk, ignore_index=True)
    mk.to_csv(HERE / 'marks.csv', index=False, encoding='utf-8-sig')

    rows = {BASE: run(df_list, keep, slopes, None)}
    for k in VARIANTS:
        rows[f'{BASE} + 右陡（{k}）'] = run(df_list, keep, slopes, k)
    s = pd.DataFrame(rows).T
    s['precision'] = s['hits'] / s['marks'] * 100
    s.to_csv(HERE / 'summary.csv', encoding='utf-8-sig')

    cols = ['hits', 'hit_rate', 'marks', 'precision', 'no_trade_pct', 'loss_med', 'loss_p90', 'loss_max']
    with pd.option_context('display.width', 200):
        print(f'{len(keep)} frames、N = {args.n}')
        print(s[cols].round(1).to_string())
        print(f'{BASE} 的點 {len(mk)} 個，算不出斜率 {int(mk["skipped"].sum())} 個'
              f'（Hit {int(mk.loc[mk["hit"], "skipped"].sum())}）')
        for k in VARIANTS:
            h, o = mk[mk['hit']], mk[~mk['hit']]
            print(f'{k}：Hit 通過 {int(h[f"pass_{k}"].sum())} / {len(h)}、其他點通過 {int(o[f"pass_{k}"].sum())} / {len(o)}')


if __name__ == '__main__':
    main()
