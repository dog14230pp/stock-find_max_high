"""量大 × 抖得厲害（242 frames）。抖動定義同 tick_jitter.py（k = 2 的比例 rate_2）。

量大        這分鐘的量 > 1.5 × 前 5 分鐘平均（借用 S3；前 5 分鐘 = 前 5 根有成交的 1 分 K）
抖得厲害    rate_2 ≥ 固定門檻（= 全部分鐘 rate_2 的第 80 百分位）
4 格        每格：最高點那分鐘落在這格的 %、一般分鐘的 %、倍數 = 前者 ÷ 後者（1 = 沒資訊）
S8_alpha    184 個點分成抖得厲害 / 不太抖（同一個門檻），各自的 Hit、點數、準度

輸出（與本檔同資料夾）：vol_jitter_grid.csv、vol_jitter_s8a.csv
執行（在專案根目錄）：python analysis/2026-10-27_tick_jitter/vol_jitter.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
sys.path.insert(0, str(ROOT / 'strategies'))
sys.path.insert(0, str(HERE))

import start_backtest as sb  # noqa: E402
from backtest_report import load_frames  # noqa: E402
from frame_filter import keep_frames, max_bar_table  # noqa: E402
from registry import STRATEGIES  # noqa: E402
from tick_jitter import BASE, minute_jitter  # noqa: E402

RATE = 'rate_2'
JIT_Q = 0.8
VOL_MULT = 1.5
VOL_N = 5


def main():
    df_list, key_list = load_frames()
    keep = keep_frames(table=max_bar_table(df_list, key_list))
    parts = []
    for i in keep:
        m = minute_jitter(df_list[i])
        mb, max_time, _ = sb.minute_bars(df_list[i])
        vol = mb['volume'].astype(float)
        m['vol_ratio'] = (vol / vol.rolling(VOL_N).mean().shift(1)).reindex(m.index)
        m['is_max'] = m.index == max_time.floor('1min')
        m['marked'] = STRATEGIES[BASE]['mask'](mb).fillna(False).reindex(m.index, fill_value=False).to_numpy(dtype=bool)
        m.insert(0, 'df_idx', i)
        parts.append(m.reset_index())
    d = pd.concat(parts, ignore_index=True)
    thr = d[RATE].quantile(JIT_Q)
    d['jit_hi'] = d[RATE] >= thr

    g = d[d[RATE].notna() & d['vol_ratio'].notna()]
    big = g['vol_ratio'] > VOL_MULT
    n_max, n_oth = int(g['is_max'].sum()), int((~g['is_max']).sum())
    cells = {'A': (False, False), 'B': (True, False), 'C': (False, True), 'D': (True, True)}
    rows = []
    for c, (b, j) in cells.items():
        inc = (big == b) & (g['jit_hi'] == j)
        a, o = (inc & g['is_max']).sum(), (inc & ~g['is_max']).sum()
        rows.append({'cell': c, 'volume': '量大' if b else '量不大', 'jitter': '抖得厲害' if j else '不太抖',
                     'max_n': int(a), 'max_pct': a / n_max * 100, 'other_n': int(o), 'other_pct': o / n_oth * 100,
                     'ratio': (a / n_max) / (o / n_oth)})
    grid = pd.DataFrame(rows)

    mk = d[d['marked']]
    s8 = []
    for name, sub in (('全部', mk), ('抖得厲害', mk[mk['jit_hi']]), ('不太抖', mk[~mk['jit_hi']])):
        s8.append({'group': name, 'marks': len(sub), 'hits': int(sub['is_max'].sum()),
                   'precision': sub['is_max'].mean() * 100 if len(sub) else np.nan})
    s8 = pd.DataFrame(s8)

    grid.to_csv(HERE / 'vol_jitter_grid.csv', index=False, encoding='utf-8-sig')
    s8.to_csv(HERE / 'vol_jitter_s8a.csv', index=False, encoding='utf-8-sig')
    with pd.option_context('display.width', 200):
        print(f'{len(keep)} frames；抖得厲害：{RATE} ≥ {thr:.2f}%（占全部分鐘 {d["jit_hi"].mean() * 100:.1f}%）；'
              f'量大 > {VOL_MULT} × 前 {VOL_N} 分鐘平均；最高點那分鐘 {n_max}、一般分鐘 {n_oth}')
        print(grid.round(2).to_string(index=False))
        print(f'{BASE} 的點')
        print(s8.round(1).to_string(index=False))


if __name__ == '__main__':
    main()
