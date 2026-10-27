"""一分鐘內的抖動：跳得特別大的成交有幾筆（242 frames）。

跳動    同一分鐘內，每筆成交跟上一筆的價差 % 取絕對值：|p[i] − p[i−1]| ÷ p[i−1] × 100（往上往下都算）
        每分鐘第一筆不算（不跟上一分鐘比）；跳動 = 0 也算進平均和標準差
門檻    前 PREV 分鐘（有成交的 1 分 K）所有跳動的 平均 + k × 標準差（只用過去資料）；前面不足 PREV 分鐘 = 空的
抖動    跳動 > 門檻的筆數（jit_k）；比例 = 筆數 ÷ 這分鐘的跳動數（rate_k）；k = 1.5 / 2 / 3

比較
    1. 最高點那分鐘 vs 一般分鐘：中位數、最高點那分鐘在 frame 內的平均百分位（50 = 沒資訊）
    2. S8_alpha 的點：Hit vs 其他點的中位數、Hit 在所有點裡的平均百分位

輸出（與本檔同資料夾）：minutes.csv、summary.csv
執行（在專案根目錄）：python analysis/2026-10-27_tick_jitter/tick_jitter.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
sys.path.insert(0, str(ROOT / 'strategies'))

import start_backtest as sb  # noqa: E402
from backtest_report import load_frames  # noqa: E402
from frame_filter import keep_frames, max_bar_table  # noqa: E402
from registry import STRATEGIES  # noqa: E402

PREV = 5
KS = [1.5, 2, 3]
BASE = 'S8_alpha'


def minute_jitter(df):
    t = df[['ts', 'close']].copy()
    t['ts'] = pd.to_datetime(t['ts'])
    t = t.sort_values('ts', kind='stable')
    t['minute'] = t['ts'].dt.floor('1min')
    p = t['close'].astype(float)
    t['jump'] = ((p - p.shift(1)).abs() / p.shift(1) * 100).where(t['minute'] == t['minute'].shift(1))
    j = t.dropna(subset=['jump'])
    g = j.groupby('minute')['jump']
    m = pd.DataFrame({'ticks': t.groupby('minute').size()})
    m['n_jumps'] = g.size().reindex(m.index).fillna(0)
    s1, s2 = g.sum().reindex(m.index).fillna(0), (j['jump'] ** 2).groupby(j['minute']).sum().reindex(m.index).fillna(0)
    n, a, b = (x.shift(1).rolling(PREV).sum() for x in (m['n_jumps'], s1, s2))
    m['prev_mean'] = a / n
    m['prev_std'] = np.sqrt(((b - a * a / n) / (n - 1)).clip(lower=0)).where(n > 1)
    for k in KS:
        thr = (m['prev_mean'] + k * m['prev_std']).reindex(j['minute']).to_numpy()
        over = pd.Series(j['jump'].to_numpy() > thr, index=j['minute'])
        cnt = over.groupby(level=0).sum().reindex(m.index).fillna(0)
        m[f'jit_{k}'] = cnt.where(m['prev_std'].notna())
        m[f'rate_{k}'] = (cnt / m['n_jumps'] * 100).where(m['prev_std'].notna() & (m['n_jumps'] > 0))
    return m


def main():
    df_list, key_list = load_frames()
    keep = keep_frames(table=max_bar_table(df_list, key_list))
    cols = [f'{p}_{k}' for k in KS for p in ('jit', 'rate')]
    parts = []
    for i in keep:
        m = minute_jitter(df_list[i])
        mb, max_time, _ = sb.minute_bars(df_list[i])
        m['is_max'] = m.index == max_time.floor('1min')
        m['marked'] = STRATEGIES[BASE]['mask'](mb).fillna(False).reindex(m.index, fill_value=False).to_numpy(dtype=bool)
        for c in cols:
            m[f'{c}_pct'] = m[c].rank(pct=True) * 100
        m.insert(0, 'df_idx', i)
        parts.append(m.reset_index())
    d = pd.concat(parts, ignore_index=True)
    d.to_csv(HERE / 'minutes.csv', index=False, encoding='utf-8-sig')

    top, other = d[d['is_max']], d[~d['is_max']]
    mk = d[d['marked']].copy()
    out = []
    for c in cols:
        mk[f'{c}_mpct'] = mk[c].rank(pct=True) * 100
        h, o = mk[mk['is_max']], mk[~mk['is_max']]
        out.append({'feature': c, 'max_median': top[c].median(), 'other_median': other[c].median(),
                    'max_pctile_mean': top[f'{c}_pct'].mean(),
                    'hit_median': h[c].median(), 'other_mark_median': o[c].median(),
                    'hit_pctile_in_marks': h[f'{c}_mpct'].mean()})
    s = pd.DataFrame(out)
    s.to_csv(HERE / 'summary.csv', index=False, encoding='utf-8-sig')

    with pd.option_context('display.width', 200):
        print(f'{len(keep)} frames、{len(d)} 分鐘（有門檻 {int(d["prev_std"].notna().sum())}）、'
              f'{BASE} {len(mk)} 個點（Hit {int(mk["is_max"].sum())}）')
        print('跳動中位數（%）', round(d['prev_mean'].median(), 5), ' 標準差中位數', round(d['prev_std'].median(), 5))
        print(s.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
