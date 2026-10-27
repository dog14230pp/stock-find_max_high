"""不同分鐘之間的斜率變化（242 frames）。斜率定義、N、忽略規則同 analysis/2026-10-26_minute_v_slope。

A  以最高點那分鐘為 0，看前後各 WIN 分鐘（有成交的 1 分 K）左 / 右迴歸斜率的中位數與中間 50%
B  這分鐘跟前 5 分鐘比（只用過去資料）：
     左比  = left_reg[t] ÷ 前 5 分鐘 |left_reg| 的平均
     右比  = |right_reg[t]| ÷ 前 5 分鐘 |right_reg| 的平均
     前 5 分鐘至少要有 MIN_PREV 分鐘算得出斜率，否則空的；這分鐘算不出斜率也是空的
   比較：最高點那分鐘在 frame 內的平均百分位（50 = 沒資訊）；比值 > 門檻的比例，最高點 vs 一般分鐘、倍數

輸出（與本檔同資料夾）：event.csv、event.png、ratio_pctile.csv、ratio_thresh.csv
執行（在專案根目錄）：python analysis/2026-10-26_slope_across_minutes/slope_across.py [--n 3]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
sys.path.insert(0, str(ROOT / 'analysis' / '2026-10-26_minute_v_slope'))

from backtest_report import bars_of, load_frames  # noqa: E402
from frame_filter import keep_frames, max_bar_table  # noqa: E402
from minute_v_slope import N_MIN, minute_slopes  # noqa: E402

WIN = 10           # A：最高點前後各幾分鐘
PREV = 5           # B：跟前幾分鐘比
MIN_PREV = 3       # B：前 PREV 分鐘至少幾分鐘算得出斜率
THRESH = [1.5, 2, 3]


def plot_event(ev, base, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, k, name in ((axes[0], 'left_reg', '左邊斜率（爬升，% / 秒）'), (axes[1], 'right_reg', '右邊斜率（下墜，% / 秒）')):
        x = ev.index
        ax.fill_between(x, ev[f'{k}_q25'], ev[f'{k}_q75'], color='#2a78d6', alpha=0.15, linewidth=0)
        ax.plot(x, ev[f'{k}_med'], color='#2a78d6', linewidth=2, marker='o', markersize=4)
        ax.axhline(base[k], color='#999', linestyle='--', linewidth=1)
        ax.text(x[0], base[k], ' 全部分鐘中位數', color='#52514e', fontsize=8, va='bottom')
        ax.axvline(0, color='#0b0b0b', linestyle=':', linewidth=1)
        ax.set_title(name, loc='left', fontsize=11)
        ax.set_xlabel('離最高點那分鐘幾分鐘（0 = 最高點）')
        ax.set_xticks(range(-WIN, WIN + 1, 2))
        ax.grid(axis='y', color='#e5e5e5', linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    fig.suptitle('242 frames：最高點前後的斜率（線 = 中位數，色帶 = 中間 50%）', x=0.01, ha='left', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=N_MIN, help='每一邊至少幾筆成交')
    args = ap.parse_args()

    df_list, key_list = load_frames()
    keep = keep_frames(table=max_bar_table(df_list, key_list))
    parts = []
    for i in keep:
        m = minute_slopes(df_list[i], args.n)
        _, max_time, _ = bars_of(df_list[i])
        m.insert(0, 'df_idx', i)
        m['is_max'] = m['minute'] == max_time.floor('1min')
        m['offset'] = np.arange(len(m)) - int(np.flatnonzero(m['is_max'])[0])
        for k, name in (('left_reg', 'left_ratio'), ('right_reg', 'right_ratio')):
            prev = m[k].abs().shift(1).rolling(PREV, min_periods=MIN_PREV).mean()
            cur = m[k] if k == 'left_reg' else m[k].abs()
            m[name] = cur / prev.where(prev > 0)
            m[f'{name}_pct'] = m[name].rank(pct=True) * 100
        parts.append(m)
    d = pd.concat(parts, ignore_index=True)

    # A
    w = d[d['offset'].between(-WIN, WIN)]
    ev = w.groupby('offset').agg(n=('left_reg', 'count'),
                                 **{f'{k}_{s}': (k, f) for k in ('left_reg', 'right_reg')
                                    for s, f in (('med', 'median'), ('q25', lambda x: x.quantile(.25)),
                                                 ('q75', lambda x: x.quantile(.75)))})
    base = {k: d[k].median() for k in ('left_reg', 'right_reg')}
    ev.to_csv(HERE / 'event.csv', encoding='utf-8-sig')
    plot_event(ev, base, HERE / 'event.png')

    # B
    top, other = d[d['is_max']], d[~d['is_max']]
    pc, th = [], []
    for k in ('left_ratio', 'right_ratio'):
        t = top[top[k].notna()]
        pc.append({'ratio': k, 'max_n': len(t), 'max_median': t[k].median(), 'other_median': other[k].median(),
                   'max_pctile_mean': t[f'{k}_pct'].mean(),
                   'max_in_low20': (t[f'{k}_pct'] <= 20).mean() * 100,
                   'max_in_high20': (t[f'{k}_pct'] > 80).mean() * 100})
        o = other[other[k].notna()]
        for c in THRESH:
            a, b = (t[k] > c).mean() * 100, (o[k] > c).mean() * 100
            th.append({'ratio': k, 'thresh': c, 'max_pct': a, 'other_pct': b, 'mult': a / b})
    pc, th = pd.DataFrame(pc), pd.DataFrame(th)
    pc.to_csv(HERE / 'ratio_pctile.csv', index=False, encoding='utf-8-sig')
    th.to_csv(HERE / 'ratio_thresh.csv', index=False, encoding='utf-8-sig')

    with pd.option_context('display.width', 200):
        print(f'{len(keep)} frames、{len(d)} 分鐘、N = {args.n}')
        print('A：全部分鐘中位數', {k: round(v, 5) for k, v in base.items()})
        print(ev[['n', 'left_reg_med', 'right_reg_med']].round(5).to_string())
        print('B：百分位')
        print(pc.round(2).to_string(index=False))
        print('B：比值 > 門檻的比例（%）')
        print(th.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
