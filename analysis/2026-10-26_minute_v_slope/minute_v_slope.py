"""一分鐘內的倒 V：用最高點把每分鐘切成左右兩段，各算價格斜率（242 frames）。

每一分鐘（只用這一分鐘自己的 tick，收盤時就算得出來）：
    最高點  這一分鐘的最高價；出現好幾次就合併，左邊到第一次、右邊從最後一次，中間那段不看
    %       (成交價 − 最高價) ÷ 最高價 × 100（最高點 = 0%）
    左邊    第一筆 ~ 第一次最高價；右邊  最後一次最高價 ~ 最後一筆（兩邊都含最高價那筆）
    斜率    單位 % / 秒，兩種算法：
              頭尾  (最後一筆 % − 第一筆 %) ÷ 時間差
              迴歸  所有成交 (秒, %) 的最小平方迴歸線斜率
    忽略    任一邊不足 N 筆，或頭尾時間差 = 0 → 這一分鐘不算（4 個斜率都是空的）

比較：最高點那一分鐘（frame 最高價第一次出現的那分鐘）vs 一般分鐘
    百分位  在每個 frame 內，把有算出斜率的分鐘排名次（小 → 大），最高點那分鐘落在第幾百分位，242 個平均
            50 = 沒資訊；右邊斜率越小（越負）= 掉得越陡

輸出（與本檔同資料夾）：minutes.csv、summary.csv、slopes.png
執行（在專案根目錄）：python analysis/2026-10-26_minute_v_slope/minute_v_slope.py [--n 3]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'lib'))

from backtest_report import bars_of, load_frames  # noqa: E402
from frame_filter import keep_frames, max_bar_table  # noqa: E402

N_MIN = 3          # 每一邊至少幾筆成交（含最高價那筆）
SLOPES = ['left_end', 'left_reg', 'right_end', 'right_reg']
LABELS = {'left_end': '左・頭尾', 'left_reg': '左・迴歸', 'right_end': '右・頭尾', 'right_reg': '右・迴歸'}


def side_slopes(sec, pct):
    """回傳 (頭尾斜率, 迴歸斜率)；時間差 = 0 回傳 (nan, nan)。"""
    dt = sec[-1] - sec[0]
    if dt <= 0:
        return np.nan, np.nan
    x = sec - sec.mean()
    return (pct[-1] - pct[0]) / dt, (x * (pct - pct.mean())).sum() / (x * x).sum()


def minute_slopes(df, n_min):
    t = df[['ts', 'close', 'volume']].copy()
    t['ts'] = pd.to_datetime(t['ts'])
    t = t.sort_values('ts', kind='stable')
    rows = []
    for m, g in t.groupby(t['ts'].dt.floor('1min')):
        p = g['close'].to_numpy(dtype=float)
        v = g['volume'].to_numpy(dtype=float)
        sec = (g['ts'] - m).dt.total_seconds().to_numpy()
        hi = p.max()
        on_hi = np.flatnonzero(p == hi)
        first, last = on_hi[0], on_hi[-1]
        pct = (p - hi) / hi * 100
        r = {'minute': m, 'ticks': len(p), 'n_left': first + 1, 'n_right': len(p) - last,
             'hi_first_sec': sec[first], 'hi_last_sec': sec[last],
             'volume': v.sum(), 'right_vol': v[last + 1:].sum()}      # right_vol 不含最高價那筆
        r.update(dict.fromkeys(SLOPES, np.nan))
        if r['n_left'] >= n_min and r['n_right'] >= n_min:
            le, lr = side_slopes(sec[:first + 1], pct[:first + 1])
            re, rr = side_slopes(sec[last:], pct[last:])
            if not np.isnan(le) and not np.isnan(re):
                r.update(left_end=le, left_reg=lr, right_end=re, right_reg=rr)
        r['skipped'] = np.isnan(r['left_end'])
        rows.append(r)
    return pd.DataFrame(rows)


def plot(d, path):
    """2×2：4 個斜率的分布（密度），一般分鐘 vs 最高點那分鐘。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    ok = d[~d['skipped']]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, k in zip(axes.ravel(), SLOPES):
        lo, hi = ok[k].quantile([0.01, 0.99])
        bins = np.linspace(lo, hi, 50)
        for sub, name, color in ((ok[~ok['is_max']], '一般分鐘', '#2a78d6'),
                                 (ok[ok['is_max']], '最高點那分鐘', '#eb6834')):
            ax.hist(sub[k].clip(lo, hi), bins=bins, density=True, histtype='step', linewidth=2, color=color,
                    label=f'{name}（{len(sub)}）')
            ax.axvline(sub[k].median(), color=color, linestyle=':', linewidth=1.5)
        ax.axvline(0, color='#999', linewidth=0.8)
        ax.set_title(f'{LABELS[k]}（% / 秒）', loc='left', fontsize=11)
        ax.grid(axis='y', color='#e5e5e5', linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_yticks([])
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
    axes[0, 0].legend(loc='upper left', fontsize=9, frameon=False)
    fig.suptitle('242 frames：一分鐘內左右兩段的斜率分布（虛線 = 中位數，頭尾 1% 夾到邊上）', x=0.01, ha='left',
                 fontsize=12)
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
        for k in SLOPES:
            m[f'{k}_pct'] = m[k].rank(pct=True) * 100
        parts.append(m)
    d = pd.concat(parts, ignore_index=True)
    d.to_csv(HERE / 'minutes.csv', index=False, encoding='utf-8-sig')

    top, other = d[d['is_max']], d[~d['is_max']]
    top_ok = top[~top['skipped']]
    out = []
    for k in SLOPES:
        out.append({'slope': k, 'max_median': top_ok[k].median(), 'other_median': other[k].median(),
                    'max_pctile_mean': top_ok[f'{k}_pct'].mean(),
                    'max_in_low20': (top_ok[f'{k}_pct'] <= 20).mean() * 100,
                    'max_in_high20': (top_ok[f'{k}_pct'] > 80).mean() * 100})
    s = pd.DataFrame(out)
    s.to_csv(HERE / 'summary.csv', index=False, encoding='utf-8-sig')
    plot(d, HERE / 'slopes.png')

    with pd.option_context('display.width', 200):
        print(f'{len(keep)} frames、{len(d)} 分鐘、N = {args.n}')
        print(f'忽略：一般分鐘 {int(other["skipped"].sum())} / {len(other)}'
              f'（{other["skipped"].mean() * 100:.1f}%）、最高點那分鐘 {int(top["skipped"].sum())} / {len(top)}'
              f'（{top["skipped"].mean() * 100:.1f}%）')
        print(s.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
