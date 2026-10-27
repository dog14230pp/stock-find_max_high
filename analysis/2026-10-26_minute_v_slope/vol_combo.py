"""右邊斜率 × 成交量（242 frames）。切法、N、忽略規則同 minute_v_slope.py。

想法 1  右邊量占比 = 最後一次最高價之後的成交量 ÷ 這一分鐘總量（最高價那幾筆不算右邊）
        比較：最高點那分鐘在 frame 內平均第幾百分位（50 = 沒資訊）
想法 2  4 格：右邊平平 / 掉得陡 × 量大 / 量不大
        平平  右邊迴歸斜率 ≥ 固定門檻（= 全部分鐘右邊迴歸斜率的第 80 百分位，最平的 20%）
        量大  這分鐘的量 > 1.5 × 前 5 分鐘平均（借用 S3 的定義；前 5 分鐘 = 前 5 根有成交的 1 分 K）
        每格：最高點那分鐘落在這格的 %、一般分鐘的 %、倍數 = 前者 ÷ 後者（1 = 沒資訊）

輸出（與本檔同資料夾）：vol_minutes.csv、vol_share.csv、vol_grid.csv、vol_share.png
執行（在專案根目錄）：python analysis/2026-10-26_minute_v_slope/vol_combo.py [--n 3]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
sys.path.insert(0, str(HERE))

from backtest_report import bars_of, load_frames  # noqa: E402
from frame_filter import keep_frames, max_bar_table  # noqa: E402
from minute_v_slope import N_MIN, minute_slopes  # noqa: E402

FLAT_Q = 0.8       # 右邊迴歸斜率 ≥ 第 80 百分位 = 平平
VOL_MULT = 1.5     # 量 > 1.5 × 前 5 分鐘平均 = 量大
VOL_N = 5


def plot(ok, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 100, 41)
    for sub, name, color in ((ok[~ok['is_max']], '一般分鐘', '#2a78d6'),
                             (ok[ok['is_max']], '最高點那分鐘', '#eb6834')):
        ax.hist(sub['right_share'], bins=bins, density=True, histtype='step', linewidth=2, color=color,
                label=f'{name}（{len(sub)}）')
        ax.axvline(sub['right_share'].median(), color=color, linestyle=':', linewidth=1.5)
    ax.set_title('242 frames：右邊量占整分鐘的 %（虛線 = 中位數）', loc='left', fontsize=11)
    ax.set_xlabel('右邊量占比（%）')
    ax.set_yticks([])
    ax.grid(axis='y', color='#e5e5e5', linewidth=0.6)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.legend(loc='upper right', fontsize=9, frameon=False)
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
        m['vol_ratio'] = m['volume'] / m['volume'].rolling(VOL_N).mean().shift(1)
        m['right_share'] = (m['right_vol'] / m['volume'] * 100).where(~m['skipped'])
        m['right_share_pct'] = m['right_share'].rank(pct=True) * 100
        parts.append(m)
    d = pd.concat(parts, ignore_index=True)
    ok = d[~d['skipped']]

    # 想法 1
    top = ok[ok['is_max']]
    share = pd.DataFrame([{'max_median': top['right_share'].median(),
                           'other_median': ok.loc[~ok['is_max'], 'right_share'].median(),
                           'max_pctile_mean': top['right_share_pct'].mean(),
                           'max_in_low20': (top['right_share_pct'] <= 20).mean() * 100,
                           'max_in_high20': (top['right_share_pct'] > 80).mean() * 100}])

    # 想法 2
    flat_thr = ok['right_reg'].quantile(FLAT_Q)
    g = ok[ok['vol_ratio'].notna()].copy()
    g['flat'] = np.where(g['right_reg'] >= flat_thr, '平平', '掉得陡')
    g['big'] = np.where(g['vol_ratio'] > VOL_MULT, '量大', '量不大')
    cells = {'A': ('平平', '量不大'), 'B': ('平平', '量大'), 'C': ('掉得陡', '量不大'), 'D': ('掉得陡', '量大')}
    n_max, n_oth = int(g['is_max'].sum()), int((~g['is_max']).sum())
    rows = []
    for c, (f, b) in cells.items():
        inc = (g['flat'] == f) & (g['big'] == b)
        a, o = (inc & g['is_max']).sum(), (inc & ~g['is_max']).sum()
        rows.append({'cell': c, 'slope': f, 'volume': b, 'max_n': int(a), 'max_pct': a / n_max * 100,
                     'other_n': int(o), 'other_pct': o / n_oth * 100,
                     'ratio': (a / n_max) / (o / n_oth)})
    grid = pd.DataFrame(rows)

    d.to_csv(HERE / 'vol_minutes.csv', index=False, encoding='utf-8-sig')
    share.to_csv(HERE / 'vol_share.csv', index=False, encoding='utf-8-sig')
    grid.to_csv(HERE / 'vol_grid.csv', index=False, encoding='utf-8-sig')
    plot(ok, HERE / 'vol_share.png')

    with pd.option_context('display.width', 200):
        print(f'{len(keep)} frames、{len(d)} 分鐘、N = {args.n}、算得出斜率 {len(ok)}')
        print('想法 1：右邊量占比（%）')
        print(share.round(2).to_string(index=False))
        print(f'想法 2：平平門檻 右邊迴歸斜率 ≥ {flat_thr:.5f} % / 秒；量大 > {VOL_MULT} × 前 {VOL_N} 分鐘平均；'
              f'最高點那分鐘 {n_max}、一般分鐘 {n_oth}（前 {VOL_N} 分鐘沒有量基準，不算）')
        print(grid.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
