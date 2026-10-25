"""每個 frame 最高點所在的那一分鐘：該分鐘內每一秒 type1_ratio 的分布。

最高點 = frame_max.csv 的 max_minute（最高價第一次出現的那筆 tick 所在分鐘）。
最高點落在 frame 資料前 SKIP_BARS 根 1 分 K（從 min/ 第一分鐘起算）的 frame 不看。
只算有成交的秒。

輸入：analysis/2026-10-21_type1_ratio/sec/、min/、frame_max.csv（make.py 產生）
輸出（與本檔同資料夾）：
    max_sec_ratio.csv   每列 = 一個 frame（排除後）：最高點那一分鐘各秒 type1_ratio 的統計
        df_idx, max_minute, max_time, bar_pos（最高點是第幾根 K，0 起算）, n_sec, minute_ratio（該分鐘成交量加權）,
        mean, std, min, q25, median, q75, max, share_ge_0.5 / 0.7 / 0.9（秒數佔比）,
        before_mean / after_mean（最高點那一秒之前 / 之後各秒 ratio 平均，不含那一秒）
    plots/df_XXX.png    單一 frame 的圖（--plot 指定）

執行（在專案根目錄）：
    python analysis/2026-10-23_max_minute_sec_ratio/max_sec_ratio.py            只產生 CSV
    python analysis/2026-10-23_max_minute_sec_ratio/max_sec_ratio.py --plot 0   另外畫 df 0
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / '2026-10-21_type1_ratio'
SKIP_BARS = 10       # 最高點落在前幾根 K 的 frame 不看
SHARES = [0.5, 0.7, 0.9]
CONTEXT = (10, 3)    # 圖 3：最高點前 / 後各畫幾分鐘

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load(i):
    s = pd.read_csv(SRC / 'sec' / f'df_{i:03d}.csv')
    m = pd.read_csv(SRC / 'min' / f'df_{i:03d}.csv')
    return s, m


def frame_stats(i, fmax):
    s, m = load(i)
    mm, mt = fmax.loc[i, 'max_minute'], fmax.loc[i, 'max_time']
    bar_pos = int((m['minute'] < mm).sum())
    if bar_pos < SKIP_BARS:
        return None
    sec = s[s['minute'] == mm]
    r = sec['type1_ratio']
    max_sec = fmax.loc[i, 'max_second']
    row = {'df_idx': i, 'max_minute': mm, 'max_time': mt, 'bar_pos': bar_pos, 'n_sec': len(sec),
           'minute_ratio': m.loc[m['minute'] == mm, 'type1_ratio'].iloc[0],
           'mean': r.mean(), 'std': r.std(), 'min': r.min(), 'q25': r.quantile(.25), 'median': r.median(),
           'q75': r.quantile(.75), 'max': r.max()}
    for th in SHARES:
        row[f'share_ge_{th}'] = (r >= th).mean()
    row['before_mean'] = r[sec['second'] < max_sec].mean()
    row['after_mean'] = r[sec['second'] > max_sec].mean()
    return row


def plot_frame(i, fmax):
    s, m = load(i)
    mm, mt = fmax.loc[i, 'max_minute'], pd.Timestamp(fmax.loc[i, 'max_time'])
    sec = s[s['minute'] == mm].copy()
    sec['t'] = pd.to_datetime(sec['second'])
    up = sec['type1_ratio'] >= 0.5

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
    fig.suptitle(f'df {i:03d}　最高點 {fmax.loc[i, "max_price"]:,.0f} @ {mt:%Y-%m-%d %H:%M:%S.%f}'[:-3]
                 + f'　（第 {int((m["minute"] < mm).sum())} 根 K）', fontsize=14)

    # 1. 最高點那一分鐘：逐秒 ratio + 價格
    ax = fig.add_subplot(gs[0, :])
    ax.bar(sec['t'], sec['type1_ratio'], width=1 / 86400 * 0.8, align='edge',
           color=np.where(up, '#d62728', '#2ca02c'), alpha=0.75)
    ax.axhline(0.5, color='gray', lw=0.8, ls=':')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('每秒 type1_ratio（紅 >= 0.5）')
    ax2 = ax.twinx()
    ax2.step(sec['t'] + pd.Timedelta(seconds=1), sec['close'], where='pre', color='black', lw=1.4, label='每秒收盤')
    ax2.plot(sec['t'] + pd.Timedelta(milliseconds=500), sec['high'], '.', color='black', ms=3, label='每秒最高')
    ax2.set_ylabel('價格')
    ax.axvline(mt, color='orange', lw=2)
    ax2.annotate(f'High {fmax.loc[i, "max_price"]:,.0f}\n{mt:%H:%M:%S}', (mt, fmax.loc[i, 'max_price']),
                 xytext=(8, -4), textcoords='offset points', color='darkorange', va='top')
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M:%S'))
    ax2.legend(loc='upper left', fontsize=9)
    ax.set_title(f'最高點那一分鐘 {mm[11:16]}：{len(sec)} 個有成交的秒', loc='left')

    # 2. 分布：這一分鐘的秒 vs 整個 frame 其他分鐘的秒
    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0, 1, 11)
    rest = s.loc[s['minute'] != mm, 'type1_ratio']
    ax.hist(rest, bins=bins, density=True, color='lightgray', edgecolor='gray', label=f'frame 其他秒（{len(rest):,}）')
    ax.hist(sec['type1_ratio'], bins=bins, density=True, histtype='step', lw=2.2, color='darkorange',
            label=f'最高點那一分鐘（{len(sec)}）')
    for v, c, ls in [(sec['type1_ratio'].median(), 'darkorange', '--'), (rest.median(), 'gray', '--')]:
        ax.axvline(v, color=c, ls=ls, lw=1)
    ax.set_xlabel('每秒 type1_ratio')
    ax.set_ylabel('密度')
    ax.set_title(f'分布（虛線 = 中位數：這分鐘 {sec["type1_ratio"].median():.2f}、其他 {rest.median():.2f}）', loc='left')
    ax.legend(fontsize=9)

    # 3. 前後幾分鐘：每分鐘各秒 ratio 的盒鬚圖 + 該分鐘成交量加權 ratio
    ax = fig.add_subplot(gs[1, 1])
    k = int((m['minute'] < mm).sum())
    mins = m['minute'].iloc[max(0, k - CONTEXT[0]):k + CONTEXT[1] + 1].tolist()
    data = [s.loc[s['minute'] == x, 'type1_ratio'] for x in mins]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
    for patch, x in zip(bp['boxes'], mins):
        patch.set_facecolor('#fdd49e' if x == mm else '#e5e5e5')
    w = m.set_index('minute').loc[mins, 'type1_ratio']
    ax.plot(range(1, len(mins) + 1), w, 'o-', color='#1f77b4', ms=4, lw=1, label='該分鐘 ratio（成交量加權）')
    ax.axhline(0.5, color='gray', lw=0.8, ls=':')
    ax.set_xticks(range(1, len(mins) + 1), [x[11:16] for x in mins], rotation=60, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'最高點前 {CONTEXT[0]} / 後 {CONTEXT[1]} 分鐘：各秒 ratio 的盒鬚圖（橘 = 最高點那分鐘）', loc='left',
                 fontsize=10)
    ax.legend(fontsize=9, loc='lower left')

    fig.tight_layout()
    out = HERE / 'plots' / f'df_{i:03d}.png'
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main(argv):
    fmax = pd.read_csv(SRC / 'frame_max.csv').set_index('df_idx')
    rows = [r for i in fmax.index if (r := frame_stats(i, fmax)) is not None]
    res = pd.DataFrame(rows)
    res.to_csv(HERE / 'max_sec_ratio.csv', index=False)
    print(f'{len(fmax)} 個 frame，最高點在前 {SKIP_BARS} 根 K 的排除 {len(fmax) - len(res)} 個 → '
          f'{len(res)} 個寫到 {HERE / "max_sec_ratio.csv"}')
    print(res.drop(columns=['df_idx', 'max_minute', 'max_time']).describe().round(3).to_string())
    if '--plot' in argv:
        for i in argv[argv.index('--plot') + 1:]:
            print('圖：', plot_frame(int(i), fmax))


if __name__ == '__main__':
    main(sys.argv)
