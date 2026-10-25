"""畫單一起點的回測案例：起點 s、最壞買點（s ~ 標記之間最高價）、標記、賣點（標記下一根 open）。

三層：K 線、成交量（+ 門檻線）、type1_ratio。策略參數與 backtest.py 相同。
輸出：cases/df_XXX_sYY.png

執行（在專案根目錄）：python analysis/2026-10-24_start_backtest/plot_case.py 45 0
    參數：df_idx  起點是第幾根（start_pos，見 detail.csv）
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from backtest import MULT, N, RATIO_MIN, RISE, bars_of, load_frames, tv  # noqa: E402

plt.rcParams['font.family'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def main(i, s):
    df_list, key_list = load_frames()
    df, key = df_list[i], key_list[i]
    bars, max_time, max_price = bars_of(df)
    m = tv.minute_table(df).reindex(bars.index)
    mark, thr = tv.vol_mask(m, N, RATIO_MIN, MULT, RISE)
    marks = np.flatnonzero(mark.to_numpy())
    ms = marks[marks >= s]
    x = np.arange(len(bars))
    max_x = bars.index.get_loc(max_time.floor('1min'))

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(14, 8.5), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1.2, 1.2]})
    col = np.where(bars['Close'] >= bars['Open'], '#d62728', '#2ca02c')     # 紅漲綠跌
    a1.vlines(x, bars['Low'], bars['High'], color=col, linewidth=1)
    a1.bar(x, (bars['Close'] - bars['Open']).abs().clip(lower=0.5),
           bottom=bars[['Open', 'Close']].min(axis=1), color=col, width=0.7)
    in_frame = np.flatnonzero((bars.index >= key[0]) & (bars.index <= key[1]))
    for a in (a1, a2, a3):
        if len(in_frame):
            a.axvspan(in_frame[0] - 0.5, in_frame[-1] + 0.5, color='#888', alpha=0.08, linewidth=0)
        a.axvline(s - 0.5, color='#9467bd', linewidth=1.5)
    span = bars['High'].max() - bars['Low'].min()
    a1.text(s - 0.3, bars['High'].max() + span * 0.10, f'起點 s {bars.index[s]:%H:%M}', color='#9467bd', fontsize=10)
    a1.scatter(marks, bars['High'].iloc[marks] + span * 0.04, marker='*', s=180, color='#1f77b4', zorder=5)
    a1.scatter([max_x], [max_price], marker='D', s=50, color='black', zorder=6)

    title = f'df {i:03d}   {key[0]:%Y-%m-%d %H:%M} ~ {key[1]:%H:%M}   起點 {bars.index[s]:%H:%M}'
    if len(ms) and ms[0] + 1 < len(bars):
        t = ms[0]
        b = s + int(bars['High'].iloc[s:t + 1].to_numpy().argmax())
        buy, sell = bars['High'].iloc[b], bars['Open'].iloc[t + 1]
        a1.scatter([b], [buy], marker='v', s=120, color='#ff7f0e', zorder=7)
        a1.scatter([t + 1], [sell], marker='o', s=90, facecolor='none', edgecolor='#ff7f0e', linewidth=2, zorder=7)
        a1.annotate(f'最壞買點 {buy:g}（{bars.index[b]:%H:%M}）', (b, buy), xytext=(8, 8),
                    textcoords='offset points', color='#ff7f0e', fontsize=10)
        a1.annotate(f'賣出 {sell:g}（{bars.index[t + 1]:%H:%M} open）', (t + 1, sell), xytext=(8, -16),
                    textcoords='offset points', color='#ff7f0e', fontsize=10)
        a1.plot([b, t + 1], [buy, buy], color='#ff7f0e', linestyle='--', linewidth=1)
        a1.annotate('', (t + 1, sell), (t + 1, buy), arrowprops=dict(arrowstyle='<->', color='#ff7f0e'))
        title += f'   第一個標記 {bars.index[t]:%H:%M}   虧損 {buy - sell:g} 點'
    a1.set_ylim(top=bars['High'].max() + span * 0.16)
    a1.set_title(title)

    a2.bar(x, m['volume'], color=np.where(mark, '#1f77b4', '#bbb'), width=0.8)
    a2.plot(x, thr, color='#ff7f0e', linewidth=1.2, label=f'{MULT:g} × 前 {N} 根平均')
    a2.set_ylabel('volume')
    a2.legend(loc='upper left', fontsize=8)
    a3.plot(x, m['ratio'], color='#555', linewidth=1)
    a3.scatter(marks, m['ratio'].iloc[marks], color='#1f77b4', s=18, zorder=5)
    a3.axhline(RATIO_MIN, color='#ff7f0e', linewidth=1.2, label=f'ratio = {RATIO_MIN:g}')
    a3.set_ylabel('type1_ratio')
    a3.set_ylim(0, 1)
    a3.legend(loc='upper left', fontsize=8)
    step = max(1, len(bars) // 15)
    a3.set_xticks(x[::step])
    a3.set_xticklabels([f'{t:%H:%M}' for t in bars.index[::step]], fontsize=8)
    a3.set_xlim(-1, len(bars))
    fig.tight_layout()
    out = HERE / 'cases' / f'df_{i:03d}_s{s:02d}.png'
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=80)
    print(out)


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
