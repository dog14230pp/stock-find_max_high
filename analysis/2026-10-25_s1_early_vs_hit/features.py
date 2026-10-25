"""S1 找到的點：Hit vs 太早的點，特徵比較（242 frames）。

每個 S1 標記 t 依「離最高點幾根」分組：
    hit      最高點那根
    early    最高點前 1～10 根
    earlier  最高點前 > 10 根
    after    最高點之後
每個點算特徵（只用 t 與之前的資料；不用 frame_start / frame_end，因為 frame 是知道最高點後才標的）：
    vol_x5        成交量 / 前 5 根平均
    vol_x20       成交量 / 前 20 根平均
    vol_vs_max    成交量 / 之前所有分鐘的最大成交量（> 1 = 資料開始以來最大量）
    ratio         type1_ratio
    body          close − open（點）
    up_wick_pct   上影線 / 整根長度（%）
    rise_10/20/30 close − 前 N 根（含 t）最低價（點）= 這波已經漲了多少
    rise_all      close − 資料開始以來最低價（點）
    new_high_20   這根 high >= 前 20 根最高價（1 / 0）
    new_high_all  這根 high >= 資料開始以來最高價（1 / 0）
    up_run        close 連漲幾次（到 t 為止）
    prior_marks   這個 frame 在 t 之前已經有幾個 S1 標記
    pos           t 是資料的第幾根

輸出（與本檔同資料夾）：marks.csv（每個點一列）、summary.csv（各組中位數）、
                       threshold.csv（單一門檻能刷掉多少太早的點、保留多少 Hit）、index.html、box.png
執行（在專案根目錄）：python analysis/2026-10-25_s1_early_vs_hit/features.py
"""
import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
sys.path.insert(0, str(ROOT / 'strategies'))

from backtest_report import load_frames  # noqa: E402
from frame_filter import keep_frames, max_bar_table  # noqa: E402
from registry import STRATEGIES  # noqa: E402
from start_backtest import CSS, minute_bars  # noqa: E402

GROUPS = ['hit', 'early', 'earlier', 'after']
GNAME = {'hit': 'Hit（最高點那根）', 'early': '太早（前 1～10 根）', 'earlier': '更早（前 > 10 根）',
         'after': '最高點之後'}
FEATS = {
    'vol_x5': ('成交量 / 前 5 根平均', '{:.2f}'), 'vol_x20': ('成交量 / 前 20 根平均', '{:.2f}'),
    'vol_vs_max': ('成交量 / 之前最大量', '{:.2f}'), 'ratio': ('type1_ratio', '{:.2f}'),
    'body': ('K 棒實體（點）', '{:.0f}'), 'up_wick_pct': ('上影線 %', '{:.0f}'),
    'rise_10': ('離前 10 根低點（點）', '{:.0f}'), 'rise_20': ('離前 20 根低點（點）', '{:.0f}'),
    'rise_30': ('離前 30 根低點（點）', '{:.0f}'), 'rise_all': ('離資料開始最低點（點）', '{:.0f}'),
    'new_high_20': ('創前 20 根新高', '{:.0%}'), 'new_high_all': ('創資料開始以來新高', '{:.0%}'),
    'up_run': ('close 連漲次數', '{:.1f}'), 'prior_marks': ('之前已有幾個標記', '{:.1f}'),
    'pos': ('第幾根', '{:.0f}'),
}
BINARY = {'new_high_20', 'new_high_all'}


def features(mb, t, marks):
    v, h, lo, c = mb['volume'].to_numpy(), mb['high'].to_numpy(), mb['low'].to_numpy(), mb['close'].to_numpy()
    o = mb['open'].to_numpy()
    rng = h[t] - lo[t]
    ups = 0
    while t - ups - 1 >= 0 and c[t - ups] > c[t - ups - 1]:
        ups += 1
    return {
        'vol_x5': v[t] / v[t - 5:t].mean() if t >= 5 else np.nan,
        'vol_x20': v[t] / v[max(0, t - 20):t].mean() if t >= 1 else np.nan,
        'vol_vs_max': v[t] / v[:t].max() if t >= 1 else np.nan,
        'ratio': mb['ratio'].iloc[t],
        'body': c[t] - o[t],
        'up_wick_pct': (h[t] - max(o[t], c[t])) / rng * 100 if rng else 0,
        'rise_10': c[t] - lo[max(0, t - 9):t + 1].min(),
        'rise_20': c[t] - lo[max(0, t - 19):t + 1].min(),
        'rise_30': c[t] - lo[max(0, t - 29):t + 1].min(),
        'rise_all': c[t] - lo[:t + 1].min(),
        'new_high_20': int(h[t] >= h[max(0, t - 20):t].max()) if t >= 1 else np.nan,
        'new_high_all': int(h[t] >= h[:t].max()) if t >= 1 else np.nan,
        'up_run': ups,
        'prior_marks': int((marks < t).sum()),
        'pos': t,
    }


def threshold_table(d):
    """每個特徵：只保留 >= 門檻（或 <=）的點時，Hit 留幾個、太早的點刷掉幾個。門檻取 Hit 的分位數。"""
    hit, early = d[d['group'] == 'hit'], d[d['group'] == 'early']
    rows = []
    for f in FEATS:
        if f in BINARY:
            cands = [('>=', 1)]
        else:
            qs = hit[f].quantile([.1, .25, .5]).tolist() + hit[f].quantile([.9, .75, .5]).tolist()
            cands = [('>=', x) for x in qs[:3]] + [('<=', x) for x in qs[3:]]
        for op, x in cands:
            keep = (lambda s: s >= x) if op == '>=' else (lambda s: s <= x)
            rows.append({'feature': f, 'rule': f'{op} {x:.4g}', 'hit_kept': int(keep(hit[f]).sum()),
                         'early_kept': int(keep(early[f]).sum()),
                         'all_kept': int(keep(d[f]).sum())})
    t = pd.DataFrame(rows)
    t['hit_kept_pct'] = t['hit_kept'] / len(hit) * 100
    t['early_removed_pct'] = 100 - t['early_kept'] / len(early) * 100
    t['all_removed_pct'] = 100 - t['all_kept'] / len(d) * 100
    return t


def box_plot(d, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    show = ['vol_x5', 'vol_x20', 'vol_vs_max', 'ratio', 'rise_10', 'rise_30', 'rise_all', 'up_wick_pct']
    groups = ['hit', 'early', 'earlier']
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for ax, f in zip(axes.flat, show):
        data = [d.loc[d['group'] == g, f].dropna() for g in groups]
        ax.boxplot(data, widths=0.5, showfliers=False, medianprops=dict(color='#1f6fd1', linewidth=2))
        for k, s in enumerate(data, 1):
            ax.scatter(np.random.default_rng(0).normal(k, 0.06, len(s)), s, s=8, color='#555', alpha=0.35)
        ax.set_xticks([1, 2, 3], ['Hit', '太早\n前1～10根', '更早\n前>10根'], fontsize=9)
        ax.set_title(FEATS[f][0], fontsize=10)
        ax.grid(axis='y', color='#e5e5e5', linewidth=0.6)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    df_list, key_list = load_frames()
    keep = keep_frames(table=max_bar_table(df_list, key_list))
    mask_fn = STRATEGIES['S1']['mask']
    rows = []
    for i in keep:
        mb, max_time, _ = minute_bars(df_list[i])
        marks = np.flatnonzero(mask_fn(mb).fillna(False).to_numpy(dtype=bool))
        max_bar = mb.index.get_loc(max_time.floor('1min'))
        for t in marks:
            off = t - max_bar
            g = 'hit' if off == 0 else 'early' if -10 <= off < 0 else 'earlier' if off < -10 else 'after'
            rows.append({'df_idx': i, 'time': mb.index[t], 'bars_from_max': off, 'group': g,
                         **features(mb, t, marks)})
    d = pd.DataFrame(rows)
    d.to_csv(HERE / 'marks.csv', index=False, encoding='utf-8-sig')
    summ = d.groupby('group')[list(FEATS)].median().reindex(GROUPS)
    for f in BINARY:
        summ[f] = d.groupby('group')[f].mean().reindex(GROUPS)
    summ.insert(0, 'n', d['group'].value_counts().reindex(GROUPS))
    summ.to_csv(HERE / 'summary.csv', encoding='utf-8-sig')
    thr = threshold_table(d)
    thr.to_csv(HERE / 'threshold.csv', index=False, encoding='utf-8-sig')
    box_plot(d, HERE / 'box.png')

    head = ''.join(f'<th>{GNAME[g]}<br><span class="sub">{int(summ.at[g, "n"])} 個</span></th>' for g in GROUPS)
    body = ''.join(f'<tr><td>{FEATS[f][0]}</td>'
                   + ''.join(f'<td>{FEATS[f][1].format(summ.at[g, f]) if pd.notna(summ.at[g, f]) else ""}</td>'
                             for g in GROUPS) + '</tr>' for f in FEATS)
    best = thr[(thr['hit_kept_pct'] >= 70)].sort_values('early_removed_pct', ascending=False).head(12)
    tb = ''.join(f'<tr><td>{FEATS[r.feature][0]}</td><td>{html.escape(r.rule)}</td><td>{r.hit_kept} / 24</td>'
                 f'<td>{r.early_removed_pct:.0f}%</td><td>{r.all_removed_pct:.0f}%</td></tr>'
                 for r in best.itertuples())
    page = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>S1 太早 vs Hit</title><style>{CSS}</style></head>
<body><main><h1>S1 找到的點：Hit vs 太早</h1>
<p class="sub">242 frames · S1 共 {len(d)} 個點 · 特徵只用標記那根與之前的資料</p>
<h2>各組中位數</h2><p class="sub">「創新高」兩列是比例。</p>
<div class="scroll"><table><thead><tr><th>特徵</th>{head}</tr></thead><tbody>{body}</tbody></table></div>
<h2>分布（Hit / 太早 / 更早）</h2><p class="sub">箱子 = 中間 50%，藍線 = 中位數，灰點 = 每個標記。</p>
<img src="box.png" style="width:100%;background:#fff;border:1px solid var(--line);border-radius:6px">
<h2>單一門檻能刷掉多少（保留 ≥ 70% Hit 的規則中，刷掉最多太早點的前 12 個）</h2>
<p class="sub">Hit 只有 24 個，門檻是從 Hit 的分位數挑的，有過度配適的風險；只當作方向參考。</p>
<div class="scroll"><table><thead><tr><th>特徵</th><th>保留條件</th><th>Hit 留下</th><th>太早的點刷掉</th>
<th>全部點刷掉</th></tr></thead><tbody>{tb}</tbody></table></div>
</main></body></html>"""
    (HERE / 'index.html').write_text(page, encoding='utf-8')
    with pd.option_context('display.width', 220, 'display.max_columns', 30):
        print(summ.round(2).T)
        print(best.to_string(index=False))


if __name__ == '__main__':
    main()
