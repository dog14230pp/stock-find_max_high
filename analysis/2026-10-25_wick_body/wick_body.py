"""每個策略標記的那根 1 分 K：上影線長度 vs 實體長度（242 frames）。

上影線 = high − max(open, close)；實體 = |close − open|；紅棒 = close > open，綠棒 = close < open，十字 = 相等
wick_body = 上影線 / 實體（實體 = 0 時為空白，另外標「十字」）
Hit = 標記那根就是 frame 最高點那根

輸出（與本檔同資料夾）：marks.csv（每個策略 × 每個標記一列）、index.html、box.png
執行（在專案根目錄）：python analysis/2026-10-25_wick_body/wick_body.py
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

BINS = [-np.inf, 0.25, 0.5, 1, 2, np.inf]
BIN_NAMES = ['< 0.25', '0.25～0.5', '0.5～1', '1～2', '≥ 2']


def collect(df_list, keep):
    rows = []
    for i in keep:
        mb, max_time, _ = minute_bars(df_list[i])
        max_bar = mb.index.get_loc(max_time.floor('1min'))
        o, h, lo, c = (mb[k].to_numpy() for k in ('open', 'high', 'low', 'close'))
        for sid, st in STRATEGIES.items():
            for t in np.flatnonzero(st['mask'](mb).fillna(False).to_numpy(dtype=bool)):
                body = abs(c[t] - o[t])
                wick = h[t] - max(o[t], c[t])
                rows.append({'strategy': sid, 'df_idx': i, 'time': mb.index[t], 'hit': t == max_bar,
                             'bars_from_max': t - max_bar,
                             'color': '紅' if c[t] > o[t] else '綠' if c[t] < o[t] else '十字',
                             'open': o[t], 'high': h[t], 'low': lo[t], 'close': c[t],
                             'wick': wick, 'body': body, 'range': h[t] - lo[t],
                             'wick_body': wick / body if body else np.nan})
    return pd.DataFrame(rows)


def group_summary(d):
    out = {}
    for name, g in (('Hit', d[d['hit']]), ('非 Hit', d[~d['hit']])):
        wb = g['wick_body']
        dist = pd.cut(wb, BINS, labels=BIN_NAMES, right=False).value_counts(normalize=True).reindex(BIN_NAMES)
        out[name] = {'n': len(g), 'red_pct': (g['color'] == '紅').mean() * 100,
                     'wick_med': g['wick'].median(), 'body_med': g['body'].median(),
                     'wb_med': wb.median(), 'wb_p25': wb.quantile(.25), 'wb_p75': wb.quantile(.75),
                     **{f'bin_{b}': v * 100 for b, v in dist.items()}}
    return pd.DataFrame(out).T


def box_plot(d, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    sids = list(STRATEGIES)
    fig, axes = plt.subplots(1, len(sids), figsize=(4.5 * len(sids), 4.2), sharey=True)
    rng = np.random.default_rng(0)
    for ax, sid in zip(np.atleast_1d(axes), sids):
        s = d[d['strategy'] == sid]
        data = [s.loc[s['hit'], 'wick_body'].dropna().clip(upper=3),
                s.loc[~s['hit'], 'wick_body'].dropna().clip(upper=3)]
        ax.boxplot(data, widths=0.5, showfliers=False, medianprops=dict(color='#1f6fd1', linewidth=2))
        for k, v in enumerate(data, 1):
            ax.scatter(rng.normal(k, 0.06, len(v)), v, s=9, color='#555', alpha=0.35)
        ax.set_xticks([1, 2], [f'Hit（{len(data[0])}）', f'非 Hit（{len(data[1])}）'])
        ax.set_title(sid, loc='left')
        ax.axhline(1, color='#999', linewidth=0.8, linestyle=':')
        ax.grid(axis='y', color='#e5e5e5', linewidth=0.6)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel('上影線 / 實體（> 3 畫在 3）')
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    df_list, key_list = load_frames()
    keep = keep_frames(table=max_bar_table(df_list, key_list))
    d = collect(df_list, keep)
    d.to_csv(HERE / 'marks.csv', index=False, encoding='utf-8-sig')
    box_plot(d, HERE / 'box.png')

    summ_html = ''
    for sid in STRATEGIES:
        s = group_summary(d[d['strategy'] == sid])
        rows = ''.join(
            f'<tr><td>{k}</td><td>{r.n:.0f}</td><td>{r.red_pct:.0f}%</td><td>{r.wick_med:.0f}</td>'
            f'<td>{r.body_med:.0f}</td><td>{r.wb_med:.2f}</td><td>{r.wb_p25:.2f} ～ {r.wb_p75:.2f}</td>'
            + ''.join(f'<td>{r[f"bin_{b}"]:.0f}%</td>' for b in BIN_NAMES) + '</tr>'
            for k, r in s.iterrows())
        summ_html += (f'<h3>{sid}：{html.escape(STRATEGIES[sid]["name"])}</h3><div class="scroll"><table><thead>'
                      '<tr><th></th><th>點數</th><th>紅棒比例</th><th>上影線 中位（點）</th><th>實體 中位（點）</th>'
                      '<th>上影線 / 實體 中位</th><th>中間 50%</th>'
                      + ''.join(f'<th>{b}</th>' for b in BIN_NAMES) + f'</tr></thead><tbody>{rows}</tbody></table></div>')

    def fmt_wb(v):
        return '十字' if pd.isna(v) else f'{v:.2f}'
    detail = ''.join(
        f'<tr data-s="{r.strategy}"><td>{r.strategy}</td><td>{r.df_idx:03d}</td><td>{r.time:%Y-%m-%d %H:%M}</td>'
        f'<td>{"Hit" if r.hit else ""}</td><td>{r.bars_from_max:+d}</td><td>{r.color}</td>'
        f'<td>{r.open:g}</td><td>{r.high:g}</td><td>{r.low:g}</td><td>{r.close:g}</td>'
        f'<td>{r.wick:g}</td><td>{r.body:g}</td><td>{fmt_wb(r.wick_body)}</td></tr>'
        for r in d.sort_values(['strategy', 'hit', 'df_idx', 'time'], ascending=[True, False, True, True]).itertuples())
    buttons = ''.join(f'<button data-s="{s}"{" class=on" if k == 0 else ""}>{s}</button>'
                      for k, s in enumerate(STRATEGIES))
    page = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>上影線 vs 實體</title><style>{CSS}</style></head>
<body><main><h1>標記那根：上影線 vs 實體</h1>
<p class="sub">242 frames · 上影線 = high − max(open, close)，實體 = |close − open|，比例 = 上影線 / 實體（1 = 一樣長）</p>
<h2>摘要</h2><p class="sub">右邊五欄是「上影線 / 實體」落在各區間的比例（不含十字）。</p>{summ_html}
<h2>分布</h2><p class="sub">箱子 = 中間 50%，藍線 = 中位數，灰點 = 每個標記，虛線 = 1（上影線和實體一樣長）。</p>
<img src="box.png" style="width:100%;background:#fff;border:1px solid var(--line);border-radius:6px">
<h2>每個標記</h2><div>{buttons}</div>
<div class="scroll"><table><thead><tr><th>策略</th><th>df</th><th>時間</th><th>Hit</th><th>離最高點</th><th>顏色</th>
<th>open</th><th>high</th><th>low</th><th>close</th><th>上影線</th><th>實體</th><th>上影線 / 實體</th></tr></thead>
<tbody>{detail}</tbody></table></div>
<script>
function show(s){{document.querySelectorAll('tr[data-s]').forEach(r=>r.style.display=r.dataset.s===s?'':'none');
document.querySelectorAll('button[data-s]').forEach(b=>b.classList.toggle('on',b.dataset.s===s));}}
document.querySelectorAll('button[data-s]').forEach(b=>b.onclick=()=>show(b.dataset.s));
show('{list(STRATEGIES)[0]}');
</script></main></body></html>"""
    (HERE / 'index.html').write_text(page, encoding='utf-8')
    with pd.option_context('display.width', 220, 'display.max_columns', 30):
        for sid in STRATEGIES:
            print(sid)
            print(group_summary(d[d['strategy'] == sid]).round(2))


if __name__ == '__main__':
    main()
