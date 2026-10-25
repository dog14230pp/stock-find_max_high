"""每一分鐘內「每秒 type1_ratio」的分布，依「這分鐘漲 / 跌」×「下一分鐘漲 / 跌」分組比較。

漲 / 跌：d_close = 這分鐘 close − 上一分鐘 close（> 0 漲、< 0 跌、= 0 平；間隔 > MAX_GAP 分鐘不算）
分組：這分鐘 + 下一分鐘，例如「漲→跌」= 這分鐘漲、下一分鐘跌；有「平」或算不出來的不分組
分布：該分鐘內有成交的秒，每秒權重相同
    mean, q25, median, q75, ge_0.7（ratio ≥ 0.7 的秒數佔比）, le_0.3（≤ 0.3 的佔比）,
    b0..b4（落在 [0,0.2) [0.2,0.4) [0.4,0.6) [0.6,0.8) [0.8,1] 的秒數佔比）

輸入：analysis/2026-10-21_type1_ratio/sec/、min/（make.py 產生）
輸出（與本檔同資料夾）：
    sec_ratio_dist.csv    每列 = 一個 frame 的一分鐘
    sec_ratio_dist.html   報告，用瀏覽器打開

執行（在專案根目錄）：
    python analysis/2026-10-23_sec_ratio_dist/sec_ratio_dist.py 0 1      指定 frame（預設 0 1）→ sec_ratio_dist.*
    python analysis/2026-10-23_sec_ratio_dist/sec_ratio_dist.py all      全部 frame → sec_ratio_dist_all.*
    python analysis/2026-10-23_sec_ratio_dist/sec_ratio_dist.py all 0 1  全部 frame，另外放 df 0、1 的每分鐘明細
"""
import html
import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / '2026-10-21_type1_ratio'
sys.path.insert(0, str(SRC))
from sec_ratio_count import CSS  # noqa: E402

MAX_GAP = 5
EDGES = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]
BIN_NAMES = ['0–0.2', '0.2–0.4', '0.4–0.6', '0.6–0.8', '0.8–1']
GROUPS = ['漲→漲', '漲→跌', '跌→漲', '跌→跌']
STATS = ['mean', 'q25', 'median', 'q75', 'ge_0.7', 'le_0.3'] + [f'b{k}' for k in range(len(BIN_NAMES))]


def direction(x):
    return np.select([x > 0, x < 0, x == 0], ['漲', '跌', '平'], default='')


def frame_minutes(i):
    s = pd.read_csv(SRC / 'sec' / f'df_{i:03d}.csv')
    m = pd.read_csv(SRC / 'min' / f'df_{i:03d}.csv').sort_values('minute')
    gap = pd.to_datetime(m['minute']).diff().dt.total_seconds() / 60
    out = m[['minute', 'close', 'volume', 'type1_ratio']].copy()
    out['d_close'] = out['close'].diff().where(gap <= MAX_GAP)
    out['next_d'] = out['d_close'].shift(-1)
    g = s.groupby('minute')['type1_ratio']
    st = pd.DataFrame({'n_sec': g.size(), 'mean': g.mean(), 'q25': g.quantile(.25), 'median': g.median(),
                       'q75': g.quantile(.75), 'ge_0.7': g.apply(lambda r: (r >= 0.7).mean()),
                       'le_0.3': g.apply(lambda r: (r <= 0.3).mean())})
    b = pd.cut(s['type1_ratio'], EDGES, right=False, labels=False)
    st = st.join(pd.crosstab(s['minute'], b, normalize='index').reindex(columns=range(len(BIN_NAMES)), fill_value=0)
                 .add_prefix('b'))
    out = out.merge(st, left_on='minute', right_index=True, how='left')
    now, nxt = direction(out['d_close']), direction(out['next_d'])
    out['group'] = [f'{a}→{b}' if f'{a}→{b}' in GROUPS else '' for a, b in zip(now, nxt)]
    out.insert(0, 'df_idx', i)
    return out


def group_table(res):
    """每組：分鐘數、各統計量的平均（= 該組每分鐘的值再平均）。"""
    g = res[res['group'] != ''].groupby('group')
    t = g[STATS].mean().reindex(GROUPS)
    t.insert(0, 'n', g.size().reindex(GROUPS).fillna(0).astype(int))
    return t


def compare(res, stat='mean'):
    """漲→漲 vs 漲→跌、跌→漲 vs 跌→跌：分鐘層級的 Welch t（n 大，p 用常態近似）＋ 逐 frame 方向一致的比例。"""
    from math import erf, sqrt
    rows = []
    for a, b in [('漲→漲', '漲→跌'), ('跌→漲', '跌→跌')]:
        x, y = res.loc[res['group'] == a, stat].dropna(), res.loc[res['group'] == b, stat].dropna()
        diff = x.mean() - y.mean()
        t = diff / sqrt(x.var() / len(x) + y.var() / len(y))
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
        per = res[res['group'].isin([a, b])].groupby(['df_idx', 'group'])[stat].mean().unstack().dropna()
        rows.append({'比較': f'{a} − {b}', 'n_a': len(x), 'n_b': len(y), 'diff': diff, 't': t, 'p': p,
                     'frames': len(per), 'frames_pos': int((per[a] > per[b]).sum())})
    return pd.DataFrame(rows)


def stat_cells(r):
    cells = [f'{r["mean"]:.2f}', f'{r["q25"]:.2f}', f'{r["median"]:.2f}', f'{r["q75"]:.2f}',
             f'{r["ge_0.7"] * 100:.0f}%', f'{r["le_0.3"] * 100:.0f}%']
    bins = [r[f'b{k}'] for k in range(len(BIN_NAMES))]
    # 5 格分布：數字 + 一條迷你長條，眼睛可以直接看形狀
    bars = ''.join(f'<span class="bar" style="height:{v * 100:.0f}%"></span>' for v in bins)
    return cells + [f'{v * 100:.0f}%' for v in bins] + [f'<span class="spark">{bars}</span>']


STAT_HEAD = ('<th>平均</th><th>q25</th><th>中位數</th><th>q75</th><th>≥ 0.7</th><th>≤ 0.3</th>'
             + ''.join(f'<th>{n}</th>' for n in BIN_NAMES) + '<th>形狀</th>')

EXTRA_CSS = """
.spark{display:inline-flex;align-items:flex-end;gap:2px;height:22px;vertical-align:middle}
.spark .bar{width:7px;background:var(--ink2);border-radius:1px;min-height:1px}
td.g-漲→漲{color:#c0392b;font-weight:600}td.g-漲→跌{color:#d68910;font-weight:600}
td.g-跌→漲{color:#2471a3;font-weight:600}td.g-跌→跌{color:#1e8449;font-weight:600}
.legend{color:var(--ink2)}
"""


def build_html(res, frames, detail):
    names = f'全部 {len(frames)} 個' if len(frames) > 10 else ', '.join(f'df {i:03d}' for i in frames)
    parts = ['<main><h1>每分鐘內每秒 type1_ratio 的分布</h1>',
             f'<p class="sub">frame：{names}　·　{len(res):,} 分鐘</p>',
             '<p class="sub">漲 / 跌 = 這分鐘 close − 上一分鐘 close；組別「漲→跌」= 這分鐘漲、下一分鐘跌。'
             '分布只算有成交的秒、每秒權重相同。「≥ 0.7」「≤ 0.3」= 達到門檻的秒數佔比；'
             '0–0.2 … 0.8–1 = 落在各區間的秒數佔比（最右的形狀圖就是這 5 格）。</p>']

    def gtable(t, title):
        rows = ''.join(f'<tr><td class="g-{k}">{k}</td><td>{int(r["n"])}</td>'
                       + ''.join(f'<td>{c}</td>' for c in stat_cells(r)) + '</tr>'
                       for k, r in t.iterrows() if r['n'] > 0)
        return (f'<h2>{title}</h2><div class="card"><table><thead><tr><th>組別</th><th>分鐘數</th>{STAT_HEAD}'
                f'</tr></thead><tbody>{rows}</tbody></table></div>')

    parts.append(gtable(group_table(res), f'分組比較（{names} 合併）'))
    c = compare(res)
    crow = ''.join(f'<tr><td>{r["比較"]}</td><td>{r["n_a"]:,} / {r["n_b"]:,}</td><td>{r["diff"]:+.3f}</td>'
                   f'<td>{r["t"]:+.2f}</td><td>{r["p"]:.3g}</td><td>{r["frames_pos"]} / {r["frames"]}'
                   f'（{r["frames_pos"] / max(r["frames"], 1) * 100:.0f}%）</td></tr>' for _, r in c.iterrows())
    parts.append('<h2>下一分鐘漲 vs 跌：差距是不是雜訊</h2>'
                 '<p class="sub">比的是每分鐘各秒 ratio 的平均。差距 = 前一組平均 − 後一組平均；t、p 用分鐘層級的 Welch t 檢定'
                 '（p 為常態近似）；「frame 方向一致」= 兩組都有分鐘的 frame 中，前一組平均比較高的 frame 數。</p>'
                 '<div class="card"><table><thead><tr><th>比較</th><th>分鐘數</th><th>差距</th><th>t</th><th>p</th>'
                 f'<th>frame 方向一致</th></tr></thead><tbody>{crow}</tbody></table></div>')
    if len(frames) <= 10:
        for i in frames:
            parts.append(gtable(group_table(res[res['df_idx'] == i]), f'分組比較：df {i:03d}'))
    for i in detail:
        g = res[res['df_idx'] == i]
        rows = ''.join(
            f'<tr><td>{r["minute"][11:16]}</td><td>{r["close"]:,.0f}</td><td>{r["d_close"]:+.0f}</td>'
            f'<td>{r["next_d"]:+.0f}</td><td class="g-{r["group"]}">{r["group"]}</td><td>{int(r["n_sec"])}</td>'
            f'<td>{r["type1_ratio"]:.2f}</td>' + ''.join(f'<td>{c}</td>' for c in stat_cells(r)) + '</tr>'
            for _, r in g.iterrows())
        parts.append(f'<h2>每分鐘明細：df {i:03d}</h2><div class="card scroll"><table><thead><tr><th>分鐘</th>'
                     '<th>收盤</th><th>這分鐘漲跌</th><th>下一分鐘漲跌</th><th>組別</th><th>有成交秒數</th>'
                     f'<th>分鐘 ratio（量加權）</th>{STAT_HEAD}</tr></thead><tbody>{rows}</tbody></table></div>')
    parts.append('</main>')
    return (f'<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape("每秒 ratio 分布")}</title><style>{CSS}{EXTRA_CSS}</style></head>'
            f'<body>{"".join(parts).replace("nan", "")}</body></html>')


def main(argv):
    picked = [int(x) for x in argv[1:] if x.isdigit()] or [0, 1]
    if 'all' in argv:     # 全部 frame 一起分組；只有另外列出的 frame 才放每分鐘明細
        frames = list(range(len(list((SRC / 'min').glob('df_*.csv')))))
        detail = [int(x) for x in argv[1:] if x.isdigit()]
        name = 'sec_ratio_dist_all'
    else:
        frames = detail = picked
        name = 'sec_ratio_dist'
    res = pd.concat([frame_minutes(i) for i in frames], ignore_index=True)
    res.to_csv(HERE / f'{name}.csv', index=False)
    report = HERE / f'{name}.html'
    report.write_text(build_html(res, frames, detail), encoding='utf-8')
    print(f'{len(res)} 分鐘 → {HERE / f"{name}.csv"}；報告 {report}')
    pd.set_option('display.width', 200)
    print(group_table(res).round(3).to_string())
    print(compare(res).round(4).to_string())
    if '--no-open' not in argv:
        webbrowser.open(report.as_uri())


if __name__ == '__main__':
    main(sys.argv)
