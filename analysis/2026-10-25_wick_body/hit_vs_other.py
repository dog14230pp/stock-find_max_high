"""簡化版：只看紅棒，Hit vs 其他找到的點，上影線 / 紅實體 的統計與分布（242 frames）。

讀 wick_body.py 產生的 marks.csv（先跑 wick_body.py）。
紅實體 = close − open（> 0）；比例 = 上影線 / 紅實體。
另外算「刪掉比例 < X 的點」時 Hit 留幾個、其他刪掉幾個。

輸出（與本檔同資料夾）：hit_vs_other.html、hist_<策略>.png
執行（在專案根目錄）：python analysis/2026-10-25_wick_body/hit_vs_other.py
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

from registry import STRATEGIES  # noqa: E402
from start_backtest import CSS  # noqa: E402

BINS = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, np.inf]
LABELS = ['0～0.25', '0.25～0.5', '0.5～0.75', '0.75～1', '1～1.5', '1.5～2', '2～3', '≥ 3']
CUTS = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2]
HIT_C, OTHER_C = '#1f6fd1', '#9a9a9a'


def hist(d, sid, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    x = np.arange(len(LABELS))
    fig, ax = plt.subplots(figsize=(11, 4))
    for k, (name, g, c) in enumerate((('Hit', d[d['hit']], HIT_C), ('其他', d[~d['hit']], OTHER_C))):
        pct = pd.cut(g['wick_body'], BINS, labels=LABELS, right=False).value_counts(normalize=True) \
            .reindex(LABELS).fillna(0) * 100
        bars = ax.bar(x + (k - 0.5) * 0.38, pct, width=0.36, color=c, label=f'{name}（{len(g)} 個）')
        for b, v in zip(bars, pct):
            if v >= 1:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f'{v:.0f}%', ha='center', fontsize=8, color='#333')
    ax.set_xticks(x, LABELS)
    ax.set_xlabel('上影線 / 紅實體')
    ax.set_ylabel('佔該組的 %')
    ax.set_title(f'{sid}：上影線 / 紅實體 的分布（紅棒，242 frames）', loc='left', fontsize=11)
    ax.grid(axis='y', color='#e5e5e5', linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def stats(g):
    wb = g['wick_body']
    return {'n': len(g), 'mean': wb.mean(), 'p10': wb.quantile(.1), 'p25': wb.quantile(.25), 'med': wb.median(),
            'p75': wb.quantile(.75), 'p90': wb.quantile(.9)}


def section(d, sid):
    hits, other = d[d['hit']], d[~d['hit']]
    st = ''.join(f'<tr><td>{name}</td>' + ''.join(f'<td>{v:.0f}</td>' if k == 'n' else f'<td>{v:.2f}</td>'
                                                  for k, v in stats(g).items()) + '</tr>'
                 for name, g in (('Hit', hits), ('其他', other)))
    cut = ''.join(
        f'<tr><td>&lt; {x:g}</td><td>{(hits["wick_body"] >= x).sum()} / {len(hits)}'
        f'（{(hits["wick_body"] >= x).mean() * 100:.0f}%）</td>'
        f'<td>{(other["wick_body"] < x).sum()} / {len(other)}（{(other["wick_body"] < x).mean() * 100:.0f}%）</td>'
        f'<td>{(d["wick_body"] >= x).sum()}</td>'
        f'<td>{(hits["wick_body"] >= x).sum() / max(1, (d["wick_body"] >= x).sum()) * 100:.0f}%</td></tr>'
        for x in CUTS)
    base = len(hits) / len(d) * 100
    return f"""<div class="sec" data-s="{sid}">
<h2>{sid}：{html.escape(STRATEGIES[sid]['name'])}</h2>
<p class="sub">{html.escape(STRATEGIES[sid]['desc'])}</p>
<img src="hist_{sid}.png" style="width:100%;background:#fff;border:1px solid var(--line);border-radius:6px">
<h3>統計</h3>
<div class="scroll"><table><thead><tr><th></th><th>點數</th><th>平均</th><th>p10</th><th>p25</th><th>中位數</th>
<th>p75</th><th>p90</th></tr></thead><tbody>{st}</tbody></table></div>
<h3>刪掉「上影線 / 紅實體 &lt; X」的點</h3>
<p class="sub">刪之前：{len(d)} 個點，其中 Hit {len(hits)} 個（{base:.0f}%）。</p>
<div class="scroll"><table><thead><tr><th>刪掉</th><th>Hit 留下</th><th>其他刪掉</th><th>剩下的點</th>
<th>剩下的點中 Hit 佔</th></tr></thead><tbody>{cut}</tbody></table></div></div>"""


def main():
    d = pd.read_csv(HERE / 'marks.csv')
    d = d[d['color'] == '紅']
    secs, buttons = '', ''
    for k, sid in enumerate(STRATEGIES):
        s = d[d['strategy'] == sid]
        hist(s, sid, HERE / f'hist_{sid}.png')
        secs += section(s, sid)
        buttons += f'<button data-s="{sid}"{" class=on" if k == 0 else ""}>{sid}</button>'
    page = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Hit vs 其他：上影線 / 紅實體</title>
<style>{CSS}</style></head><body><main>
<h1>Hit vs 其他找到的點：上影線 / 紅實體</h1>
<p class="sub">242 frames · 只看紅棒（close &gt; open）· 上影線 = high − close，紅實體 = close − open ·
比例 1 = 上影線和實體一樣長</p>
<div>{buttons}</div>{secs}
<script>
function show(s){{document.querySelectorAll('.sec').forEach(x=>x.style.display=x.dataset.s===s?'':'none');
document.querySelectorAll('button[data-s]').forEach(b=>b.classList.toggle('on',b.dataset.s===s));}}
document.querySelectorAll('button[data-s]').forEach(b=>b.onclick=()=>show(b.dataset.s));
show('{list(STRATEGIES)[0]}');
</script></main></body></html>"""
    (HERE / 'hit_vs_other.html').write_text(page, encoding='utf-8')
    for sid in STRATEGIES:
        s = d[d['strategy'] == sid]
        print(sid, {k: round(v, 2) for k, v in stats(s[s['hit']]).items()},
              {k: round(v, 2) for k, v in stats(s[~s['hit']]).items()})
        for x in CUTS:
            h, o = s[s['hit']], s[~s['hit']]
            print(f'  cut<{x}: hit kept {(h.wick_body >= x).sum()}/{len(h)}, other cut {(o.wick_body < x).sum()}/{len(o)}')


if __name__ == '__main__':
    main()
