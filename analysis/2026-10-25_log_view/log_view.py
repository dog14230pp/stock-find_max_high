"""看看取 log 之後的圖：每個 frame 左邊原本、右邊 log（全部 290 frames，使用者要求看全部）。

左：open / high / low / close 原值，成交量原值
右：ln(open / high / low / close)，ln(成交量)
★ = 正式策略 1（S5）找到的點、◆ = 最高點、灰底 = frame_start ~ frame_end；K 線紅漲綠跌

輸出（與本檔同資料夾）：plots/df_XXX.png、index.html
執行（在專案根目錄）：python analysis/2026-10-25_log_view/log_view.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
sys.path.insert(0, str(ROOT / 'strategies'))

from backtest_report import load_frames  # noqa: E402
from frame_filter import keep_frames, max_bar_table  # noqa: E402
from registry import STRATEGIES  # noqa: E402
from start_backtest import CSS, minute_bars  # noqa: E402

STRATEGY = 'S5'
plt.rcParams['font.family'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def candles(ax, o, h, lo, c, marks, max_x, inside):
    x = np.arange(len(o))
    col = np.where(c >= o, '#d62728', '#2ca02c')
    ax.vlines(x, lo, h, color=col, linewidth=1)
    body = np.abs(c - o)
    ax.bar(x, np.maximum(body, (h.max() - lo.min()) * 0.002), bottom=np.minimum(o, c), color=col, width=0.7)
    span = h.max() - lo.min()
    ax.scatter(marks, h[marks] + span * 0.05, marker='*', s=130, color='#1f6fd1', zorder=5)
    ax.scatter([max_x], [h[max_x] + span * (0.12 if max_x in set(marks) else 0.05)], marker='D', s=35,
               color='black', zorder=6)
    ax.set_ylim(lo.min() - span * 0.05, h.max() + span * 0.2)
    if len(inside):
        ax.axvspan(inside[0] - 0.5, inside[-1] + 0.5, color='#888', alpha=0.10, linewidth=0)


def volume(ax, v, marks, inside):
    col = np.full(len(v), '#bbbbbb', dtype=object)
    col[marks] = '#1f6fd1'
    ax.bar(np.arange(len(v)), v, color=col, width=0.75)
    if len(inside):
        ax.axvspan(inside[0] - 0.5, inside[-1] + 0.5, color='#888', alpha=0.10, linewidth=0)


def plot_one(i, df, key, mask_fn, path):
    mb, max_time, _ = minute_bars(df)
    marks = np.flatnonzero(mask_fn(mb).fillna(False).to_numpy(dtype=bool))
    max_x = mb.index.get_loc(max_time.floor('1min'))
    inside = np.flatnonzero((mb.index >= key[0]) & (mb.index <= key[1]))
    ohlc = [mb[k].to_numpy(dtype=float) for k in ('open', 'high', 'low', 'close')]
    v = mb['volume'].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(15, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    candles(axes[0, 0], *ohlc, marks, max_x, inside)
    candles(axes[0, 1], *[np.log(a) for a in ohlc], marks, max_x, inside)
    volume(axes[1, 0], v, marks, inside)
    volume(axes[1, 1], np.log(v), marks, inside)
    axes[0, 0].set_title('原本：價格', loc='left', fontsize=10)
    axes[0, 1].set_title('log：ln(價格)', loc='left', fontsize=10)
    axes[1, 0].set_ylabel('volume', fontsize=9)
    axes[1, 1].set_ylabel('ln(volume)', fontsize=9)
    step = max(1, len(mb) // 8)
    x = np.arange(len(mb))
    for ax in axes.flat:
        ax.grid(axis='y', color='#e5e5e5', linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    for ax in axes[1]:
        ax.set_xticks(x[::step])
        ax.set_xticklabels([f'{t:%H:%M}' for t in mb.index[::step]], fontsize=8)
        ax.set_xlim(-1, len(mb))
    hit = max_x in set(marks)
    fig.suptitle(f'df {i:03d}   {key[0]:%Y-%m-%d %H:%M} ~ {key[1]:%H:%M}   {"Hit" if hit else "Miss"}'
                 f'   ★ {STRATEGY} 找到的點  ◆ 最高點', x=0.01, ha='left', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=75)
    plt.close(fig)


def main():
    df_list, key_list = load_frames()
    keep = set(keep_frames(table=max_bar_table(df_list, key_list)))
    mask_fn = STRATEGIES[STRATEGY]['mask']
    (HERE / 'plots').mkdir(exist_ok=True)
    cards = []
    for i, (df, key) in enumerate(zip(df_list, key_list)):
        plot_one(i, df, key, mask_fn, HERE / 'plots' / f'df_{i:03d}.png')
        g = 'keep' if i in keep else 'excluded'
        cards.append(f'<div class="pic" data-g="{g}"><a href="plots/df_{i:03d}.png" target="_blank">'
                     f'<img loading="lazy" src="plots/df_{i:03d}.png" alt="df {i}"></a></div>')
    page = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>log 價格 / 成交量</title>
<style>{CSS}</style></head><body><main>
<h1>原本 vs 取 log</h1>
<p class="sub">290 frames · 左：價格與成交量原值，右：ln(open / high / low / close) 與 ln(成交量) ·
★ = 正式策略 1（{STRATEGY}）找到的點、◆ = 最高點、灰底 = frame 區間</p>
<label><input type="checkbox" id="only242" checked> 只看 242</label>
{''.join(cards)}
<script>
const only=document.getElementById('only242');
function show(){{document.querySelectorAll('.pic').forEach(p=>p.style.display=
(!only.checked||p.dataset.g==='keep')?'':'none');}}
only.onchange=show;show();
</script></main></body></html>"""
    (HERE / 'index.html').write_text(page, encoding='utf-8')
    print(f'290 frames 的圖 → {HERE}')


if __name__ == '__main__':
    main()
