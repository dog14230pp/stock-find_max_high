"""畫 type1_vol 策略的圖（242 frames），看標記實際長什麼樣。

每張圖三層：
    1. K 線 + ★ 標記 + ◆ 最高點（灰底 = frame_start ~ frame_end）
    2. 成交量（被標記的那根藍色，與 ★ 同色；不代表買賣方向）+ 門檻線 = mult × 前 n 根平均
    3. type1_ratio + 門檻線（有 --rise 時，標題會寫 close 連漲幾次）
輸出：<本資料夾>/<策略名>/plots/df_XXX.png、index.html（Hit 在前，可篩選）；策略名與 marks/ 相同

執行（在專案根目錄），參數與 strategies/type1_vol.py 相同：
    python analysis/2026-10-24_vol_x2_plots/plot.py --mult 2 --above 0.55        → type1_vol5_x2_r55/
    python analysis/2026-10-24_vol_x2_plots/plot.py --mult 2 --rise 3            → type1_vol5_x2_up3/
"""
import argparse
import html
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
sys.path.insert(0, str(ROOT / 'strategies'))

from backtest_report import bars_of, load_frames  # noqa: E402
from frame_filter import keep_frames, max_bar_table  # noqa: E402
import type1_vol as tv  # noqa: E402

plt.rcParams['font.family'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_one(i, df, key, path, cfg):
    bars, max_time, max_price = bars_of(df)
    m = tv.minute_table(df).reindex(bars.index)
    mark, thr = tv.vol_mask(m, cfg.n, cfg.above, cfg.mult, cfg.rise)
    x = range(len(bars))
    max_x = bars.index.get_loc(max_time.floor('1min'))
    hit = bool(mark.iloc[max_x])

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1.3, 1.3]})
    up = bars['Close'] >= bars['Open']
    col = up.map({True: '#d62728', False: '#2ca02c'})            # 台灣：紅漲綠跌
    a1.vlines(x, bars['Low'], bars['High'], color=col, linewidth=1)
    a1.bar(x, (bars['Close'] - bars['Open']).abs().clip(lower=0.5),
           bottom=bars[['Open', 'Close']].min(axis=1), color=col, width=0.7)
    in_frame = (bars.index >= key[0]) & (bars.index <= key[1])
    if in_frame.any():
        xs = [j for j, f in enumerate(in_frame) if f]
        for a in (a1, a2, a3):
            a.axvspan(xs[0] - 0.5, xs[-1] + 0.5, color='#888', alpha=0.08, linewidth=0)
    span = bars['High'].max() - bars['Low'].min()
    mx = [j for j, f in enumerate(mark) if f]
    a1.scatter(mx, bars['High'].iloc[mx] + span * 0.05, marker='*', s=180, color='#1f77b4', zorder=5)
    a1.scatter([max_x], [max_price], marker='D', s=50, color='black', zorder=6)
    a1.axhline(max_price, color='black', linestyle=':', linewidth=0.8)
    a1.set_ylim(top=bars['High'].max() + span * 0.12)
    a1.set_title(f'df {i:03d}   {key[0]:%Y-%m-%d %H:%M} ~ {key[1]:%H:%M}   [{"HIT" if hit else "MISS"}]'
                 f'   標記 {int(mark.sum())} 個   最高點 {max_time:%H:%M:%S} {max_price:g}')

    a2.bar(x, m['volume'], color=mark.map({True: '#1f77b4', False: '#bbb'}), width=0.8)
    a2.plot(x, thr, color='#ff7f0e', linewidth=1.2, label=f'{cfg.mult:g} × 前 {cfg.n} 根平均')
    a2.axvline(max_x, color='black', linestyle=':', linewidth=0.8)
    a2.set_ylabel('volume')
    a2.legend(loc='upper left', fontsize=8)

    a3.plot(x, m['ratio'], color='#555', linewidth=1)
    a3.scatter(mx, m['ratio'].iloc[mx], color='#1f77b4', s=18, zorder=5)
    a3.axhline(cfg.above, color="#ff7f0e", linewidth=1.2, label=f"ratio = {cfg.above:g}")
    a3.axvline(max_x, color='black', linestyle=':', linewidth=0.8)
    a3.set_ylabel('type1_ratio')
    a3.set_ylim(0, 1)
    a3.legend(loc='upper left', fontsize=8)

    step = max(1, len(bars) // 15)
    a3.set_xticks(list(x)[::step])
    a3.set_xticklabels([f'{t:%H:%M}' for t in bars.index[::step]], fontsize=8)
    a3.set_xlim(-1, len(bars))
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return hit, int(mark.sum())


def main():
    ap = argparse.ArgumentParser(description='畫 type1_vol 策略的圖（242 frames）')
    ap.add_argument('--n', type=int, default=tv.N_WINDOW)
    ap.add_argument('--above', type=float, default=tv.RATIO_MIN)
    ap.add_argument('--mult', type=float, default=tv.VOL_MULT)
    ap.add_argument('--rise', type=int, default=tv.RISE)
    a = ap.parse_args()
    name = tv.default_name(a.n, a.above, a.mult, a.rise)
    out = HERE / name
    (out / 'plots').mkdir(parents=True, exist_ok=True)
    df_list, key_list = load_frames()
    frames = keep_frames(table=max_bar_table(df_list, key_list))
    res = []
    for i in frames:
        hit, n = plot_one(i, df_list[i], key_list[i], out / 'plots' / f'df_{i:03d}.png', a)
        res.append((i, hit, n))
    res.sort(key=lambda r: (not r[1], r[0]))
    hits = sum(r[1] for r in res)
    cards = ''.join(
        f'<div class="f" data-k="{"hit" if h else ("none" if n == 0 else "miss")}">'
        f'<h3>df {i:03d} <span class="{"h" if h else "m"}">{"Hit" if h else "Miss"}</span> 標記 {n} 個</h3>'
        f'<a href="plots/df_{i:03d}.png" target="_blank"><img loading="lazy" src="plots/df_{i:03d}.png"></a></div>'
        for i, h, n in res)
    title = (f'{name}：成交量 > {a.mult:g} 倍前 {a.n} 根平均、ratio ≥ {a.above:g}'
             + (f'、close 連漲 {a.rise} 次' if a.rise else ''))
    page = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>
:root{{color-scheme:light;--page:#f9f9f7;--ink:#0b0b0b;--sub:#52514e;--line:#e1e0d9;--good:#006300;--bad:#b42f2f}}
@media (prefers-color-scheme:dark){{:root{{color-scheme:dark;--page:#0d0d0d;--ink:#f0efec;--sub:#c3c2b7;--line:#2c2c2a;--good:#0ca30c;--bad:#e66767}}}}
body{{margin:0;background:var(--page);color:var(--ink);font-family:system-ui,"Microsoft JhengHei",sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:24px 16px}}
.sub{{color:var(--sub)}} img{{width:100%;background:#fff;border:1px solid var(--line)}}
.f{{margin:28px 0}} h3{{margin:0 0 6px;font-size:16px}} .h{{color:var(--good)}} .m{{color:var(--bad)}}
button{{font:inherit;padding:4px 12px;margin-right:6px;cursor:pointer}} button.on{{font-weight:700}}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="sub">242 frames · Hit {hits} / {len(res)} · ★ / 藍色柱 = 標記、◆ = 最高點、灰底 = frame_start ~ frame_end</p>
<div><button class="on" data-f="all">全部</button><button data-f="hit">Hit</button>
<button data-f="miss">Miss（有標記）</button><button data-f="none">沒有標記</button></div>
{cards}</main>
<script>
document.querySelectorAll('button').forEach(b=>b.onclick=()=>{{
document.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b));
document.querySelectorAll('.f').forEach(f=>f.style.display=(b.dataset.f==='all'||f.dataset.k===b.dataset.f)?'':'none');}});
</script></body></html>"""
    (out / 'index.html').write_text(page, encoding='utf-8')
    print(f'{name}（242 frames）：Hit {hits} / {len(res)}，總覽 {out / "index.html"}')


if __name__ == '__main__':
    main()
