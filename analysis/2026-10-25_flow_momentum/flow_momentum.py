"""買賣壓動量（類 ML momentum）vs 價格移動：有沒有機會用來找高點（242 frames）。

每根 1 分 K t（只用 t 與之前的資料）：
    F       推力 = (外盤量 − 內盤量) / 前 5 根平均量
    v_b     累積買賣壓 = β·v[t-1] + (1−β)·F[t]，β = 0.5 / 0.7 / 0.9（pandas ewm, adjust=False）
    dv_b    v_b[t] − v_b[t-1]（累積買壓在增加還是減少）
    dclose  close[t] − close[t-1]
    acc     價格加速度 = dclose[t] − dclose[t-1]
訊號
    A_b     dv_b < 0 且 dclose > 0（買壓轉弱，價格還在漲）
    B       F > 0 且 acc < 0（買壓還在，價格推不動）
    B2      B 連續 2 根

檢查
    連續值：最高點那根在該 frame 內的平均百分位（50 = 沒資訊）；在 CHECK_STRATEGY 的點裡 Hit 的平均百分位
    訊號：最高點那根符合的比例 / 所有 K 棒符合的比例（倍數 1 = 沒資訊）；在 CHECK_STRATEGY 的點裡 Hit vs 其他 的符合比例

輸出（與本檔同資料夾）：bars.csv、summary.csv、index.html（公式 + 結果 + 每個 frame 的圖）、plots/df_XXX.png
執行（在專案根目錄）：python analysis/2026-10-25_flow_momentum/flow_momentum.py
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

CHECK_STRATEGY = 'S8'
BETAS = [0.5, 0.7, 0.9]


def features(df, mb):
    t = df[['ts', 'volume', 'tick_type']].copy()
    t['ts'] = pd.to_datetime(t['ts'])
    t['v2'] = t['volume'].where(t['tick_type'] == 2, 0)
    v2 = t.groupby(t['ts'].dt.floor('1min'))['v2'].sum().reindex(mb.index).fillna(0)
    vol = mb['volume'].astype(float)
    F = (mb['type1_volume'] - v2) / vol.rolling(5).mean().shift(1)
    dclose = mb['close'].diff()
    out = {'F': F, 'dclose': dclose, 'acc': dclose.diff()}
    for b in BETAS:
        v = F.ewm(alpha=1 - b, adjust=False, ignore_na=True).mean()
        out[f'v_{b}'], out[f'dv_{b}'] = v, v.diff()
        out[f'A_{b}'] = ((v.diff() < 0) & (dclose > 0)).astype(float).where(v.diff().notna())
    B = ((F > 0) & (out['acc'] < 0)).astype(float).where(F.notna() & out['acc'].notna())
    out['B'], out['B2'] = B, (B * B.shift(1)).where(B.shift(1).notna())
    return pd.DataFrame(out, index=mb.index)


def plot_frame(i, key, mb, f, max_x, marks, path):
    """三層：K 線（★ 策略的點、◆ 最高點、▲ 訊號 B）、推力 F、累積買賣壓 v（β = 0.9）。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    x = np.arange(len(mb))
    fig, (ak, af, av) = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True,
                                     gridspec_kw={'height_ratios': [2.6, 1, 1]})
    col = np.where(mb['close'] >= mb['open'], '#d62728', '#2ca02c')
    ak.vlines(x, mb['low'], mb['high'], color=col, linewidth=1)
    ak.bar(x, (mb['close'] - mb['open']).abs().clip(lower=0.5), bottom=mb[['open', 'close']].min(axis=1),
           color=col, width=0.7)
    span = mb['high'].max() - mb['low'].min()
    b = np.flatnonzero(f['B'].to_numpy() == 1)
    ak.scatter(b, mb['low'].iloc[b] - span * 0.04, marker='^', s=30, color='#e8780c', zorder=5,
               label='B：買壓 > 0 但價格減速')
    ak.scatter(marks, mb['high'].iloc[marks] + span * 0.05, marker='*', s=150, color='#1f6fd1', zorder=5,
               label=f'{CHECK_STRATEGY} 找到的點')
    ak.scatter([max_x], [mb['high'].iloc[max_x] + span * (0.12 if max_x in set(marks) else 0.05)],
               marker='D', s=40, color='black', zorder=6, label='最高點')
    ak.set_ylim(mb['low'].min() - span * 0.09, mb['high'].max() + span * 0.2)
    af.bar(x, f['F'], color=np.where(f['F'] >= 0, '#d62728', '#2ca02c'), width=0.75)
    af.set_ylabel('F 推力', fontsize=9)
    av.plot(x, f['v_0.9'], color='#555', linewidth=1.5)
    av.set_ylabel('v 累積（β=0.9）', fontsize=9)
    for a in (ak, af, av):
        a.axvline(max_x, color='black', linestyle=':', linewidth=0.8)
        a.grid(axis='y', color='#e5e5e5', linewidth=0.6)
        a.set_axisbelow(True)
        for side in ('top', 'right'):
            a.spines[side].set_visible(False)
    for a in (af, av):
        a.axhline(0, color='#999', linewidth=0.8)
    step = max(1, len(mb) // 12)
    av.set_xticks(x[::step])
    av.set_xticklabels([f'{t:%H:%M}' for t in mb.index[::step]], fontsize=9)
    av.set_xlim(-1, len(mb))
    ak.legend(loc='upper left', fontsize=8, frameon=False, ncol=3)
    hit = max_x in set(marks)
    ak.set_title(f'df {i:03d}   {key[0]:%Y-%m-%d %H:%M} ~ {key[1]:%H:%M}   {CHECK_STRATEGY} {"Hit" if hit else "Miss"}',
                 loc='left', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=75)
    plt.close(fig)


FORMULAS = """
<table><thead><tr><th>名稱</th><th>公式</th><th>白話</th></tr></thead><tbody>
<tr><td>F 推力</td><td>（外盤量[t] − 內盤量[t]）÷ 前 5 根（t-5 ~ t-1）平均成交量</td>
<td>這分鐘買方淨推了多少，用「平常的量」當單位。&gt; 0 = 買方在推，&lt; 0 = 賣方在推</td></tr>
<tr><td>v 累積買賣壓</td><td>v[t] = β × v[t-1] + (1 − β) × F[t]，β = 0.5 / 0.7 / 0.9</td>
<td>= ML momentum 的「速度」：最近幾分鐘推力的指數移動平均。β 越大，記得越久、越平滑</td></tr>
<tr><td>dv</td><td>v[t] − v[t-1]</td><td>累積買壓在增加（&gt; 0）還是減少（&lt; 0）</td></tr>
<tr><td>dclose</td><td>close[t] − close[t-1]</td><td>價格這分鐘移動多少（價格的速度）</td></tr>
<tr><td>acc 加速度</td><td>dclose[t] − dclose[t-1]</td><td>價格越漲越快（&gt; 0）還是在減速（&lt; 0）</td></tr>
<tr><td>訊號 A</td><td>dv &lt; 0 且 dclose &gt; 0</td><td>買壓開始轉弱，但價格還在漲（靠慣性）</td></tr>
<tr><td>訊號 B</td><td>F &gt; 0 且 acc &lt; 0</td><td>買方還在推，但價格推不動、在減速（上面有人接賣單）</td></tr>
<tr><td>訊號 B2</td><td>B 連續 2 根</td><td></td></tr>
</tbody></table>"""


def build_html(s, order, n_marks, n_hit):
    fmt = lambda v, p: '' if pd.isna(v) else f'{v:.{p}f}'  # noqa: E731
    cont = s[s['type'] == '連續']
    sig = s[s['type'] != '連續']
    t1 = ''.join(f'<tr><td>{r.feature}</td><td>{fmt(r.max_bar, 0)}</td><td>{fmt(r.hit_in_marks, 0)}</td>'
                 f'<td>{fmt(r.other_in_marks, 0)}</td></tr>' for r in cont.itertuples())
    t2 = ''.join(f'<tr><td>{r.feature}</td><td>{fmt(r.max_bar, 1)}%</td><td>{fmt(r.all_bars, 1)}%</td>'
                 f'<td>{fmt(r.ratio, 2)}</td><td>{fmt(r.hit_in_marks, 0)}%</td><td>{fmt(r.other_in_marks, 0)}%</td></tr>'
                 for r in sig.itertuples())
    cards = ''.join(f'<div class="pic" data-k="{k}"><a href="plots/df_{i:03d}.png" target="_blank">'
                    f'<img loading="lazy" src="plots/df_{i:03d}.png" alt="df {i}"></a></div>' for i, k in order)
    return f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>買賣壓動量</title>
<style>{CSS}.pic{{margin:14px 0}}.pic img{{width:100%;background:#fff;border:1px solid var(--line);border-radius:6px}}
</style></head><body><main>
<h1>買賣壓動量（類 ML momentum）</h1>
<p class="sub">242 frames · 每根 1 分 K 只用當下以前的資料 · 對照策略 {CHECK_STRATEGY}（{n_marks} 個點，Hit {n_hit}）·
結論：有料，但已被「新高 + 上影線」涵蓋，不採用</p>
<h2>公式</h2><div class="scroll">{FORMULAS}</div>
<h2>連續特徵：百分位（50 = 沒資訊）</h2>
<p class="sub">「最高點那根」= 最高點那根在它的 frame 裡排第幾百分位，242 個平均。右兩欄 = 在 {CHECK_STRATEGY} 的點裡的百分位。</p>
<div class="scroll"><table><thead><tr><th>特徵</th><th>最高點那根</th><th>{CHECK_STRATEGY} 的 Hit</th>
<th>{CHECK_STRATEGY} 的其他點</th></tr></thead><tbody>{t1}</tbody></table></div>
<h2>訊號：出現比例</h2>
<div class="scroll"><table><thead><tr><th>訊號</th><th>最高點那根</th><th>一般 K 棒</th><th>倍數</th>
<th>{CHECK_STRATEGY} 的 Hit</th><th>{CHECK_STRATEGY} 的其他點</th></tr></thead><tbody>{t2}</tbody></table></div>
<h2>每個 frame 的圖</h2>
<p class="sub">上：K 線（★ {CHECK_STRATEGY} 的點、◆ 最高點、▲ 訊號 B）；中：推力 F（紅 = 買方推、綠 = 賣方推）；
下：累積買賣壓 v（β = 0.9）。虛線 = 最高點那根。{CHECK_STRATEGY} Hit 排前面。</p>
<div><button class="on" data-f="all">全部</button><button data-f="hit">{CHECK_STRATEGY} Hit</button>
<button data-f="miss">{CHECK_STRATEGY} Miss</button></div>
{cards}
<script>
document.querySelectorAll('button[data-f]').forEach(b=>b.onclick=()=>{{
document.querySelectorAll('button[data-f]').forEach(x=>x.classList.toggle('on',x===b));
document.querySelectorAll('.pic').forEach(p=>p.style.display=(b.dataset.f==='all'||p.dataset.k===b.dataset.f)?'':'none');}});
</script></main></body></html>"""


def main():
    df_list, key_list = load_frames()
    keep = keep_frames(table=max_bar_table(df_list, key_list))
    mask_fn = STRATEGIES[CHECK_STRATEGY]['mask']
    cont = ['F', 'acc'] + [f'{p}_{b}' for b in BETAS for p in ('v', 'dv')]
    sig = [f'A_{b}' for b in BETAS] + ['B', 'B2']
    rows, order = [], []
    (HERE / 'plots').mkdir(exist_ok=True)
    for i in keep:
        mb, max_time, _ = minute_bars(df_list[i])
        f = features(df_list[i], mb)
        max_x = mb.index.get_loc(max_time.floor('1min'))
        marks = np.flatnonzero(mask_fn(mb).fillna(False).to_numpy(dtype=bool))
        plot_frame(i, key_list[i], mb, f, max_x, marks, HERE / 'plots' / f'df_{i:03d}.png')
        order.append((i, 'hit' if max_x in set(marks) else 'miss'))
        f['df_idx'] = i
        f['is_max'] = np.arange(len(mb)) == mb.index.get_loc(max_time.floor('1min'))
        f['marked'] = mask_fn(mb).fillna(False).to_numpy(dtype=bool)
        for k in cont:
            f[f'{k}_pct'] = f[k].rank(pct=True) * 100
        rows.append(f.reset_index(names='time'))
    d = pd.concat(rows, ignore_index=True)
    d.to_csv(HERE / 'bars.csv', index=False, encoding='utf-8-sig')
    top, marks = d[d['is_max']], d[d['marked']].copy()
    out = []
    for k in cont:
        marks[f'{k}_mpct'] = marks[k].rank(pct=True) * 100
        out.append({'feature': k, 'type': '連續', 'max_bar': top[f'{k}_pct'].mean(), 'all_bars': 50.0,
                    'ratio': np.nan, 'hit_in_marks': marks.loc[marks['is_max'], f'{k}_mpct'].mean(),
                    'other_in_marks': marks.loc[~marks['is_max'], f'{k}_mpct'].mean()})
    for k in sig:
        a, m = top[k].mean() * 100, d[k].mean() * 100
        out.append({'feature': k, 'type': '訊號 %', 'max_bar': a, 'all_bars': m, 'ratio': a / m,
                    'hit_in_marks': marks.loc[marks['is_max'], k].mean() * 100,
                    'other_in_marks': marks.loc[~marks['is_max'], k].mean() * 100})
    s = pd.DataFrame(out)
    s.to_csv(HERE / 'summary.csv', index=False, encoding='utf-8-sig')
    order.sort(key=lambda r: (r[1] != 'hit', r[0]))
    (HERE / 'index.html').write_text(build_html(s, order, len(marks), int(marks['is_max'].sum())),
                                     encoding='utf-8')
    with pd.option_context('display.width', 200):
        print(f'242 frames、{len(d)} 根 K、{CHECK_STRATEGY} {len(marks)} 個點（Hit {int(marks["is_max"].sum())}）')
        print(s.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
