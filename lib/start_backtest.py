"""起點回測（所有策略共用）：策略標記 + 每根 1 分 K 當起點 s 的虧損 / 獲利。

策略介面
--------
    mask_fn(mb) -> bool Series（index 與 mb 相同），True = 標記這根。
    mb = minute_bars(df)：每分鐘一列，欄位 open, high, low, close, volume, type1_volume, type2_volume, ratio。
    只能用 t 與之前的資料（sliding window），所以標記與起點無關，每個 frame 算一次。

定義
----
    Hit          最高點那根有被標記（看整個 frame，與起點無關）
    找到的點     整個 frame 的標記數；多找的點 = 找到的點 − Hit
    隨機期望 Hit 每個 frame「標記數 / K 根數」加總；倍數 = Hit / 隨機期望
    起點 s       每一根 1 分 K（剩餘根數 >= MIN_REMAIN）
    賣點         s 之後（含 s）第一個標記 t 的下一根 open；沒有標記或 t 是最後一根 = 沒出手
    loss         s ~ t 之間的最高價（最壞買點）− 賣點（點）
    gain         賣點 − s ~ t 之間的最低價（最好買點）（點）
    最高點 = 整段 tick 中最高價第一次出現的那筆（與 backtest_report.bars_of 相同）。
"""
import html

import numpy as np
import pandas as pd

from backtest_report import bars_of

MIN_REMAIN = 10      # 起點之後（含）至少要剩幾根


def minute_bars(df):
    """tick → 每分鐘一列（open, high, low, close, volume, type1_volume, type2_volume, ratio），以及最高點 (time, price)。"""
    bars, max_time, max_price = bars_of(df)
    t = df[['ts', 'volume', 'tick_type']].copy()
    t['ts'] = pd.to_datetime(t['ts'])
    t['v1'] = t['volume'].where(t['tick_type'] == 1, 0)
    t['v2'] = t['volume'].where(t['tick_type'] == 2, 0)
    g = t.groupby(t['ts'].dt.floor('1min'))
    mb = bars.rename(columns=str.lower)
    mb['type1_volume'] = g['v1'].sum().reindex(mb.index)
    mb['type2_volume'] = g['v2'].sum().reindex(mb.index)
    mb['ratio'] = mb['type1_volume'] / mb['volume']
    return mb, max_time, max_price


def run_frame(i, df, mask_fn):
    """一個 frame → (每個起點一列的 list, frame 資訊 dict)。"""
    mb, max_time, max_price = minute_bars(df)
    marks = np.flatnonzero(mask_fn(mb).fillna(False).to_numpy(dtype=bool))
    high, low, open_ = mb['high'].to_numpy(), mb['low'].to_numpy(), mb['open'].to_numpy()
    idx, n = mb.index, len(mb)
    max_bar = idx.get_loc(max_time.floor('1min'))
    sellable = marks[marks + 1 < n]
    rows = []
    for s in range(n - MIN_REMAIN + 1):
        row = {'df_idx': i, 'start': idx[s], 'start_pos': s, 'bars_from_max': s - max_bar,
               'marks_after_s': int((marks >= s).sum())}
        ts = sellable[sellable >= s]
        if len(ts):
            t = ts[0]
            b = s + int(high[s:t + 1].argmax())
            lo = s + int(low[s:t + 1].argmin())
            sell = open_[t + 1]
            row.update({'mark_time': idx[t], 'sell_time': idx[t + 1], 'sell_price': sell,
                        'buy_time': idx[b], 'buy_price': high[b], 'loss': high[b] - sell,
                        'loss_pct': (high[b] - sell) / high[b] * 100,
                        'low_time': idx[lo], 'low_price': low[lo], 'gain': sell - low[lo],
                        'gain_pct': (sell - low[lo]) / low[lo] * 100})
        rows.append(row)
    info = {'df_idx': i, 'n_bars': n, 'max_bar': max_bar, 'max_time': max_time, 'max_price': max_price,
            'n_marks': len(marks), 'hit': max_bar in set(marks),
            'marks_before_max': int((marks < max_bar).sum()), 'marks_after_max': int((marks > max_bar).sum()),
            'mark_times': ' '.join(f'{idx[t]:%H:%M}' for t in marks)}
    return rows, info


def run(mask_fn, df_list, keep):
    """所有 frame → (detail, frames)。keep = 分析用的 df_idx（其他標為 excluded）。"""
    rows, infos = [], []
    for i, df in enumerate(df_list):
        r, info = run_frame(i, df, mask_fn)
        rows += r
        infos.append(info)
    detail, frames = pd.DataFrame(rows), pd.DataFrame(infos)
    for d in (detail, frames):
        d.insert(1, 'excluded', ~d['df_idx'].isin(keep))
    return detail, frames


def q(x, p):
    x = x.dropna()
    return x.quantile(p) if len(x) else np.nan


def start_summary(d):
    """一組起點 → 虧損 / 獲利摘要。"""
    return pd.Series({
        'starts': len(d), 'no_trade_pct': d['loss'].isna().mean() * 100,
        'loss_med': q(d['loss'], .5), 'loss_p90': q(d['loss'], .9), 'loss_max': d['loss'].max(),
        'loss_pct_med': q(d['loss_pct'], .5), 'loss_pct_max': d['loss_pct'].max(),
        'gain_med': q(d['gain'], .5), 'gain_p10': q(d['gain'], .1), 'gain_min': d['gain'].min(),
    })


def frame_summary(f):
    """一組 frame → Hit / 找到的點摘要。"""
    hits, marks = int(f['hit'].sum()), int(f['n_marks'].sum())
    rnd = (f['n_marks'] / f['n_bars']).clip(upper=1).sum()
    return pd.Series({
        'frames': len(f), 'hits': hits, 'hit_rate': hits / len(f) * 100 if len(f) else np.nan,
        'marks': marks, 'extra_marks': marks - hits, 'marks_per_frame': marks / len(f) if len(f) else np.nan,
        'marks_before_max': int(f['marks_before_max'].sum()), 'marks_after_max': int(f['marks_after_max'].sum()),
        'frames_no_marks': int((f['n_marks'] == 0).sum()),
        'random_hits': rnd, 'lift': hits / rnd if rnd else np.nan,
    })


GROUPS = [('all', '全部 290'), ('keep', '242（分析用）'), ('excluded', '48（被排除）')]


def select(d, group):
    return d if group == 'all' else d[d['excluded'] == (group == 'excluded')]


def summary(detail, frames):
    """每組一列：Hit / 找到的點 + 虧損 / 獲利。"""
    return pd.DataFrame({g: pd.concat([frame_summary(select(frames, g)), start_summary(select(detail, g))])
                         for g, _ in GROUPS}).T.rename_axis('group')


# ---------- HTML ----------

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--sub:#52514e;--line:#e1e0d9;--link:#2a78d6}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--sub:#c3c2b7;--line:#2c2c2a;--link:#3987e5}}
body{margin:0;background:var(--page);color:var(--ink);font-family:system-ui,"Microsoft JhengHei",sans-serif;line-height:1.55}
main{max-width:1280px;margin:0 auto;padding:24px 16px} .sub{color:var(--sub)} a{color:var(--link)}
.scroll{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;width:100%}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left} thead th{position:sticky;top:0;background:var(--surface)}
td.t{text-align:left;white-space:normal;min-width:160px;color:var(--sub)}
h2{margin-top:32px} li{margin:3px 0}
button{font:inherit;padding:4px 12px;margin:0 6px 10px 0;cursor:pointer} button.on{font-weight:700}
.pic{margin:14px 0} .pic img{width:100%;background:#fff;border:1px solid var(--line);border-radius:6px}
"""


def fmt(v, f):
    return '' if pd.isna(v) else f.format(v)


def table(rows, head, label):
    th = f'<tr><th>{label}</th>' + ''.join(f'<th>{h}</th>' for h in head) + '</tr>'
    return f'<div class="scroll"><table><thead>{th}</thead><tbody>{"".join(rows)}</tbody></table></div>'


F_COLS = [('frames', '{:.0f}', 'frame 數'), ('hits', '{:.0f}', 'Hit'), ('hit_rate', '{:.1f}%', 'Hit rate'),
          ('marks', '{:.0f}', '找到的點'), ('extra_marks', '{:.0f}', '多找的點'),
          ('marks_per_frame', '{:.2f}', '每個 frame 平均'), ('marks_before_max', '{:.0f}', '點在最高點前'),
          ('marks_after_max', '{:.0f}', '點在最高點後'), ('frames_no_marks', '{:.0f}', '沒有點的 frame')]
S_COLS = [('starts', '{:.0f}', '起點數'), ('no_trade_pct', '{:.1f}%', '沒出手'),
          ('loss_med', '{:.0f}', '虧損 中位'), ('loss_p90', '{:.0f}', '虧損 p90'), ('loss_max', '{:.0f}', '虧損 最大'),
          ('loss_pct_med', '{:.2f}%', '虧損% 中位'), ('loss_pct_max', '{:.2f}%', '虧損% 最大'),
          ('gain_med', '{:.0f}', '獲利 中位'), ('gain_p10', '{:.0f}', '獲利 p10'), ('gain_min', '{:.0f}', '獲利 最小')]


def df_table(df, cols, label, names=None):
    rows = [f'<tr><td>{html.escape(str((names or {}).get(k, k)))}</td>'
            + ''.join(f'<td>{fmt(r[c], f)}</td>' for c, f, _ in cols) + '</tr>' for k, r in df.iterrows()]
    return table(rows, [h for _, _, h in cols], label)


DEFINITIONS = """<ul>
<li><b>Hit</b> = 最高點那根有被標記（看整個 frame，與起點無關）。<b>多找的點</b> = 找到的點 − Hit</li>
<li>每根 1 分 K 當起點 s（剩餘 ≥ {mr} 根）。<b>賣點</b> = s 之後（含 s）第一個標記的<b>下一根 open</b></li>
<li><b>虧損</b> = s ~ 標記之間的<b>最高價</b>（最壞買點）− 賣點。負數 = 最壞的買點也賺</li>
<li><b>獲利</b> = 賣點 − s ~ 標記之間的<b>最低價</b>（最好買點）。小 = 找到的點不是真的高點</li>
<li><b>沒出手</b> = s 之後沒有可賣出的標記，不算虧損 / 獲利</li>
</ul>""".format(mr=MIN_REMAIN)


def case(r, kind):
    if kind == 'loss':
        return (f'df {r.df_idx:03d}，起點 {r.start:%H:%M}：最壞買點 {r.buy_time:%H:%M} 的最高價 {r.buy_price:g}'
                f' − 賣點 {r.sell_time:%H:%M} open {r.sell_price:g} = <b>{r.loss:g} 點</b>（標記在 {r.mark_time:%H:%M}）')
    return (f'df {r.df_idx:03d}，起點 {r.start:%H:%M}：賣點 {r.sell_time:%H:%M} open {r.sell_price:g}'
            f' − 最好買點 {r.low_time:%H:%M} 的最低價 {r.low_price:g} = <b>{r.gain:g} 點</b>（標記在 {r.mark_time:%H:%M}）')


def plot_frame(i, df, key, mask_fn, path, worst=None):
    """簡單版 K 線：★ 策略找到的點、◆ 最高點、灰底 = frame_start ~ frame_end。
    worst = 這個 frame 最大虧損那一列（detail 的一列）：畫 ▼ 買點、● 賣點和虛線。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    mb, max_time, max_price = minute_bars(df)
    marks = np.flatnonzero(mask_fn(mb).fillna(False).to_numpy(dtype=bool))
    x = np.arange(len(mb))
    max_x = mb.index.get_loc(max_time.floor('1min'))
    hit = max_x in set(marks)
    fig, (ax, av) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    col = np.where(mb['close'] >= mb['open'], '#d62728', '#2ca02c')        # 紅漲綠跌
    ax.vlines(x, mb['low'], mb['high'], color=col, linewidth=1)
    ax.bar(x, (mb['close'] - mb['open']).abs().clip(lower=0.5), bottom=mb[['open', 'close']].min(axis=1),
           color=col, width=0.7)
    vcol = np.full(len(mb), '#bbbbbb', dtype=object)
    vcol[marks] = '#1f6fd1'                                                 # 標記那根 = 藍（同 ★）
    av.bar(x, mb['volume'], color=vcol, width=0.75)
    av.set_ylabel('volume', fontsize=9)
    inside = np.flatnonzero((mb.index >= key[0]) & (mb.index <= key[1]))
    if len(inside):
        for a in (ax, av):
            a.axvspan(inside[0] - 0.5, inside[-1] + 0.5, color='#888', alpha=0.10, linewidth=0)
    # 符號都放在 K 棒外：★ ◆ ▼ 疊在 high 上方，● 在 low 下方，不蓋到上影線
    span = mb['high'].max() - mb['low'].min()
    gap, step_y = span * 0.05, span * 0.075
    stack = {}                                                              # 每根已經疊了幾個符號

    def above(bar):
        k = stack.get(bar, 0)
        stack[bar] = k + 1
        return mb['high'].iloc[bar] + gap + k * step_y

    ax.scatter(marks, [above(t) for t in marks], marker='*', s=170, color='#1f6fd1', zorder=5,
               label='策略找到的點')
    ax.scatter([max_x], [above(max_x)], marker='D', s=45, color='black', zorder=6, label='最高點')
    if worst is not None:
        bx, sx = mb.index.get_loc(worst['buy_time']), mb.index.get_loc(worst['sell_time'])
        bp, sp = worst['buy_price'], worst['sell_price']
        ax.plot([bx, sx], [bp, sp], color='#e8780c', linestyle='--', linewidth=1, alpha=0.6, zorder=0.5)
        ax.scatter([bx], [above(bx)], marker='v', s=90, color='#e8780c', zorder=7)
        ax.scatter([sx], [mb['low'].iloc[sx] - gap], marker='o', s=60, color='#e8780c', zorder=7)
        ax.plot([], [], color='#e8780c', linestyle='--', marker='v',
                label=f'最大虧損 {worst["loss"]:g} 點：▼ {worst["buy_time"]:%H:%M} 買 {bp:g} → '
                      f'● {worst["sell_time"]:%H:%M} 賣 {sp:g}')
    top = max(mb['high'].iloc[b] + gap + (k - 1) * step_y for b, k in stack.items()) if stack else mb['high'].max()
    ax.set_ylim(mb['low'].min() - span * 0.10, max(top, mb['high'].max()) + span * 0.06)
    step = max(1, len(mb) // 12)
    av.set_xticks(x[::step])
    av.set_xticklabels([f'{t:%H:%M}' for t in mb.index[::step]], fontsize=9)
    av.set_xlim(-1, len(mb))
    for a in (ax, av):
        a.grid(axis='y', color='#e5e5e5', linewidth=0.6)
        a.set_axisbelow(True)
        for side in ('top', 'right'):
            a.spines[side].set_visible(False)
    av.legend(*ax.get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(0, -0.28), ncol=3, fontsize=9,
              frameon=False)
    ax.set_title(f'df {i:03d}   {key[0]:%Y-%m-%d %H:%M} ~ {key[1]:%H:%M}   {"Hit" if hit else "Miss"}'
                 f'   找到 {len(marks)} 個點', fontsize=11, loc='left')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def plot_all(df_list, key_list, mask_fn, out, detail):
    traded = detail.dropna(subset=['loss'])
    worst = traded.loc[traded.groupby('df_idx')['loss'].idxmax()].set_index('df_idx')
    for i, (df, key) in enumerate(zip(df_list, key_list)):
        w = worst.loc[i] if i in worst.index else None
        plot_frame(i, df, key, mask_fn, out / 'plots' / f'df_{i:03d}.png', w)


def strategy_html(sid, name, desc, detail, frames, plots=True):
    """單一策略的完整報告；plots = 有沒有 plots/df_XXX.png。"""
    summ = summary(detail, frames)
    gname = dict(GROUPS)
    bins = [-10**6, -31, -11, -1, 0, 10, 10**6]
    labels = ['最高點前 > 30 根', '前 11～30 根', '前 1～10 根', '最高點那根', '後 1～10 根', '後 > 10 根']
    d242 = select(detail, 'keep').copy()
    d242['rel'] = pd.cut(d242['bars_from_max'], bins, labels=labels)
    by_rel = d242.groupby('rel', observed=True).apply(start_summary, include_groups=False)
    cases = ''
    for g, gn in GROUPS:
        d = select(detail, g)
        if d['loss'].notna().any():
            cases += (f'<li>{gn} 最大虧損：{case(d.loc[d["loss"].idxmax()], "loss")}</li>'
                      f'<li>{gn} 最小獲利：{case(d.loc[d["gain"].idxmin()], "gain")}</li>')
    per = detail.groupby(detail['df_idx'].rename('frame')).apply(start_summary).rename_axis('df_idx')
    ft = frames.set_index('df_idx').join(per).reset_index()
    link = '<a href="#df{0:03d}">{0:03d}</a>' if plots else '{0:03d}'
    f_rows = [
        f'<tr><td>{link.format(r.df_idx)}</td><td>{"48" if r.excluded else "242"}</td><td>{"Hit" if r.hit else ""}</td>'
        f'<td>{r.n_bars}</td><td>{r.max_bar}</td><td>{r.max_time:%H:%M}</td><td>{r.n_marks}</td>'
        f'<td class="t">{r.mark_times}</td><td>{fmt(r.no_trade_pct, "{:.0f}%")}</td>'
        f'<td>{fmt(r.loss_med, "{:.0f}")}</td><td>{fmt(r.loss_max, "{:.0f}")}</td>'
        f'<td>{fmt(r.gain_med, "{:.0f}")}</td><td>{fmt(r.gain_min, "{:.0f}")}</td></tr>'
        for r in ft.itertuples()]
    f_tbl = table(f_rows, ['組', 'Hit', 'K 根數', '最高點第幾根', '最高點時間', '找到的點', '點的時間', '沒出手',
                           '虧損 中位', '虧損 最大', '獲利 中位', '獲利 最小'], 'df')
    pics = ''
    if plots:
        order = frames.assign(k=~frames['hit']).sort_values(['k', 'excluded', 'df_idx'])
        cards = ''.join(
            f'<div class="pic" id="df{r.df_idx:03d}" data-k="{"hit" if r.hit else ("miss" if r.n_marks else "none")}"'
            f' data-g="{"excluded" if r.excluded else "keep"}"><a href="plots/df_{r.df_idx:03d}.png" target="_blank">'
            f'<img loading="lazy" src="plots/df_{r.df_idx:03d}.png" alt="df {r.df_idx}"></a></div>'
            for r in order.itertuples())
        pics = f"""<h2>4. 每個 frame 的圖</h2>
<p class="sub">★ = 策略找到的點　◆ = 最高點　橘色 ▼ → ● = 這個 frame 的最大虧損（▼ 最壞買點的最高價、● 賣點 = 標記下一根 open）
灰底 = frame 區間　K 線紅漲綠跌。Hit 排在最前面，點圖可放大。</p>
<div><button class="on" data-f="all">全部</button><button data-f="hit">Hit</button>
<button data-f="miss">Miss（有找到點）</button><button data-f="none">沒有找到點</button>
<label><input type="checkbox" id="only242" checked> 只看 242</label></div>
{cards}
<script>
let f='all';const only=document.getElementById('only242');
function show(){{document.querySelectorAll('.pic').forEach(p=>p.style.display=
((f==='all'||p.dataset.k===f)&&(!only.checked||p.dataset.g==='keep'))?'':'none');}}
document.querySelectorAll('button[data-f]').forEach(b=>b.onclick=()=>{{f=b.dataset.f;
document.querySelectorAll('button[data-f]').forEach(x=>x.classList.toggle('on',x===b));show();}});
only.onchange=show;show();
</script>"""
    return f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(sid)} 回測</title>
<style>{CSS}</style></head><body><main>
<p><a href="../index.html">← 策略比較</a></p>
<h1>{html.escape(sid)}：{html.escape(name)}</h1>
<p>{html.escape(desc)}</p>
<p class="sub">290 frames（242 分析用 + 48 被排除）· {pd.Timestamp.now():%Y-%m-%d %H:%M}</p>
{DEFINITIONS}
<h2>1. Hit 與找到的點</h2>{df_table(summ, F_COLS, '組別', gname)}
<h2>2. 虧損與獲利</h2><ul>{cases}</ul>{df_table(summ, S_COLS, '組別', gname)}
<h3>依起點與最高點的距離（242 frames）</h3>{df_table(by_rel, S_COLS, '起點位置')}
<h2>3. 每個 frame</h2>
<p class="sub">虧損 / 獲利是該 frame 所有起點的中位數與極值；每一筆的買點、賣點在 detail.csv。</p>
{f_tbl}
{pics}
</main></body></html>"""


def compare_html(summ):
    """summ = summary.csv（每個策略 × 組別一列）→ 比較總表。"""
    blocks, buttons = [], []
    for k, (g, gn) in enumerate(GROUPS):
        s = summ[summ['group'] == g].set_index('id')
        cols = [('name', '{}', '策略')] + F_COLS[1:] + S_COLS[1:]
        rows = [f'<tr><td><a href="{sid}/index.html">{html.escape(sid)}</a></td>'
                + ''.join(f'<td class="{"t" if c == "name" else ""}">{fmt(r[c], f)}</td>' for c, f, _ in cols)
                + '</tr>' for sid, r in s.iterrows()]
        hide = '' if k == 0 else ' style="display:none"'
        blocks.append(f'<div class="g" data-g="{g}"{hide}>{table(rows, [h for _, _, h in cols], "編號")}</div>')
        buttons.append(f'<button data-g="{g}"{" class=on" if k == 0 else ""}>{gn}</button>')
    desc = ''.join(f'<li><b>{html.escape(r.id)}</b> {html.escape(r["name"])}：{html.escape(r.desc)}'
                   f'<span class="sub">（{r.run_time}）</span></li>'
                   for _, r in summ[summ['group'] == 'all'].iterrows())
    return f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>策略比較</title><style>{CSS}</style></head>
<body><main><h1>策略比較</h1>
<p class="sub">目標：242 個 frame 中抓到 10%～20% 的最高點。同一套起點回測，點編號看各策略的完整報告。</p>
{DEFINITIONS}
<div>{''.join(buttons)}</div>{''.join(blocks)}
<h2>策略說明</h2><ul>{desc}</ul>
</main><script>
document.querySelectorAll('button').forEach(b=>b.onclick=()=>{{
document.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b));
document.querySelectorAll('.g').forEach(d=>d.style.display=d.dataset.g===b.dataset.g?'':'none');}});
</script></body></html>"""
