"""起點回測：每個 frame 的每一根 1 分 K 都當作起點 s（最早可能買進的時間），看 s 之後第一個賣點的虧損 / 獲利。

策略（strategies/type1_vol.py 的 vol_mask，三個條件同時成立就標記 t）：
    1. 成交量 t > MULT × 前 N 根（t-N ~ t-1）平均
    2. type1_ratio t >= RATIO_MIN
    3. close 連漲 RISE 次：c[t-RISE] < … < c[t-1] < c[t]
    標記只看 t 與之前的資料（sliding window），與起點無關：每個 frame 算一次。
    → Hit（最高點那根有沒有被標記）、標記數都看整個 frame 就好，與起點無關。

每個 frame × 起點 s（剩餘根數 >= MIN_REMAIN）：
    賣點     s 之後（含 s）第一個標記 t 的下一根 open（sell_time = t+1）；沒有標記或 t 是最後一根 = 沒出手
    最壞買點 s ~ t 之間的最高價（buy_time / buy_price）
    最好買點 s ~ t 之間的最低價（low_time / low_price）
    loss = buy_price − sell_price（點，負數 = 最壞買點也賺）
    gain = sell_price − low_price（點，買在最低時賺多少）
最高點 = 整段 tick 中最高價第一次出現的那筆（與 backtest_report.bars_of 相同）；價格用 1 分 K 的 High / Low / Open。

輸出（與本檔同資料夾）：detail.csv（frame × 起點）、frames.csv（每個 frame 一列）、index.html
執行（在專案根目錄）：python analysis/2026-10-24_start_backtest/backtest.py
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

from backtest_report import bars_of, load_frames  # noqa: E402
from frame_filter import keep_frames, max_bar_table  # noqa: E402
import type1_vol as tv  # noqa: E402

N, MULT, RATIO_MIN, RISE = 5, 2.0, 0.5, 3
MIN_REMAIN = 10      # 起點之後（含）至少要剩幾根


def frame_marks(df):
    bars, max_time, max_price = bars_of(df)
    m = tv.minute_table(df).reindex(bars.index)
    mark, _ = tv.vol_mask(m, N, RATIO_MIN, MULT, RISE)
    return bars, max_time, max_price, np.flatnonzero(mark.to_numpy())


def frame_starts(i, df):
    bars, max_time, max_price, marks = frame_marks(df)
    high, low, open_ = bars['High'].to_numpy(), bars['Low'].to_numpy(), bars['Open'].to_numpy()
    idx, n = bars.index, len(bars)
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


def q(x, p):
    x = x.dropna()
    return x.quantile(p) if len(x) else np.nan


def start_summary(d):
    """一組起點 → 一列摘要（虧損 / 獲利）。"""
    return pd.Series({
        'starts': len(d), 'no_trade_pct': d['loss'].isna().mean() * 100,
        'loss_med': q(d['loss'], .5), 'loss_p90': q(d['loss'], .9), 'loss_max': d['loss'].max(),
        'loss_pct_med': q(d['loss_pct'], .5), 'loss_pct_max': d['loss_pct'].max(),
        'gain_med': q(d['gain'], .5), 'gain_p10': q(d['gain'], .1), 'gain_min': d['gain'].min(),
    })


def fmt(v, f):
    return '' if pd.isna(v) else f.format(v)


S_COLS = [('starts', '{:.0f}'), ('no_trade_pct', '{:.1f}%'), ('loss_med', '{:.0f}'), ('loss_p90', '{:.0f}'),
          ('loss_max', '{:.0f}'), ('loss_pct_med', '{:.2f}%'), ('loss_pct_max', '{:.2f}%'),
          ('gain_med', '{:.0f}'), ('gain_p10', '{:.0f}'), ('gain_min', '{:.0f}')]
S_HEAD = ['起點數', '沒出手', '虧損 中位', '虧損 p90', '虧損 最大', '虧損% 中位', '虧損% 最大',
          '獲利 中位', '獲利 p10', '獲利 最小']


def table(rows, head, label):
    th = f'<tr><th>{label}</th>' + ''.join(f'<th>{h}</th>' for h in head) + '</tr>'
    return f'<div class="scroll"><table><thead>{th}</thead><tbody>{"".join(rows)}</tbody></table></div>'


def s_table(df, label):
    rows = [f'<tr><td>{html.escape(str(k))}</td>' + ''.join(f'<td>{fmt(r[c], f)}</td>' for c, f in S_COLS)
            + '</tr>' for k, r in df.iterrows()]
    return table(rows, S_HEAD, label)


def case(r, kind):
    """一筆最大虧損 / 最小獲利的算式。"""
    if kind == 'loss':
        return (f'df {r.df_idx:03d}，起點 {r.start:%H:%M}：最壞買點 {r.buy_time:%H:%M} 的最高價 {r.buy_price:g}'
                f' − 賣點 {r.sell_time:%H:%M} open {r.sell_price:g} = <b>{r.loss:g} 點</b>（標記在 {r.mark_time:%H:%M}）')
    return (f'df {r.df_idx:03d}，起點 {r.start:%H:%M}：賣點 {r.sell_time:%H:%M} open {r.sell_price:g}'
            f' − 最好買點 {r.low_time:%H:%M} 的最低價 {r.low_price:g} = <b>{r.gain:g} 點</b>（標記在 {r.mark_time:%H:%M}）')


def build_html(detail, frames):
    groups = [('242（分析用）', ~frames['excluded']), ('48（被排除）', frames['excluded']), ('全部 290', None)]
    g_rows = []
    for name, sel in groups:
        f = frames if sel is None else frames[sel]
        g_rows.append(f'<tr><td>{name}</td><td>{len(f)}</td><td>{int(f["hit"].sum())}</td>'
                      f'<td>{f["hit"].mean() * 100:.1f}%</td><td>{int(f["n_marks"].sum())}</td>'
                      f'<td>{f["n_marks"].mean():.2f}</td><td>{int((f["n_marks"] == 0).sum())}</td>'
                      f'<td>{int(f["marks_before_max"].sum())}</td><td>{int(f["marks_after_max"].sum())}</td></tr>')
    hit_tbl = table(g_rows, ['frame 數', 'Hit frame', 'Hit rate', '找到的點（總數）', '每個 frame 平均',
                             '沒有任何點的 frame', '點在最高點之前', '點在最高點之後'], '組別')

    d_groups = {'242（分析用）': detail[~detail['excluded']], '48（被排除）': detail[detail['excluded']],
                '全部 290': detail}
    s_sum = pd.DataFrame({k: start_summary(v) for k, v in d_groups.items()}).T
    bins = [-10**6, -31, -11, -1, 0, 10, 10**6]
    names = ['最高點前 > 30 根', '前 11～30 根', '前 1～10 根', '最高點那根', '後 1～10 根', '後 > 10 根']
    d242 = detail[~detail['excluded']].copy()
    d242['rel'] = pd.cut(d242['bars_from_max'], bins, labels=names)
    by_rel = d242.groupby('rel', observed=True).apply(start_summary, include_groups=False)

    cases = ''
    for name, d in d_groups.items():
        if d['loss'].notna().any():
            cases += (f'<li>{name} 最大虧損：{case(d.loc[d["loss"].idxmax()], "loss")}</li>'
                      f'<li>{name} 最小獲利：{case(d.loc[d["gain"].idxmin()], "gain")}</li>')

    per = detail.groupby(detail['df_idx'].rename('frame')).apply(start_summary).rename_axis('df_idx')
    ft = frames.set_index('df_idx').join(per).reset_index()
    f_rows = [
        f'<tr><td>{r.df_idx:03d}</td><td>{"48" if r.excluded else "242"}</td><td>{"Hit" if r.hit else ""}</td>'
        f'<td>{r.n_bars}</td><td>{r.max_bar}</td><td>{r.max_time:%H:%M}</td><td>{r.n_marks}</td>'
        f'<td class="t">{r.mark_times}</td><td>{fmt(r.no_trade_pct, "{:.0f}%")}</td>'
        f'<td>{fmt(r.loss_med, "{:.0f}")}</td><td>{fmt(r.loss_max, "{:.0f}")}</td>'
        f'<td>{fmt(r.gain_med, "{:.0f}")}</td><td>{fmt(r.gain_min, "{:.0f}")}</td></tr>'
        for r in ft.itertuples()]
    f_tbl = table(f_rows, ['組', 'Hit', 'K 根數', '最高點第幾根', '最高點時間', '找到的點', '點的時間', '沒出手',
                           '虧損 中位', '虧損 最大', '獲利 中位', '獲利 最小'], 'df')

    css = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--sub:#52514e;--line:#e1e0d9}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--sub:#c3c2b7;--line:#2c2c2a}}
body{margin:0;background:var(--page);color:var(--ink);font-family:system-ui,"Microsoft JhengHei",sans-serif;line-height:1.55}
main{max-width:1280px;margin:0 auto;padding:24px 16px} .sub{color:var(--sub)}
.scroll{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;width:100%}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left} thead th{position:sticky;top:0;background:var(--surface)}
td.t{text-align:left;white-space:normal;min-width:160px;color:var(--sub)}
h2{margin-top:32px} li{margin:3px 0} code{font-size:13px}
"""
    return f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>起點回測</title><style>{css}</style></head>
<body><main><h1>起點回測：{tv.default_name(N, RATIO_MIN, MULT, RISE)}</h1>
<p class="sub">290 frames（242 分析用 + 48 被排除）· {pd.Timestamp.now():%Y-%m-%d %H:%M}</p>
<p>策略：成交量 &gt; {MULT:g} × 前 {N} 根平均、type1_ratio ≥ {RATIO_MIN:g}、close 連漲 {RISE} 次 → 標記這根。
標記與起點無關，每個 frame 一組。</p>

<h2>1. Hit rate 與找到的點（看整個 frame，與起點無關）</h2>
<p class="sub">Hit = 最高點那根有被標記。找到的點 = 整個 frame 的標記數（= 起點 s 為第一根時）。</p>
{hit_tbl}

<h2>2. 虧損與獲利（每根 1 分 K 當起點 s，剩餘 ≥ {MIN_REMAIN} 根）</h2>
<ul>
<li><b>賣點</b> = s 之後（含 s）第一個標記的<b>下一根 open</b></li>
<li><b>虧損</b> = s ~ 標記之間的<b>最高價</b>（最壞買點）− 賣點 open。負數 = 最壞的買點也賺</li>
<li><b>獲利</b> = 賣點 open − s ~ 標記之間的<b>最低價</b>（最好買點）。小或負 = 就算買在最低也賺不多，找到的點不是真的高點</li>
<li>沒出手 = s 之後沒有可賣出的標記，不算虧損 / 獲利</li>
</ul>
<ul>{cases}</ul>
{s_table(s_sum, '組別')}
<h3>依起點與最高點的距離（242 frames）</h3>
{s_table(by_rel, '起點位置')}

<h2>3. 每個 frame</h2>
<p class="sub">虧損 / 獲利是該 frame 所有起點的中位數與極值。每一筆的買點、賣點時間與價格在 detail.csv。</p>
{f_tbl}
</main></body></html>"""


def main():
    df_list, key_list = load_frames()
    keep = set(keep_frames(table=max_bar_table(df_list, key_list)))
    rows, infos = [], []
    for i, df in enumerate(df_list):
        r, info = frame_starts(i, df)
        rows += r
        infos.append(info)
    detail = pd.DataFrame(rows)
    detail.insert(1, 'excluded', ~detail['df_idx'].isin(keep))
    frames = pd.DataFrame(infos)
    frames.insert(1, 'excluded', ~frames['df_idx'].isin(keep))
    detail.to_csv(HERE / 'detail.csv', index=False, encoding='utf-8-sig')
    frames.to_csv(HERE / 'frames.csv', index=False, encoding='utf-8-sig')
    (HERE / 'index.html').write_text(build_html(detail, frames), encoding='utf-8')
    for name, sel in (('242', ~frames['excluded']), ('48', frames['excluded'])):
        f = frames[sel]
        print(f'{name}: Hit {int(f["hit"].sum())} / {len(f)}，找到的點 {int(f["n_marks"].sum())}')
    g = detail.groupby('excluded').apply(start_summary, include_groups=False)
    with pd.option_context('display.width', 200, 'display.max_columns', 30):
        print(g.round(2).T)
    print(f'{len(df_list)} frames、{len(detail)} 個起點 → {HERE}')


if __name__ == '__main__':
    main()
