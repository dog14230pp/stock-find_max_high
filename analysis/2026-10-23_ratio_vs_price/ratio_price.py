"""type1_ratio 大雜燴：把 2026-10-21 的 sec_ratio_count（高 ratio 秒數、梯度 / 速度、成交量相關）
和每分鐘價格變動合在同一份報告。

價格變動（點數）
    body     = 這分鐘 close − 這分鐘 open（分鐘內漲跌）
    d_close  = 這分鐘 close − 上一分鐘 close（間隔 > MAX_GAP 分鐘，例如休市，不算）
    next_d   = 下一分鐘的 d_close（這分鐘的 ratio 能不能預測下一分鐘）

輸入：analysis/2026-10-21_type1_ratio/sec/、min/、frame_max.csv（make.py 產生）
輸出（與本檔同資料夾）：
    ratio_price.csv       每列 = 一個 frame 的一分鐘：sec_ratio_count.csv 的欄位 + open, body, d_close, next_d
    ratio_price_corr.csv  每列 = 一個 frame：type1_ratio vs volume / body / d_close / next_d 的 Pearson、Spearman
    ratio_price.html      報告，用瀏覽器打開

執行（在專案根目錄）：python analysis/2026-10-23_ratio_vs_price/ratio_price.py
"""
import html
import json
import sys
import webbrowser
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / '2026-10-21_type1_ratio'
sys.path.insert(0, str(SRC))
from sec_ratio_count import BETA, CSS, THRESHOLDS, count_frame  # noqa: E402

MAX_GAP = 5          # 相鄰分鐘間隔超過幾分鐘就不算 d_close（同 sec_ratio_count.py）
TARGETS = {'volume': '成交量', 'body': '分鐘內漲跌 (close − open)', 'd_close': 'Δclose (− 上一分鐘 close)',
           'next_d': '下一分鐘 Δclose'}
PRICE = ['body', 'd_close', 'next_d']
BINS = [0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 1.0001]


def frame_minutes(i):
    m = pd.read_csv(SRC / 'min' / f'df_{i:03d}.csv').sort_values('minute')
    gap = pd.to_datetime(m['minute']).diff().dt.total_seconds() / 60
    p = pd.DataFrame({'minute': m['minute'],
                      'body': m['close'] - m['open'],
                      'd_close': m['close'].diff().where(gap <= MAX_GAP)})
    p['next_d'] = p['d_close'].shift(-1)
    return count_frame(i).merge(p, on='minute', how='left')       # count_frame 已有 open


def correlations(res):
    """每個 frame 各自：type1_ratio vs 成交量 / 各種價格變動的 Pearson / Spearman。"""
    rows = []
    for i, g in res.groupby('df_idx'):
        row = {'df_idx': i, 'n_min': len(g)}
        for t in TARGETS:
            row[f'{t}_pearson'] = g['type1_ratio'].corr(g[t])
            row[f'{t}_spearman'] = g['type1_ratio'].rank().corr(g[t].rank())   # = 排名後的 Pearson
        rows.append(row)
    return pd.DataFrame(rows)


def ratio_bins(res):
    """所有分鐘合併：依 type1_ratio 分組，看平均漲跌與上漲比例。"""
    g = res.groupby(pd.cut(res['type1_ratio'], BINS, right=False), observed=True)
    out = pd.DataFrame({'n': g.size(), 'volume_mean': g['volume'].mean()})
    for t in PRICE:
        out[f'{t}_mean'] = g[t].mean()
        out[f'{t}_up'] = g[t].apply(lambda s: (s.dropna() > 0).mean())
    return out


JS = """
const DATA = __DATA__;
const sel = document.getElementById('frame'), body = document.getElementById('rows'),
      info = document.getElementById('info');
DATA.frames.forEach((f, k) => sel.add(new Option(`df ${String(f.i).padStart(3,'0')}　${f.label}`, k)));
function show(k) {
  const f = DATA.frames[k];
  info.innerHTML = `${f.rows.length} 分鐘　·　ratio 的 Pearson：成交量 ${f.c[0]}、分鐘內漲跌 ${f.c[1]}、` +
    `Δclose ${f.c[2]}、下一分鐘 ${f.c[3]}　·　<span class="maxtag">最高價 ${f.max_price} @ ${f.max_time}</span>`;
  body.innerHTML = f.rows.map(r => (r[0] === f.max_minute ? '<tr class="maxrow">' : '<tr>') +
    r.map(v => `<td>${v}</td>`).join('') + '</tr>').join('');
}
sel.addEventListener('change', () => show(sel.value));
show(0);

const ct = document.getElementById('corr');
ct.querySelectorAll('th').forEach((th, c) => th.addEventListener('click', () => {
  const tb = ct.tBodies[0], rows = [...tb.rows], asc = th.dataset.dir !== 'asc';
  ct.querySelectorAll('th').forEach(x => delete x.dataset.dir); th.dataset.dir = asc ? 'asc' : 'desc';
  rows.sort((a, b) => {
    const x = a.cells[c].dataset.v ?? a.cells[c].textContent, y = b.cells[c].dataset.v ?? b.cells[c].textContent;
    const r = (!isNaN(+x) && !isNaN(+y)) ? x - y : x.localeCompare(y);
    return asc ? r : -r; });
  rows.forEach(r => tb.appendChild(r));
}));
ct.tBodies[0].addEventListener('click', e => {
  const tr = e.target.closest('tr'); if (!tr) return;
  sel.value = tr.dataset.k; show(sel.value);
  document.getElementById('frame-sec').scrollIntoView({behavior: 'smooth'});
});
"""


def fmt(v, spec):
    return '' if pd.isna(v) else format(v, spec)


def build_html(res, key_list, per, bins, fmax):
    label = lambda i: f'{key_list[i][0]:%Y-%m-%d %H:%M} ~ {key_list[i][1]:%H:%M}'
    th_cols = [c for th in THRESHOLDS for c in (f'ge_{th}', f'vol_ge_{th}')]
    th_head = ''.join(f'<th>≥ {th} 秒數</th><th>≥ {th} 成交量</th>' for th in THRESHOLDS)
    p_head = ''.join(f'<th>{TARGETS[t]}</th>' for t in PRICE)

    # 1. 所有分鐘合計（同 sec_ratio_count）
    cols = ['n_sec', 'volume'] + th_cols
    summ = res[cols].agg(['mean', 'median', 'max'])
    share = res['vol_share_ge_0.9'].agg(['mean', 'median', 'max'])
    names = {'mean': '平均', 'median': '中位數', 'max': '最大'}
    total_rows = ''.join(
        f'<tr><td>{names[k]}</td>' + ''.join(f'<td>{summ.loc[k, c]:,.1f}</td>' for c in cols)
        + f'<td>{share[k] * 100:.1f}%</td></tr>' for k in names)

    # 2. 每個 frame 相關係數的分布
    summ_rows = ''
    for t, name in TARGETS.items():
        p, s = per[f'{t}_pearson'].dropna(), per[f'{t}_spearman'].dropna()
        summ_rows += (f'<tr><td>{name}</td><td>{p.median():+.3f}</td><td>{p.mean():+.3f}</td>'
                      f'<td>{(p > 0).sum()} / {len(p)}（{(p > 0).mean() * 100:.0f}%）</td>'
                      f'<td>{(p > 0.3).sum()}</td><td>{(p < -0.3).sum()}</td><td>{s.median():+.3f}</td>'
                      f'<td>{res["type1_ratio"].corr(res[t]):+.3f}</td></tr>')

    # 3. ratio 分組
    bin_rows = ''.join(
        f'<tr><td>{iv.left:.2f} – {min(iv.right, 1):.2f}</td><td>{int(r.n):,}</td><td>{r.volume_mean:,.0f}</td>'
        + ''.join(f'<td>{getattr(r, t + "_mean"):+.2f}</td><td>{getattr(r, t + "_up") * 100:.1f}%</td>'
                  for t in PRICE) + '</tr>'
        for iv, r in zip(bins.index, bins.itertuples(index=False)))

    # 4. 每個 frame 的相關係數表
    ccols = [f'{t}_{k}' for t in TARGETS for k in ('pearson', 'spearman')]
    corr_rows = ''.join(
        f'<tr data-k="{k}"><td>{int(r["df_idx"]):03d}</td><td>{label(int(r["df_idx"]))}</td><td>{int(r["n_min"])}</td>'
        + ''.join(f'<td data-v="{fmt(r[c], ".6f")}">{fmt(r[c], "+.2f")}</td>' for c in ccols) + '</tr>'
        for k, r in per.iterrows())

    # 5. 每分鐘明細
    pc = per.set_index('df_idx')
    data = {'frames': []}
    for i, g in res.groupby('df_idx'):
        data['frames'].append({
            'i': int(i), 'label': label(i),
            'max_minute': fmax.loc[i, 'max_minute'][11:16],
            'max_time': fmax.loc[i, 'max_time'][11:19],
            'max_price': f'{fmax.loc[i, "max_price"]:,.0f}',
            'c': [fmt(pc.loc[i, f'{t}_pearson'], '+.2f') for t in TARGETS],
            'rows': [[r['minute'][11:16], f'{r["open"]:,.0f}', f'{r["close"]:,.0f}', f'{r["high"]:,.0f}',
                      int(r['n_sec']), f'{int(r["volume"]):,}', f'{r["type1_ratio"]:.3f}']
                     + [int(r[c]) for c in th_cols]
                     + [f'{r["vol_share_ge_0.9"] * 100:.1f}%', f'{r["grad"]:+.3f}', f'{r["velocity"]:+.3f}']
                     + [fmt(r[t], '+.0f') for t in PRICE]
                     for _, r in g.iterrows()],
        })

    body = (
        '<main><h1>type1_ratio 大雜燴</h1>'
        f'<p class="sub">{res["df_idx"].nunique()} frames · {len(res):,} 分鐘 · 價格變動單位：點</p>'
        '<p class="sub">type1_ratio = 外盤成交量 / 全部成交量。「≥ x 秒數」= 該分鐘內每秒 type1_ratio 達到門檻的秒數；'
        '「≥ x 成交量」= 那些秒的成交量合計；「≥ 0.9 成交量佔比」= ≥ 0.9 成交量 / 該分鐘成交量。</p>'
        f'<p class="sub">梯度 = 這分鐘 type1_ratio − 上一分鐘；速度 = {BETA} × 上一分鐘速度 + {1 - BETA:.1f} × 梯度'
        f'（frame 起點 = 0，斷超過 {MAX_GAP} 分鐘重新起算）。</p>'
        f'<p class="sub">分鐘內漲跌 = close − open；Δclose = close − 上一分鐘 close（間隔 > {MAX_GAP} 分鐘不算）；'
        '下一分鐘 Δclose = 用這分鐘的 ratio 對下一分鐘的漲跌，只有這一欄是「事前」能用的。</p>'
        '<h2>所有分鐘合計</h2><div class="card"><table><thead><tr><th>每分鐘</th><th>有成交的秒數</th>'
        f'<th>成交量</th>{th_head}<th>≥ 0.9 成交量佔比</th></tr></thead><tbody>{total_rows}</tbody></table></div>'
        '<h2>type1_ratio vs 成交量 / 價格：每個 frame 各自算相關係數，再看 290 個的分布</h2>'
        '<div class="card"><table><thead><tr><th>ratio vs</th><th>Pearson 中位數</th><th>Pearson 平均</th>'
        '<th>Pearson &gt; 0 的 frame</th><th>&gt; 0.3</th><th>&lt; −0.3</th><th>Spearman 中位數</th>'
        '<th>所有分鐘混在一起的 Pearson</th></tr></thead>'
        f'<tbody>{summ_rows}</tbody></table></div>'
        '<h2>依 type1_ratio 分組（所有分鐘合併）</h2>'
        '<p class="sub">每組的平均成交量、平均漲跌（點）與上漲比例（&gt; 0 的分鐘佔比，平盤算不漲）。</p>'
        '<div class="card"><table><thead><tr><th>type1_ratio</th><th>分鐘數</th><th>平均成交量</th>'
        + ''.join(f'<th>{TARGETS[t]} 平均</th><th>上漲比例</th>' for t in PRICE)
        + f'</tr></thead><tbody>{bin_rows}</tbody></table></div>'
        '<h2>每個 frame 的相關係數（type1_ratio vs …）</h2>'
        '<p class="sub">點欄位標題可排序，點一列會在下方顯示那個 frame 的每分鐘明細。</p>'
        '<div class="card scroll"><table id="corr"><thead><tr><th>df</th><th>frame</th><th>分鐘數</th>'
        + ''.join(f'<th>{n}<br>Pearson</th><th>{n}<br>Spearman</th>' for n in TARGETS.values())
        + f'</tr></thead><tbody>{corr_rows}</tbody></table></div>'
        '<h2 id="frame-sec">每個 frame 的每分鐘明細</h2>'
        '<p class="sub">標色那一列 = 該 frame 最高價（第一次出現）所在的分鐘。</p>'
        '<p class="sub"><select id="frame"></select>　<span id="info"></span></p>'
        '<div class="card scroll"><table><thead><tr><th>分鐘</th><th>開盤</th><th>收盤</th><th>最高價</th>'
        f'<th>有成交的秒數</th><th>成交量</th><th>type1_ratio</th>{th_head}<th>≥ 0.9 成交量佔比</th>'
        f'<th>梯度 Δratio</th><th>速度</th>{p_head}</tr></thead><tbody id="rows"></tbody></table></div></main>'
    )
    js = JS.replace('__DATA__', json.dumps(data, ensure_ascii=False))
    return (f'<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape("type1_ratio 大雜燴")}</title><style>{CSS}</style></head>'
            f'<body>{body}<script>{js}</script></body></html>')


def main(open_browser=True):
    sys.path.insert(0, str(HERE.parents[1] / 'lib'))
    from backtest_report import load_frames
    _, key_list = load_frames()
    res = pd.concat([frame_minutes(i) for i in range(len(key_list))], ignore_index=True)
    res.to_csv(HERE / 'ratio_price.csv', index=False)
    per = correlations(res)
    per.to_csv(HERE / 'ratio_price_corr.csv', index=False)
    bins = ratio_bins(res)
    fmax = pd.read_csv(SRC / 'frame_max.csv').set_index('df_idx')
    report = HERE / 'ratio_price.html'
    report.write_text(build_html(res, key_list, per, bins, fmax), encoding='utf-8')
    print(f'{len(res)} 分鐘 → {HERE / "ratio_price.csv"}；報告 {report}')
    for t in TARGETS:
        p = per[f'{t}_pearson']
        print(f'  ratio vs {t:8s} Pearson 中位數 {p.median():+.3f}，> 0 的 frame {(p > 0).sum()}/{p.notna().sum()}')
    if open_browser:
        webbrowser.open(report.as_uri())


if __name__ == '__main__':
    main(open_browser='--no-open' not in sys.argv)
