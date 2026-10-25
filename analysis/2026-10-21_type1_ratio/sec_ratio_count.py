"""每個 frame 的每一分鐘：該分鐘內有成交的秒，type1_ratio >= 0.7 / 0.8 / 0.9 各有幾秒、那些秒的成交量；
以及每分鐘成交量 vs 每分鐘 type1_ratio 的相關係數。

輸入：sec/、min/df_XXX.csv（make.py 產生）
輸出（與本檔同資料夾）：
    sec_ratio_count.csv   每列 = 一個 frame 的一分鐘
        df_idx, minute, open（該分鐘開盤價）, close（該分鐘收盤價）, high（該分鐘最高價）, n_sec（有成交的秒數）, volume（該分鐘成交量）, type1_ratio（該分鐘）,
        ge_0.7, vol_ge_0.7, ge_0.8, vol_ge_0.8, ge_0.9, vol_ge_0.9,
        vol_share_ge_0.9（= vol_ge_0.9 / volume）,
        grad（梯度 = 這分鐘 type1_ratio − 上一分鐘）,
        velocity（速度 = BETA × 上一分鐘速度 + (1 − BETA) × grad，frame 起點 = 0）
        （ge_x = 達到門檻的秒數；vol_ge_x = 那些秒的成交量合計）
    sec_ratio_corr.csv    每列 = 一個 frame：該 frame 內每分鐘 volume vs type1_ratio 的 Pearson / Spearman
    sec_ratio_count.html  報告，用瀏覽器打開

執行（在專案根目錄）：python analysis/2026-10-21_type1_ratio/sec_ratio_count.py
"""
import html
import json
import webbrowser
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
THRESHOLDS = [0.7, 0.8, 0.9]
MAX_GAP = 5          # 相鄰分鐘間隔超過幾分鐘（例如休市）就重新起算速度
BETA = 0.1           # 動量係數（同 NN 的 momentum β）


def count_frame(i):
    s = pd.read_csv(HERE / 'sec' / f'df_{i:03d}.csv')
    m = pd.read_csv(HERE / 'min' / f'df_{i:03d}.csv').set_index('minute')
    out = pd.DataFrame({'open': m['open'], 'close': m['close'], 'high': m['high'], 'n_sec': s.groupby('minute').size(),
                        'volume': m['volume'], 'type1_ratio': m['type1_ratio']})
    for th in THRESHOLDS:
        hit = s['type1_ratio'] >= th
        out[f'ge_{th}'] = hit.groupby(s['minute']).sum().astype(int)
        out[f'vol_ge_{th}'] = s['volume'].where(hit, 0).groupby(s['minute']).sum().astype(int)
    out['vol_share_ge_0.9'] = out['vol_ge_0.9'] / out['volume']   # ≥0.9 的秒佔該分鐘成交量的比例

    # 類 NN 的 momentum（以分鐘為 time slot）
    #   梯度 grad(t) = (ratio(t) - ratio(t-1)) / 間隔分鐘數；只用到過去，每分鐘結束時就算得出來
    #   速度 velocity(t) = BETA * velocity(t-1) + (1 - BETA) * grad(t)；frame 起點 velocity = 0
    #   間隔 > MAX_GAP 分鐘（例如休市）：grad = 0、velocity 從 0 重新起算
    out = out.sort_index()
    gap = (pd.Series(pd.to_datetime(out.index), index=out.index).diff().dt.total_seconds() / 60).to_numpy()
    grad = (out['type1_ratio'].diff().to_numpy() / gap)
    vel, v = [], 0.0
    for k in range(len(out)):
        if k == 0 or gap[k] > MAX_GAP:
            grad[k], v = 0.0, 0.0
        else:
            v = BETA * v + (1 - BETA) * grad[k]
        vel.append(v)
    out['grad'] = grad
    out['velocity'] = vel
    out.index.name = 'minute'
    out.insert(0, 'df_idx', i)
    return out.reset_index()


def correlations(res):
    """每個 frame 各自：該 frame 內每分鐘 volume vs type1_ratio 的相關係數。"""
    def spearman(x, y):                              # = 排名後的 Pearson（不需要 scipy）
        return x.rank().corr(y.rank())

    rows = [{'df_idx': i, 'n_min': len(g),
             'pearson': g['volume'].corr(g['type1_ratio']),
             'spearman': spearman(g['volume'], g['type1_ratio'])}
            for i, g in res.groupby('df_idx')]
    return pd.DataFrame(rows)


CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e1e0d9;
--border:rgba(11,11,11,.10);--max-bg:#fdebc8;--max-ink:#8a5300}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;
--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--border:rgba(255,255,255,.10);--max-bg:#4a3510;--max-ink:#f5c46b}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--grid:#2c2c2a;--border:rgba(255,255,255,.10);--max-bg:#4a3510;--max-ink:#f5c46b}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
font:14px/1.5 system-ui,-apple-system,"Segoe UI","Microsoft JhengHei",sans-serif}
main{max-width:1240px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px}
.sub{color:var(--ink2);margin:0 0 4px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;overflow:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 12px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600;position:sticky;top:0;background:var(--surface)}
.scroll{max-height:640px}
tr.maxrow td{background:var(--max-bg);font-weight:600}
.maxtag{background:var(--max-bg);color:var(--max-ink);padding:0 8px;border-radius:999px;font-weight:600}
select{font:inherit;padding:4px 8px;border-radius:8px;border:1px solid var(--border);
background:var(--surface);color:var(--ink)}
"""

JS = """
const DATA = __DATA__;
const sel = document.getElementById('frame'), body = document.getElementById('rows'),
      info = document.getElementById('info');
DATA.frames.forEach((f, k) => sel.add(new Option(`df ${String(f.i).padStart(3,'0')}　${f.label}`, k)));
function show(k) {
  const f = DATA.frames[k];
  info.textContent = `${f.rows.length} 分鐘　·　這個 frame 的相關係數：Pearson ${f.pearson}、Spearman ${f.spearman}`;
  body.innerHTML = f.rows.map(r => (r[0] === f.max_minute ? '<tr class="maxrow">' : '<tr>') +
    r.map(v => `<td>${v}</td>`).join('') + '</tr>').join('');
  info.innerHTML = info.textContent + `　·　<span class="maxtag">最高價 ${f.max_price} @ ${f.max_time}</span>`;
}
sel.addEventListener('change', () => show(sel.value));
show(0);

// 相關係數表：點欄位標題排序；點一列 → 下方切到那個 frame
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


def build_html(res, key_list, per, fmax):
    th_cols = [c for th in THRESHOLDS for c in (f'ge_{th}', f'vol_ge_{th}')]
    cols = ['n_sec', 'volume'] + th_cols
    summ = res[cols].agg(['mean', 'median', 'max'])
    names = {'mean': '平均', 'median': '中位數', 'max': '最大'}
    share = res['vol_share_ge_0.9'].agg(['mean', 'median', 'max'])
    summary_rows = ''.join(
        f'<tr><td>{names[k]}</td>' + ''.join(f'<td>{summ.loc[k, c]:,.1f}</td>' for c in cols)
        + f'<td>{share[k] * 100:.1f}%</td></tr>' for k in ['mean', 'median', 'max'])
    pc = per.set_index('df_idx')
    data = {'frames': []}
    for i, g in res.groupby('df_idx'):
        data['frames'].append({
            'i': int(i),
            'max_minute': fmax.loc[i, 'max_minute'][11:16],
            'max_time': fmax.loc[i, 'max_time'][11:19],
            'max_price': f'{fmax.loc[i, "max_price"]:,.0f}',
            'label': f'{key_list[i][0]:%Y-%m-%d %H:%M} ~ {key_list[i][1]:%H:%M}',
            'pearson': f'{pc.loc[i, "pearson"]:.2f}', 'spearman': f'{pc.loc[i, "spearman"]:.2f}',
            'rows': [[r[0][11:16], f'{r[1]:,.0f}', f'{r[2]:,.0f}', f'{r[3]:,.0f}', int(r[4]), f'{int(r[5]):,}',
                      f'{r[6]:.2f}']
                     + [int(v) for v in r[7:-3]]
                     + [f'{r[-3] * 100:.1f}%', f'{r[-2]:+.3f}', f'{r[-1]:+.3f}']
                     for r in g[['minute', 'open', 'close', 'high', 'n_sec', 'volume', 'type1_ratio'] + th_cols
                                + ['vol_share_ge_0.9', 'grad', 'velocity']].itertuples(index=False)],
        })
    th_head = ''.join(f'<th>≥ {th} 秒數</th><th>≥ {th} 成交量</th>' for th in THRESHOLDS)
    corr_rows = ''.join(
        f'<tr data-k="{k}"><td>{int(r.df_idx):03d}</td>'
        f'<td>{key_list[int(r.df_idx)][0]:%Y-%m-%d %H:%M} ~ {key_list[int(r.df_idx)][1]:%H:%M}</td>'
        f'<td>{int(r.n_min)}</td><td data-v="{r.pearson:.6f}">{r.pearson:.2f}</td>'
        f'<td data-v="{r.spearman:.6f}">{r.spearman:.2f}</td></tr>'
        for k, r in enumerate(per.itertuples(index=False)))
    corr_html = (
        '<h2>每個 frame：每分鐘成交量 vs 每分鐘 type1_ratio 的相關係數</h2>'
        f'<p class="sub">{len(per)} 個 frame 中 Pearson 為正的有 {int((per["pearson"] > 0).sum())} 個，'
        f'中位數 {per["pearson"].median():.2f}。點欄位標題可排序，點一列會在下方顯示那個 frame 的每分鐘明細。</p>'
        '<div class="card scroll"><table id="corr"><thead><tr><th>df</th><th>frame</th><th>分鐘數</th>'
        f'<th>Pearson</th><th>Spearman</th></tr></thead><tbody>{corr_rows}</tbody></table></div>')
    body = (
        '<main><h1>每分鐘內 type1_ratio 高的秒數</h1>'
        f'<p class="sub">{res["df_idx"].nunique()} frames · {len(res):,} 分鐘</p>'
        '<p class="sub">每一秒的 type1_ratio = 該秒外盤成交量 / 該秒全部成交量；只計算有成交的秒。'
        '「≥ x 秒數」= 該分鐘內 type1_ratio 達到門檻的秒數；「≥ x 成交量」= 那些秒的成交量合計；「≥ 0.9 成交量佔比」= ≥ 0.9 成交量 / 該分鐘成交量。</p>'
        '<p class="sub">梯度 = 這分鐘 type1_ratio − 上一分鐘（只用過去資料）；速度 = 0.1 × 上一分鐘速度 + 0.9 × 梯度（同 NN 的 momentum，β = 0.1）。每個 frame 第一分鐘速度 = 0，中間斷超過 5 分鐘也從 0 重新起算；因為從 0 起算，前幾分鐘的速度會偏小。</p>'
        '<h2>所有分鐘合計</h2><div class="card"><table><thead><tr><th>每分鐘</th><th>有成交的秒數</th>'
        f'<th>成交量</th>{th_head}<th>≥ 0.9 成交量佔比</th></tr></thead><tbody>{summary_rows}</tbody></table></div>'
        f'{corr_html}'
        '<h2 id="frame-sec">每個 frame 的每分鐘明細</h2>'
        '<p class="sub">標色那一列 = 該 frame 最高價（第一次出現）所在的分鐘。</p>'
        '<p class="sub"><select id="frame"></select>　<span id="info"></span></p>'
        '<div class="card scroll"><table><thead><tr><th>分鐘</th><th>開盤價</th><th>收盤價</th><th>最高價</th>'
        '<th>有成交的秒數</th><th>成交量</th>'
        f'<th>type1_ratio</th>{th_head}<th>≥ 0.9 成交量佔比</th><th>梯度 Δratio</th><th>速度</th></tr></thead><tbody id="rows"></tbody></table></div></main>'
    )
    js = JS.replace('__DATA__', json.dumps(data, ensure_ascii=False))
    return (f'<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape("每分鐘高外盤秒數")}</title><style>{CSS}</style></head>'
            f'<body>{body}<script>{js}</script></body></html>')


def main(open_browser=True):
    import sys
    sys.path.insert(0, str(HERE.parents[1] / 'lib'))
    from backtest_report import load_frames
    _, key_list = load_frames()
    res = pd.concat([count_frame(i) for i in range(len(key_list))], ignore_index=True)
    res.to_csv(HERE / 'sec_ratio_count.csv', index=False)
    per = correlations(res)
    per.to_csv(HERE / 'sec_ratio_corr.csv', index=False)
    report = HERE / 'sec_ratio_count.html'
    fmax = pd.read_csv(HERE / 'frame_max.csv').set_index('df_idx')   # make.py 產生
    report.write_text(build_html(res, key_list, per, fmax), encoding='utf-8')
    print(f'{len(res)} 分鐘 → {HERE / "sec_ratio_count.csv"}；報告 {report}')
    if open_browser:
        webbrowser.open(report.as_uri())


if __name__ == '__main__':
    main(open_browser='--no-open' not in __import__('sys').argv)
