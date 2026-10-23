"""讀策略輸出的標記 CSV，對每個 frame 判斷 frame 最高點有沒有被標記到（Hit / Miss），輸出 HTML 報告。

CSV 格式
--------
    資料夾 : 一個策略一個資料夾，例如 marks/type1/
    檔名   : df_000.csv, df_001.csv, ...（編號 = df_list 的位置）
    內容   : 一欄 time，一列一個標記時間，例如
                time
                2021-05-13 09:44:00
                2021-05-13 09:53:27
    沒有標記的 frame：不放檔案，或只放標題列

判斷
----
    標記時間與 frame 最高點時間都對齊到所在的 1 分 K；
    最高點那一分鐘（最高價第一次出現的那筆 tick）有被標記 = Hit，否則 Miss。

用法
----
    python lib/backtest_report.py marks/type1             報告輸出到 reports/type1/index.html
    python lib/backtest_report.py marks/type1 --name v2   報告輸出到 reports/v2/index.html
    （相對路徑都以專案根目錄 D:\stock 為準，從哪裡執行都可以）
    寫 CSV 可以用 save_marks(times, df_idx, 'marks/type1')
"""
from __future__ import annotations

import argparse
import html
import time
import webbrowser
from pathlib import Path

import pandas as pd

FREQ = '1min'
ROOT = Path(__file__).resolve().parents[1]          # 專案根目錄（lib 的上一層）
DATA_PATH = ROOT / 'data' / '20261017高點研究' / 'df_dict_ticks.pkl.gz'
REPORT_DIR = ROOT / 'reports'


def load_frames(path=DATA_PATH):
    """讀 pkl.gz，回傳 (df_list, key_list)。"""
    d = pd.read_pickle(path, compression='gzip')
    return list(d.values()), list(d.keys())


def save_marks(times, df_idx, marks_dir):
    """把一個 frame 的標記時間寫成 <marks_dir>/df_XXX.csv。"""
    marks_dir = Path(marks_dir)
    marks_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'time': pd.to_datetime(list(times))}).to_csv(
        marks_dir / f'df_{df_idx:03d}.csv', index=False)


def read_marks(marks_dir, df_idx):
    """讀 <marks_dir>/df_XXX.csv，回傳排序過的標記時間；沒有檔案回傳空的。"""
    path = Path(marks_dir) / f'df_{df_idx:03d}.csv'
    if not path.exists():
        return pd.DatetimeIndex([])
    t = pd.read_csv(path)
    if 'time' not in t.columns:
        raise ValueError(f'{path} 沒有 time 欄位')
    return pd.DatetimeIndex(pd.to_datetime(t['time'])).dropna().sort_values()


def bars_of(df):
    """tick -> 1 分 K（Open, High, Low, Close, Volume），以及最高價第一次出現的 tick。"""
    t = df[['ts', 'close', 'volume']].copy()
    t['ts'] = pd.to_datetime(t['ts'])
    t = t.sort_values('ts', kind='stable')
    g = t.groupby(t['ts'].dt.floor(FREQ))
    bars = pd.DataFrame({'Open': g['close'].first(), 'High': g['close'].max(),
                         'Low': g['close'].min(), 'Close': g['close'].last(),
                         'Volume': g['volume'].sum()})
    i = t['close'].to_numpy().argmax()
    return bars, t['ts'].iloc[i], t['close'].iloc[i]


def plot_frame(bars, marks, max_time, max_price, title, path):
    """K 線 + ★ 標記（上方寫時間）+ ◆ 最高點。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import mplfinance as mpf

    style = mpf.make_mpf_style(base_mpf_style='yahoo', rc={
        'font.family': ['Microsoft JhengHei', 'Microsoft YaHei', 'DejaVu Sans'],
        'axes.unicode_minus': False})
    fig, axes = mpf.plot(bars, type='candle', volume=True, style=style, returnfig=True,
                         figsize=(14, 6), title=title)
    ax = axes[0]
    span = (bars['High'].max() - bars['Low'].min()) * 0.07
    for m in marks.floor(FREQ).unique():
        if m not in bars.index:
            continue
        x = bars.index.get_loc(m)
        y = bars['High'].iloc[x] + span
        ax.scatter([x], [y], marker='*', color='red', s=200, zorder=5)
        ax.annotate(f'{m:%H:%M}', (x, y), xytext=(0, 9), textcoords='offset points',
                    ha='center', fontsize=8, color='red')
    x = bars.index.get_loc(max_time.floor(FREQ))
    ax.axhline(max_price, color='black', linestyle=':', linewidth=0.8)
    ax.scatter([x], [max_price], marker='D', color='black', s=45, zorder=6)
    ax.text(0.005, max_price, f' Max {max_price:g} @ {max_time:%H:%M:%S}',
            transform=ax.get_yaxis_transform(), va='bottom', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', ec='black', alpha=0.9))
    ax.set_ylim(top=bars['High'].max() + span * 2.5)       # 留空間給 ★ 和時間
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def evaluate(marks_dir, df_list, key_list, out=None):
    """回傳每個 frame 一列的 DataFrame；out 有給就同時畫圖到 out/plots/。"""
    rows = []
    for i, df in enumerate(df_list):
        bars, max_time, max_price = bars_of(df)
        marks = read_marks(marks_dir, i)
        minutes = marks.floor(FREQ)
        hit = max_time.floor(FREQ) in set(minutes)
        rows.append({
            'df_idx': i, 'frame_start': key_list[i][0], 'frame_end': key_list[i][1],
            'max_price': max_price, 'max_time': max_time, 'hit': hit,
            'n_marks': len(marks),
            'marks': list(marks),
            'marks_no_tick': [m for m in marks if m.floor(FREQ) not in bars.index],
        })
        if out is not None:
            title = f'df {i}   {key_list[i][0]} ~ {key_list[i][1]}   [{"HIT" if hit else "MISS"}]'
            plot_frame(bars, marks, max_time, max_price, title, out / 'plots' / f'df_{i:03d}.png')
    return pd.DataFrame(rows)


CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--border:rgba(11,11,11,.10);--link:#2a78d6;--good:#006300;--good-bg:#e3f4e3;
--bad:#b42f2f;--bad-bg:#fbe9e9}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;
--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--border:rgba(255,255,255,.10);
--link:#3987e5;--good:#0ca30c;--good-bg:#12301a;--bad:#e66767;--bad-bg:#3a1a1a}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--grid:#2c2c2a;--border:rgba(255,255,255,.10);--link:#3987e5;--good:#0ca30c;--good-bg:#12301a;
--bad:#e66767;--bad-bg:#3a1a1a}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
font:14px/1.5 system-ui,-apple-system,"Segoe UI","Microsoft JhengHei",sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 8px}
.sub{color:var(--ink2);margin:0}
.big{font-size:30px;font-weight:600;margin:16px 0 0}
.toolbar{display:flex;gap:8px;margin:8px 0}
.toolbar button{font:inherit;padding:4px 14px;border-radius:999px;border:1px solid var(--border);
background:var(--surface);color:var(--ink2);cursor:pointer}
.toolbar button.on{background:var(--ink);color:var(--surface)}
.scroll{max-height:520px;overflow:auto;border:1px solid var(--border);border-radius:10px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;border-bottom:1px solid var(--grid);text-align:left;vertical-align:top}
th{color:var(--ink2);font-weight:600;position:sticky;top:0;background:var(--surface)}
td.marks{color:var(--ink2);white-space:normal}
td a,.frame a{color:var(--link)}
.pill{display:inline-block;padding:0 8px;border-radius:999px;font-size:12px;font-weight:600}
.pill.hit{color:var(--good);background:var(--good-bg)}.pill.miss{color:var(--bad);background:var(--bad-bg)}
.frame{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px;margin:12px 0}
.frame h3{font-size:14px;margin:0 0 6px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.frame img{width:100%;border-radius:6px;background:#fff}
.frame p{margin:4px 0;color:var(--ink2);font-size:13px}
"""

JS = """
document.querySelectorAll('.toolbar button').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('.toolbar button').forEach(x=>x.classList.toggle('on',x.dataset.f===b.dataset.f));
  document.querySelectorAll('[data-hit]').forEach(el=>{
    el.style.display=(b.dataset.f==='all'||el.dataset.hit===b.dataset.f)?'':'none';});
}));
"""


def _times(marks, no_tick):
    if not marks:
        return '—'
    return '、'.join(f'{m:%H:%M:%S}' + ('（無 tick）' if m in no_tick else '') for m in marks)


def build_html(name, marks_dir, res, plots):
    n, hits = len(res), int(res['hit'].sum())
    rows, frames = [], []
    for _, r in res.iterrows():
        i, tag = int(r['df_idx']), 'hit' if r['hit'] else 'miss'
        pill = f'<span class="pill {tag}">{"Hit" if r["hit"] else "Miss"}</span>'
        times = html.escape(_times(r['marks'], r['marks_no_tick']))
        link = f'<a href="#df{i}">{i:03d}</a>' if plots else f'{i:03d}'
        rows.append(f'<tr data-hit="{tag}"><td>{link}</td>'
                    f'<td>{r["frame_start"]:%Y-%m-%d %H:%M} ~ {r["frame_end"]:%H:%M}</td>'
                    f'<td>{pill}</td><td>{r["max_time"]:%H:%M:%S}</td><td>{r["max_price"]:,.0f}</td>'
                    f'<td class="marks">{times}</td></tr>')
        if plots:
            frames.append(f'<div class="frame" id="df{i}" data-hit="{tag}"><h3>df {i:03d} {pill}'
                          f'<span class="sub">{r["frame_start"]:%Y-%m-%d %H:%M} ~ {r["frame_end"]:%H:%M}</span></h3>'
                          f'<p>最高點 {r["max_time"]:%H:%M:%S}（{r["max_price"]:,.0f}）　標記：{times}</p>'
                          f'<a href="plots/df_{i:03d}.png" target="_blank">'
                          f'<img loading="lazy" src="plots/df_{i:03d}.png" alt="df {i}"></a></div>')
    toolbar = ('<div class="toolbar"><button class="on" data-f="all">全部</button>'
               '<button data-f="hit">Hit</button><button data-f="miss">Miss</button></div>')
    body = (f'<main><h1>{html.escape(name)}</h1>'
            f'<p class="sub">標記來源：{html.escape(str(marks_dir))}　·　{n} frames　·　'
            f'{pd.Timestamp.now():%Y-%m-%d %H:%M}</p>'
            f'<p class="big">Hit {hits} / {n}（{hits / n * 100:.1f}%）</p>'
            f'<p class="sub">Hit = frame 最高點那一分鐘有被標記</p>{toolbar}'
            '<h2>每個 frame</h2><div class="scroll"><table><thead><tr><th>df</th><th>frame</th>'
            '<th>結果</th><th>最高點時間</th><th>最高價</th><th>標記時間</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></div>'
            + (('<h2>圖（★ 標記、◆ 最高點）</h2>' + ''.join(frames)) if plots else '') + '</main>')
    return (f'<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(name)}</title><style>{CSS}</style></head>'
            f'<body>{body}<script>{JS}</script></body></html>')


def run_report(marks_dir, name=None, df_list=None, key_list=None, plots=True, open_browser=True):
    """讀 marks_dir 的 CSV，輸出 reports/<name>/index.html，回傳 (報告路徑, 每個 frame 的結果)。"""
    t0 = time.time()
    marks_dir = Path(marks_dir)
    if not marks_dir.is_absolute():
        marks_dir = ROOT / marks_dir
    if df_list is None:
        df_list, key_list = load_frames()
    name = name or marks_dir.name
    out = REPORT_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    res = evaluate(marks_dir, df_list, key_list, out if plots else None)
    csv = res.assign(marks=res['marks'].map(lambda m: ' '.join(f'{x:%Y-%m-%d %H:%M:%S}' for x in m)))
    csv.drop(columns='marks_no_tick').to_csv(out / 'result.csv', index=False, encoding='utf-8-sig')
    report = out / 'index.html'
    report.write_text(build_html(name, marks_dir, res, plots), encoding='utf-8')

    print(f'Hit {int(res["hit"].sum())} / {len(res)}　報告：{report}（{time.time() - t0:.0f} 秒）')
    if open_browser:
        webbrowser.open(report.as_uri())
    return report, res


def main():
    ap = argparse.ArgumentParser(description='讀標記 CSV 資料夾，輸出 Hit / Miss 報告')
    ap.add_argument('marks_dir', help='標記 CSV 的資料夾，例如 marks/type1')
    ap.add_argument('--name', help='報告名稱（預設 = 資料夾名稱）')
    ap.add_argument('--no-plots', action='store_true', help='不畫圖，只出表格')
    ap.add_argument('--no-open', action='store_true', help='跑完不自動打開瀏覽器')
    a = ap.parse_args()
    run_report(a.marks_dir, a.name, plots=not a.no_plots, open_browser=not a.no_open)


if __name__ == '__main__':
    main()
