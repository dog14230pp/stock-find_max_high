"""type1_ratio 與漲幅的背離：ratio 說該漲、價格卻沒漲。

預期漲幅：每個 frame 各自，只用「這分鐘之前」的分鐘做線性回歸 d_close ≈ a + b × type1_ratio
    （過去有效分鐘 < MIN_PAST 不算）。不偷看未來，之後可以直接當策略用。
背離 z = (實際 d_close − 預期) / 過去殘差的標準差；z 很負 = ratio 高但漲得比預期少。
直觀版：absorb = type1_ratio ≥ HIGH_RATIO 且 d_close ≤ 0（買方主動吃、價格卻沒上去）。

d_close = 這分鐘 close − 上一分鐘 close（間隔 > MAX_GAP 分鐘不算）；next_d = 下一分鐘的 d_close。
bar_pos = 這分鐘是 frame 資料的第幾根 1 分 K（0 起算）；to_max = 最高點那根 − 這根。
max_in_N = 最高點在「接下來 1~N 根」出現（不含這一根：這分鐘結束才看得到訊號）。

輸入：analysis/2026-10-21_type1_ratio/min/、frame_max.csv（make.py 產生）
輸出（與本檔同資料夾）：
    divergence.csv    每列 = 一個 frame 的一分鐘
    divergence.html   報告，用瀏覽器打開

執行（在專案根目錄）：python analysis/2026-10-23_ratio_divergence/divergence.py
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
MIN_PAST = 10        # 回歸至少要幾個過去分鐘
HIGH_RATIO = 0.6
SKIP_BARS = 10       # 事件研究：最高點在前幾根 K 的 frame 不看（同 max_sec_ratio.py）
EVENT = range(-10, 6)
Z_BINS = [-np.inf, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, np.inf]
HORIZONS = [3, 5]


def causal_z(x, y):
    """逐分鐘：只用前面（有效的）分鐘回歸 y ≈ a + b x，回傳預期值、背離 z。"""
    n = len(x)
    pred, z = np.full(n, np.nan), np.full(n, np.nan)
    sx = sy = sxx = sxy = syy = 0.0
    k = 0
    for t in range(n):
        if np.isnan(y[t]):
            continue
        if k >= MIN_PAST:
            vx, cxy = sxx - sx * sx / k, sxy - sx * sy / k
            if vx > 0:
                b = cxy / vx
                a = (sy - b * sx) / k
                sse = (syy - sy * sy / k) - b * cxy
                s = np.sqrt(max(sse, 0) / (k - 2))
                pred[t] = a + b * x[t]
                if s > 0:
                    z[t] = (y[t] - pred[t]) / s
        sx += x[t]; sy += y[t]; sxx += x[t] ** 2; sxy += x[t] * y[t]; syy += y[t] ** 2
        k += 1
    return pred, z


def frame_minutes(i, fmax):
    m = pd.read_csv(SRC / 'min' / f'df_{i:03d}.csv').sort_values('minute').reset_index(drop=True)
    gap = pd.to_datetime(m['minute']).diff().dt.total_seconds() / 60
    out = m[['minute', 'close', 'volume', 'type1_ratio']].copy()
    out['d_close'] = out['close'].diff().where(gap <= MAX_GAP)
    out['next_d'] = out['d_close'].shift(-1)
    out['pred'], out['z'] = causal_z(out['type1_ratio'].to_numpy(), out['d_close'].to_numpy())
    out['absorb'] = (out['type1_ratio'] >= HIGH_RATIO) & (out['d_close'] <= 0)
    out['bar_pos'] = np.arange(len(out))
    max_pos = int((m['minute'] < fmax.loc[i, 'max_minute']).sum())
    out['to_max'] = max_pos - out['bar_pos']
    for h in HORIZONS:
        out[f'max_in_{h}'] = out['to_max'].between(1, h)
    out.insert(0, 'df_idx', i)
    out['max_bar_pos'] = max_pos
    return out


def event_table(res):
    """最高點前後：以最高點那根為 0，各位置的平均。只看最高點 bar_pos ≥ SKIP_BARS 的 frame。"""
    e = res[(res['max_bar_pos'] >= SKIP_BARS) & res['to_max'].between(-max(EVENT), -min(EVENT))].copy()
    e['k'] = -e['to_max']
    g = e.groupby('k')
    t = pd.DataFrame({'n': g.size(), 'ratio': g['type1_ratio'].mean(), 'd_close': g['d_close'].mean(),
                      'pred': g['pred'].mean(), 'z': g['z'].mean(), 'z_lt_-1': g['z'].apply(lambda s: (s.dropna() < -1).mean()),
                      'absorb': g['absorb'].mean()})
    base = {'ratio': res['type1_ratio'].mean(), 'd_close': res['d_close'].mean(), 'pred': res['pred'].mean(),
            'z': res['z'].mean(), 'z_lt_-1': (res['z'].dropna() < -1).mean(), 'absorb': res['absorb'].mean()}
    return t.reindex(list(EVENT)), base


def outcome_table(res, by):
    """依 by 分組：下一分鐘漲跌、最高點是否在接下來 1~N 根出現。"""
    g = res.groupby(by, observed=True)
    t = pd.DataFrame({'n': g.size(), 'ratio': g['type1_ratio'].mean(), 'd_close': g['d_close'].mean(),
                      'next_d': g['next_d'].mean(),
                      'next_up': g['next_d'].apply(lambda s: (s.dropna() > 0).mean())})
    for h in HORIZONS:
        t[f'max_in_{h}'] = g[f'max_in_{h}'].mean()
    return t


def build_html(res, ev, base, zt, at, zt_hi):
    base_rate = {h: res[f'max_in_{h}'].mean() for h in HORIZONS}
    up_rate = (res['next_d'].dropna() > 0).mean()

    def pct(v):
        return '' if pd.isna(v) else f'{v * 100:.1f}%'

    def num(v, spec):
        return '' if pd.isna(v) else format(v, spec)

    ev_rows = ''.join(
        f'<tr{" class=maxrow" if k == 0 else ""}><td>{"最高點" if k == 0 else f"{k:+d}"}</td><td>{int(r.n)}</td>'
        f'<td>{num(r.ratio, ".3f")}</td><td>{num(r.d_close, "+.1f")}</td><td>{num(r.pred, "+.1f")}</td>'
        f'<td>{num(r.z, "+.2f")}</td><td>{pct(r.z_lt_m1)}</td><td>{pct(r.absorb)}</td></tr>'
        for k, r in zip(ev.index, ev.rename(columns={'z_lt_-1': 'z_lt_m1'}).itertuples(index=False)))
    ev_rows += (f'<tr><td>全部分鐘（基準）</td><td>{len(res):,}</td><td>{base["ratio"]:.3f}</td>'
                f'<td>{base["d_close"]:+.1f}</td><td>{base["pred"]:+.1f}</td><td>{base["z"]:+.2f}</td>'
                f'<td>{pct(base["z_lt_-1"])}</td><td>{pct(base["absorb"])}</td></tr>')

    def otable(t, label_fn):
        rows = ''.join(
            f'<tr><td>{label_fn(k)}</td><td>{int(r["n"]):,}</td><td>{num(r["ratio"], ".3f")}</td>'
            f'<td>{num(r["d_close"], "+.1f")}</td><td>{num(r["next_d"], "+.2f")}</td><td>{pct(r["next_up"])}</td>'
            + ''.join(f'<td>{pct(r[f"max_in_{h}"])}</td><td>{r[f"max_in_{h}"] / base_rate[h]:.2f}×</td>' for h in HORIZONS)
            + '</tr>' for k, r in t.iterrows())
        head = ''.join(f'<th>最高點在接下來 1~{h} 根</th><th>倍數</th>' for h in HORIZONS)
        return ('<div class="card"><table><thead><tr><th>組別</th><th>分鐘數</th><th>平均 ratio</th><th>這分鐘漲跌</th>'
                f'<th>下一分鐘平均漲跌</th><th>下一分鐘上漲比例</th>{head}</tr></thead><tbody>{rows}</tbody></table></div>')

    zlabel = lambda iv: f'{iv.left:g} ~ {iv.right:g}'.replace('-inf ~ ', '< ').replace(' ~ inf', ' 以上')
    alabel = {True: f'ratio ≥ {HIGH_RATIO} 且沒漲（背離）', False: '其他'}
    body = (
        '<main><h1>type1_ratio 與漲幅的背離</h1>'
        f'<p class="sub">{res["df_idx"].nunique()} frames · {len(res):,} 分鐘 · 有背離 z 的 {res["z"].notna().sum():,} 分鐘</p>'
        '<p class="sub">預期漲幅：每個 frame 只用這分鐘之前的分鐘回歸 Δclose ≈ a + b × ratio（過去至少 '
        f'{MIN_PAST} 分鐘）。背離 z = (實際 Δclose − 預期) / 過去殘差標準差，z 很負 = ratio 說該漲、價格卻沒漲。'
        f'「背離」直觀版 = ratio ≥ {HIGH_RATIO} 且這分鐘 Δclose ≤ 0。</p>'
        '<h2>1. 最高點前後的分鐘</h2>'
        f'<p class="sub">以最高點那根 K 為 0，−1 = 前一根。只看最高點不在前 {SKIP_BARS} 根的 frame。最下面是全部分鐘的基準。</p>'
        '<div class="card"><table><thead><tr><th>位置</th><th>分鐘數</th><th>平均 ratio</th><th>平均 Δclose</th>'
        '<th>預期 Δclose</th><th>平均 z</th><th>z &lt; −1 比例</th><th>背離比例</th></tr></thead>'
        f'<tbody>{ev_rows}</tbody></table></div>'
        '<h2>2. 依背離 z 分組：之後發生什麼</h2>'
        f'<p class="sub">基準：下一分鐘上漲比例 {up_rate * 100:.1f}%；最高點在接下來 1~3 根 {base_rate[3] * 100:.1f}%、'
        f'1~5 根 {base_rate[5] * 100:.1f}%。倍數 = 該組比例 / 基準。</p>'
        + otable(zt, zlabel)
        + f'<h2>3. 只看 ratio ≥ {HIGH_RATIO} 的分鐘，依背離 z 分組</h2>' + otable(zt_hi, zlabel)
        + '<h2>4. 直觀版背離</h2>' + otable(at, lambda k: alabel[k])
        + '</main>')
    return (f'<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape("ratio 背離")}</title><style>{CSS}</style></head><body>{body}</body></html>')


def main(argv):
    fmax = pd.read_csv(SRC / 'frame_max.csv').set_index('df_idx')
    res = pd.concat([frame_minutes(i, fmax) for i in fmax.index], ignore_index=True)
    res.to_csv(HERE / 'divergence.csv', index=False)
    ev, base = event_table(res)
    zb = pd.cut(res['z'], Z_BINS)
    zt = outcome_table(res, zb)
    hi = res['type1_ratio'] >= HIGH_RATIO
    zt_hi = outcome_table(res[hi], zb[hi])
    at = outcome_table(res[res['d_close'].notna()], 'absorb').sort_index(ascending=False)
    report = HERE / 'divergence.html'
    report.write_text(build_html(res, ev, base, zt, at, zt_hi), encoding='utf-8')
    pd.set_option('display.width', 220)
    print(f'{len(res)} 分鐘 → {HERE / "divergence.csv"}；報告 {report}')
    print(ev.round(3).to_string()); print('base', {k: round(v, 3) for k, v in base.items()})
    print(zt.round(3).to_string()); print(zt_hi.round(3).to_string()); print(at.round(3).to_string())
    print('base max_in', {h: round(res[f'max_in_{h}'].mean(), 4) for h in HORIZONS},
          'next_up', round((res['next_d'].dropna() > 0).mean(), 4))
    if '--no-open' not in argv:
        webbrowser.open(report.as_uri())


if __name__ == '__main__':
    main(sys.argv)
