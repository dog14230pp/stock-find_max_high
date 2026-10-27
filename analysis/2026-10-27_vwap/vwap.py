"""前 10 根 VWAP 與乖離（242 frames）。

VWAP        t-9 ~ t 這 10 根 1 分 K 所有成交的 Σ(價 × 量) ÷ Σ量（含當根；前面不足 10 根 = NaN）
外盤 VWAP   同上，只算外盤（tick_type = 1）的成交
乖離        (收盤價 − VWAP) ÷ VWAP × 100；dev_high 用最高價，bdev_close 用外盤 VWAP
百分位      每個 frame 內把乖離排名（50 = 沒資訊）；看最高點那分鐘排第幾
S12         S12 的點分成 Hit / 沒中，比較乖離（只分組，不加條件）

輸出（與本檔同資料夾）：minutes.csv、pctile.csv、s12.csv、dev.png
執行（在專案根目錄）：python analysis/2026-10-27_vwap/vwap.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
sys.path.insert(0, str(ROOT / 'strategies'))

import start_backtest as sb  # noqa: E402
from backtest_report import load_frames  # noqa: E402
from frame_filter import keep_frames, max_bar_table  # noqa: E402
from registry import STRATEGIES  # noqa: E402

N = 10
BASE = 'S12'
DEVS = ['dev_close', 'dev_high', 'bdev_close']
BLUE, ORANGE = '#2a78d6', '#eb6834'
plt.rcParams['font.family'] = 'Microsoft JhengHei'
plt.rcParams['axes.unicode_minus'] = False


def minute_vwap(df, n=N):
    """tick → 每分鐘一列：1 分 K + 前 n 根 VWAP、外盤 VWAP、乖離。"""
    mb, max_time, _ = sb.minute_bars(df)
    t = df[['ts', 'close', 'volume', 'tick_type']].copy()
    t['ts'] = pd.to_datetime(t['ts'])
    t['pv'] = t['close'] * t['volume']
    t['bv'] = t['volume'].where(t['tick_type'] == 1, 0)
    t['bpv'] = t['close'] * t['bv']
    g = t.groupby(t['ts'].dt.floor('1min'))[['pv', 'volume', 'bpv', 'bv']].sum().reindex(mb.index)
    r = g.rolling(n, min_periods=n).sum()
    m = mb[['open', 'high', 'low', 'close', 'volume']].copy()
    m['vwap'] = r['pv'] / r['volume']
    m['bvwap'] = r['bpv'] / r['bv'].where(r['bv'] > 0)
    m['dev_close'] = (m['close'] - m['vwap']) / m['vwap'] * 100
    m['dev_high'] = (m['high'] - m['vwap']) / m['vwap'] * 100
    m['bdev_close'] = (m['close'] - m['bvwap']) / m['bvwap'] * 100
    for k in DEVS:
        m[f'{k}_pct'] = m[k].rank(pct=True) * 100
    m['is_max'] = m.index == max_time.floor('1min')
    m['marked'] = STRATEGIES[BASE]['mask'](mb).fillna(False).to_numpy(dtype=bool)
    return m


def plot_dev(d, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(-0.6, 1.2, 37)
    panels = [(axes[0], d, '全部分鐘', '最高點那分鐘', '一般分鐘'),
              (axes[1], d[d['marked']], f'{BASE} 的點', 'Hit', '沒中')]
    for ax, sub, title, a_lab, b_lab in panels:
        sub = sub[sub['dev_close'].notna()]
        a, b = sub[sub['is_max']], sub[~sub['is_max']]
        for x, c, lab in ((b, BLUE, b_lab), (a, ORANGE, a_lab)):
            ax.hist(x['dev_close'].clip(bins[0], bins[-1]), bins=bins, density=True, histtype='step',
                    lw=2, color=c, label=f'{lab}（{len(x)}，中位 {x["dev_close"].median():.2f}%）')
        ax.set_title(title, loc='left')
        ax.set_xlabel('乖離 =（收盤 - 前 10 根 VWAP）÷ VWAP（%）')
        ax.set_yticks([])
        ax.grid(axis='x', color='#e5e5e5', lw=0.8)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f'{d["df_idx"].nunique()} frames', x=0.01, ha='left', fontsize=10, color='#666')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    df_list, key_list = load_frames()
    keep = keep_frames(table=max_bar_table(df_list, key_list))
    parts = []
    for i in keep:
        m = minute_vwap(df_list[i])
        m.insert(0, 'df_idx', i)
        parts.append(m.reset_index())
    d = pd.concat(parts, ignore_index=True)

    top, other = d[d['is_max']], d[~d['is_max']]
    pc = []
    for k in DEVS:
        t = top[top[k].notna()]
        pc.append({'dev': k, 'max_n': len(t), 'max_median': t[k].median(), 'other_median': other[k].median(),
                   'max_pctile_mean': t[f'{k}_pct'].mean(),
                   'max_in_low20': (t[f'{k}_pct'] <= 20).mean() * 100,
                   'max_in_high20': (t[f'{k}_pct'] > 80).mean() * 100})
    pc = pd.DataFrame(pc)

    mk = d[d['marked']]
    s12 = []
    for k in DEVS:
        for name, sub in (('Hit', mk[mk['is_max']]), ('沒中', mk[~mk['is_max']])):
            s12.append({'dev': k, 'group': name, 'n': int(sub[k].notna().sum()), 'mean': sub[k].mean(),
                        'median': sub[k].median(), 'pctile_mean': sub[f'{k}_pct'].mean()})
    s12 = pd.DataFrame(s12)

    d.to_csv(HERE / 'minutes.csv', index=False, encoding='utf-8-sig')
    pc.to_csv(HERE / 'pctile.csv', index=False, encoding='utf-8-sig')
    s12.to_csv(HERE / 's12.csv', index=False, encoding='utf-8-sig')
    plot_dev(d, HERE / 'dev.png')
    with pd.option_context('display.width', 200):
        print(f'{len(keep)} frames、{len(d)} 分鐘（有 VWAP 的 {d["vwap"].notna().sum()}）、'
              f'最高點那分鐘有 VWAP 的 {top["vwap"].notna().sum()} / {len(top)}')
        print('最高點那分鐘的百分位')
        print(pc.round(3).to_string(index=False))
        print(f'{BASE} 的點（{len(mk)} 個、Hit {int(mk["is_max"].sum())}）')
        print(s12.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
