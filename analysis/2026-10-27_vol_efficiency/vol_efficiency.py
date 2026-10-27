"""量能效率：每 1 口量把價格推了多少（242 frames）。

eff         實體 |close − open| ÷ 這根的量（十字線 = 0）
eff_range   全長 (high − low) ÷ 這根的量
eff_ratio   eff ÷ 前 5 根 eff 的平均（前面不足 5 根 = NaN）
百分位      每個 frame 內排名（50 = 沒資訊）；這次看最高點那分鐘是不是偏低
S12         S12 的點分成 Hit / 沒中，比較效率（只分組，不加條件）

輸出（與本檔同資料夾）：minutes.csv、pctile.csv、s12.csv、eff.png
執行（在專案根目錄）：python analysis/2026-10-27_vol_efficiency/vol_efficiency.py
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

PREV = 5
BASE = 'S12'
EFFS = ['eff', 'eff_range', 'eff_ratio']
BLUE, ORANGE = '#2a78d6', '#eb6834'
plt.rcParams['font.family'] = 'Microsoft JhengHei'
plt.rcParams['axes.unicode_minus'] = False


def minute_eff(df, prev=PREV):
    """tick → 每分鐘一列：1 分 K + 效率、全長效率、效率比。"""
    mb, max_time, _ = sb.minute_bars(df)
    m = mb[['open', 'high', 'low', 'close', 'volume']].copy()
    vol = m['volume'].astype(float)
    m['eff'] = (m['close'] - m['open']).abs() / vol
    m['eff_range'] = (m['high'] - m['low']) / vol
    base = m['eff'].rolling(prev).mean().shift(1)
    m['eff_ratio'] = m['eff'] / base.where(base > 0)
    for k in EFFS:
        m[f'{k}_pct'] = m[k].rank(pct=True) * 100
    m['is_max'] = m.index == max_time.floor('1min')
    m['marked'] = STRATEGIES[BASE]['mask'](mb).fillna(False).to_numpy(dtype=bool)
    return m


def plot_eff(d, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, 3, 31)
    panels = [(axes[0], d, '全部分鐘', '最高點那分鐘', '一般分鐘'),
              (axes[1], d[d['marked']], f'{BASE} 的點', 'Hit', '沒中')]
    for ax, sub, title, a_lab, b_lab in panels:
        sub = sub[sub['eff_ratio'].notna()]
        a, b = sub[sub['is_max']], sub[~sub['is_max']]
        for x, c, lab in ((b, BLUE, b_lab), (a, ORANGE, a_lab)):
            ax.hist(x['eff_ratio'].clip(bins[0], bins[-1]), bins=bins, density=True, histtype='step',
                    lw=2, color=c, label=f'{lab}（{len(x)}，中位 {x["eff_ratio"].median():.2f}）')
        ax.set_title(title, loc='left')
        ax.set_xlabel('效率比 = 效率 ÷ 前 5 根效率平均（> 3 併入 3）')
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
        m = minute_eff(df_list[i])
        m.insert(0, 'df_idx', i)
        parts.append(m.reset_index())
    d = pd.concat(parts, ignore_index=True)

    top, other = d[d['is_max']], d[~d['is_max']]
    pc = []
    for k in EFFS:
        t = top[top[k].notna()]
        pc.append({'eff': k, 'max_n': len(t), 'max_median': t[k].median(), 'other_median': other[k].median(),
                   'max_pctile_mean': t[f'{k}_pct'].mean(),
                   'max_in_low20': (t[f'{k}_pct'] <= 20).mean() * 100,
                   'max_in_high20': (t[f'{k}_pct'] > 80).mean() * 100})
    pc = pd.DataFrame(pc)

    mk = d[d['marked']]
    s12 = []
    for k in EFFS:
        for name, sub in (('Hit', mk[mk['is_max']]), ('沒中', mk[~mk['is_max']])):
            s12.append({'eff': k, 'group': name, 'n': int(sub[k].notna().sum()), 'mean': sub[k].mean(),
                        'median': sub[k].median(), 'pctile_mean': sub[f'{k}_pct'].mean()})
    s12 = pd.DataFrame(s12)

    d.to_csv(HERE / 'minutes.csv', index=False, encoding='utf-8-sig')
    pc.to_csv(HERE / 'pctile.csv', index=False, encoding='utf-8-sig')
    s12.to_csv(HERE / 's12.csv', index=False, encoding='utf-8-sig')
    plot_eff(d, HERE / 'eff.png')
    with pd.option_context('display.width', 200):
        print(f'{len(keep)} frames、{len(d)} 分鐘')
        print('最高點那分鐘的百分位')
        print(pc.round(3).to_string(index=False))
        print(f'{BASE} 的點（{len(mk)} 個、Hit {int(mk["is_max"].sum())}）')
        print(s12.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
