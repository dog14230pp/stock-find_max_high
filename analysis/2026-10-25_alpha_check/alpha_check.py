"""快速檢查三個文獻因子有沒有機會用來找高點（242 frames）。

因子（每根 1 分 K t，只用 t 與之前的資料）：
    a12        WorldQuant Alpha#12：sign(V[t] − V[t-1]) × (−(C[t] − C[t-1]))；放量上漲 → 負值
    a6         WorldQuant Alpha#6：−corr(Open, Volume, 10)（t-9 ~ t）；價量正相關（追價放量）→ 負值
    illiq10    Amihud：mean(|C[t]/C[t-1] − 1| / V, 10)（× 1e6）；值大 = 少量就能推動價格
    illiq_rel  |報酬[t]| / V[t] ÷ 前 10 根的 |報酬| / V 平均；Amihud 的相對版，不受各年份量水準影響

檢查
    1. 最高點那根在該 frame 所有 K 棒中的百分位（0 = 最小、100 = 最大），242 個 frame 平均。50 = 沒資訊
    2. 策略 CHECK_STRATEGY 找到的點：Hit vs 其他 的中位數，以及 Hit 在這些點中的平均百分位

輸出（與本檔同資料夾）：bars.csv（每根一列）、summary.csv
執行（在專案根目錄）：python analysis/2026-10-25_alpha_check/alpha_check.py
"""
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
from start_backtest import minute_bars  # noqa: E402

CHECK_STRATEGY = 'S6'
FEATS = ['a12', 'a6', 'illiq10', 'illiq_rel']


def factors(mb):
    c, v, o = mb['close'], mb['volume'].astype(float), mb['open']
    ret = c.pct_change().abs()
    imp = ret / v * 1e6
    return pd.DataFrame({
        'a12': np.sign(v.diff()) * -c.diff(),
        'a6': -o.rolling(10).corr(v),
        'illiq10': imp.rolling(10).mean(),
        'illiq_rel': imp / imp.rolling(10).mean().shift(1),
    }, index=mb.index)


def main():
    df_list, key_list = load_frames()
    keep = keep_frames(table=max_bar_table(df_list, key_list))
    mask_fn = STRATEGIES[CHECK_STRATEGY]['mask']
    rows = []
    for i in keep:
        mb, max_time, _ = minute_bars(df_list[i])
        f = factors(mb)
        f['df_idx'], f['pos'] = i, np.arange(len(mb))
        f['is_max'] = f['pos'] == mb.index.get_loc(max_time.floor('1min'))
        f['marked'] = mask_fn(mb).fillna(False).to_numpy(dtype=bool)
        for k in FEATS:
            f[f'{k}_pct'] = f[k].rank(pct=True) * 100           # frame 內百分位
        rows.append(f.reset_index(names='time'))
    d = pd.concat(rows, ignore_index=True)
    d.to_csv(HERE / 'bars.csv', index=False, encoding='utf-8-sig')

    top = d[d['is_max']]
    marks = d[d['marked']].copy()
    for k in FEATS:
        marks[f'{k}_mpct'] = marks[k].rank(pct=True) * 100              # 在所有標記中的百分位
    out = []
    for k in FEATS:
        out.append({
            'feature': k,
            'max_bar_pct_mean': top[f'{k}_pct'].mean(),
            'max_bar_pct_median': top[f'{k}_pct'].median(),
            'all_bars_median': d[k].median(), 'max_bar_median': top[k].median(),
            f'{CHECK_STRATEGY}_hit_median': marks.loc[marks['is_max'], k].median(),
            f'{CHECK_STRATEGY}_other_median': marks.loc[~marks['is_max'], k].median(),
            f'{CHECK_STRATEGY}_hit_pct_among_marks': marks.loc[marks['is_max'], f'{k}_mpct'].mean(),
        })
    s = pd.DataFrame(out)
    s.to_csv(HERE / 'summary.csv', index=False, encoding='utf-8-sig')
    with pd.option_context('display.width', 220, 'display.max_columns', 20):
        print(f'242 frames、{len(d)} 根 K、{CHECK_STRATEGY} 找到 {len(marks)} 個點（Hit {int(marks["is_max"].sum())}）')
        print(s.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
