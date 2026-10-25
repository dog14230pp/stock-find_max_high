"""最高點那根 1 分 K 的形狀 vs type1_ratio > 0.5 的比例（242 frames）。

每個 frame 只看最高點那根，分 3 組：
    綠棒或十字   close <= open
    一般紅棒     close > open，且 上影線 + 下影線 >= 實體 / 2
    大紅棒       close > open，且 上影線 + 下影線 <  實體 / 2
上影線 = high − max(open, close)，下影線 = min(open, close) − low，實體 = |close − open|

輸出（與本檔同資料夾）：max_bars.csv（每個 frame 一列）、summary.csv
執行（在專案根目錄）：python analysis/2026-10-25_max_bar_shape/max_bar_shape.py
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'lib'))

from backtest_report import load_frames  # noqa: E402
from frame_filter import keep_frames, max_bar_table  # noqa: E402
from start_backtest import minute_bars  # noqa: E402

GROUPS = ['綠棒或十字', '一般紅棒', '大紅棒']


def shape(r):
    if r['close'] <= r['open']:
        return '綠棒或十字'
    return '大紅棒' if r['upper'] + r['lower'] < r['body'] / 2 else '一般紅棒'


def main():
    df_list, key_list = load_frames()
    keep = keep_frames(table=max_bar_table(df_list, key_list))
    rows = []
    for i in keep:
        mb, max_time, _ = minute_bars(df_list[i])
        r = mb.loc[max_time.floor('1min')]
        rows.append({'df_idx': i, 'minute': max_time.floor('1min'),
                     'open': r['open'], 'high': r['high'], 'low': r['low'], 'close': r['close'],
                     'upper': r['high'] - max(r['open'], r['close']), 'lower': min(r['open'], r['close']) - r['low'],
                     'body': abs(r['close'] - r['open']), 'ratio': r['ratio']})
    d = pd.DataFrame(rows)
    d['group'] = d.apply(shape, axis=1)
    d['ratio_gt_05'] = d['ratio'] > 0.5
    d.to_csv(HERE / 'max_bars.csv', index=False, encoding='utf-8-sig')
    s = d.groupby('group').agg(n=('df_idx', 'size'), ratio_gt_05=('ratio_gt_05', 'sum'),
                               ratio_median=('ratio', 'median')).reindex(GROUPS)
    s['pct'] = s['ratio_gt_05'] / s['n'] * 100
    s.loc['全部'] = [len(d), d['ratio_gt_05'].sum(), d['ratio'].median(), d['ratio_gt_05'].mean() * 100]
    s.to_csv(HERE / 'summary.csv', encoding='utf-8-sig')
    print('242 frames，最高點那根')
    print(s.round(2).to_string())
    for g in GROUPS:
        print(f'\n{g}：', ' '.join(f'{i:03d}' for i in d.loc[d['group'] == g, 'df_idx']))


if __name__ == '__main__':
    main()
