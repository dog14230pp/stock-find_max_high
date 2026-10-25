"""依「frame 最高點落在第幾根 1 分 K」過濾 frame。

最高點定義與 backtest_report.bars_of 相同：整段 tick（含 frame_start 之前的資料）中，最高價第一次出現的那筆。
bar_pos = 最高點那根 1 分 K 是 frame 資料的第幾根（0 起算，只數有成交的分鐘；與 analysis/*/min/ 的列數一致）。

用法
----
    import sys; sys.path.insert(0, 'lib')
    from frame_filter import keep_frames, max_bar_table
    frames = keep_frames()          # 最高點不在前 MIN_BAR_POS 根的 df_idx
    t = max_bar_table()             # 每個 frame 一列：bar_pos、最高點是否在 frame_start 之前…
"""
import pandas as pd

from backtest_report import bars_of, load_frames

MIN_BAR_POS = 5      # 最高點落在前幾根 K 的 frame 不分析


def max_bar_table(df_list=None, key_list=None):
    if df_list is None:
        df_list, key_list = load_frames()
    rows = []
    for i, (df, key) in enumerate(zip(df_list, key_list)):
        bars, max_time, max_price = bars_of(df)
        rows.append({'df_idx': i, 'frame_start': key[0], 'frame_end': key[1],
                     'data_start': bars.index[0], 'n_bars': len(bars),
                     'max_time': max_time, 'max_price': max_price,
                     'bar_pos': int((bars.index < max_time.floor('min')).sum()),
                     'before_frame_start': max_time < key[0]})
    return pd.DataFrame(rows)


def keep_frames(min_bar_pos=MIN_BAR_POS, table=None):
    t = max_bar_table() if table is None else table
    return t.loc[t['bar_pos'] >= min_bar_pos, 'df_idx'].tolist()
