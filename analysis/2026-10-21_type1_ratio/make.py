"""每個 frame 的秒級 / 分級 type1 ratio，以及每個 frame 最高點的資訊。

type1_ratio = type1（外盤）成交量 / 全部成交量（分母含 type 0）

輸出（與本檔同資料夾）
----
sec/df_XXX.csv  秒級，每個 frame 一個檔，每列 = 有成交的一秒
    second, minute（所在分鐘，方便對到 min/），n_ticks, open, high, low, close, volume,
    type1_volume, type2_volume, type0_volume, type1_ratio,
    bid_price, ask_price, bid_volume, ask_volume（後四個取該秒最後一筆）

min/df_XXX.csv  分級，每個 frame 一個檔，每列 = 有成交的一分鐘
    minute, type1_ratio, n_ticks, open, high, low, close, close_mean, close_std,
    volume, volume_mean, volume_max, type1_volume, type2_volume, type0_volume,
    bid_price_mean, ask_price_mean, bid_volume_mean, ask_volume_mean,
    sec_ratio_mean / std / min / max（該分鐘內各秒 type1_ratio 的統計）

frame_max.csv  每列 = 一個 frame 的最高點（最高價第一次出現的那筆 tick）
    df_idx, frame_start, frame_end, max_price, max_time, max_second, max_minute,
    max_tick_type, max_tick_volume,
    sec_volume, sec_type1_ratio（最高點那一秒）, min_volume, min_type1_ratio（最高點那一分鐘）

執行（在 D:\\stock）：python analysis/2026-10-21_type1_ratio/make.py
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
from backtest_report import load_frames  # noqa: E402

TICK_COLS = ['ts', 'close', 'volume', 'tick_type', 'bid_price', 'ask_price', 'bid_volume', 'ask_volume']


def prepare(df):
    t = df[TICK_COLS].copy()
    t['ts'] = pd.to_datetime(t['ts'])
    t = t.sort_values('ts', kind='stable').reset_index(drop=True)
    t['second'] = t['ts'].dt.floor('1s')
    t['minute'] = t['ts'].dt.floor('1min')
    for k in (0, 1, 2):
        t[f'v{k}'] = t['volume'].where(t['tick_type'] == k, 0)
    return t


def per_second(t):
    s = t.groupby('second').agg(
        minute=('minute', 'first'), n_ticks=('close', 'size'),
        open=('close', 'first'), high=('close', 'max'), low=('close', 'min'), close=('close', 'last'),
        volume=('volume', 'sum'), type1_volume=('v1', 'sum'), type2_volume=('v2', 'sum'),
        type0_volume=('v0', 'sum'),
        bid_price=('bid_price', 'last'), ask_price=('ask_price', 'last'),
        bid_volume=('bid_volume', 'last'), ask_volume=('ask_volume', 'last'),
    )
    s.insert(s.columns.get_loc('type0_volume') + 1, 'type1_ratio', s['type1_volume'] / s['volume'])
    return s.reset_index()


def per_minute(t, s):
    m = t.groupby('minute').agg(
        n_ticks=('close', 'size'),
        open=('close', 'first'), high=('close', 'max'), low=('close', 'min'), close=('close', 'last'),
        close_mean=('close', 'mean'), close_std=('close', 'std'),
        volume=('volume', 'sum'), volume_mean=('volume', 'mean'), volume_max=('volume', 'max'),
        type1_volume=('v1', 'sum'), type2_volume=('v2', 'sum'), type0_volume=('v0', 'sum'),
        bid_price_mean=('bid_price', 'mean'), ask_price_mean=('ask_price', 'mean'),
        bid_volume_mean=('bid_volume', 'mean'), ask_volume_mean=('ask_volume', 'mean'),
    )
    m.insert(0, 'type1_ratio', m['type1_volume'] / m['volume'])
    sec = s.groupby('minute')['type1_ratio'].agg(['mean', 'std', 'min', 'max']).add_prefix('sec_ratio_')
    return m.join(sec)


def frame_max(i, key, t, s, m):
    j = t['close'].to_numpy().argmax()          # 同價取最早
    tick = t.iloc[j]
    sec = s.set_index('second').loc[tick['second']]
    mi = m.loc[tick['minute']]
    return {
        'df_idx': i, 'frame_start': key[0], 'frame_end': key[1],
        'max_price': tick['close'], 'max_time': tick['ts'],
        'max_second': tick['second'], 'max_minute': tick['minute'],
        'max_tick_type': int(tick['tick_type']), 'max_tick_volume': int(tick['volume']),
        'sec_volume': int(sec['volume']), 'sec_type1_ratio': sec['type1_ratio'],
        'min_volume': int(mi['volume']), 'min_type1_ratio': mi['type1_ratio'],
    }


def main():
    df_list, key_list = load_frames()
    for d in ('sec', 'min'):
        (HERE / d).mkdir(parents=True, exist_ok=True)
    maxes = []
    for i, df in enumerate(df_list):
        t = prepare(df)
        s = per_second(t)
        m = per_minute(t, s)
        s.to_csv(HERE / 'sec' / f'df_{i:03d}.csv', index=False)
        m.to_csv(HERE / 'min' / f'df_{i:03d}.csv')
        maxes.append(frame_max(i, key_list[i], t, s, m))
    pd.DataFrame(maxes).to_csv(HERE / 'frame_max.csv', index=False)
    print(f'寫好 {len(df_list)} 個 frame 的 sec/、min/ CSV，以及 frame_max.csv（{HERE}）')


if __name__ == '__main__':
    main()
