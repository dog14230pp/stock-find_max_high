"""策略清單：給 strategies/compare.py 用的編號、名稱、說明、標記函式。

正式策略（2026-10-25 定案，說明見 README「正式策略」）：
    策略 1 = S5

新增策略：在 STRATEGIES 加一筆，然後 python strategies/compare.py <編號>。
標記函式 mask(mb) -> bool Series：mb 每分鐘一列（open, high, low, close, volume, type1_volume, type2_volume, ratio），
只能用 t 與之前的資料（不可偷看未來）。
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import type1_vol as tv


def softmax_base(volume, n=5, temp=1.0):
    """前 n 根（t-n ~ t-1）成交量的 softmax 加權平均：權重 = softmax((v / 前 n 根平均) / temp)。
    temp 越小越接近前 n 根最大量，越大越接近一般平均。前面不足 n 根 = NaN。"""
    v = volume.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) > n:
        w = sliding_window_view(v[:-1], n)                  # 第 k 列 = v[k : k+n]，對應 t = k+n
        z = w / w.mean(axis=1, keepdims=True) / temp
        e = np.exp(z - z.max(axis=1, keepdims=True))
        out[n:] = (e * w).sum(axis=1) / e.sum(axis=1)
    return pd.Series(out, index=volume.index)


def price_vol_corr(mb, n=10, price='open'):
    """corr(price, Volume, n)：t-n+1 ~ t 這 n 根的價格（open / close）與成交量相關係數
    （price='open' = WorldQuant Alpha#6 去掉負號）。越高 = 價格越高的那幾根量也越大（追價放量）。前面不足 n 根 = NaN。"""
    return mb[price].rolling(n).corr(mb['volume'].astype(float))


def vol_rise(n=5, mult=2.0, ratio_min=0.5, rise=3, new_high=0, wick_min=0.0, wick_body_min=0.0,
             vol_base='mean', temp=1.0, corr_min=None, corr_n=10, corr_close_min=None):
    """量放大 + 外盤比例（+ close 連漲 rise 次）（+ 創前 new_high 根新高）（+ 上影線 >= wick_min）
    （+ 上影線 >= wick_body_min × 實體）（+ corr(Open, Volume, corr_n) >= corr_min）。

    vol_base：量的基準。'mean' = 前 n 根平均；'softmax' = 前 n 根 softmax 加權平均（見 softmax_base，溫度 temp）；
              'none' = 不看量（不用 n、mult）。
    new_high：這根 high >= 前 new_high 根的最高價；前面不足 new_high 根時用現有的根數。
    wick_min：上影線 /（high − low）的下限（0.15 = 15%）；high == low 時上影線 = 0。
    wick_body_min：上影線 >= wick_body_min × |close − open|；十字線（實體 = 0）一定符合。
    corr_min：corr(Open, Volume, corr_n) 的下限（見 price_vol_corr）；None = 不看。
    corr_close_min：corr(Close, Volume, corr_n) 的下限；None = 不看。
    """
    def mask(mb):
        if vol_base in ('softmax', 'none'):
            ok = mb['ratio'] >= ratio_min
            if vol_base == 'softmax':
                ok &= mb['volume'] > mult * softmax_base(mb['volume'], n, temp)
            if rise:
                ok &= (mb['close'].diff() > 0).rolling(rise).sum().eq(rise)
        else:
            ok = tv.vol_mask(mb, n, ratio_min, mult, rise)[0]
        if corr_min is not None:
            ok &= price_vol_corr(mb, corr_n, 'open') >= corr_min
        if corr_close_min is not None:
            ok &= price_vol_corr(mb, corr_n, 'close') >= corr_close_min
        if new_high:
            ok &= mb['high'] >= mb['high'].rolling(new_high, min_periods=1).max().shift(1)
        if wick_min:
            rng = mb['high'] - mb['low']
            wick = (mb['high'] - mb[['open', 'close']].max(axis=1)) / rng.where(rng > 0)
            ok &= wick.fillna(0) >= wick_min
        if wick_body_min:
            wick = mb['high'] - mb[['open', 'close']].max(axis=1)
            ok &= wick >= wick_body_min * (mb['close'] - mb['open']).abs()
        return ok
    return mask


STRATEGIES = {
    'S1': {
        'name': '量放大 + 外盤過半 + 連漲 3 次',
        'desc': '成交量 > 2 × 前 5 根平均、type1_ratio ≥ 0.5、close 連漲 3 次（c[t-3] < c[t-2] < c[t-1] < c[t]）'
                '→ 標記 t（= type1_vol5_x2_up3）',
        'mask': vol_rise(n=5, mult=2.0, ratio_min=0.5, rise=3),
    },
    'S2': {
        'name': '2 倍量 + 外盤過半 + 創新高 + 上影線',
        'desc': '成交量 > 2 × 前 5 根平均、type1_ratio ≥ 0.5、high ≥ 前 60 根最高價（不足 60 根用現有的）、'
                '上影線 ≥ 整根長度的 15% → 標記 t',
        'mask': vol_rise(n=5, mult=2.0, ratio_min=0.5, rise=0, new_high=60, wick_min=0.15),
    },
    'S3': {
        'name': '1.5 倍量 + 外盤過半 + 創新高 + 上影線',
        'desc': '同 S2，但成交量 > 1.5 × 前 5 根平均',
        'mask': vol_rise(n=5, mult=1.5, ratio_min=0.5, rise=0, new_high=60, wick_min=0.15),
    },
    'S4': {
        'name': 'S3 + 上影線 ≥ 0.9 倍實體',
        'desc': '同 S3（成交量 > 1.5 × 前 5 根平均、type1_ratio ≥ 0.5、high ≥ 前 60 根最高價、上影線 ≥ 整根長度的 15%），'
                '再加上影線 ≥ 0.9 × 實體（|close − open|）→ 標記 t',
        'mask': vol_rise(n=5, mult=1.5, ratio_min=0.5, rise=0, new_high=60, wick_min=0.15, wick_body_min=0.9),
    },
    'S5': {
        'name': '【正式策略 1】S3 + 上影線 ≥ 0.8 倍實體',
        'desc': '成交量 > 1.5 × 前 5 根平均、type1_ratio ≥ 0.5、high ≥ 前 60 根最高價（不足 60 根用現有的）、'
                '上影線 ≥ 整根長度的 15%、上影線 ≥ 0.8 × 實體（|close − open|）→ 標記 t',
        'mask': vol_rise(n=5, mult=1.5, ratio_min=0.5, rise=0, new_high=60, wick_min=0.15, wick_body_min=0.8),
    },
    'S6': {
        'name': '策略 1，量的基準改 softmax 平均（×1.2）',
        'desc': '同 S5，但條件 1 改成：成交量 > 1.2 × 前 5 根的 softmax 加權平均'
                '（權重 = softmax(量 / 前 5 根平均)，溫度 1；量越大的那根權重越大）',
        'mask': vol_rise(n=5, mult=1.2, ratio_min=0.5, rise=0, new_high=60, wick_min=0.15, wick_body_min=0.8,
                         vol_base='softmax', temp=1.0),
    },
    'S7': {
        'name': 'S6 的量換成價量相關：corr(Open, Vol) ≥ 0.2 且 corr(Close, Vol) ≥ 0.2',
        'desc': '同 S6，但條件 1 改成：corr(Open, Volume, 10) ≥ 0.2 且 corr(Close, Volume, 10) ≥ 0.2'
                '（t-9 ~ t 這 10 根「價格越高量越大」的程度；Open 版 = WorldQuant Alpha#6 去掉負號），不看量放大',
        'mask': vol_rise(ratio_min=0.5, rise=0, new_high=60, wick_min=0.15, wick_body_min=0.8,
                         vol_base='none', corr_min=0.2, corr_n=10, corr_close_min=0.2),
    },
    'S8_alpha': {
        'name': 'S6 + corr(Close, Vol) ≥ 0.2（含 S5 的條件 2～5）',
        'desc': '同 S6（成交量 > 1.2 × 前 5 根 softmax 加權平均、type1_ratio ≥ 0.5、high ≥ 前 60 根最高價、'
                '上影線 ≥ 整根 15%、上影線 ≥ 0.8 × 實體），再加 corr(Close, Volume, 10) ≥ 0.2。'
                '原本叫 S8，2026-10-25 改名',
        'mask': vol_rise(n=5, mult=1.2, ratio_min=0.5, rise=0, new_high=60, wick_min=0.15, wick_body_min=0.8,
                         vol_base='softmax', temp=1.0, corr_n=10, corr_close_min=0.2),
    },
    'S8': {
        'name': '純粹：softmax 量 × 1.2 + corr(Close, Vol) ≥ 0.2',
        'desc': '只有兩個條件：成交量 > 1.2 × 前 5 根 softmax 加權平均（溫度 1）、corr(Close, Volume, 10) ≥ 0.2。'
                '不帶其他策略的條件',
        'mask': vol_rise(n=5, mult=1.2, ratio_min=0.0, rise=0, vol_base='softmax', temp=1.0,
                         corr_n=10, corr_close_min=0.2),
    },
    'S9': {
        'name': 'S8_alpha 拿掉 ratio ≥ 0.5 和 上影線 ≥ 0.8 × 實體',
        'desc': '4 個條件：成交量 > 1.2 × 前 5 根 softmax 加權平均（溫度 1）、high ≥ 前 60 根最高價、'
                '上影線 ≥ 整根長度的 15%、corr(Close, Volume, 10) ≥ 0.2',
        'mask': vol_rise(n=5, mult=1.2, ratio_min=0.0, rise=0, new_high=60, wick_min=0.15, wick_body_min=0.0,
                         vol_base='softmax', temp=1.0, corr_n=10, corr_close_min=0.2),
    },
}
