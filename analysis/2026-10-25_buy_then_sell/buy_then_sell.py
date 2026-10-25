"""「前買後賣」檢查：S8 找到的點，再看那一分鐘是不是前段買、後段賣（242 frames）。

每一分鐘（只用這分鐘內的 tick，分鐘收盤時就知道）：
    把有成交的秒照時間排好，試每個切點 k（前段 = 第 1 ~ k 秒，後段 = 第 k+1 秒 ~ 最後；兩段都至少 MIN_SEC 秒）
    分數(k) = （前段外盤量 + 後段內盤量）÷ 這分鐘總成交量
    bts = 最大的分數（0 ~ 1）；有成交的秒不到 2 × MIN_SEC 的分鐘 = 空白（切不出來，不保留）
規則：S8 標記 且 bts >= 門檻 → 標記

輸出（與本檔同資料夾）：minute_bts.csv（每分鐘的 bts）、sweep.csv（各門檻的結果）
執行（在專案根目錄）：python analysis/2026-10-25_buy_then_sell/buy_then_sell.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
sys.path.insert(0, str(ROOT / 'strategies'))

import start_backtest as sb  # noqa: E402
from backtest_report import load_frames  # noqa: E402
from frame_filter import keep_frames, max_bar_table  # noqa: E402
from registry import STRATEGIES  # noqa: E402

BASE = 'S8'
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
MIN_SEC = 3          # 前段、後段各至少幾秒有成交


def minute_bts(df):
    """tick → 每分鐘的 bts（index = 分鐘）。"""
    t = df[['ts', 'volume', 'tick_type']].copy()
    t['ts'] = pd.to_datetime(t['ts'])
    t['v1'] = t['volume'].where(t['tick_type'] == 1, 0)
    t['v2'] = t['volume'].where(t['tick_type'] == 2, 0)
    s = t.groupby(t['ts'].dt.floor('1s'))[['volume', 'v1', 'v2']].sum()
    out = {}
    for m, g in s.groupby(s.index.floor('1min')):
        n = len(g)
        if n < 2 * MIN_SEC:
            out[m] = np.nan
            continue
        front_buy = g['v1'].cumsum().to_numpy()[:-1]                    # [k-1] = 前段第 1 ~ k 秒的外盤
        back_sell = g['v2'].to_numpy()[::-1].cumsum()[::-1][1:]          # [k-1] = 後段第 k+1 秒 ~ 最後的內盤
        score = (front_buy + back_sell)[MIN_SEC - 1:n - MIN_SEC]         # k = MIN_SEC ~ n − MIN_SEC
        out[m] = score.max() / g['volume'].sum()
    return pd.Series(out)


def main():
    df_list, key_list = load_frames()
    keep = set(keep_frames(table=max_bar_table(df_list, key_list)))
    bts = pd.concat([minute_bts(df) for df in df_list])
    bts = bts[~bts.index.duplicated()]
    bts.rename('bts').to_csv(HERE / 'minute_bts.csv', index_label='minute', encoding='utf-8-sig')
    base = STRATEGIES[BASE]['mask']

    rows = []
    for th in [None] + THRESHOLDS:
        def mask(mb, th=th):
            ok = base(mb)
            return ok if th is None else ok & (bts.reindex(mb.index) >= th)
        d, f = sb.run(mask, df_list, keep)
        s = sb.summary(d, f).loc['keep']
        rows.append({'rule': BASE if th is None else f'{BASE} + bts >= {th}', 'hits': int(s.hits),
                     'hit_pct': s.hits / s.frames * 100, 'marks': int(s.marks), 'extra': int(s.extra_marks),
                     'prec_pct': s.hits / s.marks * 100, 'frames_no_marks': int(s.frames_no_marks),
                     'loss_med': s.loss_med, 'loss_p90': s.loss_p90, 'loss_max': s.loss_max})
    r = pd.DataFrame(rows)
    r.to_csv(HERE / 'sweep.csv', index=False, encoding='utf-8-sig')

    # S8 的點裡，Hit 與其他點的 bts
    hit_b, other_b = [], []
    for i in keep:
        mb, max_time, _ = sb.minute_bars(df_list[i])
        m = base(mb).fillna(False).to_numpy(dtype=bool)
        mx = mb.index.get_loc(max_time.floor('1min'))
        b = bts.reindex(mb.index).to_numpy()
        for k in np.flatnonzero(m):
            (hit_b if k == mx else other_b).append(b[k])
    with pd.option_context('display.width', 200):
        print(f'242 frames；{BASE} 的點：Hit {len(hit_b)} 個 bts 中位 {np.nanmedian(hit_b):.2f}，'
              f'其他 {len(other_b)} 個 bts 中位 {np.nanmedian(other_b):.2f}')
        print(r.round(1).to_string(index=False))


if __name__ == '__main__':
    main()
