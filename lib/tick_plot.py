import pandas as pd
import mplfinance as mpf


def plot_tick_frame(df, freq='1min', mark_time=None, title=None, volume=True,
                    mark_high=True, start=None, end=None):
    """把 tick df 轉成 K 線並標出「當下」的點位。

    df        : 含 ts, close, volume 欄位的 tick DataFrame
    freq      : K 線週期，例如 '1min'、'5min'、'30s'
    mark_time : 要標的時間點；None（預設）就不標
    start, end: 只畫這段時間（前後都含）。可給完整時間 '2021-05-13 08:50:00'，
                或只給時間 '08:50'（日期取 df 第一筆 tick 的日期）；任一個可省略
    mark_high : True（預設）在整張 K 線圖的最高點標紅點，註明價位與時間
    回傳 (fig, axes, mark_price)，沒標的話 mark_price 為 None
    """
    ticks = df.set_index(pd.to_datetime(df['ts'])).sort_index()
    if start is not None or end is not None:
        base = ticks.index[0].normalize()

        def resolve(x):
            x = str(x).strip()
            return pd.Timestamp(x) if len(x) > 8 else pd.Timestamp(f'{base:%Y-%m-%d} {x}')

        lo = resolve(start) if start is not None else ticks.index[0]
        hi = resolve(end) if end is not None else ticks.index[-1]
        ticks = ticks.loc[lo:hi]
        if ticks.empty:
            raise ValueError(f'{lo} ~ {hi} 沒有 tick')

    ohlc = ticks['close'].resample(freq).ohlc()
    ohlc['Volume'] = ticks['volume'].resample(freq).sum()
    ohlc = ohlc.dropna(subset=['open']).rename(columns=str.capitalize)

    fig, axes = mpf.plot(
        ohlc, type='candle', volume=volume, style='yahoo',
        title=title or f'{ticks.index[0]} ~ {ticks.index[-1]}  ({freq})',
        returnfig=True, figsize=(12, 6),
    )
    ax = axes[0]
    if mark_high:
        # 整張 K 線圖的最高點：最高價、發生的 tick 時間（同價取最早）
        i = ticks['close'].to_numpy().argmax()
        hi_ts, hi_price = ticks.index[i], ticks['close'].iloc[i]
        hx = ohlc.index.searchsorted(hi_ts, side='right') - 1
        ax.scatter([hx], [hi_price], color='red', zorder=5)
        ax.annotate(f'High {hi_price:g}\n{hi_ts:%Y-%m-%d %H:%M:%S}', (hx, hi_price),
                    xytext=(0, 18), textcoords='offset points', ha='center',
                    color='red', arrowprops=dict(arrowstyle='-', color='red'),
                    bbox=dict(boxstyle='round', fc='white', ec='red', alpha=0.9))

    if mark_time is None:
        return fig, axes, None

    mark_ts = pd.Timestamp(mark_time)
    mark_price = ticks.loc[:mark_ts, 'close'].iloc[-1]
    # 該時間落在第幾根 K 棒（mplfinance 的 x 軸是 0..n-1）
    x = ohlc.index.searchsorted(mark_ts, side='right') - 1

    ax.axhline(mark_price, color='red', linestyle='--', linewidth=0.8)
    ax.scatter([x], [mark_price], color='red', zorder=5)
    ax.annotate(f'{mark_ts:%H:%M:%S}  {mark_price:g}', (x, mark_price),
                xytext=(-10, 12), textcoords='offset points',
                ha='right', color='red',
                bbox=dict(boxstyle='round', fc='white', ec='red', alpha=0.9))
    return fig, axes, mark_price
