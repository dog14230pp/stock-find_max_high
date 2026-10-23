from pathlib import Path

import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parents[1] / 'analysis'


def minute_profile(df, freq='1min'):
    """每根 K 棒內，成交價 (close) 的成交量分布。

    回傳 long format 的 DataFrame，每列 = (某一分鐘, 某個價位)：
      minute  : K 棒起始時間
      price   : 該分鐘內出現過的成交價
      volume  : 該價位在該分鐘的成交量加總
      n_ticks : 該價位在該分鐘的成交筆數
    """
    t = df[['ts', 'close', 'volume']].copy()
    t['minute'] = pd.to_datetime(t['ts']).dt.floor(freq)
    prof = (
        t.groupby(['minute', 'close'])['volume']
        .agg(volume='sum', n_ticks='size')
        .reset_index()
        .rename(columns={'close': 'price'})
    )
    return prof


def save_minute_profile(df, name, out_dir=ANALYSIS_DIR, freq='1min'):
    """算出 minute_profile 並存成 <out_dir>/<name>.csv，回傳 (profile, 路徑)。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prof = minute_profile(df, freq)
    path = out_dir / f'{name}.csv'
    prof.to_csv(path, index=False)
    return prof, path


def plot_minute_distribution(df, minute=None, freq='1min'):
    """畫單一根 K 棒內的分布：左邊是成交價折線（時間 vs 價），
    右邊是各價位成交量的 histogram（與左圖共用價格軸）。

    minute : 要看的那根 K 棒起始時間；None 代表 df 的第一根
    回傳 (fig, (ax_line, ax_hist))
    """
    import matplotlib.pyplot as plt

    t = df[['ts', 'close', 'volume']].copy()
    t['ts'] = pd.to_datetime(t['ts'])
    t['minute'] = t['ts'].dt.floor(freq)
    minute = t['minute'].iloc[0] if minute is None else pd.Timestamp(minute).floor(freq)
    sub = t[t['minute'] == minute]
    if sub.empty:
        raise ValueError(f'{minute} 沒有 tick')

    vol_by_price = sub.groupby('close')['volume'].sum()

    fig, (ax_line, ax_hist) = plt.subplots(
        1, 2, sharey=True, figsize=(11, 5),
        gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.03},
    )
    ax_line.plot(sub['ts'], sub['close'], drawstyle='steps-post', lw=1)
    ax_line.set_xlabel('time')
    ax_line.set_ylabel('price')
    ax_line.tick_params(axis='x', rotation=30)
    ax_line.grid(alpha=0.3)

    ax_hist.barh(vol_by_price.index, vol_by_price.values, height=0.8,
                 color='tab:orange', alpha=0.8)
    ax_hist.set_xlabel('volume')
    ax_hist.grid(alpha=0.3)

    fig.suptitle(f'{minute}  ({freq})   ticks={len(sub)}  volume={sub["volume"].sum()}')
    return fig, (ax_line, ax_hist)


def plot_minute_overlay(df, minute=None, freq='1min', bin='1s'):
    """單一張圖疊加：x 軸是時間，
    左 Y 軸 = close（成交價折線），右 Y 軸 = volume（每 bin 秒的成交量長條）。

    minute : 要看的那根 K 棒起始時間；None 代表 df 的第一根
    bin    : volume 長條的時間間隔，例如 '1s'、'5s'
    回傳 (fig, (ax_close, ax_volume))
    """
    import matplotlib.pyplot as plt

    t = df[['ts', 'close', 'volume']].copy()
    t['ts'] = pd.to_datetime(t['ts'])
    t['minute'] = t['ts'].dt.floor(freq)
    minute = t['minute'].iloc[0] if minute is None else pd.Timestamp(minute).floor(freq)
    sub = t[t['minute'] == minute]
    if sub.empty:
        raise ValueError(f'{minute} 沒有 tick')

    vol = sub.set_index('ts')['volume'].resample(bin).sum()
    width = pd.Timedelta(bin) / pd.Timedelta('1D')   # matplotlib 日期軸以「天」為單位

    fig, ax_close = plt.subplots(figsize=(11, 5))
    ax_vol = ax_close.twinx()

    ax_vol.bar(vol.index, vol.values, width=width, align='edge',
               color='tab:orange', alpha=0.45, label='volume')
    ax_vol.set_ylabel('volume', color='tab:orange')
    ax_vol.tick_params(axis='y', colors='tab:orange')

    # 價格線畫在上層，不被長條蓋住
    ax_close.set_zorder(ax_vol.get_zorder() + 1)
    ax_close.patch.set_visible(False)
    ax_close.plot(sub['ts'], sub['close'], drawstyle='steps-post', lw=1.2,
                  color='tab:blue', label='close')
    ax_close.set_ylabel('close', color='tab:blue')
    ax_close.tick_params(axis='y', colors='tab:blue')
    ax_close.tick_params(axis='x', rotation=30)
    ax_close.set_xlabel('time')
    ax_close.grid(alpha=0.3)

    fig.suptitle(f'{minute}  ({freq})   ticks={len(sub)}  volume={sub["volume"].sum()}')
    return fig, (ax_close, ax_vol)


QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def minute_stats(df, freq='1min'):
    """每根 K 棒內，close 與 volume(單筆 tick 成交量) 各自的描述統計。

    回傳 DataFrame，index 是 K 棒起始時間，欄位為 close_* 與 volume_*：
      count, mean, std, min, q01, q05, q10, q25, q50(median), q75, q90, q95, q99,
      max, range, iqr, skew, kurt；volume 另有 volume_sum
    """
    t = df[['ts', 'close', 'volume']].copy()
    t['minute'] = pd.to_datetime(t['ts']).dt.floor(freq)

    parts = {}
    for col in ['close', 'volume']:
        g = t.groupby('minute')[col]
        cols = {
            'count': g.count(), 'mean': g.mean(), 'std': g.std(), 'min': g.min(),
        }
        for q in QUANTILES:
            cols[f'q{round(q * 100):02d}'] = g.quantile(q)
        cols['max'] = g.max()
        cols['range'] = cols['max'] - cols['min']
        cols['iqr'] = cols['q75'] - cols['q25']
        cols['skew'] = g.skew()
        cols['kurt'] = g.apply(pd.Series.kurt)
        if col == 'volume':
            cols['sum'] = g.sum()
        parts.update({f'{col}_{k}': v for k, v in cols.items()})
    return pd.DataFrame(parts)


def save_minute_stats(df, name, out_dir=ANALYSIS_DIR, freq='1min'):
    """算出 minute_stats 並存成 <out_dir>/<name>_stats.csv，回傳 (stats, 路徑)。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = minute_stats(df, freq)
    path = out_dir / f'{name}_stats.csv'
    stats.to_csv(path)
    return stats, path


def minute_summary(df, freq='1min'):
    """每根 K 棒的摘要：開高低收、總成交量、內外盤(type 1 / type 2)的筆數與成交量。

    欄位：open, high, low, close, volume,
          type1_count, type1_volume, type2_count, type2_volume,
          type1_ratio（外盤成交量 / 該分鐘總成交量，分母含 type 0）
    （type 1 = 外盤、type 2 = 內盤；每個 df 內依 ts 順序決定 open / close）
    """
    t = df[['ts', 'close', 'volume', 'tick_type']].copy()
    t['ts'] = pd.to_datetime(t['ts'])
    t = t.sort_values('ts', kind='stable')
    t['minute'] = t['ts'].dt.floor(freq)
    t['v1'] = t['volume'].where(t['tick_type'] == 1, 0)
    t['v2'] = t['volume'].where(t['tick_type'] == 2, 0)
    t['c1'] = (t['tick_type'] == 1).astype(int)
    t['c2'] = (t['tick_type'] == 2).astype(int)

    summary = t.groupby('minute').agg(
        open=('close', 'first'),
        high=('close', 'max'),
        low=('close', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
        type1_count=('c1', 'sum'),
        type1_volume=('v1', 'sum'),
        type2_count=('c2', 'sum'),
        type2_volume=('v2', 'sum'),
    )
    summary['type1_ratio'] = summary['type1_volume'] / summary['volume']
    return summary


def save_minute_summary(df, name, out_dir=ANALYSIS_DIR, freq='1min'):
    """算出 minute_summary 並存成 <out_dir>/<name>_summary.csv，回傳 (summary, 路徑)。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = minute_summary(df, freq)
    path = out_dir / f'{name}_summary.csv'
    summary.to_csv(path)
    return summary, path
