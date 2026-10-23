from pathlib import Path

import pandas as pd
import mplfinance as mpf

from minute_profile import ANALYSIS_DIR, minute_summary


def find_type1_segments(df, freq='1min', threshold=0.45, mark_threshold=0.5):
    """用每分鐘的 type1_ratio（外盤成交量 / 總成交量）切出區間。

    規則（以每根 K 棒為一個 time slot）：
      起點 : type1_ratio > threshold 的 K 棒
      終點 : 起點之後，出現「連續兩根 type1_ratio < threshold」時，區間停在第二根
      標記 : 區間內「最後一次 type1_ratio > mark_threshold」的 K 棒（區間內都沒有則為 NaT）
    ratio 剛好等於 threshold 時，不算起點也不算低比例（區間不因它而結束）。
    資料結束前還沒等到連續兩根低比例的區間，closed=False（標記點只用目前已有的資料找）。

    回傳 (segments, summary)：
      segments: 每列一個區間，欄位 start, end, mark, n_slots, closed, start_ratio, mark_ratio
      summary : minute_summary 的結果（含 type1_ratio）
    """
    summary = minute_summary(df, freq)
    summary['type1_ratio'] = summary['type1_volume'] / summary['volume']
    ratio = summary['type1_ratio'].to_numpy()
    idx = summary.index
    n = len(summary)

    rows = []
    i = 0
    while i < n:
        if not ratio[i] > threshold:
            i += 1
            continue
        start, j, end = i, i + 1, None
        while j + 1 < n:
            if ratio[j] < threshold and ratio[j + 1] < threshold:
                end = j + 1
                break
            j += 1
        closed = end is not None
        last = end if closed else n - 1
        high = [k for k in range(start, last + 1) if ratio[k] > mark_threshold]
        mark = high[-1] if high else None
        rows.append({
            'start': idx[start],
            'end': idx[last],
            'mark': idx[mark] if mark is not None else pd.NaT,
            'n_slots': last - start + 1,
            'closed': closed,
            'start_ratio': ratio[start],
            'mark_ratio': ratio[mark] if mark is not None else float('nan'),
        })
        i = last + 1
    return pd.DataFrame(rows), summary


def plot_type1_segments(df, freq='1min', threshold=0.45, mark_threshold=0.5, title=None,
                        mark_max=True):
    """畫 K 線並標出 find_type1_segments 的結果。

    綠▲ = 起點；紅★ = 區間內最後一次 type1_ratio > mark_threshold；灰▼ = 區間結束；淺色底 = 區間範圍
    mark_max=True 時另外用黑色菱形＋水平虛線標出整張圖（整個 df）的最高價與發生的 tick 時間
    回傳 (fig, axes, segments)
    """
    segments, _ = find_type1_segments(df, freq, threshold, mark_threshold)

    t = df[['ts', 'close', 'volume']].copy()
    t = t.set_index(pd.to_datetime(t['ts'])).sort_index()
    ohlc = t['close'].resample(freq).ohlc()
    ohlc['Volume'] = t['volume'].resample(freq).sum()
    ohlc = ohlc.dropna(subset=['open']).rename(columns=str.capitalize)

    fig, axes = mpf.plot(
        ohlc, type='candle', volume=True, style='yahoo', returnfig=True, figsize=(14, 6),
        title=title or f'type1_ratio > {threshold} segments  ({freq})',
    )
    ax = axes[0]
    span = (ohlc['High'].max() - ohlc['Low'].min()) * 0.07   # 標記與 K 線的距離
    for _, s in segments.iterrows():
        a, b = ohlc.index.get_loc(s['start']), ohlc.index.get_loc(s['end'])
        ax.axvspan(a - 0.5, b + 0.5, color='tab:blue', alpha=0.08)
        ax.scatter([a], [ohlc['Low'].iloc[a] - span], marker='^', color='green', s=60, zorder=5)
        if pd.notna(s['mark']):
            m = ohlc.index.get_loc(s['mark'])
            ax.scatter([m], [ohlc['High'].iloc[m] + span], marker='*', color='red', s=220, zorder=5)
        if s['closed']:
            ax.scatter([b], [ohlc['High'].iloc[b] + span], marker='v', color='gray', s=60, zorder=5)
    if mark_max:
        i = t['close'].to_numpy().argmax()               # 同價取最早
        mx_ts, mx_price = t.index[i], t['close'].iloc[i]
        x = ohlc.index.get_loc(mx_ts.floor(freq))
        ax.axhline(mx_price, color='black', linestyle=':', linewidth=0.8)
        ax.scatter([x], [mx_price], marker='D', color='black', s=45, zorder=6)
        ax.text(0.005, mx_price, f' Max {mx_price:g}  @ {mx_ts:%H:%M:%S}',
                transform=ax.get_yaxis_transform(), va='bottom', ha='left', fontsize=9,
                bbox=dict(boxstyle='round', fc='white', ec='black', alpha=0.9))
    return fig, axes, segments


def save_type1_segments(df, name, out_dir=ANALYSIS_DIR, freq='1min', threshold=0.45,
                        mark_threshold=0.5):
    """把區間存成 <out_dir>/<name>_type1_segments.csv，回傳 (segments, 路徑)。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    segments, _ = find_type1_segments(df, freq, threshold, mark_threshold)
    path = out_dir / f'{name}_type1_segments.csv'
    segments.to_csv(path, index=False)
    return segments, path


def type1_segments_all(df_list, key_list=None, freq='1min', threshold=0.45, mark_threshold=0.5):
    """對 df_list 每個 df 都做一次 find_type1_segments，合併成一張表。

    多出的欄位：
      df_idx        : 在 df_list 的位置
      frame_start/end : 該 df 的 key（有給 key_list 才有）
      seg_max_price : 區間內（start ~ end 那幾根 K 棒）的最高價
      mark_high     : 標記那根 K 棒的最高價
      mark_is_max   : 標記點所在的 K 棒 high == 區間最高價（與區間最高價重疊；同價也算）
    """
    out = []
    for i, df in enumerate(df_list):
        seg, summ = find_type1_segments(df, freq, threshold, mark_threshold)
        if seg.empty:
            continue
        seg = seg.copy()
        seg.insert(0, 'df_idx', i)
        if key_list is not None:
            seg.insert(1, 'frame_start', key_list[i][0])
            seg.insert(2, 'frame_end', key_list[i][1])
        high = summ['high']
        seg['seg_max_price'] = [high.loc[a:b].max() for a, b in zip(seg['start'], seg['end'])]
        seg['mark_high'] = [high.loc[m] if pd.notna(m) else float('nan') for m in seg['mark']]
        seg['mark_is_max'] = seg['mark_high'] == seg['seg_max_price']
        out.append(seg)
    return pd.concat(out, ignore_index=True)


def frame_max_vs_marks(df_list, key_list=None, freq='1min', threshold=0.45, mark_threshold=0.5):
    """每個 df：整個 df 的最高價發生在哪一分鐘，有沒有被標記點（★）剛好標到。

    max_minute      : 整個 df 最高價（close）第一次出現的那一分鐘
    max_minutes_all : 最高價出現過的所有分鐘（同價平手時會有多個）
    hit             : max_minute 與任一個標記點的分鐘完全相同
    hit_any_tie     : 最高價出現過的任一分鐘與標記點相同（較寬鬆）
    回傳 (per_frame, all_segments)
    """
    seg = type1_segments_all(df_list, key_list, freq, threshold, mark_threshold)
    marks = seg[seg['mark'].notna()].groupby('df_idx')['mark'].apply(set).to_dict()

    rows = []
    for i, df in enumerate(df_list):
        t = df[['ts', 'close']].copy()
        t['ts'] = pd.to_datetime(t['ts'])
        t = t.sort_values('ts', kind='stable')
        mx = t['close'].max()
        t['minute'] = t['ts'].dt.floor(freq)
        first = t.loc[t['close'] == mx, 'minute'].iloc[0]
        all_min = sorted(t.loc[t['close'] == mx, 'minute'].unique())
        m = marks.get(i, set())
        row = {'df_idx': i}
        if key_list is not None:
            row.update(frame_start=key_list[i][0], frame_end=key_list[i][1])
        row.update(
            max_price=mx,
            max_minute=first,
            n_max_minutes=len(all_min),
            n_marks=len(m),
            hit=first in m,
            hit_any_tie=any(x in m for x in all_min),
        )
        rows.append(row)
    return pd.DataFrame(rows), seg
