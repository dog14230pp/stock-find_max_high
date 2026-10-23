import pandas as pd


def slice_ticks(df, start, end):
    """回傳 df 中 start <= ts <= end 的 tick（前後都含）。

    start / end : 可以是字串或 Timestamp，例如 '2021-05-13 09:44:32'
    """
    ts = pd.to_datetime(df['ts'])
    return df[(ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))]
