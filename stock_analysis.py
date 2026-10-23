"""
股價分析腳本（由 讀取資料.ipynb 整理而來）

主要功能：
1. 讀取股價 CSV，以還原價格 (adj*) 繪製 K 線圖與收盤價走勢
2. 找出每月最低點、每 N 日最低點
3. 找出價格低點、成交量低點、成交量 V 型反轉、連續下跌、區間回檔等訊號
4. （選用）批次將各訊號前後的走勢圖存成 PNG

執行方式：
    .venv\\Scripts\\python.exe stock_analysis.py

需要套件：pandas, numpy, matplotlib, mplfinance
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "goog_prices.csv"
OUTPUT_DIR = BASE_DIR / "analysis_plots"

# True 會產生上千張圖片，執行時間較長
RUN_BATCH_SAVE = False


# ---------------------------------------------------------------------------
# 資料處理
# ---------------------------------------------------------------------------
def load_data(path=DATA_PATH):
    """讀取 CSV，日期轉為 datetime 並依日期排序"""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def add_mid(df):
    """新增 mid 欄位

    注意：沿用 notebook 原本的公式 low + (low + high) / 2，
    與一般的中間價 (low + high) / 2 不同。若要改為中間價，修改這一行即可。
    """
    df["mid"] = df["low"] + (df["low"] + df["high"]) / 2
    return df


def to_adjusted_ohlcv(df):
    """取出還原價格，並轉成 mplfinance 需要的格式（DatetimeIndex + Open/High/Low/Close/Volume）"""
    out = df.set_index("date")[["adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"]].copy()
    out.columns = ["Open", "High", "Low", "Close", "Volume"]
    return out


# ---------------------------------------------------------------------------
# 基本繪圖
# ---------------------------------------------------------------------------
def plot_kline(df_adj, title="Adjusted Stock Price"):
    """K 線圖加成交量"""
    mpf.plot(df_adj, type="candle", volume=True, style="yahoo", title=title, figsize=(12, 8))


def plot_adj_close_trend(df):
    """還原收盤價走勢"""
    plt.figure(figsize=(16, 8))
    plt.plot(df["date"], df["adjClose"], color="b", linestyle="-", label="trend")
    plt.title("adjClose chart")
    plt.xlabel("date")
    plt.ylabel("adjClose")
    plt.grid(True)
    plt.legend()
    plt.show()


# ---------------------------------------------------------------------------
# 低點分析
# ---------------------------------------------------------------------------
def plot_monthly_lows(df):
    """每月最低點（以還原收盤價計）"""
    df = df.sort_values("date")
    min_indices = df.groupby(df["date"].dt.to_period("M"))["adjClose"].idxmin()
    monthly_min = df.loc[min_indices]
    print("每月最低點資料：")
    print(monthly_min[["date", "adjClose"]])

    plt.figure(figsize=(14, 6))
    plt.plot(df["date"], df["adjClose"], color="dodgerblue", linewidth=1.5, label="adjClose")
    plt.scatter(monthly_min["date"], monthly_min["adjClose"], color="red", s=30, zorder=5, label="Monthly Low")
    plt.title("Stock Adjusted Close Price & Monthly Lows", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_group_lows_with_volume(df, n_days=5):
    """每 n_days 個交易日取一個最低點（mid），上圖為價格，下圖為成交量"""
    plot_df = df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    if "mid" not in plot_df.columns:
        plot_df["mid"] = plot_df["adjLow"] + (plot_df["adjHigh"] - plot_df["adjLow"]) / 2

    group_key = np.arange(len(plot_df)) // n_days
    min_indices = plot_df.groupby(group_key)["mid"].idxmin()
    min_rows = plot_df.loc[min_indices]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(plot_df["date"], plot_df["adjClose"], color="dodgerblue", alpha=0.5, label="Adjusted Close")
    ax1.scatter(min_rows["date"], min_rows["adjClose"], color="red", s=25, zorder=5,
                label=f"{n_days}-Day Group Low")
    ax1.set_title(f"Stock Analysis: {n_days}-Day Group Lows & Volume", fontsize=14)
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(plot_df["date"], plot_df["adjVolume"], color="gray", alpha=0.4, label="Volume")
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return min_indices


def plot_focus_around_index(df, row_idx, offset=50):
    """以 row_idx 為中心，前後各 offset 筆資料畫 mid 與成交量"""
    start_idx = max(0, row_idx - offset)
    end_idx = min(len(df), row_idx + offset)
    subset = df.iloc[start_idx:end_idx]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    ax1.plot(subset.index, subset["mid"], marker="o", color="royalblue", label="Mid Price")
    ax1.scatter(row_idx, df.loc[row_idx, "mid"], color="red", s=100, zorder=5, label="Target Point")
    ax1.set_title(f"Focus View: Price & Volume around Index {row_idx}")
    ax1.set_ylabel("Price (Mid)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(subset.index, subset["volume"], color="gray", alpha=0.5, label="Volume")
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Row Index")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 訊號偵測
# ---------------------------------------------------------------------------
def find_market_indices(df, method="price_low", n_days=60, prominence=0.2, window=5, tolerance=3):
    """回傳符合條件的列索引

    method:
        'price_low'    每 n_days 天的 mid 最低點
        'vol_low'      每 n_days 天的成交量最低點
        'vol_v_shape'  成交量 V 型反轉（左右各 window 天的均量都高出 prominence）
        'combined'     price_low 與 vol_v_shape 在 tolerance 天內同時出現的點
    """
    temp_df = df.copy()
    if "mid" not in temp_df.columns:
        temp_df["mid"] = temp_df["adjLow"] + (temp_df["adjHigh"] - temp_df["adjLow"]) / 2

    if method == "price_low":
        group_key = np.arange(len(temp_df)) // n_days
        return temp_df.groupby(group_key)["mid"].idxmin().values

    if method == "vol_low":
        group_key = np.arange(len(temp_df)) // n_days
        return temp_df.groupby(group_key)["volume"].idxmin().values

    if method == "vol_v_shape":
        temp_df["vol_smooth"] = temp_df["volume"].rolling(window=3, center=True).mean()
        v_indices = []
        for i in range(window, len(temp_df) - window):
            mid = temp_df.iloc[i]["vol_smooth"]
            left = temp_df.iloc[i - window:i]["vol_smooth"].mean()
            right = temp_df.iloc[i + 1:i + window + 1]["vol_smooth"].mean()
            if mid < left * (1 - prominence) and mid < right * (1 - prominence):
                if mid == temp_df.iloc[i - window:i + window + 1]["vol_smooth"].min():
                    v_indices.append(i)
        return np.array(v_indices)

    if method == "combined":
        p_indices = find_market_indices(df, method="price_low", n_days=n_days)
        v_indices = find_market_indices(df, method="vol_v_shape", prominence=prominence, window=window)
        combined = []
        for p in p_indices:
            for v in v_indices:
                if abs(p - v) <= tolerance:
                    combined.append(p)
                    break
        return np.array(combined)

    raise ValueError(f"未知的 method: {method}")


def find_consecutive_drops(df, n_days=5):
    """找出連續下跌恰好 n_days 天的位置（不重疊）"""
    close = df["adjClose"].values
    indices = []
    i = 1
    while i < len(df):
        count = 0
        while i < len(df) and close[i] < close[i - 1]:
            count += 1
            i += 1
            if count == n_days:
                indices.append(i - 1)
                break  # 避免重複計算同一段下跌
        i += 1
    return np.array(indices)


def find_price_drawdowns(df, window=5, threshold=0.05):
    """找出 window 天內跌幅達 threshold 的位置（不重疊）"""
    close = df["adjClose"].values
    indices = []
    last_idx = -1
    for i in range(window, len(df)):
        change = (close[i] - close[i - window]) / close[i - window]
        if change <= -threshold and i > last_idx + window:
            indices.append(i)
            last_idx = i
    return np.array(indices)


def plot_market_analysis(df, target_indices, title="Market Analysis", window_size=None, focus_idx=None):
    """畫出價格與成交量，並標示指定的訊號點

    window_size 與 focus_idx 同時給定時，只畫 focus_idx 前後 window_size 天
    """
    plot_df = df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])

    if focus_idx is not None and window_size is not None:
        start = max(0, focus_idx - window_size)
        end = min(len(plot_df), focus_idx + window_size)
        plot_df = plot_df.iloc[start:end]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1.plot(plot_df["date"], plot_df["adjClose"], color="dodgerblue", alpha=0.6, label="Price")
    valid_p_indices = [i for i in target_indices if i in plot_df.index]
    ax1.scatter(df.loc[valid_p_indices, "date"], df.loc[valid_p_indices, "adjClose"],
                color="red", s=50, zorder=5, label="Markers")
    ax1.set_title(title, fontsize=14)
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    ax2.bar(plot_df["date"], plot_df["volume"], color="gray", alpha=0.3, label="Volume")
    ax2.plot(plot_df["date"], plot_df["volume"].rolling(7, center=True).mean(),
             color="orange", alpha=0.7, label="Vol 7MA")
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 批次存圖
# ---------------------------------------------------------------------------
def plot_and_save(df, idx, offset, title_suffix, save_dir):
    """以 idx 為中心畫一張圖並存檔"""
    subset = df.iloc[idx - offset:idx + offset].copy()
    subset["date"] = pd.to_datetime(subset["date"])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    ax1.plot(subset["date"], subset["adjClose"], marker=".", color="blue", alpha=0.5)
    ax1.scatter(df.loc[idx, "date"], df.loc[idx, "adjClose"], color="red", s=80, zorder=5)
    ax1.set_title(title_suffix)
    ax1.grid(True, alpha=0.2)

    ax2.bar(subset["date"], subset["volume"], color="gray", alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / f"{title_suffix}.png")
    plt.close(fig)


def batch_save_plots(df, method, param_list, offset=20):
    """依 method 與參數列表找訊號，並批次存圖

    method: 'price_low', 'vol_low', 'vol_v_shape'
    param_list: price_low/vol_low 為 n_days 列表；vol_v_shape 為 prominence 列表
    """
    base_dir = OUTPUT_DIR / method
    base_dir.mkdir(parents=True, exist_ok=True)

    for p in param_list:
        if method in ("price_low", "vol_low"):
            indices = find_market_indices(df, method=method, n_days=p)
            title_prefix = f"{method}_win{p}"
        else:
            indices = find_market_indices(df, method=method, prominence=p, window=10)
            title_prefix = f"{method}_prom{p}"

        print(f"Processing {method} with param {p}, found {len(indices)} points...")

        for i, idx in enumerate(indices):
            if idx < offset or idx > len(df) - offset:
                continue
            plot_and_save(df, idx, offset, f"{title_prefix}_pt{i}", base_dir)


def batch_save_drop_signals(df, offset=20):
    """連續下跌與區間回檔訊號的批次存圖"""
    for n in [5, 10, 15]:
        indices = find_consecutive_drops(df, n_days=n)
        save_dir = OUTPUT_DIR / "consecutive_drop" / f"n{n}"
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Consecutive {n} days drop: found {len(indices)} points.")
        for i, idx in enumerate(indices):
            if idx < offset or idx > len(df) - offset:
                continue
            plot_and_save(df, idx, offset, f"Consecutive_{n}d_Drop_pt{i}", save_dir)

    params = [(w, t) for w in (5, 10, 20, 30) for t in (0.05, 0.1, 0.2)]
    for win, thr in params:
        indices = find_price_drawdowns(df, window=win, threshold=thr)
        save_dir = OUTPUT_DIR / "drawdown" / f"win{win}_thr{int(thr * 100)}"
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Drawdown (Win:{win}, Thr:{thr}): found {len(indices)} points.")
        for i, idx in enumerate(indices):
            if idx < offset or idx > len(df) - offset:
                continue
            plot_and_save(df, idx, offset, f"Drawdown_{win}d_{int(thr * 100)}pct_pt{i}", save_dir)


def batch_save_combined_plots(df, n_days_list, prominence_list, window=5, tolerance=3, offset=30):
    """組合訊號（價格低點 + 成交量 V 型）的批次存圖"""
    method = "combined"
    base_dir = OUTPUT_DIR / method
    base_dir.mkdir(parents=True, exist_ok=True)

    for n_days in n_days_list:
        for prom in prominence_list:
            indices = find_market_indices(df, method=method, n_days=n_days, prominence=prom,
                                          window=window, tolerance=tolerance)
            title_prefix = f"{method}_ndays{n_days}_prom{prom}_tol{tolerance}"
            print(f"Processing {method} (n_days={n_days}, prom={prom}), found {len(indices)} points...")

            for i, idx in enumerate(indices):
                if idx < offset or idx > len(df) - offset:
                    continue
                plot_and_save(df, idx, offset, f"{title_prefix}_pt{i}", base_dir)


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------
def main():
    df = add_mid(load_data())
    df_adj = to_adjusted_ohlcv(df)

    # 基本走勢
    plot_kline(df_adj)
    plot_kline(df_adj.iloc[-50:], title="Adjusted Stock Price (Last 50)")
    plot_adj_close_trend(df)

    # 低點分析
    plot_monthly_lows(df)
    min_indices = plot_group_lows_with_volume(df, n_days=100)
    print(df.iloc[min_indices])
    plot_focus_around_index(df, min_indices[0], offset=50)

    # 訊號偵測（畫圖）
    lows = find_market_indices(df, method="price_low", n_days=100)
    plot_market_analysis(df, lows, title="100-Day Price Lows Analysis")

    v_shapes = find_market_indices(df, method="vol_v_shape", prominence=0.2)
    if len(v_shapes) > 0:
        plot_market_analysis(df, v_shapes, title="Focus View: Volume V-Shape Turnaround",
                             window_size=60, focus_idx=v_shapes[-1])

    combined_lows = find_market_indices(df, method="combined", n_days=60, prominence=0.2,
                                        window=5, tolerance=3)
    plot_market_analysis(df, combined_lows, title="Combined: 60-Day Price Low + Vol V-Shape")

    v_shapes_strict = find_market_indices(df, method="vol_v_shape", prominence=0.3, window=30)
    if len(v_shapes_strict) > 1:
        plot_market_analysis(df, v_shapes_strict, title="Focus View: Volume V-Shape Turnaround (Strict)",
                             window_size=20, focus_idx=v_shapes_strict[1])

    # 批次存圖
    if RUN_BATCH_SAVE:
        batch_save_plots(df, "price_low", [10, 50, 100])
        batch_save_plots(df, "vol_low", [10, 50, 100])
        batch_save_plots(df, "vol_v_shape", [0.2, 0.5, 0.8])
        batch_save_drop_signals(df)
        batch_save_combined_plots(df, n_days_list=[60, 100], prominence_list=[0.2, 0.5])
        print(f"所有分析圖已儲存至 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
