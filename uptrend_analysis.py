"""
上漲區間分析（收盤價）

1. forward_hits：每個交易日當起點，找出之後第一次收盤價漲幅 >= THRESHOLD 的日期
2. zigzag_swings：上漲 THRESHOLD 確認低點、回落 THRESHOLD 確認高點，
   輸出每一段「低點(起點) -> 追高點(終點)」的上漲波段

結果存到 results/，圖存為 PNG。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stock_analysis import DATA_PATH, load_data

THRESHOLD = 0.03
RESULT_DIR = Path(__file__).resolve().parent / "results"


def forward_hits(df, threshold=THRESHOLD, price_col="adjClose"):
    """每個起點 s：往後第一次 close[t] / close[s] - 1 >= threshold 的 t（沒有則為空）"""
    close = df[price_col].values
    dates = df["date"].values
    rows = []
    for s in range(len(df)):
        target = close[s] * (1 + threshold)
        hit = np.flatnonzero(close[s + 1:] >= target)
        if len(hit) == 0:
            rows.append((dates[s], close[s], pd.NaT, np.nan, np.nan, np.nan))
            continue
        e = s + 1 + hit[0]
        rows.append((dates[s], close[s], dates[e], close[e], close[e] / close[s] - 1, e - s))
    out = pd.DataFrame(rows, columns=["start_date", "start_close", "hit_date", "hit_close",
                                      "hit_return", "days_to_hit"])
    out.insert(0, "start_idx", np.arange(len(df)))
    return out


def zigzag_pivots(close, threshold=THRESHOLD):
    """回傳交替的轉折點 [(idx, 'low'|'high', confirmed)]

    從低點上漲 threshold 才確認低點；從高點回落 threshold 才確認高點。
    最後一個還沒被反向 threshold 確認的極值以 confirmed=False 附上。
    """
    n = len(close)
    lo = hi = 0
    direction = None  # 'up' 代表目前追蹤高點，'down' 代表目前追蹤低點
    pivots = []

    for t in range(1, n):
        if direction is None:
            if close[t] < close[lo]:
                lo = t
            if close[t] > close[hi]:
                hi = t
            if close[t] >= close[lo] * (1 + threshold):
                pivots.append((lo, "low", True))
                direction, hi = "up", t
            elif close[t] <= close[hi] * (1 - threshold):
                pivots.append((hi, "high", True))
                direction, lo = "down", t
        elif direction == "up":
            if close[t] > close[hi]:
                hi = t
            elif close[t] <= close[hi] * (1 - threshold):
                pivots.append((hi, "high", True))
                direction = "down"
                lo = hi + 1 + int(np.argmin(close[hi + 1:t + 1]))
        else:
            if close[t] < close[lo]:
                lo = t
            elif close[t] >= close[lo] * (1 + threshold):
                pivots.append((lo, "low", True))
                direction = "up"
                hi = lo + 1 + int(np.argmax(close[lo + 1:t + 1]))

    if direction == "up":
        pivots.append((hi, "high", False))
    elif direction == "down":
        pivots.append((lo, "low", False))
    return pivots


def zigzag_swings(df, threshold=THRESHOLD, price_col="adjClose"):
    """把轉折點整理成相鄰兩點的波段表（up = 低點 -> 高點，down = 高點 -> 低點）"""
    close = df[price_col].values
    pivots = zigzag_pivots(close, threshold)
    rows = []
    for (i0, k0, _), (i1, k1, confirmed) in zip(pivots, pivots[1:]):
        rows.append({
            "direction": "up" if k0 == "low" else "down",
            "start_idx": i0, "start_date": df["date"].iloc[i0], "start_close": close[i0],
            "end_idx": i1, "end_date": df["date"].iloc[i1], "end_close": close[i1],
            "return": close[i1] / close[i0] - 1,
            "days": i1 - i0,
            "end_confirmed": confirmed,
        })
    return pd.DataFrame(rows)


def plot_swings(df, swings, path, price_col="adjClose", title="Close with 3% zigzag"):
    ups = swings[swings["direction"] == "up"]
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.plot(df["date"], df[price_col], color="lightgray", linewidth=1, label=price_col)
    for _, r in ups.iterrows():
        seg = df.iloc[int(r.start_idx):int(r.end_idx) + 1]
        ax.plot(seg["date"], seg[price_col], color="tab:red", linewidth=1.8)
    ax.scatter(ups["start_date"], ups["start_close"], color="tab:green", s=30, zorder=5, label="start (low)")
    ax.scatter(ups["end_date"], ups["end_close"], color="tab:red", s=30, zorder=5, label="peak (high)")
    ax.set_title(title)
    ax.set_ylabel(price_col)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(df, threshold):
    tag = f"{round(threshold * 100)}pct"

    hits = forward_hits(df, threshold)
    hits.to_csv(RESULT_DIR / f"forward_hits_{tag}.csv", index=False)
    reached = hits["hit_date"].notna()
    print(f"[{tag} forward_hits] 起點 {len(hits)} 個，{reached.sum()} 個之後有達標，{(~reached).sum()} 個沒有")
    print(f"  達標天數：中位數 {hits['days_to_hit'].median():.0f}，平均 {hits['days_to_hit'].mean():.1f}，"
          f"最長 {hits['days_to_hit'].max():.0f}")

    swings = zigzag_swings(df, threshold)
    swings.to_csv(RESULT_DIR / f"zigzag_swings_{tag}.csv", index=False)
    ups = swings[swings["direction"] == "up"]
    print(f"[{tag} zigzag] 上漲波段 {len(ups)} 段、下跌波段 {(swings['direction'] == 'down').sum()} 段")
    print(f"  上漲波段漲幅：中位數 {ups['return'].median():.1%}，平均 {ups['return'].mean():.1%}，"
          f"最大 {ups['return'].max():.1%}；天數中位數 {ups['days'].median():.0f}")

    plot_swings(df, swings, RESULT_DIR / f"zigzag_{tag}.png", title=f"Close with {tag[:-3]}% zigzag")


def main(thresholds=(0.03, 0.05, 0.10)):
    RESULT_DIR.mkdir(exist_ok=True)
    df = load_data(DATA_PATH)
    for th in thresholds:
        run(df, th)
    print(f"結果已存到 {RESULT_DIR}")


if __name__ == "__main__":
    main()
