"""最高點落在前 MIN_BAR_POS 根 1 分 K 的 frame：列出來，之後的分析都排除。

輸出（與本檔同資料夾）：
    frame_max_bar.csv   每個 frame 一列（lib/frame_filter.max_bar_table），加上 excluded 欄
    excluded.csv        被排除的 frame，加上 last_close（資料最後收盤）、drop_pts（最高價 − 最後收盤）

執行（在專案根目錄）：python analysis/2026-10-24_max_early_frames/early_max.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'lib'))
from backtest_report import load_frames  # noqa: E402
from frame_filter import MIN_BAR_POS, max_bar_table  # noqa: E402


def main():
    df_list, key_list = load_frames()
    t = max_bar_table(df_list, key_list)
    t['excluded'] = t['bar_pos'] < MIN_BAR_POS
    t.to_csv(HERE / 'frame_max_bar.csv', index=False)

    e = t[t['excluded']].copy()
    e['last_close'] = [df_list[i]['close'].iloc[-1] for i in e['df_idx']]
    e['drop_pts'] = e['max_price'] - e['last_close']
    e.to_csv(HERE / 'excluded.csv', index=False)

    print(f'{len(t)} 個 frame，最高點在前 {MIN_BAR_POS} 根 K → 排除 {len(e)} 個，剩 {len(t) - len(e)} 個')
    print('bar_pos 0~9 的 frame 數：', t['bar_pos'].value_counts().sort_index().loc[:9].to_dict())
    print(f'排除的 frame 中，最高點在 frame_start 之前：{int(e["before_frame_start"].sum())} / {len(e)}')
    print(f'排除的 frame：最高價到資料最後收盤，中位數跌 {e["drop_pts"].median():.0f} 點')
    print('排除的 df_idx：', e['df_idx'].tolist())


if __name__ == '__main__':
    main()
