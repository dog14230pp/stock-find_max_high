"""策略比較：對 registry.py 的策略跑起點回測（全部 290 frames），寫報告並更新比較總表。

輸出（reports/compare/）
    <編號>/detail.csv、frames.csv、index.html   單一策略的完整報告
    <編號>/plots/df_XXX.png                    每個 frame 的 K 線圖（★ 找到的點、◆ 最高點）
    summary.csv                                每個策略 × 組別（all / keep / excluded）一列；只更新這次跑的策略
    index.html                                 比較總表

執行（在專案根目錄）：
    python strategies/compare.py          跑 registry 裡全部策略
    python strategies/compare.py S1 S3    只跑指定的策略
    python strategies/compare.py S1 --no-plots   不畫圖（快很多）
"""
import sys
import time
import webbrowser
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
sys.path.insert(0, str(ROOT / 'strategies'))

import start_backtest as sb  # noqa: E402
from backtest_report import load_frames  # noqa: E402
from frame_filter import keep_frames, max_bar_table  # noqa: E402
from registry import STRATEGIES  # noqa: E402

OUT = ROOT / 'reports' / 'compare'


def main(ids, plots=True):
    df_list, key_list = load_frames()
    keep = set(keep_frames(table=max_bar_table(df_list, key_list)))
    path = OUT / 'summary.csv'
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    new = []
    for sid in ids:
        t0 = time.time()
        st = STRATEGIES[sid]
        detail, frames = sb.run(st['mask'], df_list, keep)
        out = OUT / sid
        out.mkdir(parents=True, exist_ok=True)
        detail.to_csv(out / 'detail.csv', index=False, encoding='utf-8-sig')
        frames.to_csv(out / 'frames.csv', index=False, encoding='utf-8-sig')
        if plots:
            sb.plot_all(df_list, key_list, st['mask'], out, detail)
        has_plots = (out / 'plots' / 'df_000.png').exists()
        (out / 'index.html').write_text(sb.strategy_html(sid, st['name'], st['desc'], detail, frames, has_plots),
                                        encoding='utf-8')
        s = sb.summary(detail, frames).reset_index()
        s.insert(0, 'id', sid)
        s.insert(1, 'name', st['name'])
        s.insert(2, 'desc', st['desc'])
        s['run_time'] = f'{pd.Timestamp.now():%Y-%m-%d %H:%M}'
        new.append(s)
        k = s.set_index('group')
        print(f'{sid}：Hit {k.at["keep", "hits"]:.0f} / 242、{k.at["all", "hits"]:.0f} / 290，'
              f'找到的點 {k.at["all", "marks"]:.0f}，虧損最大 {k.at["all", "loss_max"]:.0f}（{time.time() - t0:.0f} 秒）')
    if len(old):
        old = old[~old['id'].isin(ids) & old['id'].isin(STRATEGIES)]
    summ = pd.concat([old, *new], ignore_index=True)
    order = {sid: k for k, sid in enumerate(STRATEGIES)}
    summ = summ.sort_values('id', key=lambda c: c.map(order), kind='stable')
    summ.to_csv(path, index=False, encoding='utf-8-sig')
    (OUT / 'index.html').write_text(sb.compare_html(summ), encoding='utf-8')
    print(f'比較總表：{OUT / "index.html"}')
    return OUT / 'index.html'


if __name__ == '__main__':
    plots = '--no-plots' not in sys.argv
    ids = [a for a in sys.argv[1:] if not a.startswith('--')] or list(STRATEGIES)
    bad = [i for i in ids if i not in STRATEGIES]
    if bad:
        sys.exit(f'registry.py 沒有：{bad}；可用：{list(STRATEGIES)}')
    webbrowser.open(main(ids, plots).as_uri())
