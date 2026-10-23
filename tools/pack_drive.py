"""把不進 git 的檔案（原始資料 + 可重建的輸出）打包成 zip，給 Google Drive 搬到另一台電腦。

執行（在專案資料夾）：python tools/pack_drive.py
輸出：_transfer/stock_drive_YYYYMMDD.zip（_transfer/ 不進 git）
到另一台電腦：先 git pull，再把 zip 解壓縮到專案資料夾（選「取代」）
"""
import datetime
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ['data', 'marks', 'reports', 'analysis']
SKIP_DIRS = [ROOT / 'analysis' / '2026-10-21_type1_ratio' / 'frames']   # 舊的合併版，已被 sec/、min/ 取代


def main():
    out = ROOT / '_transfer' / f'stock_drive_{datetime.date.today():%Y%m%d}.zip'
    out.parent.mkdir(parents=True, exist_ok=True)
    n, raw = 0, 0
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for top in INCLUDE:
            if not (ROOT / top).exists():
                continue
            for p in sorted((ROOT / top).rglob('*')):
                if p.is_dir() or any(d in p.parents for d in SKIP_DIRS):
                    continue
                z.write(p, p.relative_to(ROOT).as_posix())
                n += 1
                raw += p.stat().st_size
    with zipfile.ZipFile(out) as z:
        bad = z.testzip()
    print(f'{n} 個檔案，原始 {raw / 1e6:.0f} MB → 壓縮後 {out.stat().st_size / 1e6:.0f} MB；'
          f'完整性檢查：{"OK" if bad is None else bad}')
    print(out)


if __name__ == '__main__':
    main()
