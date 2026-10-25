# Build / update the Python environment on this computer (run from anywhere):
#   powershell -ExecutionPolicy Bypass -File tools\setup.ps1
# Creates .venv (Python 3.13 / 3.12 / 3.11 via the py launcher) if missing, installs requirements.txt,
# then checks the packages and, if data\ is present, that the 242 analysis frames load.
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$venvPy = Join-Path $root '.venv\Scripts\python.exe'

if (-not (Test-Path $venvPy)) {
    $launcher = (Get-Command py -ErrorAction SilentlyContinue).Source
    if (-not $launcher) { $launcher = Join-Path $env:LOCALAPPDATA 'Programs\Python\Launcher\py.exe' }
    if (-not (Test-Path $launcher)) { throw 'Python launcher (py.exe) not found. Install Python 3.11-3.13 from python.org first.' }
    $list = (& $launcher --list) | Out-String
    $ver = @('3.13', '3.12', '3.11') | Where-Object { $list -match "-V:$([regex]::Escape($_))\b" } | Select-Object -First 1
    if (-not $ver) { throw "No Python 3.11-3.13 found. Installed:`n$list" }
    Write-Host "Creating .venv with Python $ver"
    & $launcher "-$ver" -m venv .venv
}

& $venvPy -m pip install --upgrade pip -q
& $venvPy -m pip install -r requirements.txt -q
if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }

& $venvPy -c "import pandas, numpy, matplotlib, mplfinance, ipykernel; print('packages OK: pandas', pandas.__version__, '/ numpy', numpy.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw 'Package import failed. If the error says a DLL is blocked by an application control policy, Windows Smart App Control is blocking it (see README).'
}

& $venvPy -c @"
import sys; sys.path.insert(0, 'lib')
from backtest_report import DATA_PATH
if not DATA_PATH.exists():
    print('data/ not found yet: unzip the latest stock_drive_*.zip from Google Drive into the project folder, then rerun this script.')
else:
    from frame_filter import keep_frames
    print('data OK: analysis frames =', len(keep_frames()))
"@
