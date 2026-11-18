# Rebuild the Brill board table of one replayed model (see data\brill-replay\README.md).
#
#   powershell -File ben\scripts\match\brill\rebuild_board_table.ps1 [-Model old] [-Name GIB-BBO-8730-base] [-WaitForPid 1234] [-Publish]
#
# Steps, each skipping what is already saved:
#   1. Brill's autoplay of the boards whose contract the Brill entrant did not play
#   2. single-dummy probabilities of new deals
#   3. the board table (boards.parquet, boards.csv, events.csv)
#   4. the page (brill-board-table.html, leader-board.html) and groups.csv
# -Publish     then put leader-board.html on the web server (the ops folder's
#              publish_brill_leaderboard.ps1, which is not part of this repository)
# -WaitForPid  first wait for a running brill_autoplay.py to end (two must not
#              run at once: the Brill service gets one request at a time).
param(
    [string]$Model = "old",
    [string]$Name = "GIB-BBO-8730-base",
    [int]$WaitForPid = 0,
    [switch]$Publish
)
$ErrorActionPreference = "Stop"
$src = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$python = Join-Path $src "elo\.venv\Scripts\python.exe"
$scripts = Join-Path $src "ben\scripts\match\brill"
$data = Join-Path $src "data\brill-replay"
$out = Join-Path $data "board-table"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
Set-Location $src

function Step([string]$What, [string[]]$Arguments) {
    Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $What"
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$What failed with exit code $LASTEXITCODE" }
}

Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') start"
if ($WaitForPid) {
    Write-Host "waiting for process $WaitForPid ..."
    Wait-Process -Id $WaitForPid -ErrorAction SilentlyContinue
}
Step "Brill autoplay" @("$scripts\brill_autoplay.py", "$data\field.jsonl", "$data\runs\$Model.jsonl", "$data\autoplay\$Model.jsonl")
Step "single-dummy probabilities" @("$scripts\brill_sd_probs.py", "$data\field.jsonl", "$out\sd.json", "--workers", "16")
Step "board table" @("$scripts\brill_board_table.py", "$data\field.jsonl", "$data\runs\$Model.jsonl", "$data\autoplay\$Model.jsonl", "$out\sd.json", $out)
Step "page" @("$scripts\brill_board_page.py", $out, $Name)
if ($Publish) {
    Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') publish"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $src "ops\publish_brill_leaderboard.ps1")
    if ($LASTEXITCODE -ne 0) { throw "publish failed with exit code $LASTEXITCODE" }
}
Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') end"
