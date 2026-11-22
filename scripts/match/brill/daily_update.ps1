# Daily update of one replayed model's Brill board table and leaderboard page.
#
#   powershell -File ben\scripts\match\brill\daily_update.ps1 [-Model prodnow] [-Conf /var/lib/ben/brill/conf/prodnow.conf]
#       [-Name ...] [-Engine checkout|image] [-Workers 8] [-NoPublish] [-SkipRebuild]
#
# Defaults since 2026-11-22: prod's bidding (the self-play 012 network with the two rules), run from
# this checkout. The page's model until then: -Model old -Conf /var/lib/ben/ab/conf/old.conf
# -Name GIB-BBO-8730-base -Engine image
#
# Brill's events close at 06:00; run this after that. Steps:
#   1. fetch the events that closed since the last run (fetch_brill_field.py adds only new ones)
#   2. have the model bid the new boards as South (brill_replay.py --resume, in a
#      temporary container of the BEN image, with this folder's scripts mounted).
#      Events Morty AI played itself are not bid: their own table is the result.
#      -Engine image runs the image's BEN code; -Engine checkout (default) mounts
#      this BEN checkout at /ben (and the auction model's code and runs at /code
#      and /data, as the self-play farm container does), for settings that need
#      code or networks newer than the image, such as prod's of 2026-11-22.
#   3. rebuild_board_table.ps1: Brill's autoplay of the new contracts, the
#      single-dummy probabilities, the board table and the page; with -Publish
#      unless -NoPublish is given
# One run at a time (a lock file). It waits for any other brill_autoplay.py to
# end first: the Brill service gets one request at a time.
# Each run writes data\brill-replay\daily\<date_time>.log and one line in
# data\brill-replay\daily\history.log. Exit code 1 when a step failed.
param(
    [string]$Model = "prodnow",
    [string]$Conf = "/var/lib/ben/brill/conf/prodnow.conf",
    [string]$Name = "self-play 012 bidding with the jump and 2NT-to-3NT rules (prod 2026-11-22)",
    [string]$Image = "ben:latest",
    [ValidateSet("image", "checkout")]
    [string]$Engine = "checkout",
    [string]$AuctionModelData = "E:\bridge\data\auction-model",
    [int]$Workers = 8,
    [switch]$NoPublish,
    [switch]$SkipRebuild
)
$ErrorActionPreference = "Stop"
$src = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$python = Join-Path $src "elo\.venv\Scripts\python.exe"
$scripts = $PSScriptRoot
$data = Join-Path $src "data\brill-replay"
$daily = Join-Path $data "daily"
$container = "ben-brill-daily"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
Set-Location $src
if (-not (Test-Path -LiteralPath $daily)) { New-Item -ItemType Directory -Path $daily | Out-Null }

$started = Get-Date
$lock = Join-Path $daily "running.lock"
if (Test-Path -LiteralPath $lock) {
    $other = (Get-Content -LiteralPath $lock -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($other -match '^\d+$' -and (Get-Process -Id ([int]$other) -ErrorAction SilentlyContinue)) {
        Add-Content -LiteralPath (Join-Path $daily "history.log") -Value ("{0:yyyy-MM-dd HH:mm:ss} skipped: run {1} is still going" -f $started, $other) -Encoding utf8
        exit 0
    }
}
Set-Content -LiteralPath $lock -Value $PID
Start-Transcript -Path (Join-Path $daily ("{0:yyyy-MM-dd_HHmm}.log" -f $started)) -Force | Out-Null

function Say([string]$Text) { Write-Host ("{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date), $Text) }
function Run-Python([string]$What, [string[]]$Arguments) {
    Say $What
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$What failed with exit code $LASTEXITCODE" }
}
function Stop-Container {
    # wslc writes to stderr when there is nothing to stop; that must not end the run
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & wslc stop $container *> $null } catch { }
    $ErrorActionPreference = $previous
}
# boards of the field file without a good line in the model's replay
$countMissing = @'
import json, sys
field, replay = sys.argv[1:3]
done = set()
try:
    for line in open(replay, encoding='utf-8'):
        r = json.loads(line)
        if 'error' not in r:
            done.add((r['tournament_id'], r['board']))
except FileNotFoundError:
    pass
print(sum(1 for line in open(field, encoding='utf-8') for b in [json.loads(line)] if (b['tournament_id'], b['board']) not in done))
'@
# the boards to bid: those of the field file outside the events Morty AI played itself
$boardsToBid = @'
import json, sys
scripts, field, target = sys.argv[1:4]
sys.path.insert(0, scripts)
from brill_autoplay import entered_events
boards = [json.loads(line) for line in open(field, encoding='utf-8')]
entered = entered_events(boards)
with open(target, 'w', encoding='utf-8', newline='\n') as f:
    for b in boards:
        if b['tournament_id'] not in entered:
            f.write(json.dumps(b) + '\n')
print(f"{len(entered)} events played by Morty AI itself are not bid")
'@
# the replay file with one line per board: the last good one, else the last error
$tidyReplay = @'
import json, os, sys
source, target = sys.argv[1:3]
rows = {}
for line in open(source, encoding='utf-8'):
    r = json.loads(line)
    key = (r['tournament_id'], r['board'])
    if 'error' not in r or key not in rows or 'error' in rows[key]:
        rows[key] = r
with open(target + '.new', 'w', encoding='utf-8') as f:
    for r in rows.values():
        f.write(json.dumps(r) + '\n')
os.replace(target + '.new', target)
print(f"{len(rows)} boards in {target}, {sum(1 for r in rows.values() if 'error' in r)} with an error")
'@

$result = "ok"
$newBoards = 0
try {
    Say "start: model $Model"
    $field = Join-Path $data "field.jsonl"
    $replay = Join-Path $data "runs\$Model.jsonl"
    Run-Python "fetch new events" @("$scripts\fetch_brill_field.py", $field)
    $toBid = Join-Path $daily "field-to-bid.jsonl"
    $boardsToBid | & $python - $scripts $field $toBid
    if ($LASTEXITCODE -ne 0) { throw "listing the boards to bid failed with exit code $LASTEXITCODE" }

    $newBoards = [int]($countMissing | & $python - $toBid $replay)
    Say "$newBoards boards to bid"
    if ($newBoards -gt 0) {
        $work = Join-Path $data "runs\$Model.replay"
        if (-not (Test-Path -LiteralPath $work)) { New-Item -ItemType Directory -Path $work | Out-Null }
        if (Test-Path -LiteralPath $replay) { Copy-Item -LiteralPath $replay -Destination (Join-Path $work "replay.jsonl") -Force }
        Stop-Container      # one left over from a run that was cut short
        Say "starting container $container from $Image (engine: $Engine)"
        if ($Engine -eq "checkout") {
            $mounts = @("-v", "$src\ben:/ben", "-v", "$src\auction-model:/code", "-v", "${AuctionModelData}:/data")
            $workdir = "/ben/src"
            $replayScript = "/ben/scripts/match/brill/brill_replay.py"
        } else {
            $mounts = @("-v", "$src\ben\scripts\match:/opt/ben/scripts/match")
            $workdir = "/opt/ben/src"
            $replayScript = "/opt/ben/scripts/match/brill/brill_replay.py"
        }
        & wslc run -d --rm --name $container -m 32G -v "ben-state:/var/lib/ben" @mounts `
            -v "${data}:/brilldata" -w $workdir --entrypoint sleep $Image infinity | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "wslc run failed with exit code $LASTEXITCODE" }
        try {
            & wslc exec $container python3 $replayScript --conf $Conf --field /brilldata/daily/field-to-bid.jsonl `
                --out "/brilldata/runs/$Model.replay" --all-nodes --resume --workers $Workers
            $replayExit = $LASTEXITCODE
        } finally {
            Stop-Container
        }
        Say "tidy the replay file"
        $tidyReplay | & $python - (Join-Path $work "replay.jsonl") $replay
        if ($LASTEXITCODE -ne 0) { throw "tidying the replay file failed with exit code $LASTEXITCODE" }
        if ($replayExit -ne 0) { throw "brill_replay.py failed on some boards (exit code $replayExit)" }
        $left = [int]($countMissing | & $python - $toBid $replay)
        if ($left -gt 0) { throw "$left boards still have no replay" }
    }

    if (-not $SkipRebuild) {
        $waited = $false
        while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'brill_autoplay\.py' }) {
            if (-not $waited) { Say "another brill_autoplay.py is running; waiting for it"; $waited = $true }
            Start-Sleep -Seconds 60
        }
        Say "rebuild the board table and the page"
        & (Join-Path $scripts "rebuild_board_table.ps1") -Model $Model -Name $Name -Publish:(-not $NoPublish)
    }
} catch {
    $result = "FAILED: $($_.Exception.Message)"
    Say $result
} finally {
    $line = "{0:yyyy-MM-dd HH:mm:ss} elapsed {1:hh\:mm\:ss} new boards {2} {3}" -f $started, ((Get-Date) - $started), $newBoards, $result
    Add-Content -LiteralPath (Join-Path $daily "history.log") -Value $line -Encoding utf8
    Say "end: $result"
    Remove-Item -LiteralPath $lock -Force -ErrorAction SilentlyContinue
    Stop-Transcript | Out-Null
}
if ($result -ne "ok") { exit 1 }
