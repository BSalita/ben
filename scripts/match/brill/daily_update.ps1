# Daily update of the Brill board table and leaderboard pages, and the way to replay
# another model on the past events.
#
#   powershell -File ben\scripts\match\brill\daily_update.ps1 [-Model KEY] [-Conf /var/lib/ben/.../KEY.conf]
#       [-Name ...] [-Engine checkout|image] [-MakeDefault] [-SkipEntered] [-Workers 8] [-NoPublish] [-SkipRebuild]
#
# With no -Model: the default model of data\brill-replay\models.json (brill_views.py),
# with its conf, name and engine (since 2026-11-22: prodnow, prod's bidding of that day).
#
# Replaying a model on the previous sessions (an option, not the daily task's job):
#   daily_update.ps1 -Model KEY -Conf /var/lib/ben/<path>.conf -Name "what it is" [-Engine checkout|image] [-MakeDefault]
# bids every past board the model has not bid yet (all of them for a new KEY; about
# 2 s a board with 8 workers), registers KEY in models.json, plays the new contracts on
# Brill's autoplay (copying any other model's autoplay of the same auction) and
# rebuilds every page: KEY gets its own replay page, and the saved results of the
# models replayed before stay on theirs. -MakeDefault also makes KEY the model the main
# page shows before Morty AI entered and the one the daily task replays from then on.
#
# Brill's events close at 06:00; run this after that. Steps:
#   1. fetch the events that closed since the last run (fetch_brill_field.py adds only new ones)
#   2. have the model bid the new boards as South (brill_replay.py --resume, in a
#      temporary container of the BEN image, with this folder's scripts mounted).
#      Events Morty AI played itself are bid too (its replay page covers every event;
#      the main page uses its own table there); -SkipEntered leaves them out.
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
    [string]$Model = "",
    [string]$Conf = "",
    [string]$Name = "",
    [string]$Image = "ben:latest",
    [ValidateSet("", "image", "checkout")]
    [string]$Engine = "",
    [string]$AuctionModelData = "E:\bridge\data\auction-model",
    [int]$Workers = 8,
    [switch]$MakeDefault,
    [switch]$SkipEntered,
    [switch]$NoPublish,
    [switch]$SkipRebuild
)
$ErrorActionPreference = "Stop"
$src = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$python = Join-Path $src "elo\.venv\Scripts\python.exe"
$scripts = $PSScriptRoot
$data = Join-Path $src "data\brill-replay"
$daily = Join-Path $data "daily"

# The model: -Model, else the default of models.json; what is not given comes from its entry there.
$registryPath = Join-Path $data "models.json"
$registry = if (Test-Path -LiteralPath $registryPath) { Get-Content -LiteralPath $registryPath -Raw -Encoding UTF8 | ConvertFrom-Json } else { $null }
if (-not $Model) {
    if (-not $registry -or -not $registry.default) { throw "no -Model given and ${registryPath} names no default model" }
    $Model = $registry.default
}
$entry = if ($registry -and $registry.models.PSObject.Properties[$Model]) { $registry.models.$Model } else { $null }
if (-not $Conf) { $Conf = if ($entry) { $entry.conf } else { "" } }
if (-not $Name) { $Name = if ($entry) { $entry.name } else { $Model } }
if (-not $Engine) { $Engine = if ($entry -and $entry.engine) { $entry.engine } else { "checkout" } }
if (-not $Conf) { throw "model $Model is not in ${registryPath}: give -Conf (and -Name, -Engine)" }
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
# the boards to bid: those of the field file; with "skip", not those of the events Morty AI played itself
$boardsToBid = @'
import json, sys
scripts, field, target, skip = sys.argv[1:5]
sys.path.insert(0, scripts)
from brill_autoplay import entered_events
boards = [json.loads(line) for line in open(field, encoding='utf-8')]
entered = entered_events(boards) if skip == 'skip' else set()
with open(target, 'w', encoding='utf-8', newline='\n') as f:
    for b in boards:
        if b['tournament_id'] not in entered:
            f.write(json.dumps(b) + '\n')
if entered:
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
    Say "start: model $Model ($Name), conf $Conf, engine $Engine"
    $register = @("$scripts\brill_views.py", $data, "register", $Model, "--name", $Name, "--conf", $Conf, "--engine", $Engine)
    if ($MakeDefault) { $register += "--default" }
    & $python @register | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "registering model $Model failed with exit code $LASTEXITCODE" }
    $field = Join-Path $data "field.jsonl"
    $replay = Join-Path $data "runs\$Model.jsonl"
    Run-Python "fetch new events" @("$scripts\fetch_brill_field.py", $field)
    $toBid = Join-Path $daily "field-to-bid.jsonl"
    $boardsToBid | & $python - $scripts $field $toBid $(if ($SkipEntered) { "skip" } else { "all" })
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
