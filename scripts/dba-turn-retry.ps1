$ErrorActionPreference = 'Continue'
$dbaRoot = $env:DBA_HOME
if (-not $dbaRoot) { $dbaRoot = 'D:\AI\DBA' }
$script = Join-Path $dbaRoot 'scripts\nb_turn.py'
& py -3.14 -B $script retry 2>&1 | Out-Null
exit $LASTEXITCODE
