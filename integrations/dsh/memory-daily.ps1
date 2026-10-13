[CmdletBinding()]
param(
    [switch]$ResolveKeyOnly
)

$ErrorActionPreference = 'Stop'

# ⭐ 2026-10-06：记忆数据与工具链都已搬到 DBA 仓库（D:\AI\DBA）。
#   本脚本留在 ~/.dsh/scripts 只是因为计划任务 DSH_Memory_Daily 指着它；
#   它现在只做一个转发：把工作目录切到 DBA 仓库再跑那边的脚本。
#   为什么不留一份副本在这里：两份 nb_*.py 迟早会漂移，而「数据跟着代码走」
#   是 DBA 的硬约定 —— 记忆数据必须只有一处，脚本也只有一处。
$DbaHome = if ($env:DBA_HOME) { $env:DBA_HOME } elseif (Test-Path 'D:\AI\DBA\scripts') { 'D:\AI\DBA' } else { $null }
if (-not $DbaHome) {
    throw "找不到 DBA 仓库。设 DBA_HOME 环境变量，或确认 D:\AI\DBA 存在。"
}
$ScriptsDirectory = Join-Path $DbaHome 'scripts'
$MemoryDirectory = Join-Path $DbaHome 'memory'
$env:DBA_HOME = $DbaHome

# ⚠️ cron.log **不放进 DBA 仓库**，留在 hermes 下的运行态目录。
#   理由与 mem_tasklog 一致：DBA 的 memory/ 只放三层记忆 + 索引，保持仓库干净；
#   日志/工作区/tasklog 都是运行态产物，跟记忆不是一类东西。
$HermesHome = if ($env:HERMES_HOME) { $env:HERMES_HOME } else { Join-Path $env:LOCALAPPDATA 'hermes' }
$LogPath = Join-Path (Join-Path $HermesHome 'memories') 'cron.log'
$VenvPython = Join-Path $HermesHome 'hermes-agent\venv\Scripts\python.exe'
$CredentialsPath = Join-Path $HOME '.dsh\.credentials.yaml'
$MaximumOutputCharacters = 2000

# ⚠️ 中文输出必须显式指定 UTF-8，否则日志里全是乱码。
#   Python 3 打印中文走 UTF-8，而 powershell.exe（5.1）默认按 OEM 代码页（本机
#   是 GBK）解码子进程输出，于是 `→` 变成 `鈫?`，中文变成 `鍙岃剳鏋舵瀯`。
#   乱码不只是难看：日志里「哪一步失败了」这类关键判断也会变得读不出来，
#   而日志存在的唯一理由就是在出故障时能读。所以这两行是必需的，不是修饰。
#   （PS 7+ 默认 UTF-8，无需这两行；这里用 powershell.exe 5.1 就必须设。）
$env:PYTHONIOENCODING = 'utf-8'
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
} catch {
    # 无控制台（计划任务后台跑）时会失败，属正常，不影响日志编码
}

if (-not (Test-Path -LiteralPath $MemoryDirectory -PathType Container)) {
    throw "Memory directory not found: $MemoryDirectory"
}

if (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
    $PythonCommand = $VenvPython
    $PythonPrefix = @()
}
else {
    $PythonCommand = 'py'
    $PythonPrefix = @('-3.14')
}

# --- API key resolution (2026-10-03) -----------------------------------------
# Order: new key (DEEPSEEK_API_KEY) -> old key (DEEPSEEK_API_KEY_OLD).
# The new key is validated with GET /v1/models; if the API rejects it with
# 401/402/403 the old key is used instead. Any other outcome keeps the new key
# so the cron never refuses to run because of a transient network problem.

function Get-CredentialValue {
    param([string]$Name)

    $Value = [Environment]::GetEnvironmentVariable($Name, 'Process')
    if (-not [string]::IsNullOrWhiteSpace($Value)) { return $Value.Trim() }

    if (-not (Test-Path -LiteralPath $CredentialsPath -PathType Leaf)) { return '' }
    foreach ($Line in [System.IO.File]::ReadAllLines($CredentialsPath)) {
        $CandidateLine = $Line.TrimStart([char]0xFEFF)
        if ($CandidateLine -match "^$([regex]::Escape($Name))\s*:\s*(.+?)\s*$") {
            return $Matches[1].Trim().Trim('"').Trim("'")
        }
    }
    return ''
}

function Get-DeepSeekKeyStatus {
    param([string]$Key)

    if ([string]::IsNullOrWhiteSpace($Key)) { return '000' }
    $Code = & curl.exe -s -o NUL -w '%{http_code}' --max-time 20 --noproxy '*' `
        -H "Authorization: Bearer $Key" 'https://api.deepseek.com/v1/models'
    if ($null -eq $Code) { return '000' }
    return $Code.Trim()
}

$NewKey = Get-CredentialValue 'DEEPSEEK_API_KEY'
$OldKey = Get-CredentialValue 'DEEPSEEK_API_KEY_OLD'

if ([string]::IsNullOrWhiteSpace($NewKey) -and [string]::IsNullOrWhiteSpace($OldKey)) {
    $ApiKey = ''
    $ApiKeySource = 'not found'
}
elseif ([string]::IsNullOrWhiteSpace($NewKey)) {
    $ApiKey = $OldKey
    $ApiKeySource = 'old key (DEEPSEEK_API_KEY not configured)'
}
else {
    $ApiKey = $NewKey
    $StatusCode = Get-DeepSeekKeyStatus $NewKey
    if ($StatusCode -in @('401', '402', '403')) {
        if ([string]::IsNullOrWhiteSpace($OldKey)) {
            $ApiKeySource = "new key (rejected HTTP $StatusCode, no old key configured)"
        }
        else {
            $ApiKey = $OldKey
            $ApiKeySource = "fallback to old key (new key rejected HTTP $StatusCode)"
        }
    }
    elseif ($StatusCode -eq '200') {
        $ApiKeySource = 'new key (validated)'
    }
    else {
        $ApiKeySource = "new key (unverified: HTTP $StatusCode)"
    }
}

if (-not [string]::IsNullOrWhiteSpace($ApiKey)) {
    $env:DEEPSEEK_API_KEY = $ApiKey
}
$env:PYTHONIOENCODING = 'utf-8'

if ($ResolveKeyOnly) {
    $Masked = if ($ApiKey.Length -gt 12) {
        $ApiKey.Substring(0, 8) + '...' + $ApiKey.Substring($ApiKey.Length - 4)
    }
    else { '(none)' }
    Write-Output "source: $ApiKeySource"
    Write-Output "key:    $Masked"
    exit 0
}

function Invoke-MemoryStep {
    param(
        [int]$Number,
        [string]$Script,
        [string[]]$ScriptArguments
    )

    $ExitCode = 1
    $CapturedOutput = @()

    Push-Location -LiteralPath $ScriptsDirectory
    try {
        $CapturedOutput = @(& $PythonCommand @PythonPrefix $Script @ScriptArguments 2>&1)
        if ($null -ne $LASTEXITCODE) {
            $ExitCode = [int]$LASTEXITCODE
        }
        else {
            $ExitCode = 0
        }
    }
    catch {
        $CapturedOutput = @("ERROR: $($_.Exception.Message)")
        $ExitCode = 1
    }
    finally {
        Pop-Location
    }

    $OutputText = ($CapturedOutput | ForEach-Object { $_.ToString() }) -join [Environment]::NewLine
    $WasTruncated = $OutputText.Length -gt $MaximumOutputCharacters
    if ($WasTruncated) {
        $OutputText = $OutputText.Substring($OutputText.Length - $MaximumOutputCharacters)
    }

    $DisplayArguments = $ScriptArguments -join ' '
    return [pscustomobject]@{
        Number = $Number
        Script = $Script
        Arguments = $DisplayArguments
        ExitCode = $ExitCode
        Output = $OutputText
        WasTruncated = $WasTruncated
    }
}

$RunTimestamp = Get-Date
# ⭐ 2026-10-06：原来的 STEP 3（mem_compress.py，L1→L2 级联压缩）已删除。
#   理由：DBA 的 stream/ws 不限长度，**没有分层压缩这一步**；该脚本在新架构下
#   只会打印 skip。而且它 import 的是旧五级配置 mem_config（没跟着搬到 DBA），
#   在 DBA/scripts 下会直接 ModuleNotFoundError —— 每天留一个必然失败的步骤
#   在日志里，会训练人忽略日志里的 FAIL。要回退五级架构时用 --legacy 单独跑。
$Results = @(
    Invoke-MemoryStep -Number 1 -Script 'mem_tasklog.py' -ScriptArguments @('compress')
    Invoke-MemoryStep -Number 2 -Script 'mem_tasklog.py' -ScriptArguments @('init')
    Invoke-MemoryStep -Number 3 -Script 'memory_maintenance.py' -ScriptArguments @()
)

$FailedCount = @($Results | Where-Object { $_.ExitCode -ne 0 }).Count
$SummaryStatus = if ($FailedCount -eq 0) { 'OK' } else { 'FAIL' }

$LogLines = New-Object System.Collections.Generic.List[string]
[void]$LogLines.Add('')
[void]$LogLines.Add("===== SnowFox memory daily run: $($RunTimestamp.ToString('yyyy-MM-dd HH:mm:ss zzz')) =====")
[void]$LogLines.Add("Python: $PythonCommand")
[void]$LogLines.Add("DEEPSEEK_API_KEY source: $ApiKeySource")

foreach ($Result in $Results) {
    $StepDescription = "$($Result.Script) $($Result.Arguments)".Trim()
    [void]$LogLines.Add("STEP $($Result.Number) ${StepDescription}: exit=$($Result.ExitCode)")
    if ([string]::IsNullOrEmpty($Result.Output)) {
        [void]$LogLines.Add('<no output>')
    }
    else {
        $OutputLabel = if ($Result.WasTruncated) { '<output tail, max 2000 chars>' } else { '<output>' }
        [void]$LogLines.Add($OutputLabel)
        [void]$LogLines.Add($Result.Output)
    }
}

[void]$LogLines.Add("SUMMARY: $SummaryStatus ($FailedCount of $($Results.Count) steps failed)")

$LogBlock = ($LogLines -join [Environment]::NewLine) + [Environment]::NewLine
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::AppendAllText($LogPath, $LogBlock, $Utf8NoBom)

$LogLines | ForEach-Object { Write-Output $_ }

if ($FailedCount -gt 0) {
    exit 1
}

exit 0
