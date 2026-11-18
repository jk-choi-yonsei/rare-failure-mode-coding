# MAUDE full run, pre-registered in notes/prereg-maude-2026-10-07.md.
# Encoder arms first (the 100k CE seed-42 run fixes the full-N learning rate),
# then the open-weight LLM arms. Every step skips or resumes finished work, so
# rerunning this file after an interruption continues where it stopped.
$ErrorActionPreference = 'Continue'
# PowerShell 5.1 writes UTF-16 for redirection by default.
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force ..\runs\maude | Out-Null
$log = "..\runs\maude\driver.log"
$py = "$env:USERPROFILE\miniconda3\envs\maintlog\python.exe"
$env:PYTHONUNBUFFERED = '1'
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'

# Keep the machine awake while this process lives (ES_CONTINUOUS |
# ES_SYSTEM_REQUIRED); the request ends with the process, nothing persists.
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null

function Step($name, [string[]]$argv) {
    "== $name $(Get-Date -Format o)" *>> $log
    & $py @argv *>> $log
    if ($LASTEXITCODE -ne 0) { "${name}_FAILED rc=$LASTEXITCODE" *>> $log }
}

"START $(Get-Date -Format o)" *>> $log
foreach ($size in '1k', '10k', '100k') {
    foreach ($s in 42, 43, 44) { Step "ce_${size}_seed$s" @('maude_encoder.py', 'ce', $size, $s) }
}
foreach ($s in 42, 43, 44) { Step "ce_full_seed$s" @('maude_encoder.py', 'ce', 'full', $s) }
Step 'cb_full_seed42' @('maude_encoder.py', 'cb', 'full', 42)
# Pre-registered conditional: two more CB seeds only if CE seeds disagree.
$sd = & $py maude_score.py ce_seed_sd
"CE full-N seed SD of tail macro-F1: $sd" *>> $log
if ($sd -ne 'nan' -and [double]$sd -gt 0.01) {
    foreach ($s in 43, 44) { Step "cb_full_seed$s" @('maude_encoder.py', 'cb', 'full', $s) }
}
foreach ($mk in 'gptoss20b', 'qwen38_27b', 'gemma4_31b') {
    Step "llm_$mk" @('openweight_llm.py', $mk, 'maude_cls', 'maude_novel')
}
"ALL_DONE $(Get-Date -Format o)" *>> $log
