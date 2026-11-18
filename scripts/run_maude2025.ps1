# MAUDE 2025 out-of-time test and router evaluation, pre-registered in
# notes/prereg-maude-2025-router-2026-10-08.md. The class-balanced encoders are
# retrained with the committed recipe (weights kept this time), then the three
# local LLMs classify the 2025 items, then everything is scored. Finished steps
# are skipped, so rerunning this file after an interruption continues.
$ErrorActionPreference = 'Continue'
# PowerShell 5.1 writes UTF-16 for redirection by default.
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force ..\runs\maude2025 | Out-Null
$log = "..\runs\maude2025\driver.log"
$py = "$env:USERPROFILE\miniconda3\envs\maintlog\python.exe"
$env:PYTHONUNBUFFERED = '1'
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'

# Keep the machine awake while this process lives; nothing persists after it.
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null

function Step($name, [string[]]$argv) {
    "== $name $(Get-Date -Format o)" *>> $log
    & $py @argv *>> $log
    if ($LASTEXITCODE -ne 0) { "${name}_FAILED rc=$LASTEXITCODE" *>> $log }
}

"START $(Get-Date -Format o)" *>> $log
foreach ($size in '1k', '10k', '100k') {
    foreach ($s in 42, 43, 44) { Step "cb_${size}_seed$s" @('maude_encoder.py', 'cb', $size, $s, 'eval2025') }
}
Step 'cb_full_seed42' @('maude_encoder.py', 'cb', 'full', 42, 'eval2025')
foreach ($mk in 'gptoss20b', 'qwen38_27b', 'gemma4_31b') {
    Step "llm_$mk" @('openweight_llm.py', $mk, 'maude2025_cls')
}
Step 'score' @('maude_2025_score.py')
"ALL_DONE $(Get-Date -Format o)" *>> $log
