# FAA SDR replication (notes/prereg-sdr-2026-10-24.md), GPU part. Runs the MAUDE pipeline
# under MAINT_CORPUS=sdr: class-balanced encoders with the 2024-test and 2025-test
# predictions (eval2025), then the open models' zero-shot and retrieval runs on both tests.
# Waits for the natural-sample driver to release the GPU (NATURAL_DONE). Finished steps are
# skipped, so rerunning continues. Order follows the pre-registration's priority if time
# runs short: CB-base, open LLM 2025 cells, 2024 cells, CB-large.
# -NoWait: start at once (the GPU is known to be free) instead of waiting for NATURAL_DONE.
param([switch]$NoWait)
$ErrorActionPreference = 'Continue'
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force ..\runs\sdr, ..\runs\sdr2025 | Out-Null
$log = "..\runs\sdr\driver_sdr.log"
$py = "$env:USERPROFILE\miniconda3\envs\maintlog\python.exe"
$env:MAINT_CORPUS = 'sdr'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'

Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null

function Step($name, [string[]]$argv) {
    "== $name $(Get-Date -Format o)" *>> $log
    & $py @argv *>> $log
    if ($LASTEXITCODE -ne 0) { "${name}_FAILED rc=$LASTEXITCODE" *>> $log }
}
function LlmStep($name, [string[]]$argv) {
    Step $name $argv
    Step "${name}_retry" $argv
}

"START $(Get-Date -Format o) corpus=$env:MAINT_CORPUS" *>> $log
while (-not $NoWait -and (-not (Test-Path ..\runs\maude2025nat\driver_natural.log) -or
       -not (Select-String -Path ..\runs\maude2025nat\driver_natural.log -Pattern 'NATURAL_DONE' -Quiet))) {
    Start-Sleep -Seconds 120
}
"GPU_FREE $(Get-Date -Format o)" *>> $log
foreach ($s in 42, 43, 44) { Step "cb_1k_seed$s" @('maude_encoder.py', 'cb', '1k', $s, 'eval2025') }
foreach ($s in 42, 43, 44) { Step "cb_10k_seed$s" @('maude_encoder.py', 'cb', '10k', $s, 'eval2025') }
Step 'cb_full_seed42' @('maude_encoder.py', 'cb', 'full', 42, 'eval2025')
"ENCODERS_DONE $(Get-Date -Format o)" *>> $log
foreach ($mk in 'qwen38_27b', 'gemma4_31b') {
    LlmStep "llm2025_$mk" @('openweight_llm.py', $mk, 'maude2025_cls', 'maude2025_ragfull')
}
foreach ($mk in 'qwen38_27b', 'gemma4_31b') {
    LlmStep "llm2024_$mk" @('openweight_llm.py', $mk, 'maude_cls', 'maude_ragfull')
}
"LLM_DONE $(Get-Date -Format o)" *>> $log
Step 'large_10k_seed42' @('maude_encoder.py', 'cb', '10k', 42, 'eval2025', 'large')
Step 'large_full_seed42' @('maude_encoder.py', 'cb', 'full', 42, 'eval2025', 'large')
"SDR_DONE $(Get-Date -Format o)" *>> $log
