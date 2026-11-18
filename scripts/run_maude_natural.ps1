# Natural-distribution 2025 sample (notes/prereg-maude-natural-2026-10-23.md), GPU part:
# encoder predictions from the saved weights, then the open models' zero-shot and
# retrieval runs on the sample. Launch after run_maude_selector_open.ps1 has written
# RAG2024_DONE (the llama.cpp server holds the GPU). Finished rows are skipped.
$ErrorActionPreference = 'Continue'
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force ..\runs\maude2025nat | Out-Null
$log = "..\runs\maude2025nat\driver_natural.log"
$py = "$env:USERPROFILE\miniconda3\envs\maintlog\python.exe"
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

"START $(Get-Date -Format o)" *>> $log
# Wait for the open-model replication driver to release the GPU.
while (-not (Select-String -Path ..\runs\maude\driver_selector_open.log -Pattern 'RAG2024_DONE' -Quiet)) {
    Start-Sleep -Seconds 120
}
"GPU_FREE $(Get-Date -Format o)" *>> $log
Step 'enc_large' @('maude_encoder_predict.py', 'cb-large_full_seed42', 'natural')
Step 'enc_base' @('maude_encoder_predict.py', 'cb_full_seed42', 'natural')
foreach ($mk in 'qwen38_27b', 'gemma4_31b') {
    LlmStep "nat_$mk" @('openweight_llm.py', $mk, 'maude2025nat_cls', 'maude2025nat_ragfull')
}
"NATURAL_DONE $(Get-Date -Format o)" *>> $log
