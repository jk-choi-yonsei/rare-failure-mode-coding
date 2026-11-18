# Open-model replication of the selector (notes/prereg-maude-selector-open-2026-10-23.md):
# retrieval few-shot classification of the known 2024 test items for Qwen3.8-27B and
# Gemma 4 31B on the local llama.cpp server. Finished rows are skipped, so rerunning
# continues; each LLM step runs twice (the second call retries errored rows only).
# The selector fit and the 2025 scoring are run by hand afterwards (fit, commit, score).
$ErrorActionPreference = 'Continue'
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force ..\runs\maude | Out-Null
$log = "..\runs\maude\driver_selector_open.log"
$py = "$env:USERPROFILE\miniconda3\envs\maintlog\python.exe"
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'

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
foreach ($mk in 'qwen38_27b', 'gemma4_31b') {
    LlmStep "rag2024_$mk" @('openweight_llm.py', $mk, 'maude_ragfull')
}
"RAG2024_DONE $(Get-Date -Format o)" *>> $log
