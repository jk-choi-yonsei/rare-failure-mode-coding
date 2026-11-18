# Follow-up experiments, pre-registered in notes/prereg-maude-rag-large-2026-10-09.md:
# the linear baseline, roberta-large class-balanced encoders, the few-shot
# (retrieved examples) LLM runs, scoring; then the descriptive LLM runs and
# scoring again. Finished steps are skipped, so rerunning continues.
$ErrorActionPreference = 'Continue'
# PowerShell 5.1 writes UTF-16 for redirection by default.
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force ..\runs\maude2025 | Out-Null
$log = "..\runs\maude2025\driver_extra.log"
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

# An LLM step runs twice: the second call retries only rows that errored (none -> no-op).
function LlmStep($name, [string[]]$argv) {
    Step $name $argv
    Step "${name}_retry" $argv
}

function Collapsed($size) {
    $n = 0
    foreach ($f in Get-ChildItem "..\runs\maude2025\encoder_cb-large_${size}_seed*_meta.json" -ErrorAction SilentlyContinue) {
        if ((Get-Content $f.FullName -Raw | ConvertFrom-Json).collapsed) { $n++ }
    }
    $n
}

"START $(Get-Date -Format o)" *>> $log
Step 'linear' @('maude_linear.py', '10k', '100k', 'full')
foreach ($s in 42, 43, 44) { Step "large_10k_seed$s" @('maude_encoder.py', 'cb', '10k', $s, 'eval2025', 'large') }
# A collapsed 10k seed (after the 1e-5 fallback) is replaced by the next seed, at most twice.
$extra = 45
while (((Collapsed '10k') -gt ($extra - 45)) -and ($extra -le 46)) {
    Step "large_10k_seed$extra" @('maude_encoder.py', 'cb', '10k', $extra, 'eval2025', 'large')
    $extra++
}
Step 'large_100k_seed42' @('maude_encoder.py', 'cb', '100k', 42, 'eval2025', 'large')
Step 'large_full_seed42' @('maude_encoder.py', 'cb', 'full', 42, 'eval2025', 'large')
foreach ($mk in 'qwen38_27b', 'gemma4_31b') {
    LlmStep "rag_$mk" @('openweight_llm.py', $mk, 'maude2025_rag10k', 'maude2025_ragfull')
}
Step 'score' @('maude_extra_score.py')
"CONFIRMATORY_DONE $(Get-Date -Format o)" *>> $log
# Descriptive runs: Qwen at 100k and the labels-only control, gpt-oss at full N.
LlmStep 'desc_qwen38_27b' @('openweight_llm.py', 'qwen38_27b', 'maude2025_rag100k', 'maude2025_labelsfull')
LlmStep 'desc_gptoss20b' @('openweight_llm.py', 'gptoss20b', 'maude2025_ragfull')
Step 'score_all' @('maude_extra_score.py')
"ALL_DONE $(Get-Date -Format o)" *>> $log
