# LoRA arm, pre-registered in notes/prereg-maude-lora-2026-10-09.md: the zero-shot
# size control (Qwen3-4B prompted), then Qwen3-4B-Base + LoRA as a class-balanced
# classifier at 10k (seed 42, lr grid), 100k (lr grid), 10k seeds 43 and 44, an
# interim score, full N, and the final score. Finished steps are skipped and an
# interrupted LoRA run resumes from its hourly checkpoint, so rerunning continues.
$ErrorActionPreference = 'Continue'
# PowerShell 5.1 writes UTF-16 for redirection by default.
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force ..\runs\maude2025 | Out-Null
$log = "..\runs\maude2025\driver_lora.log"
$py = "$env:USERPROFILE\miniconda3\envs\maintlog\python.exe"
$env:PYTHONUNBUFFERED = '1'
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'
$env:HF_HUB_OFFLINE = '1'  # every weight file is already local and pinned

# Keep the machine awake while this process lives; nothing persists after it.
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null

function Step($name, [string[]]$argv) {
    "== $name $(Get-Date -Format o)" *>> $log
    & $py @argv *>> $log
    if ($LASTEXITCODE -ne 0) { "${name}_FAILED rc=$LASTEXITCODE" *>> $log }
}

# A LoRA run is tried up to three times; each retry resumes from its checkpoint.
function LoraStep($size, $seed) {
    $meta = "..\runs\maude2025\encoder_cb-lora_${size}_seed${seed}_meta.json"
    for ($try = 1; $try -le 3; $try++) {
        if (Test-Path $meta) { return }
        Step "lora_${size}_seed${seed}_try$try" @('maude_lora.py', $size, $seed)
    }
}

function Collapsed($size) {
    $n = 0
    foreach ($f in Get-ChildItem "..\runs\maude2025\encoder_cb-lora_${size}_seed*_meta.json" -ErrorAction SilentlyContinue) {
        if ((Get-Content $f.FullName -Raw | ConvertFrom-Json).collapsed) { $n++ }
    }
    $n
}

"START $(Get-Date -Format o)" *>> $log
# Size control (descriptive): the same 4B release, prompted. The second call retries errored rows only.
Step 'zs_qwen3_4b' @('openweight_llm.py', 'qwen3_4b', 'maude2025_cls')
Step 'zs_qwen3_4b_retry' @('openweight_llm.py', 'qwen3_4b', 'maude2025_cls')
LoraStep '10k' 42
LoraStep '100k' 42
foreach ($s in 43, 44) { LoraStep '10k' $s }
# A collapsed 10k seed (after the fallback lr) is replaced by the next seed, at most twice.
$extra = 45
while (((Collapsed '10k') -gt ($extra - 45)) -and ($extra -le 46)) {
    LoraStep '10k' $extra
    $extra++
}
Step 'score_partial' @('maude_lora_score.py')
"PARTIAL_DONE $(Get-Date -Format o)" *>> $log
LoraStep 'full' 42
Step 'score' @('maude_lora_score.py')
"ALL_DONE $(Get-Date -Format o)" *>> $log
