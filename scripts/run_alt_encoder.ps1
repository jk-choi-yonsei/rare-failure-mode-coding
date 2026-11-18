# Another encoder under the agreement rule, aircraft reports (SDR) first (2026-11-18; the user set the start for 18:20).
# Step 0, added the same day at the user's request: time per report of the classifier and the LLM coders (cost question).
# Started by a one-time Windows Task Scheduler task, so a restart of the Claude app does not end it. Every step is safe
# to repeat: finished runs, encoder parts and LLM rows are skipped. A step waits while the GPU is busy with something
# else and retries up to three times.
#   1  train the other encoder with the paper's recipe (maude_encoder.py: 10k runs for the learning rate, then full
#      size until the validation score stops rising). The model must already be in the local Hugging Face cache
#      (no download here). If it is missing or its training fails, fall back to a second seed of the paper's encoder.
#   2  pass it over every known-code report of 2026 and 2025 (maude_encoder_year.py)
#   3  shortlist and neighbours of the reports that have no LLM answer yet (rare_year.py, in runs/sdr<year>short_<name>)
#   4  Qwen and Gemma, zero-shot and retrieval, on those reports (openweight_llm.py)
#   5  score on the CPU (rare_year.py score, alt_encoder_check.py -> results/alt_encoder/sdr_<name>.json)
param([string]$Model = 'answerdotai/ModernBERT-base', [string]$Name = 'mbert', [string]$Corpus = 'sdr')
$ErrorActionPreference = 'Continue'
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
Set-Location $PSScriptRoot
$log = "..\runs\alt_encoder.log"
$py = "$env:USERPROFILE\miniconda3\envs\maintlog\python.exe"
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'          # nothing is downloaded by this run
$env:TRANSFORMERS_OFFLINE = '1'
$env:MAINT_CORPUS = $Corpus
$env:MAINT_SELECTOR_ENC = $(if ($Corpus -eq 'sdr') { 'cb-conv' } else { 'cb-large-conv' })
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

function GpuBusy {
    $q = (& nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits) -split ','
    $mem = [int]$q[0].Trim(); $util = [int]$q[1].Trim()
    return ($mem -gt 6000) -or ($util -gt 50)
}
function WaitGpu {
    $idle = 0
    while ($idle -lt 3) {
        if (GpuBusy) {
            $idle = 0
            "WAITING_FOR_GPU $(Get-Date -Format o) $(& nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" *>> $log
            Start-Sleep -Seconds 300
        } else { $idle++; Start-Sleep -Seconds 20 }
    }
}
function Launch([string]$argline, [string]$out, [string]$err) {
    $p = Start-Process -FilePath $py -ArgumentList $argline -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
         -RedirectStandardOutput $out -RedirectStandardError $err
    $null = $p.Handle  # cache the handle, or ExitCode reads as empty once the process has exited
    Start-Sleep -Seconds 2
    try { $p.PriorityClass = 'BelowNormal' } catch {}
    return $p
}
function Step([string]$name, [string]$argline) {
    for ($try = 1; $try -le 3; $try++) {
        WaitGpu
        "== $name try $try $(Get-Date -Format o) : $argline" *>> $log
        $out = "..\runs\alt_encoder_step.out"; $err = "..\runs\alt_encoder_step.err"
        $p = Launch $argline $out $err
        $p.WaitForExit()
        Get-Content $out -Tail 14 *>> $log
        Get-Content $err -Tail 6 *>> $log
        if ($p.ExitCode -eq 0) { "${name}_OK $(Get-Date -Format o)" *>> $log; return $true }
        "${name}_FAILED rc=$($p.ExitCode) try $try $(Get-Date -Format o)" *>> $log
        Start-Sleep -Seconds 120
    }
    "${name}_GAVE_UP $(Get-Date -Format o)" *>> $log
    return $false
}
function CpuStep([string]$name, [string]$argline) {
    "== $name (cpu) $(Get-Date -Format o) : $argline" *>> $log
    $out = "..\runs\alt_encoder_cpu.out"; $err = "..\runs\alt_encoder_cpu.err"
    $p = Launch $argline $out $err
    $p.WaitForExit()
    Get-Content $out -Tail 40 *>> $log
    Get-Content $err -Tail 6 *>> $log
    "${name}_END rc=$($p.ExitCode) $(Get-Date -Format o)" *>> $log
}

"START $(Get-Date -Format o) corpus $Corpus model $Model name $Name; GPU $(& nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" *>> $log

# ---- 0 (added 2026-11-18): time per report of the classifier and of both LLM coders, both streams, first, while the
#      GPU is otherwise idle (compute_time.py -> results/compute_time/<corpus>_<phase>.json). A failed timing step is
#      logged and the run goes on with the other encoder.
foreach ($c in 'maude', 'sdr') {
    $env:MAINT_CORPUS = $c
    $env:MAINT_SELECTOR_ENC = $(if ($c -eq 'sdr') { 'cb-conv' } else { 'cb-large-conv' })
    $null = Step "time_classifier_$c" 'compute_time.py classifier'
    foreach ($mk in 'qwen38_27b', 'gemma4_31b') { $null = Step "time_llm_${c}_$mk" "compute_time.py llm $mk" }
}
$env:MAINT_CORPUS = $Corpus
$env:MAINT_SELECTOR_ENC = $(if ($Corpus -eq 'sdr') { 'cb-conv' } else { 'cb-large-conv' })
"TIMING_DONE $(Get-Date -Format o)" *>> $log

$weights = "$env:USERPROFILE\models\${Corpus}_encoders"

# ---- 1: the other encoder, or a second seed of the paper's encoder
$tag = $null
$cache = "$env:USERPROFILE\.cache\huggingface\hub\models--$($Model -replace '/', '--')\snapshots"
if (Test-Path "$cache\*\config.json") {
    $env:MAINT_ENC_MODEL = $Model
    $env:MAINT_ENC_NAME = $Name
    $null = Step "train_${Name}_10k" 'maude_encoder.py cb 10k 42 eval2025'
    $null = Step "train_${Name}_full" 'maude_encoder.py cb full 42 eval2025 conv'
    if (Test-Path "$weights\cb-$Name-conv_full_seed42\config.json") { $tag = "cb-$Name-conv_full_seed42"; $suffix = "_$Name" }
    else { "ALT_MODEL_TRAINING_DID_NOT_FINISH $(Get-Date -Format o)" *>> $log }
} else { "ALT_MODEL_NOT_IN_CACHE $Model $(Get-Date -Format o)" *>> $log }
Remove-Item Env:MAINT_ENC_MODEL -ErrorAction SilentlyContinue
Remove-Item Env:MAINT_ENC_NAME -ErrorAction SilentlyContinue
if (-not $tag) {
    "FALLBACK second seed of the paper's encoder $(Get-Date -Format o)" *>> $log
    $base = $(if ($Corpus -eq 'sdr') { 'cb-conv' } else { 'cb-large-conv' })
    $flags = $(if ($Corpus -eq 'sdr') { 'eval2025 conv' } else { 'eval2025 large conv' })
    $null = Step 'train_seed43_full' "maude_encoder.py cb full 43 $flags"
    if (Test-Path "$weights\${base}_full_seed43\config.json") { $tag = "${base}_full_seed43"; $suffix = '_seed43' }
}
if (-not $tag) { "NO_ENCODER_TRAINED, stopping $(Get-Date -Format o)" *>> $log; exit 1 }
"ENCODER $tag $(Get-Date -Format o)" *>> $log

# ---- 2: every known-code report of both test years
foreach ($y in '2026', '2025') { $null = Step "year_$y" "maude_encoder_year.py $y --tag $tag" }

# ---- 3: shortlist and neighbours in this encoder's own directories
$env:MAINT_ENC_TAG = $tag
$env:MAINT_SHORT_SUFFIX = $suffix
CpuStep 'shortlist' 'rare_year.py shortlist'
CpuStep 'neighbours' 'rare_year.py neighbours'

# ---- 4: both LLMs, zero-shot and retrieval, on the reports with a rare code in this encoder's top five
$stages = @()
foreach ($kind in 'cls', 'ragfull') { foreach ($y in '2026', '2025') {
    $d = "..\runs\${Corpus}${y}short$suffix"
    if ((Test-Path "$d\shortlist.json") -and (($kind -eq 'cls') -or (Test-Path "$d\neighbors_full.json"))) { $stages += "maude${y}short5_$kind" }
} }
if ($stages.Count -eq 0) { "NO_LLM_STAGES $(Get-Date -Format o)" *>> $log }
else { foreach ($mk in 'qwen38_27b', 'gemma4_31b') { $null = Step "llm_$mk" "openweight_llm.py $mk $($stages -join ' ')" } }

# ---- 5: scores
CpuStep 'score' 'rare_year.py score'
CpuStep 'check' 'alt_encoder_check.py'
"ALL_DONE $(Get-Date -Format o)" *>> $log
