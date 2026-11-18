# GPU runs of the redesign (notes/redesign-2026-11-06.md, notes/prereg-redesign-2026-11-06.md),
# weekend of 2026-11-06. One process, every step sets its own corpus. Order:
#   1. encoders trained to convergence under the fixed stopping rule (training-year data only):
#      MAUDE roberta-large CB full; SDR base and large CB full; MAUDE base CB 100k x3 seeds and full
#   2. gate: wait until the pre-registration is committed (no test-year output exists before it)
#   3. encoder predictions on the test year (MAUDE 2026, 2026 natural sample, 2026 never-seen set;
#      SDR 2026) and the open-weight LLMs on SDR 2026 and on the MAUDE 2026 never-seen set
# A failed step is retried up to three times, ten minutes apart; finished encoder runs and LLM
# rows are skipped on a retry. Each step waits for an idle GPU first (the user's own runs share
# the card; the combined load tripped the power on 2026-10-24). Keeps the machine awake.
$ErrorActionPreference = 'Continue'
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
Set-Location $PSScriptRoot
$log = "..\runs\redesign.log"
$py = "$env:USERPROFILE\miniconda3\envs\maintlog\python.exe"
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

function GpuBusy {
    $q = (& nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits) -split ','
    $mem = [int]$q[0].Trim(); $util = [int]$q[1].Trim()
    return ($mem -gt 3500) -or ($util -gt 15)
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
function Step($name, $corpus, [string[]]$argv) {
    for ($try = 1; $try -le 3; $try++) {
        WaitGpu
        $env:MAINT_CORPUS = $corpus
        "== $name [$corpus] try $try $(Get-Date -Format o)" *>> $log
        & $py @argv *>> $log
        $rc = $LASTEXITCODE
        if ($rc -eq 0) { "${name}_OK $(Get-Date -Format o)" *>> $log; return }
        "${name}_FAILED rc=$rc try $try $(Get-Date -Format o)" *>> $log
        Start-Sleep -Seconds 600
    }
    "${name}_GAVE_UP $(Get-Date -Format o)" *>> $log
}

"START $(Get-Date -Format o)" *>> $log
# 1. encoders (stopping rule fixed in notes/redesign-2026-11-06.md, commit b14627f)
Step 'enc_maude_cb-large-conv_full' 'maude' @('maude_encoder.py', 'cb', 'full', '42', 'eval2025', 'large', 'conv')
Step 'enc_sdr_cb-conv_full' 'sdr' @('maude_encoder.py', 'cb', 'full', '42', 'eval2025', 'conv')
Step 'enc_sdr_cb-large-conv_full' 'sdr' @('maude_encoder.py', 'cb', 'full', '42', 'eval2025', 'large', 'conv')

# 2. gate
while (-not (& git log --oneline -1 -- ..\notes\prereg-redesign-2026-11-06.md)) {
    "WAITING_FOR_PREREG $(Get-Date -Format o)" *>> $log
    Start-Sleep -Seconds 300
}
"PREREG_COMMITTED $(& git log --oneline -1 -- ..\notes\prereg-redesign-2026-11-06.md)" *>> $log

# 3a. selector encoders on the test year, then the SDR LLM outputs the selectors need
Step 'pred_maude_large_2026' 'maude' @('maude_encoder_predict.py', 'cb-large-conv_full_seed42', '2026')
Step 'pred_maude_large_2026nat' 'maude' @('maude_encoder_predict.py', 'cb-large-conv_full_seed42', '2026nat')
Step 'pred_maude_large_2026novel' 'maude' @('maude_encoder_predict.py', 'cb-large-conv_full_seed42', '2026novel')
foreach ($t in @('cb-conv_full_seed42', 'cb-large-conv_full_seed42', 'cb_10k_seed42', 'cb_10k_seed43', 'cb_10k_seed44',
                 'cb_1k_seed42', 'cb_1k_seed43', 'cb_1k_seed44')) {
    Step "pred_sdr_${t}_2026" 'sdr' @('maude_encoder_predict.py', $t, '2026')
}
Step 'llm_sdr_qwen_2026' 'sdr' @('openweight_llm.py', 'qwen38_27b', 'maude2026_cls', 'maude2026_ragfull')
Step 'llm_sdr_gemma_2026' 'sdr' @('openweight_llm.py', 'gemma4_31b', 'maude2026_cls', 'maude2026_ragfull')

# 3b. the MAUDE base encoders of the boundary tests, then their test-year predictions
foreach ($s in @('42', '43', '44')) {
    Step "enc_maude_cb-conv_100k_seed$s" 'maude' @('maude_encoder.py', 'cb', '100k', $s, 'eval2025', 'conv')
}
Step 'enc_maude_cb-conv_full' 'maude' @('maude_encoder.py', 'cb', 'full', '42', 'eval2025', 'conv')
foreach ($t in @('cb-conv_full_seed42', 'cb-conv_100k_seed42', 'cb-conv_100k_seed43', 'cb-conv_100k_seed44',
                 'cb_10k_seed42', 'cb_10k_seed43', 'cb_10k_seed44', 'cb_1k_seed42', 'cb_1k_seed43', 'cb_1k_seed44')) {
    Step "pred_maude_${t}_2026" 'maude' @('maude_encoder_predict.py', $t, '2026')
}

# 3c. the never-seen-mode test (E): the open-weight LLMs on the 2026 known test + the hidden codes' reports
Step 'llm_maude_qwen_2026novel' 'maude' @('openweight_llm.py', 'qwen38_27b', 'maude2026novel_novel')
Step 'llm_maude_gemma_2026novel' 'maude' @('openweight_llm.py', 'gemma4_31b', 'maude2026novel_novel')
"ALL_DONE $(Get-Date -Format o)" *>> $log
