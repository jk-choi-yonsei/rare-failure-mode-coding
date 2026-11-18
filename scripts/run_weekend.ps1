# Weekend GPU queue (2026-11-14, notes/selector-vs-rules-2026-11-11.md, section "주말 실행"). The user lent the GPU
# until Monday morning and asked for as many results as possible, with no stop time. Started by a one-time Windows
# Task Scheduler task, so a restart of the Claude app does not end it. Every step is safe to repeat: finished encoder
# parts and LLM rows are skipped. A step waits while the GPU is busy with something else, retries up to three times,
# and the queue goes on after a step that gave up.
#   A  SDR: the encoder over every known-code report of 2026 and 2025; encoder CSVs of the new natural samples;
#      shortlist; neighbours; Qwen and Gemma, zero-shot and retrieval, on the reports with a rare code in the top 5
#   B  SDR natural samples (splits/sdr_<year>_natural.json): Qwen and Gemma, zero-shot and retrieval
#   C  SDR: the same on the reports with a rare code at rank 6-10
#   D  MAUDE: the same on the reports with a rare code at rank 6-10
# Each block ends with rare_year.py score on the CPU.
$ErrorActionPreference = 'Continue'
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
Set-Location $PSScriptRoot
$log = "..\runs\weekend.log"
$py = "$env:USERPROFILE\miniconda3\envs\maintlog\python.exe"
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

function SetCorpus([string]$c) {
    $env:MAINT_CORPUS = $c
    $env:MAINT_SELECTOR_ENC = $(if ($c -eq 'sdr') { 'cb-conv' } else { 'cb-large-conv' })
}
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
function Step([string]$name, [string]$corpus, [string]$argline) {
    for ($try = 1; $try -le 3; $try++) {
        WaitGpu
        SetCorpus $corpus
        "== $name [$corpus] try $try $(Get-Date -Format o) : $argline" *>> $log
        $out = "..\runs\weekend_step.out"; $err = "..\runs\weekend_step.err"
        $p = Launch $argline $out $err
        $p.WaitForExit()
        Get-Content $out -Tail 12 *>> $log
        Get-Content $err -Tail 5 *>> $log
        if ($p.ExitCode -eq 0) { "${name}_OK $(Get-Date -Format o)" *>> $log; return }
        "${name}_FAILED rc=$($p.ExitCode) try $try $(Get-Date -Format o)" *>> $log
        Start-Sleep -Seconds 300
    }
    "${name}_GAVE_UP $(Get-Date -Format o)" *>> $log
}
function CpuStep([string]$name, [string]$corpus, [string]$argline) {
    SetCorpus $corpus
    "== $name [$corpus] (cpu) $(Get-Date -Format o) : $argline" *>> $log
    $out = "..\runs\weekend_cpu.out"; $err = "..\runs\weekend_cpu.err"
    $p = Launch $argline $out $err
    $p.WaitForExit()
    Get-Content $out -Tail 80 *>> $log
    Get-Content $err -Tail 5 *>> $log
    "${name}_END rc=$($p.ExitCode) $(Get-Date -Format o)" *>> $log
}
function Short([string]$corpus, [int[]]$ks, [string]$kind) {
    # stage names of the shortlisted reports, smaller k first; a year without its files is left out
    $s = @()
    foreach ($k in $ks) { foreach ($y in '2026', '2025') {
        $d = "..\runs\${corpus}${y}short"
        if ((Test-Path "$d\shortlist.json") -and (($kind -eq 'cls') -or (Test-Path "$d\neighbors_full.json"))) { $s += "maude${y}short${k}_$kind" }
    } }
    return $s  # callers wrap the result in @() so that one or no stage is still a list
}
function Llm([string]$name, [string]$corpus, [string[]]$stages) {
    $stages = @($stages | Where-Object { $_ })
    if ($stages.Count -eq 0) { "${name}_NO_STAGES $(Get-Date -Format o)" *>> $log; return }
    foreach ($mk in 'qwen38_27b', 'gemma4_31b') { Step "${name}_$mk" $corpus "openweight_llm.py $mk $($stages -join ' ')" }
}

"START $(Get-Date -Format o) $(& nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" *>> $log
# ---- A: SDR, rare code in the encoder's top 5
foreach ($y in '2026', '2025') { Step "A_enc_sdr_$y" 'sdr' "maude_encoder_year.py $y" }
Step 'A_enccsv_sdr_nat' 'sdr' 'maude_encoder_predict.py cb-conv_full_seed42 natural'
Step 'A_enccsv_sdr_2026nat' 'sdr' 'maude_encoder_predict.py cb-conv_full_seed42 2026nat'
CpuStep 'A_shortlist_sdr' 'sdr' 'rare_year.py shortlist'
CpuStep 'A_neighbours_sdr' 'sdr' 'rare_year.py neighbours --rag-k 10'
Llm 'A_llm_sdr' 'sdr' (@(Short 'sdr' @(3, 5) 'cls') + @(Short 'sdr' @(3, 5) 'ragfull'))
CpuStep 'A_score_sdr' 'sdr' 'rare_year.py score'
# ---- B: SDR natural samples
Llm 'B_llm_sdr_nat' 'sdr' @('maude2025nat_cls', 'maude2026nat_cls', 'maude2025nat_ragfull', 'maude2026nat_ragfull')
CpuStep 'B_score_sdr' 'sdr' 'rare_year.py score'
# ---- C: SDR, rank 6-10
Llm 'C_llm_sdr_k10' 'sdr' (@(Short 'sdr' @(10) 'cls') + @(Short 'sdr' @(10) 'ragfull'))
CpuStep 'C_score_sdr' 'sdr' 'rare_year.py score'
# ---- D: MAUDE, rank 6-10
CpuStep 'D_neighbours_maude' 'maude' 'rare_year.py neighbours --rag-k 10'
Llm 'D_llm_maude_k10' 'maude' (@(Short 'maude' @(10) 'cls') + @(Short 'maude' @(10) 'ragfull'))
CpuStep 'D_score_maude' 'maude' 'rare_year.py score'
"ALL_DONE $(Get-Date -Format o)" *>> $log
