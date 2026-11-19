# Reports under the withheld codes, 2026 and 2025 (withheld_check.py; asked for by the user on 2026-11-18 after the mock review).
# Started by a one-time Windows Task Scheduler task. It waits until the other-encoder run of the same night has written
# ALL_DONE to runs\alt_encoder.log (or gives up at 08:30 the next morning), then for both streams:
#   1  the paper's classifier over every report of both test years filed under a withheld code, and the shortlist (GPU)
#   2  neighbours for the retrieval LLM (CPU)
#   3  Qwen and Gemma, zero-shot and retrieval, on the shortlisted reports (at most 5,000 per stream and year; GPU)
#   4  score (CPU) -> results\withheld\<corpus>.json
$ErrorActionPreference = 'Continue'
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
Set-Location $PSScriptRoot
$log = "..\runs\withheld.log"
$py = "$env:USERPROFILE\miniconda3\envs\maintlog\python.exe"
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
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
        $out = "..\runs\withheld_step.out"; $err = "..\runs\withheld_step.err"
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
    $out = "..\runs\withheld_cpu.out"; $err = "..\runs\withheld_cpu.err"
    $p = Launch $argline $out $err
    $p.WaitForExit()
    Get-Content $out -Tail 40 *>> $log
    Get-Content $err -Tail 6 *>> $log
    "${name}_END rc=$($p.ExitCode) $(Get-Date -Format o)" *>> $log
}

"WAITING_FOR_OTHER_RUN $(Get-Date -Format o)" *>> $log
$deadline = Get-Date '2026-11-19 08:30'
while ($true) {
    $done = $false
    if (Test-Path '..\runs\alt_encoder.log') {
        $done = [bool](Select-String -Path '..\runs\alt_encoder.log' -Pattern '^ALL_DONE 2026-11-(18T(1[89]|2)|19T)' -Quiet)
    }
    if ($done) { break }
    if ((Get-Date) -gt $deadline) { "GAVE_UP_WAITING $(Get-Date -Format o)" *>> $log; exit 1 }
    Start-Sleep -Seconds 300
}
"START $(Get-Date -Format o); GPU $(& nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" *>> $log

foreach ($c in 'maude', 'sdr') {
    $env:MAINT_CORPUS = $c
    $env:MAINT_SELECTOR_ENC = $(if ($c -eq 'sdr') { 'cb-conv' } else { 'cb-large-conv' })
    Remove-Item Env:MAINT_ENC_TAG -ErrorAction SilentlyContinue
    $env:MAINT_SHORT_SUFFIX = '_withheld'
    if (Step "year_$c" 'withheld_check.py year') {
        CpuStep "neighbours_$c" 'rare_year.py neighbours'
        foreach ($mk in 'qwen38_27b', 'gemma4_31b') {
            $null = Step "llm_${c}_$mk" "openweight_llm.py $mk maude2026short5_cls maude2025short5_cls maude2026short5_ragfull maude2025short5_ragfull"
        }
        CpuStep "score_$c" 'withheld_check.py score'
    }
}
"WITHHELD_DONE $(Get-Date -Format o)" *>> $log
