# Shown-candidates runs (2026-11-16, notes/why-it-works-plan-2026-11-16.md, experiment 3). The user approved GPU use from
# 19:20; at 19:25 the GPU was busy with the user's own jobs, so every chunk waits until the GPU is idle. Started by a
# one-time Windows Task Scheduler task, so a restart of the Claude app does not end it. Each chunk runs at most 1,000
# requests and then stops the server (llama-server slowed down after ~1,500 requests in the weekend run). Finished
# rows are skipped, so every step is safe to repeat.
$ErrorActionPreference = 'Continue'
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
Set-Location $PSScriptRoot
$log = "..\runs\cand5.log"
$py = "$env:USERPROFILE\miniconda3\envs\maintlog\python.exe"
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

function SetCorpus([string]$c) { $env:MAINT_CORPUS = $c }
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
function Chunks([string]$name, [string]$corpus, [string]$mk, [string]$stage) {
    $fails = 0
    for ($i = 1; $i -le 80; $i++) {
        WaitGpu
        SetCorpus $corpus
        "== $name chunk $i $(Get-Date -Format o)" *>> $log
        $out = "..\runs\cand5_step.out"; $err = "..\runs\cand5_step.err"
        $p = Launch "openweight_llm.py $mk $stage --max-new 1000" $out $err
        $p.WaitForExit()
        Get-Content $out -Tail 6 *>> $log
        Get-Content $err -Tail 3 *>> $log
        $rc = $p.ExitCode
        if ($rc -eq 0) { "${name}_DONE $(Get-Date -Format o)" *>> $log; return }
        if ($rc -eq 3) { continue }
        $fails++
        "${name}_FAILED rc=$rc fails=$fails $(Get-Date -Format o)" *>> $log
        if ($fails -ge 3) { "${name}_GAVE_UP $(Get-Date -Format o)" *>> $log; return }
        Start-Sleep -Seconds 300
    }
}

"START $(Get-Date -Format o) $(& nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" *>> $log
foreach ($y in '2026', '2025') {
    foreach ($mk in 'qwen38_27b', 'gemma4_31b') {
        foreach ($c in 'maude', 'sdr') { Chunks "cand5_${c}_${y}_$mk" $c $mk "maude${y}cand5_cand" }
    }
}
"ALL_DONE $(Get-Date -Format o)" *>> $log
