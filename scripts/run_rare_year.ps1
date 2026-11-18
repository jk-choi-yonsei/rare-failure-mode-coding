# GPU run for the whole-year check of the rare-code rule (notes/selector-vs-rules-2026-11-11.md,
# section "한 해 전체로 재는 방법"). The user lent the GPU from 2026-11-12 20:00 until before 09:00 on
# 2026-11-13. The driver waits for the start time and a free GPU, runs the steps in order, retries a failed
# step up to three times (finished encoder parts and LLM rows are skipped on a retry), starts no GPU step
# after 07:30 on 2026-11-13 and stops a running one at 08:45. Scoring uses the CPU only and always runs.
# Each step runs at below-normal priority. Keeps the machine awake.
$ErrorActionPreference = 'Continue'
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
Set-Location $PSScriptRoot
$log = "..\runs\rare_year.log"
$py = "$env:USERPROFILE\miniconda3\envs\maintlog\python.exe"
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:MAINT_SELECTOR_ENC = 'cb-large-conv'
$env:MAINT_CORPUS = 'maude'
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

$start = Get-Date '2026-11-12 20:00'
$lastStart = Get-Date '2026-11-13 07:30'
$hardStop = Get-Date '2026-11-13 08:45'
"SCHEDULED $(Get-Date -Format o) -> start at $($start.ToString('o'))" *>> $log
while ((Get-Date) -lt $start) { Start-Sleep -Seconds 60 }

function GpuBusy {
    # a game or another model on the card; ordinary desktop use (about 3 GB, 20-30 %) does not count
    $q = (& nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits) -split ','
    $mem = [int]$q[0].Trim(); $util = [int]$q[1].Trim()
    return ($mem -gt 6000) -or ($util -gt 50)
}
function WaitGpu {
    $idle = 0
    while ($idle -lt 3) {
        if ((Get-Date) -gt $lastStart) { return $false }
        if (GpuBusy) {
            $idle = 0
            "WAITING_FOR_GPU $(Get-Date -Format o) $(& nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" *>> $log
            Start-Sleep -Seconds 300
        } else { $idle++; Start-Sleep -Seconds 20 }
    }
    return $true
}
function Launch([string]$argline, [string]$out, [string]$err) {
    $p = Start-Process -FilePath $py -ArgumentList $argline -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
         -RedirectStandardOutput $out -RedirectStandardError $err
    $null = $p.Handle  # cache the handle, or ExitCode reads as empty once the process has exited
    Start-Sleep -Seconds 2
    try { $p.PriorityClass = 'BelowNormal' } catch {}
    return $p
}
function Step($name, [string]$argline) {
    for ($try = 1; $try -le 3; $try++) {
        if (-not (WaitGpu)) { "${name}_SKIPPED_DEADLINE $(Get-Date -Format o)" *>> $log; return }
        "== $name try $try $(Get-Date -Format o) : $argline" *>> $log
        $out = "..\runs\rare_year_step.out"; $err = "..\runs\rare_year_step.err"
        $p = Launch $argline $out $err
        while (-not $p.HasExited) {
            if ((Get-Date) -gt $hardStop) {
                "HARD_STOP $name $(Get-Date -Format o)" *>> $log
                Stop-Process -Id $p.Id -Force -Confirm:$false
                Get-Process -Name 'llama-server' -ErrorAction SilentlyContinue | Stop-Process -Force -Confirm:$false
                Get-Content $out -Tail 5 *>> $log
                return
            }
            Start-Sleep -Seconds 30
        }
        Get-Content $out -Tail 12 *>> $log
        Get-Content $err -Tail 5 *>> $log
        if ($p.ExitCode -eq 0) { "${name}_OK $(Get-Date -Format o)" *>> $log; return }
        "${name}_FAILED rc=$($p.ExitCode) try $try $(Get-Date -Format o)" *>> $log
        Start-Sleep -Seconds 300
    }
    "${name}_GAVE_UP $(Get-Date -Format o)" *>> $log
}
function CpuStep($name, [string]$argline) {
    "== $name (cpu) $(Get-Date -Format o) : $argline" *>> $log
    $out = "..\runs\rare_year_cpu.out"; $err = "..\runs\rare_year_cpu.err"
    $p = Launch $argline $out $err
    $p.WaitForExit(3600000) | Out-Null
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -Confirm:$false }
    Get-Content $out -Tail 60 *>> $log
    Get-Content $err -Tail 5 *>> $log
    "${name}_END rc=$($p.ExitCode) $(Get-Date -Format o)" *>> $log
}
function Stages([string]$kind) {
    # k = 3 for both years first, then k = 5, so a cut run still has a whole k; a year without its files is left out
    $s = @()
    foreach ($k in 3, 5) { foreach ($y in '2026', '2025') {
        $d = "..\runs\maude${y}short"
        if ((Test-Path "$d\shortlist.json") -and (($kind -eq 'cls') -or (Test-Path "$d\neighbors_full.json"))) { $s += "maude${y}short${k}_$kind" }
    } }
    return ($s -join ' ')
}

"START $(Get-Date -Format o) $(& nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" *>> $log
Step 'enc_2026' 'maude_encoder_year.py 2026'
Step 'enc_2025' 'maude_encoder_year.py 2025'
CpuStep 'shortlist' 'rare_year.py shortlist'
$nb = Launch 'rare_year.py neighbours' '..\runs\rare_year_nb.out' '..\runs\rare_year_nb.err'  # CPU, beside the zero-shot steps
$zs = Stages 'cls'
if ($zs) {
    Step 'qwen_zs' "openweight_llm.py qwen38_27b $zs"
    Step 'gemma_zs' "openweight_llm.py gemma4_31b $zs"
} else { "NO_SHORTLIST $(Get-Date -Format o)" *>> $log }
CpuStep 'score_zs' 'rare_year.py score'
while ((-not $nb.HasExited) -and ((Get-Date) -lt $lastStart)) { Start-Sleep -Seconds 30 }
Get-Content '..\runs\rare_year_nb.out' -Tail 6 *>> $log
Get-Content '..\runs\rare_year_nb.err' -Tail 5 *>> $log
"neighbours_END exited=$($nb.HasExited) rc=$($nb.ExitCode) $(Get-Date -Format o)" *>> $log
$rag = Stages 'ragfull'
if ($rag) {
    Step 'qwen_rag' "openweight_llm.py qwen38_27b $rag"
    Step 'gemma_rag' "openweight_llm.py gemma4_31b $rag"
} else { "NO_NEIGHBOURS $(Get-Date -Format o)" *>> $log }
CpuStep 'score_final' 'rare_year.py score'
"ALL_DONE $(Get-Date -Format o)" *>> $log
