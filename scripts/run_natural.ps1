# GPU runs for the natural-proportion selector (notes/natural-selector-2026-11-09.md).
# The user lent the GPU from 2026-11-09 18:20 until before 09:00 on 2026-11-11. The driver waits for
# the start time and an idle GPU, runs the steps in order, retries a failed step up to three times
# (finished LLM rows are skipped on a retry), starts no step after 07:30 on 2026-11-11 and stops a
# running step at 08:45. Each step runs at below-normal priority. Keeps the machine awake.
$ErrorActionPreference = 'Continue'
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
Set-Location $PSScriptRoot
$log = "..\runs\natural.log"
$py = "$env:USERPROFILE\miniconda3\envs\maintlog\python.exe"
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:MAINT_SELECTOR_ENC = 'cb-large-conv'
$env:MAINT_CORPUS = 'maude'
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

$start = Get-Date '2026-11-09 18:20'
$lastStart = Get-Date '2026-11-11 07:30'
$hardStop = Get-Date '2026-11-11 08:45'
"SCHEDULED $(Get-Date -Format o) -> start at $($start.ToString('o'))" *>> $log
while ((Get-Date) -lt $start) { Start-Sleep -Seconds 60 }

function GpuBusy {
    $q = (& nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits) -split ','
    $mem = [int]$q[0].Trim(); $util = [int]$q[1].Trim()
    return ($mem -gt 3500) -or ($util -gt 15)
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
function Step($name, [string]$argline) {
    for ($try = 1; $try -le 3; $try++) {
        if (-not (WaitGpu)) { "${name}_SKIPPED_DEADLINE $(Get-Date -Format o)" *>> $log; return }
        "== $name try $try $(Get-Date -Format o)" *>> $log
        $out = "..\runs\natural_step.out"; $err = "..\runs\natural_step.err"
        $p = Start-Process -FilePath $py -ArgumentList $argline -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
             -RedirectStandardOutput $out -RedirectStandardError $err
        $null = $p.Handle  # cache the handle, or ExitCode reads as empty once the process has exited (2026-11-09)
        Start-Sleep -Seconds 2
        try { $p.PriorityClass = 'BelowNormal' } catch {}
        while (-not $p.HasExited) {
            if ((Get-Date) -gt $hardStop) {
                "HARD_STOP $name $(Get-Date -Format o)" *>> $log
                Stop-Process -Id $p.Id -Force -Confirm:$false
                Get-Process -Name 'llama-server' -ErrorAction SilentlyContinue | Stop-Process -Force -Confirm:$false
                Get-Content $out -Tail 5 *>> $log
                return
            }
            Start-Sleep -Seconds 60
        }
        Get-Content $out -Tail 20 *>> $log
        Get-Content $err -Tail 5 *>> $log
        if ($p.ExitCode -eq 0) { "${name}_OK $(Get-Date -Format o)" *>> $log; return }
        "${name}_FAILED rc=$($p.ExitCode) try $try $(Get-Date -Format o)" *>> $log
        Start-Sleep -Seconds 600
    }
    "${name}_GAVE_UP $(Get-Date -Format o)" *>> $log
}

"START $(Get-Date -Format o)" *>> $log
if (-not (Test-Path '..\runs\maude2024val\encoder_cb-large-conv_full_seed42.csv')) {
    Step 'enc_2024val' 'maude_encoder_predict.py cb-large-conv_full_seed42 2024val'
}
Step 'qwen_2024val' 'openweight_llm.py qwen38_27b maude2024val_cls maude2024val_ragfull'
Step 'gemma_2024val' 'openweight_llm.py gemma4_31b maude2024val_cls maude2024val_ragfull'
Step 'qwen_2026nat' 'openweight_llm.py qwen38_27b maude2026nat_cls maude2026nat_ragfull'
Step 'gemma_2026nat' 'openweight_llm.py gemma4_31b maude2026nat_cls maude2026nat_ragfull'
"ALL_DONE $(Get-Date -Format o)" *>> $log
