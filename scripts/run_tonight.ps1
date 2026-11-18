# Resume the weekend GPU work tonight (user: GPU available from the evening of 2026-10-26).
# Sleeps until the start time, then runs the SDR replication first (the paper's second
# corpus; run_sdr.ps1 -NoWait, which skips finished steps) and the remainder of the
# natural-sample runs (Gemma retrieval, 3,184 of 5,814 left). Keeps the machine awake
# while it lives. Usage: run_tonight.ps1 [-StartAt '22:00']
param([string]$StartAt = '22:00')
$ErrorActionPreference = 'Continue'
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
Set-Location $PSScriptRoot
$log = "..\runs\tonight.log"
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

$start = Get-Date $StartAt
if ($start -lt (Get-Date)) { $start = $start.AddDays(1) }
"SCHEDULED $(Get-Date -Format o) -> start at $($start.ToString('o'))" *>> $log
while ((Get-Date) -lt $start) { Start-Sleep -Seconds 60 }
# The user's own GPU experiments share this card and the combined load tripped the
# power on 2026-10-24. Start only once the GPU has been idle for three checks a minute
# apart (under 3 GB used and under 15% utilization); log every wait.
function GpuBusy {
    $q = (& nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits) -split ','
    $mem = [int]$q[0].Trim(); $util = [int]$q[1].Trim()
    return ($mem -gt 3000) -or ($util -gt 15)
}
$idle = 0
while ($idle -lt 3) {
    if (GpuBusy) { $idle = 0; "WAITING_FOR_GPU $(Get-Date -Format o) $(& nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" *>> $log; Start-Sleep -Seconds 300 }
    else { $idle++; Start-Sleep -Seconds 60 }
}
"START $(Get-Date -Format o) (GPU idle)" *>> $log
# Each driver runs in its own process: run_sdr.ps1 sets MAINT_CORPUS=sdr, and on
# 2026-10-26 that leaked into run_maude_natural.ps1 (same process), which then looked
# for SDR paths and failed every step.
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSScriptRoot\run_sdr.ps1`" -NoWait" -WindowStyle Hidden -Wait
"SDR_DRIVER_RETURNED $(Get-Date -Format o)" *>> $log
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSScriptRoot\run_maude_natural.ps1`"" -WindowStyle Hidden -Wait
"NATURAL_DRIVER_RETURNED $(Get-Date -Format o)" *>> $log
"ALL_DONE $(Get-Date -Format o)" *>> $log
