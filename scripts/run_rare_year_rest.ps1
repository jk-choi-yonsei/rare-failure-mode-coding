# Follow-up to run_rare_year.ps1 (2026-11-12 20:45, user: "강제 종료 그럼 취소. 다 돌아갈때까진 두자").
# The first driver keeps its own rules (no new GPU step after 07:30, a running step stopped at 08:45); its
# variables cannot be changed while it runs. This driver waits until the first one and its step processes
# are gone, then runs whatever is still missing, with no deadline. Every step is safe to repeat: finished
# encoder parts and LLM rows are skipped. It still waits while the GPU is busy with something else.
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

function FirstRunAlive {
    $n = @(Get-CimInstance Win32_Process | Where-Object {
        ($_.Name -eq 'powershell.exe' -and $_.CommandLine -like '*-File*run_rare_year.ps1*') -or
        ($_.Name -eq 'python.exe' -and ($_.CommandLine -like '*maude_encoder_year.py*' -or $_.CommandLine -like '*openweight_llm.py*' -or $_.CommandLine -like '*rare_year.py*')) -or
        ($_.Name -eq 'llama-server.exe') }).Count
    return ($n -gt 0)
}
"REST_SCHEDULED $(Get-Date -Format o) -> after the first driver ends" *>> $log
$quiet = 0
while ($quiet -lt 2) { if (FirstRunAlive) { $quiet = 0 } else { $quiet++ }; Start-Sleep -Seconds 60 }

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
    $null = $p.Handle
    Start-Sleep -Seconds 2
    try { $p.PriorityClass = 'BelowNormal' } catch {}
    return $p
}
function Step($name, [string]$argline) {
    for ($try = 1; $try -le 3; $try++) {
        WaitGpu
        "== rest $name try $try $(Get-Date -Format o) : $argline" *>> $log
        $out = "..\runs\rare_year_step.out"; $err = "..\runs\rare_year_step.err"
        $p = Launch $argline $out $err
        $p.WaitForExit()
        Get-Content $out -Tail 12 *>> $log
        Get-Content $err -Tail 5 *>> $log
        if ($p.ExitCode -eq 0) { "rest ${name}_OK $(Get-Date -Format o)" *>> $log; return }
        "rest ${name}_FAILED rc=$($p.ExitCode) try $try $(Get-Date -Format o)" *>> $log
        Start-Sleep -Seconds 300
    }
    "rest ${name}_GAVE_UP $(Get-Date -Format o)" *>> $log
}
function CpuStep($name, [string]$argline) {
    "== rest $name (cpu) $(Get-Date -Format o) : $argline" *>> $log
    $out = "..\runs\rare_year_cpu.out"; $err = "..\runs\rare_year_cpu.err"
    $p = Launch $argline $out $err
    $p.WaitForExit(3600000) | Out-Null
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -Confirm:$false }
    Get-Content $out -Tail 60 *>> $log
    Get-Content $err -Tail 5 *>> $log
    "rest ${name}_END rc=$($p.ExitCode) $(Get-Date -Format o)" *>> $log
}
function Stages([string]$kind) {
    $s = @()
    foreach ($k in 3, 5) { foreach ($y in '2026', '2025') {
        $d = "..\runs\maude${y}short"
        if ((Test-Path "$d\shortlist.json") -and (($kind -eq 'cls') -or (Test-Path "$d\neighbors_full.json"))) { $s += "maude${y}short${k}_$kind" }
    } }
    return ($s -join ' ')
}

"REST_START $(Get-Date -Format o) $(& nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" *>> $log
$enc = "..\runs\maude_encyear\cb-large-conv_full_seed42"
foreach ($y in '2026', '2025') { if (-not (Test-Path "$enc\$y.npz")) { Step "enc_$y" "maude_encoder_year.py $y" } }
if (-not ((Test-Path '..\runs\maude2026short\shortlist.json') -and (Test-Path '..\runs\maude2025short\shortlist.json'))) { CpuStep 'shortlist' 'rare_year.py shortlist' }
if (-not ((Test-Path '..\runs\maude2026short\neighbors_full.json') -and (Test-Path '..\runs\maude2025short\neighbors_full.json'))) { CpuStep 'neighbours' 'rare_year.py neighbours' }
$zs = Stages 'cls'
if ($zs) {  # the runner returns at once when every row of a stage is already there
    Step 'qwen_zs' "openweight_llm.py qwen38_27b $zs"
    Step 'gemma_zs' "openweight_llm.py gemma4_31b $zs"
}
$rag = Stages 'ragfull'
if ($rag) {
    Step 'qwen_rag' "openweight_llm.py qwen38_27b $rag"
    Step 'gemma_rag' "openweight_llm.py gemma4_31b $rag"
}
CpuStep 'score_final' 'rare_year.py score'
"REST_DONE $(Get-Date -Format o)" *>> $log
