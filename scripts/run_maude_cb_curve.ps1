# CB learning curve, pre-registered in notes/prereg-maude-cb-curve-2026-10-07.md.
# Nine encoder runs, then the boundary analysis. Finished runs are skipped, so
# rerunning this file after an interruption continues where it stopped.
$ErrorActionPreference = 'Continue'
# PowerShell 5.1 writes UTF-16 for redirection by default.
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force ..\runs\maude | Out-Null
$log = "..\runs\maude\driver_cb_curve.log"
$py = "$env:USERPROFILE\miniconda3\envs\maintlog\python.exe"
$env:PYTHONUNBUFFERED = '1'
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'

# Keep the machine awake while this process lives; nothing persists after it.
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null

function Step($name, [string[]]$argv) {
    "== $name $(Get-Date -Format o)" *>> $log
    & $py @argv *>> $log
    if ($LASTEXITCODE -ne 0) { "${name}_FAILED rc=$LASTEXITCODE" *>> $log }
}

"START $(Get-Date -Format o)" *>> $log
foreach ($size in '1k', '10k', '100k') {
    foreach ($s in 42, 43, 44) { Step "cb_${size}_seed$s" @('maude_encoder.py', 'cb', $size, $s) }
}
Step 'boundary' @('maude_boundary.py')
"ALL_DONE $(Get-Date -Format o)" *>> $log
