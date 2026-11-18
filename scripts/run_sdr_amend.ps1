# SDR amendment (notes/prereg-sdr-amendment-2026-10-27.md): retrain the full-N encoders
# with the 10k epoch budget ("long"), then resume the natural-sample driver (Gemma
# retrieval remainder) in a fresh process without the SDR corpus switch.
$ErrorActionPreference = 'Continue'
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
Set-Location $PSScriptRoot
$log = "..\runs\sdr\driver_sdr_amend.log"
$py = "$env:USERPROFILE\miniconda3\envs\maintlog\python.exe"
$env:MAINT_CORPUS = 'sdr'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'

Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null

function Step($name, [string[]]$argv) {
    "== $name $(Get-Date -Format o)" *>> $log
    & $py @argv *>> $log
    if ($LASTEXITCODE -ne 0) { "${name}_FAILED rc=$LASTEXITCODE" *>> $log }
}

"START $(Get-Date -Format o) corpus=$env:MAINT_CORPUS" *>> $log
Step 'cb_long_full_seed42' @('maude_encoder.py', 'cb', 'full', 42, 'eval2025', 'long')
Step 'large_long_full_seed42' @('maude_encoder.py', 'cb', 'full', 42, 'eval2025', 'large', 'long')
"AMEND_DONE $(Get-Date -Format o)" *>> $log
# Resume the natural-sample runs in a clean process (no corpus switch).
Remove-Item Env:MAINT_CORPUS -ErrorAction SilentlyContinue
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSScriptRoot\run_maude_natural.ps1`"" -WindowStyle Hidden
"NATURAL_RELAUNCHED $(Get-Date -Format o)" *>> $log
