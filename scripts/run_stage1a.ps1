param([string]$Dataset = 'fmc')
$ErrorActionPreference = 'Continue'
# PowerShell 5.1 writes UTF-16 for redirection by default, which makes the
# log unreadable to grep-based tooling.
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force ..\runs\stage1 | Out-Null
$log = "..\runs\stage1\train_$Dataset.log"
$py = "$env:USERPROFILE\miniconda3\envs\maintlog\python.exe"
$env:PYTHONUNBUFFERED = '1'
"START $Dataset $(Get-Date -Format o)" *>> $log
foreach ($s in 42, 43, 44) {
    # Direct redirection, not `| Add-Content`: a reader holding the log open
    # made the pipeline's writes fail silently and swallowed the traceback.
    & $py encoder_finetune.py $Dataset $s *>> $log
    if ($LASTEXITCODE -ne 0) { "SEED_${s}_FAILED rc=$LASTEXITCODE" *>> $log }
}
"ALL_SEEDS_DONE $Dataset $(Get-Date -Format o)" *>> $log
