param([string]$Dataset = 'fmc')
$ErrorActionPreference = 'Continue'
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force ..\runs\stage2 | Out-Null
$log = "..\runs\stage2\train_$Dataset.log"
$py = "$env:USERPROFILE\miniconda3\envs\maintlog\python.exe"
$env:PYTHONUNBUFFERED = '1'
"START $Dataset $(Get-Date -Format o)" *>> $log
foreach ($rot in 0, 1, 2) {
    foreach ($s in 42, 43, 44) {
        & $py novelmode_encoder.py $Dataset $rot $s *>> $log
        if ($LASTEXITCODE -ne 0) { "ROT${rot}_SEED${s}_FAILED rc=$LASTEXITCODE" *>> $log }
    }
}
"ALL_DONE $Dataset $(Get-Date -Format o)" *>> $log
