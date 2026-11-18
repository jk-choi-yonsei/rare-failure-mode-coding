$ErrorActionPreference = 'Continue'
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
Set-Location $PSScriptRoot
$log = "..\runs\stage2\reseed.log"
$py = "$env:USERPROFILE\miniconda3\envs\maintlog\python.exe"
$env:PYTHONUNBUFFERED = '1'
"START reseed $(Get-Date -Format o)" *>> $log
foreach ($ds in 'fmc', 'maintie_m1') {
    foreach ($rot in 0, 1, 2) {
        foreach ($s in 43, 44) {
            & $py novelmode_encoder.py $ds $rot $s *>> $log
            if ($LASTEXITCODE -ne 0) { "${ds}_ROT${rot}_SEED${s}_FAILED rc=$LASTEXITCODE" *>> $log }
        }
    }
}
"ALL_RESEED_DONE $(Get-Date -Format o)" *>> $log
