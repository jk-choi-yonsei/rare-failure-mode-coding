# Claude arms (notes/prereg-maude-claude-2026-10-22.md): submit the given cells
# one after another (each waits for its seed batch), then poll until every batch
# has ended. Detached from the session, so a restart does not interrupt it.
#   run_maude_claude.ps1 <model:stage[:seed0][:repeat][:chunkN]> ...     e.g. haiku55:maude2025_cls:seed0 opus55:maude2025_cls
# "seed0" skips the seed batch (the cell's cache entry was written within the hour).
# -SubmitOnly: submit the cells and exit without the initial or final poll loops, for
# adding cells while another driver is already polling (that driver collects every
# batch in the book, so two pollers never race on the same batch).
param([switch]$SubmitOnly, [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)][string[]]$Cells)
$ErrorActionPreference = 'Continue'
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force ..\runs\claude | Out-Null
$log = "..\runs\claude\driver.log"
$py = "$env:USERPROFILE\miniconda3\envs\maintlog\python.exe"
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'

"START $(Get-Date -Format o) cells: $($Cells -join ' ')$(if ($SubmitOnly) { ' (submit only)' })" *>> $log
# Collect anything already ended (and clear the duplicate guard for a cell whose seed is pending).
if (-not $SubmitOnly) {
    do {
        $out = & $py maude_claude.py poll 2>&1 | Where-Object { $_ -notmatch 'warn' }
        $out *>> $log
        if ($out -match 'ALL_ENDED') { break }
        Start-Sleep -Seconds 30
    } while ($true)
}
foreach ($cell in $Cells) {
    $parts = $cell.Split(':')
    $argv = @('maude_claude.py', 'submit', $parts[0], $parts[1])
    if ($parts.Length -gt 2) {
        foreach ($opt in $parts[2..($parts.Length - 1)]) {
            if ($opt -eq 'seed0') { $argv += @('--seed', '0') }
            if ($opt -eq 'repeat') { $argv += @('--repeat') }
            if ($opt -match '^chunk(\d+)$') { $argv += @('--chunk', $Matches[1]) }
        }
    }
    "== submit $cell $(Get-Date -Format o)" *>> $log
    & $py @argv 2>&1 | Where-Object { $_ -notmatch 'warn' } *>> $log
}
if ($SubmitOnly) {
    "SUBMIT_ONLY_DONE $(Get-Date -Format o)" *>> $log
    exit 0
}
do {
    Start-Sleep -Seconds 120
    $out = & $py maude_claude.py poll 2>&1 | Where-Object { $_ -notmatch 'warn' }
    $out | Where-Object { $_ -match 'ended;|WARNING|total|interrupted' } *>> $log
} while (-not ($out -match 'ALL_ENDED'))
"PHASE_DONE $(Get-Date -Format o)" *>> $log
