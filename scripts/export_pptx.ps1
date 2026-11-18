# Export each .pptx to a PDF (one page per slide) and a PNG per slide, through
# PowerPoint itself, so the PDF matches what PowerPoint shows (fonts embedded).
# All files go through one PowerPoint session; PowerPoint is quit afterwards only if
# this script started it. -Pptx takes several paths, or one string joined with ';'.
#   powershell -File scripts/export_pptx.ps1 -Pptx "a.pptx;b.pptx" -OutDir paper/figures -Dpi 300
param([Parameter(Mandatory=$true)][string[]]$Pptx, [Parameter(Mandatory=$true)][string]$OutDir, [int]$Dpi = 300)
$ErrorActionPreference = "Stop"
$files = @()
foreach ($p in $Pptx) { $files += @($p.Split(';') | Where-Object { $_ }) }
$wasRunning = @(Get-Process POWERPNT -ErrorAction SilentlyContinue).Count -gt 0
$app = New-Object -ComObject PowerPoint.Application
try {
    $out = (Resolve-Path $OutDir).Path
    foreach ($p in $files) {
        $full = (Resolve-Path $p).Path
        $name = [IO.Path]::GetFileNameWithoutExtension($full)
        # Open(FileName, ReadOnly, Untitled, WithWindow)
        $pres = $app.Presentations.Open($full, -1, 0, 0)
        try {
            $pres.SaveAs((Join-Path $out "$name.pdf"), 32)   # ppSaveAsPDF
            $wIn = $pres.PageSetup.SlideWidth / 72.0
            $hIn = $pres.PageSetup.SlideHeight / 72.0
            $i = 0
            foreach ($s in $pres.Slides) {
                $i++
                $suffix = if ($pres.Slides.Count -gt 1) { "_$i" } else { "" }
                $s.Export((Join-Path $out "$name$suffix.png"), "PNG", [int]($wIn * $Dpi), [int]($hIn * $Dpi))
            }
            Write-Output ("exported {0} ({1:N2} x {2:N2} in)" -f $name, $wIn, $hIn)
        } finally { $pres.Close() }
    }
} finally {
    if (-not $wasRunning) { $app.Quit() }
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($app)
}
