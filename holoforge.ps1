$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
Write-Host "HoloForge v1.1.0 - Webcam Spatial Computing"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    Write-Error "Virtual environment not found. Run scripts\setup_windows.bat first."
    exit 1
}
$activate = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
& $activate
& $python -m src.main @args
exit $LASTEXITCODE
