param(
    [string]$PythonLauncher = "py",
    [string]$VenvDir = ".venv-win-build",
    [string]$AppName = "MotionExtractionUI"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $VenvDir)) {
    & $PythonLauncher -3 -m venv $VenvDir
}

$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found in venv: $PythonExe"
}

& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install numpy opencv-python pyinstaller

if (Test-Path "build") {
    Remove-Item -Recurse -Force "build"
}
if (Test-Path "dist") {
    Remove-Item -Recurse -Force "dist"
}

& $PythonExe -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --onefile `
    --name $AppName `
    ui.py

Write-Host ""
Write-Host "Build complete:"
Write-Host "dist\$AppName.exe"
