$ErrorActionPreference = "SilentlyContinue"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Streamlit = Join-Path $ProjectRoot ".venv\Scripts\streamlit.exe"

Write-Host "Stopping old FishSTOP Streamlit processes..."
$processes = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match [regex]::Escape($ProjectRoot) -and
    $_.CommandLine -match "streamlit|src/app.py|src\\app.py"
}

foreach ($process in $processes) {
    try {
        Stop-Process -Id $process.ProcessId -Force
    } catch {}
}

Write-Host "Removing project Python caches..."
Get-ChildItem -Path $ProjectRoot -Recurse -Directory -Filter "__pycache__" |
    Where-Object { $_.FullName -notmatch "\\.venv\\" } |
    Remove-Item -Recurse -Force

Set-Location $ProjectRoot
Write-Host "Starting FishSTOP..."
& $Streamlit run src/app.py
