$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$VenvStreamlit = Join-Path $ProjectRoot ".venv\Scripts\streamlit.exe"
$Entrypoint = Join-Path $ProjectRoot "streamlit_app.py"

Write-Host "Stopping old FishSTOP Streamlit processes..."
try {
    $processes = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and
        $_.CommandLine -match [regex]::Escape($ProjectRoot) -and
        ($_.CommandLine -match "streamlit" -or $_.CommandLine -match "streamlit_app\.py" -or $_.CommandLine -match "src[/\\]app\.py")
    }

    foreach ($process in $processes) {
        if ($process.ProcessId -ne $PID) {
            Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
} catch {
    Write-Warning "Could not stop old Streamlit processes: $($_.Exception.Message)"
}

Write-Host "Removing project Python caches..."
try {
    Get-ChildItem -Path $ProjectRoot -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch "\\.venv\\" } |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
} catch {
    Write-Warning "Could not remove every cache folder: $($_.Exception.Message)"
}

if (-not (Test-Path $Entrypoint)) {
    throw "Streamlit entrypoint not found: $Entrypoint"
}

Set-Location $ProjectRoot
Write-Host "Starting FishSTOP from $Entrypoint ..."

if (Test-Path $VenvPython) {
    & $VenvPython -m streamlit run $Entrypoint
} elseif (Test-Path $VenvStreamlit) {
    & $VenvStreamlit run $Entrypoint
} else {
    Write-Warning "Project virtual environment not found. Falling back to system Python."
    python -m streamlit run $Entrypoint
}
