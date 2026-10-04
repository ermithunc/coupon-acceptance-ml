# Start the coupon demo from the repository root.
# The app and the saved model resolve paths from their own files, not from the shell's current directory.
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}
& $Python -m streamlit run (Join-Path $Root "app\app.py")
