# Start the Streamlit dashboard and share it.
#   LAN:      other devices on the same network open  http://<this-PC-IP>:8501
#             (Windows asks once to allow Python through the firewall - click Allow on private networks)
#   Internet: a Cloudflare quick tunnel prints a temporary https://<random>.trycloudflare.com link.
#             Anyone with the link can use the demo while this PC and this window stay running.
# Usage (PowerShell, from the project folder):  .\scripts\start_public_demo.ps1 [-Python <path to python.exe>]
param([string]$Python = "$env:USERPROFILE\.venvs\msf_fusion\Scripts\python.exe")
$root = Split-Path -Parent $PSScriptRoot
$app = Start-Process -FilePath $Python -ArgumentList @("-m", "streamlit", "run", "$root\app\streamlit_app.py",
    "--server.headless", "true", "--server.port", "8501", "--server.address", "0.0.0.0") -PassThru -NoNewWindow
Write-Host "Dashboard starting (PID $($app.Id)). LAN address:"
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.PrefixOrigin -ne "WellKnown" } | ForEach-Object { "  http://$($_.IPAddress):8501" }
$cf = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
if (-not $cf) { $cf = "${env:ProgramFiles(x86)}\cloudflared\cloudflared.exe" }
if (Test-Path $cf) {
    Write-Host "Opening public tunnel - look for the https://...trycloudflare.com line below (Ctrl+C stops it)."
    & $cf tunnel --no-autoupdate --url http://localhost:8501
} else {
    Write-Host "cloudflared not installed: winget install --id Cloudflare.cloudflared"
}
