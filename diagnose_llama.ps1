$ErrorActionPreference = "Continue"
Write-Host "=== CPU ==="
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed | Format-List

Write-Host "=== RAM ==="
$cs = Get-CimInstance Win32_ComputerSystem
"Total RAM GB: {0:N1}" -f ($cs.TotalPhysicalMemory / 1GB)

Write-Host "=== GPU ==="
Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion | Format-Table -AutoSize

Write-Host "=== llama.cpp version ==="
llama-server --version

Write-Host "=== llama.cpp devices ==="
llama-server --list-devices

Write-Host "=== server health (only works if server is already running) ==="
try {
    Invoke-RestMethod -Uri "http://127.0.0.1:1234/health" -TimeoutSec 2 | ConvertTo-Json -Depth 5
} catch {
    Write-Host "Server is not running on port 1234."
}
