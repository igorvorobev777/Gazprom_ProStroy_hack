param(
    [string]$Model = "D:\Models\Qwen3-4B-GGUF\Qwen3-4B-Q4_K_M.gguf",
    [string]$Alias = "qwen3-4b-rag",
    [int]$Context = 4096,
    [int]$Threads = 0,
    [int]$BatchThreads = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

if (-not (Get-Command llama-server -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: llama-server was not found in PATH."
    exit 1
}
if (-not (Test-Path $Model)) {
    Write-Host "ERROR: model not found: $Model"
    exit 1
}

$cpu = Get-CimInstance Win32_Processor
$physical = [int](($cpu | Measure-Object -Property NumberOfCores -Sum).Sum)
$logical = [int](($cpu | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum)

if ((Test-Path ".llama_tuning.json") -and ($Threads -le 0 -or $BatchThreads -le 0)) {
    try {
        $tuned = Get-Content ".llama_tuning.json" -Raw | ConvertFrom-Json
        if ($Threads -le 0 -and [int]$tuned.generation_threads -gt 0) { $Threads = [int]$tuned.generation_threads }
        if ($BatchThreads -le 0 -and [int]$tuned.batch_threads -gt 0) { $BatchThreads = [int]$tuned.batch_threads }
        Write-Host "Loaded .llama_tuning.json"
    } catch {
        Write-Host "WARNING: could not read .llama_tuning.json; using CPU topology defaults."
    }
}

if ($Threads -le 0) { $Threads = [Math]::Max(1, $physical) }
if ($BatchThreads -le 0) { $BatchThreads = [Math]::Max(1, $logical) }

# Explicit values protect the launcher from inherited LLAMA_ARG_* environment settings.
$env:LLAMA_ARG_TIMEOUT = "3600"
$env:LLAMA_ARG_LOAD_MODE = "mmap"
$env:LLAMA_ARG_N_GPU_LAYERS = "0"

Write-Host "CPU profile: physical=$physical logical=$logical generation_threads=$Threads batch_threads=$BatchThreads"
Write-Host "Model: $Model"
Write-Host "Context: $Context"
Write-Host ""
Write-Host "IMPORTANT: when you see 'listening on http://127.0.0.1:1234', leave this window open."
Write-Host "Do NOT press Ctrl+C until you intentionally want to stop the model server."
Write-Host ""

$argsList = @(
    "-m", $Model,
    "--alias", $Alias,
    "--host", "127.0.0.1",
    "--port", "1234",
    "-c", "$Context",
    "-np", "1",
    "-ngl", "0",
    "-t", "$Threads",
    "-tb", "$BatchThreads",
    "-fa", "auto",
    "-lm", "mmap",
    "--reasoning", "off",
    "--jinja",
    "--cache-prompt",
    "--cache-reuse", "64",
    "--no-context-shift",
    "--no-warmup",
    "--perf",
    "--timeout", "3600",
    "--sleep-idle-seconds", "-1",
    "--cors-origins", "localhost",
    "--no-ui",
    "--temp", "0.7",
    "--top-p", "0.8",
    "--top-k", "20",
    "--min-p", "0"
)

& llama-server @argsList
$code = $LASTEXITCODE
Write-Host ""
Write-Host "llama-server exited with code: $code"
if ($code -eq 0) {
    Write-Host "If you did not press Ctrl+C, tell me that the server exited by itself and include the last 20 log lines."
}
exit $code
