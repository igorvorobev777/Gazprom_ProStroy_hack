param(
    [string]$Model = "D:\Models\Qwen3-4B-GGUF\Qwen3-4B-Q4_K_M.gguf",
    [int]$Repetitions = 3
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
if (-not (Get-Command llama-bench -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: llama-bench was not found in PATH. Install/use the matching llama.cpp tools build."
    exit 1
}
if (-not (Test-Path $Model)) {
    Write-Host "ERROR: model not found: $Model"
    exit 1
}

$cpu = Get-CimInstance Win32_Processor
$physical = [int](($cpu | Measure-Object -Property NumberOfCores -Sum).Sum)
$logical = [int](($cpu | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum)
$half = [Math]::Max(1, [int][Math]::Floor($physical / 2))
$values = @(1, $half, $physical, $logical) | Sort-Object -Unique
$threadCsv = ($values -join ",")

Write-Host "Running llama-bench. Stop llama-server first for clean numbers."
Write-Host "Physical=$physical Logical=$logical Candidates=$threadCsv"
$raw = & llama-bench -m $Model -p 768 -n 96 -t $threadCsv -r $Repetitions -o json
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$rows = $raw | ConvertFrom-Json

$pp = $rows | Where-Object { [int]$_.n_prompt -gt 0 -and [int]$_.n_gen -eq 0 } | Sort-Object {[double]$_.avg_ts} -Descending
$tg = $rows | Where-Object { [int]$_.n_gen -gt 0 -and [int]$_.n_prompt -eq 0 } | Sort-Object {[double]$_.avg_ts} -Descending
if (-not $pp -or -not $tg) {
    Write-Host "ERROR: unexpected llama-bench output."
    exit 1
}

$bestPrompt = $pp | Select-Object -First 1
$bestGen = $tg | Select-Object -First 1
$result = [ordered]@{
    model = $Model
    physical_cores = $physical
    logical_processors = $logical
    generation_threads = [int]$bestGen.n_threads
    batch_threads = [int]$bestPrompt.n_threads
    generation_tps = [Math]::Round([double]$bestGen.avg_ts, 3)
    prompt_tps = [Math]::Round([double]$bestPrompt.avg_ts, 3)
}
$result | ConvertTo-Json | Set-Content -Path ".llama_tuning.json" -Encoding UTF8

Write-Host ""
Write-Host "Best generation threads: $($result.generation_threads) ($($result.generation_tps) tok/s)"
Write-Host "Best prompt threads:     $($result.batch_threads) ($($result.prompt_tps) tok/s)"
Write-Host "Saved: .llama_tuning.json"
Write-Host "run_llama_cpu_tuned.ps1 will use these values automatically."
