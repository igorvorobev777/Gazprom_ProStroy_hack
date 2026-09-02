$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

if (-not (Test-Path ".env")) {
    if (Test-Path ".env.hihub.example") {
        Copy-Item ".env.hihub.example" ".env"
    } elseif (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
    } else {
        New-Item -ItemType File -Path ".env" | Out-Null
    }
}

$email = Read-Host "HiHub account email"
$securePassword = Read-Host "HiHub password" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
try {
    $password = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
}

function Quote-DotEnv([string]$value) {
    $escaped = $value.Replace("\", "\\").Replace('"', '\"')
    return '"' + $escaped + '"'
}

$lines = [System.Collections.Generic.List[string]]::new()
Get-Content ".env" | ForEach-Object { [void]$lines.Add($_) }

function Upsert-Env([string]$key, [string]$value) {
    $pattern = '^\s*' + [regex]::Escape($key) + '\s*='
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match $pattern) {
            $lines[$i] = "$key=$value"
            return
        }
    }
    [void]$lines.Add("$key=$value")
}

Upsert-Env "HIHUB_BASE_URL" "https://hihub.ru"
Upsert-Env "HIHUB_EMAIL" (Quote-DotEnv $email)
Upsert-Env "HIHUB_PASSWORD" (Quote-DotEnv $password)
Upsert-Env "HIHUB_TOKEN_NAME" '""'
Upsert-Env "HIHUB_SECTION_ID" "0"
Upsert-Env "HIHUB_TIMEOUT_SECONDS" "30"
Upsert-Env "HIHUB_PER_PAGE" "200"
Upsert-Env "HIHUB_MAX_ARTICLES" "0"

Set-Content -Path ".env" -Value $lines -Encoding UTF8
$password = $null
$securePassword = $null

Write-Host "HiHub settings saved to .env."
Write-Host "Next: python -m scripts.check_hihub; then python -m scripts.build_index --force"
