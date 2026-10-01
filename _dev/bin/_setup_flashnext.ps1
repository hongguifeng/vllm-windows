# Rebuild Flash-Next from tracked sources, never from a surviving venv.
[CmdletBinding()]
param(
    [string]$Repo = 'D:\code\vllm-windows',
    [string]$VenvPath = '',
    [string]$RuntimeRoot = '',
    [int]$Port = 9393,
    [switch]$Execute,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
if (-not $VenvPath) { $VenvPath = Join-Path $Repo '.venv-flashnext' }
if (-not $RuntimeRoot) { $RuntimeRoot = Join-Path $Repo '.flashnext-root' }
$Repo = [IO.Path]::GetFullPath($Repo).TrimEnd('\')
$VenvPath = [IO.Path]::GetFullPath($VenvPath).TrimEnd('\')
$RuntimeRoot = [IO.Path]::GetFullPath($RuntimeRoot).TrimEnd('\')
$bin = Join-Path $Repo '_dev\bin'
if ($VenvPath -eq "$Repo\.venv" -or $RuntimeRoot -eq $Repo) {
    throw 'Use an isolated venv and a runtime root outside the source package.'
}
if (Test-Path (Join-Path $RuntimeRoot 'vllm')) {
    throw 'The runtime root contains vllm source and would shadow site-packages.'
}
$steps = @(
    "Provision $VenvPath with uv and tracked Windows constraints",
    "Create $RuntimeRoot\.venv junction -> $VenvPath (never delete a directory)",
    'Build/install vLLM from this checkout with --no-build-isolation',
    'Deploy tracked vLLM Python sources and apply the Humming Windows patch',
    'Build Humming device-info, NVRTC, cubin patcher and launcher artifacts',
    'Compile the buffered IOCP PLE reader from tracked C source',
    'Check deployed Python sources; start the service separately'
)
if ($DryRun -or -not $Execute) {
    $steps | ForEach-Object { Write-Output $_ }
    return
}
$listeners = @(Get-NetTCPConnection -State Listen -ErrorAction Stop)
if ($listeners.LocalPort -contains $Port) {
    throw "Port $Port is in use. Stop Flash-Next explicitly before rebuilding; this script never stops it."
}
Get-Command uv -ErrorAction Stop | Out-Null

function Step([string]$Name, [scriptblock]$Command) {
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    $global:LASTEXITCODE = 0
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$Name failed (exit $LASTEXITCODE)" }
}

$env:CUDA_VISIBLE_DEVICES = '1'
Step $steps[0] { & "$bin\_flashtest_provision.ps1" -Repo $Repo -VenvPath $VenvPath }
New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null
$link = Join-Path $RuntimeRoot '.venv'
$existing = Get-Item -LiteralPath $link -Force -ErrorAction SilentlyContinue
if ($existing) {
    $target = [string]($existing.Target | Select-Object -First 1)
    if (-not ($existing.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
        -not $target -or [IO.Path]::GetFullPath($target).TrimEnd('\') -ne $VenvPath) {
        throw "Refusing to replace $link; it is not the expected venv junction."
    }
} else {
    New-Item -ItemType Junction -Path $link -Target $VenvPath | Out-Null
}
Step $steps[2] { & "$bin\_flashtest_build.ps1" -Repo $Repo -VenvPath $VenvPath }
$python = Join-Path $VenvPath 'Scripts\python.exe'
Push-Location $RuntimeRoot
try {
    Step 'Deploy vLLM runtime' {
        & $python "$bin\_patch_vllm_windows_runtime.py" --venv-root $RuntimeRoot
    }
    Step 'Patch Humming runtime' {
        & $python "$bin\_patch_humming_windows.py" --repo $RuntimeRoot --apply
    }
    foreach ($helper in @('device_info', 'nvrtc', 'cubinpatch', 'launcher')) {
        Step "Build Humming $helper" { & "$bin\_humming_$helper.ps1" -Venv $RuntimeRoot }
    }
    Step 'Build buffered IOCP PLE helper' {
        & "$bin\_ple_ssd_io_win_build.ps1" -Repo $Repo -OutDir "$Repo\_dev\out\ple_ssd_io"
    }
    Step 'Check tracked runtime deployment' {
        & $python "$bin\_patch_vllm_windows_runtime.py" --venv-root $RuntimeRoot --check
    }
} finally {
    Pop-Location
}
Write-Host 'Build complete. No service was started or stopped.' -ForegroundColor Green
Write-Host "Start with: & '$Repo\start_qwen38_flash_next.ps1' -Venv '$RuntimeRoot'"
