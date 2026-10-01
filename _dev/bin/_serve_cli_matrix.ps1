# Regression matrix for start_server.ps1's argv assembly (the script lives in the repo
# root; this matrix stays in _dev\bin with the rest of the scaffolding).
#
# Changing a default changes the behaviour of EVERY existing caller, and "the
# script still parses" proves nothing about that.  This runs start_server.ps1 -DryRun
# under each combination that matters and asserts the two argv fields the
# 2026-09-22 default flip touched: the scheduling flag and --max-model-len.
#
#   pwsh -File .\_serve_cli_matrix.ps1
#
# -DryRun never execs python, so this is cheap and safe.  It DOES trip the port
# pre-flight, so stop a running server first (otherwise every case fails on
# "port 29550 is already in use" rather than on argv).
#
# Expected as of 2026-09-22 (async became the default; the context guard was
# re-scoped from "is async on" to "is the buggy flashinfer top-k back on"):
#
#   default                          --async-scheduling     71680
#   -NoAsync                         --no-async-scheduling  71680
#   -Async           (legacy alias)  --async-scheduling     71680
#   -Async:$false                    --no-async-scheduling  71680
#   + DFLASH2_TOPK_IMPL=flashinfer   --async-scheduling     60928   <- guard fires
#   ... + -AllowUnstable             --async-scheduling     71680
#   ... + -TextOnly                  --async-scheduling     71680   <- guard exempt

$repo = 'D:\code\vllm-windows'   # start_server.ps1 sits in the repo root, not next to this file
if (-not (Test-Path "$repo\start_server.ps1")) { throw "start_server.ps1 not found under $repo" }

$script:lines = @()
$script:fail = 0

function Test-Case {
    param(
        [string]$Label,
        [string[]]$Splat = @(),
        [hashtable]$Env = @{},
        [string]$WantAsync,
        [int]$WantLen
    )
    $saved = @{}
    foreach ($k in $Env.Keys) {
        $saved[$k] = [Environment]::GetEnvironmentVariable($k)
        [Environment]::SetEnvironmentVariable($k, $Env[$k])
    }
    try {
        $txt = (& pwsh -NoProfile -File "$repo\start_server.ps1" -DryRun @Splat 2>&1) -join "`n"
    } finally {
        foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
    }
    # Test --no-async-scheduling FIRST: the plain name is a substring of it, so
    # matching the async form first would report every no-async run as async.
    $gotAsync = if ($txt -match '--no-async-scheduling') { '--no-async-scheduling' }
                elseif ($txt -match '--async-scheduling') { '--async-scheduling' }
                else { '<neither -- server probably failed pre-flight>' }
    $gotLen = if ($txt -match '--max-model-len (\d+)') { [int]$Matches[1] } else { -1 }
    $ok = ($gotAsync -eq $WantAsync -and $gotLen -eq $WantLen)
    if (-not $ok) { $script:fail++ }
    $script:lines += "$(if ($ok) { 'PASS' } else { 'FAIL' })  $Label"
    $script:lines += "        want $WantAsync / $WantLen   got $gotAsync / $gotLen"
}

Test-Case 'default (no flags)'                    -WantAsync '--async-scheduling'    -WantLen 71680
Test-Case '-NoAsync'                 @('-NoAsync')            -WantAsync '--no-async-scheduling' -WantLen 71680
Test-Case '-Async (legacy alias)'    @('-Async')              -WantAsync '--async-scheduling'    -WantLen 71680
Test-Case '-Async:$false'            @('-Async:$false')       -WantAsync '--no-async-scheduling' -WantLen 71680
Test-Case '+ flashinfer (guard fires)'   -Env @{ DFLASH2_TOPK_IMPL = 'flashinfer' } -WantAsync '--async-scheduling' -WantLen 60928
Test-Case '+ flashinfer -AllowUnstable'  -Splat @('-AllowUnstable') -Env @{ DFLASH2_TOPK_IMPL = 'flashinfer' } -WantAsync '--async-scheduling' -WantLen 71680
Test-Case '+ flashinfer -TextOnly'       -Splat @('-TextOnly')      -Env @{ DFLASH2_TOPK_IMPL = 'flashinfer' } -WantAsync '--async-scheduling' -WantLen 71680

$script:lines | ForEach-Object { Write-Host $_ }
Write-Host ''
if ($script:fail -gt 0) {
    Write-Host "$($script:fail) case(s) FAILED" -ForegroundColor Red
    exit 1
}
Write-Host 'all cases PASS' -ForegroundColor Green
exit 0
