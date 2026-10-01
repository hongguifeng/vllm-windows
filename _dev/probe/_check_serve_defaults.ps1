# Regression check for start_server.ps1's *defaults*: they must keep resolving to the
# configuration that was verified on 2026-09-22 (71,680 ctx / 5.8e9 pin / capture
# ceiling 8 / vision copy released).  Run this after touching the param block:
#
#   pwsh -NoProfile -File _dev\probe\_check_serve_defaults.ps1
#
# Each probe runs in its OWN pwsh child process.  Calling the script in-process
# with `& $script @extra` returns nothing for the non-empty cases (the splatted
# arguments vanish), which looks exactly like a failure -- do not "simplify"
# this back into an in-process loop.
$script = 'D:\code\vllm-windows\start_server.ps1'
$pwsh   = 'C:\Program Files\PowerShell\7\pwsh.exe'

$errs = $null
[void][System.Management.Automation.Language.Parser]::ParseFile($script, [ref]$null, [ref]$errs)
if ($errs.Count) {
    "ParseFile: $($errs.Count) ERRORS"
    $errs | ForEach-Object { "  " + $_.Message }
    exit 1
}
"ParseFile: OK"

function Get-Field([string]$text, [string]$pattern) {
    [regex]::Match($text, $pattern).Groups[1].Value.Trim()
}

function Probe([string]$label, [string]$extra) {
    $out = & $pwsh -NoProfile -Command "& '$script' -DryRun $extra *>&1" | Out-String
    "{0,-25} maxlen={1,-6} kv={2,-11} ceil={3,-3} vision={4}" -f `
        $label,
        (Get-Field $out '--max-model-len (\d+)'),
        (Get-Field $out '--kv-cache-memory-bytes (\d+)'),
        (Get-Field $out 'max_cudagraph_capture_size":(\d+)'),
        (Get-Field $out 'vision copy ([^(]*)\(')
}

Probe 'default'                   ''
Probe '-MaxSeqs 32'               '-MaxSeqs 32'
Probe '-MaxSeqs 64'               '-MaxSeqs 64'
Probe '-MaxGraphCapture 64'       '-MaxGraphCapture 64'
Probe '-ReleaseOffloadCopy:false' '-ReleaseOffloadCopy:$false'
# A stray "  async      on" or similar may trail the last row: the child pwsh
# inherits the console handle and some of its output bypasses the capture.  It
# is cosmetic -- every field above is parsed out of the captured string.
