param([string]$Path = 'D:\code\vllm-windows\_dev\bin\_flashnext_struct_serve.ps1')
$errs = $null
$null = [System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$null, [ref]$errs)
if ($errs -and $errs.Count) {
    Write-Host "PARSE_FAIL count=$($errs.Count)"
    foreach ($e in $errs) { Write-Host ("line " + $e.Extent.StartLineNumber + ": " + $e.Message) }
    exit 1
}
Write-Host 'PARSE_OK'
exit 0
