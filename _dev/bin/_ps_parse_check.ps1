# Parse a PowerShell script without running it, so edits can be checked cheaply.
param([Parameter(Mandatory = $true)][string]$Path)

if (-not (Test-Path -LiteralPath $Path)) {
    Write-Host "missing file: $Path"
    exit 1
}
$text = Get-Content -LiteralPath $Path -Raw
if ([string]::IsNullOrWhiteSpace($text)) {
    Write-Host "empty content read from: $Path"
    exit 1
}
$errors = $null
$tokens = $null
[System.Management.Automation.Language.Parser]::ParseInput(
    $text, [ref]$tokens, [ref]$errors) > $null

if ($errors.Count -eq 0) {
    if ($tokens.Count -eq 0) {
        Write-Host "parse read nothing: $Path"
        exit 1
    }
    Write-Host "parse OK: $Path ($($tokens.Count) tokens)"
    exit 0
}
foreach ($e in $errors) {
    Write-Host ("{0}:{1} {2}" -f $e.Extent.StartLineNumber,
        $e.Extent.StartColumnNumber, $e.Message)
}
exit 1
