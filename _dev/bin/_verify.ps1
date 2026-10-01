. 'D:\code\vllm-windows\_dev\bin\_env.ps1'

$REPO = 'D:\code\vllm-windows'
$DEV  = "$REPO\_dev"
$log  = "$DEV\out\_verify.log"

'=== VERIFY START ' + (Get-Date) | Out-File $log -Encoding utf8

# Must NOT run from the repo root: the checked-out `vllm/` source tree would
# shadow the installed package and the built extension modules.
Set-Location 'C:\Users\hong'

$py = "$REPO\.venv\Scripts\python.exe"

(& $py -X faulthandler "$DEV\bin\_verify.py") *>&1 | Out-File $log -Append -Encoding utf8
$rc = $LASTEXITCODE
"=== VERIFY EXIT=$rc ===" | Out-File $log -Append -Encoding utf8

exit $rc
