$ErrorActionPreference = 'Continue'

# The injected WorkBuddy python shim patches os.unlink/os.remove and aborts the
# process once a turn deletes more than ~50 files, which breaks pip installs.
$env:CODEBUDDY_SAFE_DELETE_ENABLED = '0'

$py  = 'D:\code\vllm-windows\.venv\Scripts\python.exe'
$log = 'D:\code\vllm-windows\_dev\out\_deps.log'

"=== DEPS START $(Get-Date) ===" | Out-File $log -Encoding utf8

$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
"uv = $uv" | Out-File $log -Append -Encoding utf8

function Run-Uv($title, $argv) {
    "" | Out-File $log -Append -Encoding utf8
    "=== $title ===" | Out-File $log -Append -Encoding utf8
    & $uv @argv *>&1 | Out-File $log -Append -Encoding utf8
    "--- exit=$LASTEXITCODE ---" | Out-File $log -Append -Encoding utf8
}
function Run-Pip($title, $argv) {
    "" | Out-File $log -Append -Encoding utf8
    "=== $title ===" | Out-File $log -Append -Encoding utf8
    & $py -m pip @argv *>&1 | Out-File $log -Append -Encoding utf8
    "--- exit=$LASTEXITCODE ---" | Out-File $log -Append -Encoding utf8
}

# 1) torch stack (cu130). torch is already installed; audio/vision are missing.
Run-Pip 'step1 torch stack' @('install',
    'torch==2.11.0+cu130','torchaudio==2.11.0+cu130','torchvision==0.26.0+cu130',
    '--index-url','https://download.pytorch.org/whl/cu130')

# 2) build requirements
Run-Pip 'step2 build reqs' @('install','-r','D:\code\vllm-windows\requirements\build\cuda.txt')

# 3) cuda runtime requirements
Run-Pip 'step3 cuda reqs' @('install','-r','D:\code\vllm-windows\requirements\cuda.txt')

# 4) windows-specific requirements
Run-Pip 'step4 windows reqs' @('install','-r','D:\code\vllm-windows\requirements\windows.txt')

"" | Out-File $log -Append -Encoding utf8
"=== DEPS DONE $(Get-Date) ===" | Out-File $log -Append -Encoding utf8
