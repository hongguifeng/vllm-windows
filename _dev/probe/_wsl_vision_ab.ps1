param([int]$Port = 18020, [int]$Tokens = 256)
# A/B on the WSL-side container: text baseline -> image request -> text again.
# Mirrors the Windows-side measurement (same client, same token count) so the two
# p50_gap numbers are directly comparable.  Run with:
#   pwsh -NoProfile -File _dev\probe\_wsl_vision_ab.ps1
$ErrorActionPreference = "Continue"
$repo = "D:\code\vllm-windows"
$py   = "C:\Users\hong\.workbuddy\binaries\python\versions\3.13.12\python.exe"
$out  = "$repo\_dev\out\_wsl_vision_ab.txt"
Remove-Item $out -ErrorAction SilentlyContinue

function W($m) { Add-Content -Encoding utf8 $out $m }
function Mem { (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) }

W "=== WSL container A/B, port $Port, tokens $Tokens ==="
W ("t0 mem          : " + (Mem) + " MiB")

W ""
W "--- [1] text baseline ---"
& $py "$repo\_dev\bench\_itl_client.py" --port $Port --tokens $Tokens --warm 0 --label wsl-before 2>&1 |
    Add-Content -Encoding utf8 $out
W ("mem after text  : " + (Mem) + " MiB")

W ""
W "--- [2] vision smoke (one image, then the same image again) ---"
& "$repo\.venv\Scripts\python.exe" "$repo\_dev\test\_vision_smoke.py" --port $Port 2>&1 |
    Add-Content -Encoding utf8 $out
W ("mem after image : " + (Mem) + " MiB")

W ""
W "--- [3] text again ---"
& $py "$repo\_dev\bench\_itl_client.py" --port $Port --tokens $Tokens --warm 0 --label wsl-after 2>&1 |
    Add-Content -Encoding utf8 $out
W ("mem after text2 : " + (Mem) + " MiB")
W "done"
"written -> $out"
