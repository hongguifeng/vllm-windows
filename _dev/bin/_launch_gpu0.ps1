# Detached launcher: serve on GPU 0 only (first CMP 170HX).
# Launched via: Start-Process pwsh -ArgumentList '-NoProfile','-File',<this file> -WindowStyle Hidden
$env:CUDA_VISIBLE_DEVICES = '0'
& 'D:\code\vllm-windows\start_server.ps1'
