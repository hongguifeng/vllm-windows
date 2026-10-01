# Launch the clean production arm: the promotion baseline configuration with no
# observers at all, so step intervals measured from outside are the engine's own
# pace and not an artifact of our logging.
$ErrorActionPreference = 'Stop'

& 'D:\code\vllm-windows\_dev\bin\_serve_bg.ps1' -ServeArgs @(
    '-Venv', 'D:\code\vllm-windows',
    '-FullWeights',
    '-PleSsd', '-PleDepth', '256', '-PleCacheMb', '512',
    '-PleWorkers', '16', '-PlePrefetchTokens', '16384',
    '-Graphs',
    '-Mtp', '-MtpTokens', '2',
    '-KvGiB', '14'
)
exit $LASTEXITCODE
