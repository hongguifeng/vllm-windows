# Launch the B arm: x2 depth, step spans, and in-graph phase pairs.
# Same knobs as the x2 battle baseline, with both observers on.
$ErrorActionPreference = 'Stop'

& 'D:\code\vllm-windows\_dev\bin\_serve_bg.ps1' -ServeArgs @(
    '-Venv', 'D:\code\vllm-windows',
    '-FullWeights',
    '-PleSsd', '-PleDepth', '256', '-PleCacheMb', '512',
    '-PleWorkers', '16', '-PlePrefetchTokens', '16384',
    '-Graphs',
    '-Mtp', '-MtpTokens', '2',
    '-KvGiB', '14',
    '-StepTimingWindow', '200',
    '-PhaseEvents', '-PhaseEventsLogEvery', '500'
)
exit $LASTEXITCODE
