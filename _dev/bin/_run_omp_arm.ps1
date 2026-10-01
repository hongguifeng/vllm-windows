# Launch the OMP arm: the B arm exactly, plus pinned host threads.
# Everything else is byte-identical to _run_b_arm.ps1 so the two are comparable.
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
    '-PhaseEvents', '-PhaseEventsLogEvery', '500',
    '-OmpThreads', '1'
)
exit $LASTEXITCODE
