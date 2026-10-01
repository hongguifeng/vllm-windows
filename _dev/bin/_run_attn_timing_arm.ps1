# Launch the attention build timing arm: the B arm plus a wall clock timer on
# attention metadata construction, so its cost can be compared with the per step
# idle gap the engine reports.
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
    '-AttnBuildTiming', '-AttnBuildLogEvery', '200'
)
exit $LASTEXITCODE
