# Repro for the 2.2c crash (tower + 71,680 + --async-scheduling + C2/C4 ->
# device-side assert, engine dead) with an ACCURATE python stack.
#
# The crash logs we have report `CUDA error: invalid argument` at
# async_utils.py:168 copy_event.synchronize() -- that is only the async
# report point. CUDA_LAUNCH_BLOCKING=1 serializes every CUDA call, so the
# traceback at the first device-side assert names the true faulting kernel
# launch (embedding gather OOB vs input_ids scatter OOB vs masked_scatter
# size check -- the three candidates from upstream issues #36906 / #24799 /
# #30624).
#
# Usage (run in YOUR OWN terminal, not the agent's). Lives in _dev\probe, so
# invoke it with its full path from any cwd:
#   & D:\code\vllm-windows\_dev\probe\_repro_async_tower.ps1
#                                            # DEFAULT: true crash config, no
#                                            # blocking, full era stage order
#   .\_repro_async_tower.ps1 -Blocking 1     # only if the above crashes and you
#                                            # want the exact faulting launch
#   .\_repro_async_tower.ps1 -Spec none      # discriminator: no speculative
#                                            # decoding -- if THIS survives
#                                            # C1-C8, spec is a required
#                                            # ingredient of the crash
#   .\_repro_async_tower.ps1 -Stages cohort  # the 20:47/21:02 shape (no warmup,
#                                            # no long prefill) -- kept only to
#                                            # re-test that variable
#   .\_repro_async_tower.ps1 -IdCheck -NoDraftGraph -Repeat 4
#                                            # probe + 4 ladder passes on one
#                                            # boot; -NoDraftGraph is REQUIRED
#                                            # for the probe to see anything
#   .\_repro_async_tower.ps1 -Clamp -Repeat 4 # IN-GRAPH guard + 4 passes, with
#                                            # the drafter graph ON (default).
#                                            # This is the experiment that can
#                                            # instrument a replay; -NoDraftGraph
#                                            # must NOT be combined with it.
#
# 2026-09-21 22:56 (7th shot: -IdCheck -NoDraftGraph -Repeat 4, unconditional
# flush): CLEAN, and for the first time the probe PROVED it stayed alive --
# 403 flush lines covering steps=50 .. 20150 over 21.6 min (~15.5 step/s), all
# reading `cum cand ... neg=0 ge=0 | cum anchor ... neg=0 ge=0`.  Extreme values
# never left [0, 248069].  So the eager path never sees an out-of-range id on
# 4 ladder passes with num_reqs swinging 1..8, and 8 clean passes mean
# -NoDraftGraph has never crashed.  Conclusion: the bad id is produced inside
# the REPLAYED drafter graph, where no Python probe can run.
# => the guard itself has to live in the graph: -Clamp (see _clamp_exp.py).
#
# 2026-09-21 22:19 (6th shot, first with -NoDraftGraph): CLEAN over 4 passes,
# and the drafter really was eager (its "Capturing dflash2 CUDA graphs" bar is
# gone and the capture fingerprint drops 1.30 -> 1.21 GiB).  But the probe only
# printed 2 lines: ARMED at step 1 and one flush at step 25, then nothing for
# the remaining 22 min.  That log line was gated on "the accumulated min/max
# CHANGED", and `now` values were not printed, so silence had two readings:
# the probe stopped being called, or the running min/max simply never moved
# (candidate/anchor ids come from a narrow, high-confidence token set, so the
# running extremes can legitimately stick).  The flush is now UNCONDITIONAL and
# prints the current step's ids too, so `steps=` alone proves liveness.
#
# 2026-09-21 21:59 (5th shot of the true crash config): CLEAN again, and the
# [idcheck] probe printed nothing at all.  Two readings -- "no bad id" or "never
# armed" -- were indistinguishable, because start_server.ps1's env whitelist drops
# DFLASH2_IDCHECK.  Both ends of that are now fixed: the probe logs an
# unconditional '[idcheck] ARMED' line at step 1, start_server.ps1 prints the probe
# state in the header and dumps DFLASH2_* in --- env ---.
#
# Reading the call path afterwards showed the probe could not have worked
# anyway: the drafter captures FULL CUDA graphs per decode bucket and the
# runner REPLAYS them (dflash/speculator.py: cg_mode == FULL -> run_fullgraph),
# so _generate_draft's Python body never executes on a decode step.  Hence
# -NoDraftGraph (DFLASH2_NO_DRAFT_CUDAGRAPH=1, needs _patch_draft_eager.py
# --apply), which builds the drafter with CUDAGraphMode.NONE so the Python runs
# every step.  Expect the drafter's "Capturing ... CUDA graphs" progress bar and
# the 1.30 GiB capture fingerprint to DISAPPEAR in that mode -- that is the
# point, not a bad boot.
#
# 5 shots so far, and the two runs are otherwise byte-identical: 09-20 2/2
# crash; 09-21 1 crash / 4 clean.  Same compile-cache hashes (a8addbd9ba +
# aot 60aeee7e/ac66d4b5/cec1ef79), same 22.76 GiB free, same 1.30 GiB capture,
# same 5 JIT kernels -- so it is a nondeterministic race, and the only lever
# is exposure: -Repeat multiplies decode steps at C2/C4 per boot.
#
# 2026-09-21 results, and why the default stage set changed:
#   20:47  blocking ON,  --only cohort  -> clean (8/8 x4, 0 errors)
#   21:02  blocking OFF, --only cohort  -> clean (8/8 x4, 0 errors)   <- the true
#          crash config, so the first conclusion was "not reproducible".
#   But both of those ran the cohort on a COLD-ish cache: `_ab_bench.py`, which
#   produced the 09-20 crashes, defaults to `--only warmup,prefill,cohort` and
#   ran the long-prefill ladder (1024x16, 16384x4) BEFORE the cohort.  Every
#   crash-era JSON shows the prefill rows ahead of the failing cohort row.  A
#   cohort-only run therefore never establishes the long-context KV state the
#   crash happened in.  Default fixed to warmup,prefill,cohort + --lens
#   1024,16384 to restore that sequence.
#
# 2026-09-21 20:47 result: with blocking ON the crash did NOT reproduce --
# tower + 71,680 + async, C1-C8 all 8/8 OK, 0 errors in the server log.  So the
# blocking mode is itself suspect: it serializes kernel launches, and a crash
# that needs the async overlap window disappears.  `-Blocking 0` re-tests the
# real config; if the crash comes back there, the next tool is
# compute-sanitizer (memcheck + --target-processes all), not blocking mode.
#
# What it does: starts the server (tower + 71,680 + -Async + -AllowUnstable),
# waits for /health, runs warmup -> prefill ladder -> C1-C8 cohort, then prints
# the log excerpt around the FIRST CUDA error. Paste that excerpt back.

[CmdletBinding()]
param(
    [ValidateSet('dflash', 'none')]
    [string]$Spec = 'dflash',
    [int]$WaitMin = 8,
    # '0' = the plain crash config (async untouched) -- THE DEFAULT, because the
    #       2026-09-21 20:47 run proved blocking is not innocent here.
    # '1' = CUDA_LAUNCH_BLOCKING=1: exact python stack at the faulting launch,
    #       BUT it serializes every CUDA call, which hides the race.  Measured
    #       20:47: blocking ON, tower + 71,680 + async + C1-C8 -> NO crash
    #       (8/8 OK in all four stages), 0 errors in the server log.
    #       Use it only if -Blocking 0 crashes and you want the stack.
    [ValidateSet('0', '1')]
    [string]$Blocking = '0',
    # Stage sequence, run in the order given.  The 09-20 crash runs used the
    # A/B protocol's default order -- warmup, then the long-prefill ladder,
    # THEN the cohort -- so the KV cache held 4x16,384-token requests by the
    # time C4 ran.  The 09-21 repro runs used `--only cohort` and came back
    # clean; that missing warm state is the leading suspect.  Don't shorten
    # this to 'cohort' unless you deliberately want to re-test that variable.
    [string]$Stages = 'warmup,prefill,cohort',
    # Prefill ladder; 1024,16384 is what `_ab_bench.py --lens 1024,16384` ran.
    [string]$Lens = '1024,16384',
    # Sets DFLASH2_IDCHECK=1 for the spawned server, which makes
    # dflash2/speculator.py:_generate_draft log the min/max/out-of-range counts
    # of candidate_ids and anchor_token_ids (see _idcheck_exp.py).
    # The 21:43 crash is an out-of-range id into the [248320, 256] codebook
    # gather in qwen3_dflash2.py:181, so this says WHICH tensor carries it and
    # what the value is.  Requires `_idcheck_exp.py --apply` first.
    # Note: `grep -c '\[idcheck\]'` == 0 in the serve log now unambiguously
    # means "never armed" -- the probe logs one `[idcheck] ARMED` line at
    # step 1.  start_server.ps1's header also prints `idcheck on/off`.
    [switch]$IdCheck,
    # Probe flush cadence in decode steps (only with -IdCheck).  50 ≈ 10k steps
    # per pass -> a few hundred lines, and every line now carries `steps=`, so
    # the counter alone proves the probe is live.  1 = flush every step:
    # tightest coverage of the value that trips the assert (the assert fires
    # one line later, so every=1 is the only setting that captures THAT step),
    # but one sync per step perturbs the async overlap -- use it to CONFIRM,
    # not to hunt.
    [int]$Every = 50,
    # Run the DRAFTER eagerly (DFLASH2_NO_DRAFT_CUDAGRAPH=1; see
    # _patch_draft_eager.py).  Without this, -IdCheck is pointless: the drafter
    # captures FULL CUDA graphs for every decode bucket and REPLAYS them, so
    # _generate_draft's Python body never executes at decode time and the probe
    # cannot observe any id.  This is why the 21:59 run's probe was silent even
    # in the "value unchanged" case.  Diagnostic only -- it costs decode speed.
    [switch]$NoDraftGraph,
    # Inject the IN-GRAPH guard (DFLASH2_CLAMP=1; see _clamp_exp.py).  This is
    # the follow-up to the eager probe coming back clean over 20,150 steps:
    # if the bad id is produced inside the replayed drafter graph, no Python
    # probe can see it, so the guard itself has to live in the graph.  It
    # counts out-of-range `candidate_ids` / `anchor_token_ids` and clamps them,
    # and propose() reads the counter back every 50 steps.
    # Run it WITHOUT -NoDraftGraph: the whole point is the replay path.
    # Requires `_clamp_exp.py --apply` first.
    [switch]$Clamp,
    # Run the stage ladder N times against ONE boot, each pass into its own
    # VBENCH_OUT dir under the run directory, so a run can never overwrite a
    # previous run's evidence.  The crash is 1-in-5 on this tree (09-20: 2/2,
    # 09-21: 1 crash / 4 clean), while a boot costs ~3 min and the ladder ~6 --
    # so repeating the ladder multiplies the exposure per boot instead of paying
    # another warm-up per shot.
    [int]$Repeat = 1,
    # Run the WHOLE thing N times, each in a fresh process (fresh CUDA context,
    # fresh graph capture, fresh warm-up), into one run directory:
    #     _bench_repro\run_<yyyyMMdd_HHmmss>\boot<N>[\pass<M>]
    # -Repeat widens exposure to the RUNTIME race; -Boots widens it to the
    # STARTUP one.  2.2c had a face there too -- CUDA graph capture executes the
    # graph once, so a capture that hits the bad top-k dies inside
    # torch.cuda.graph() -- and 4 passes on a single boot say nothing about it.
    # -Boots 3 -Repeat 1 is the acceptance shape for the 2026-09-22 fix: three
    # startups, each followed by one full ladder.
    [int]$Boots = 1,
    # Set by the -Boots driver so every child lands in the same run directory
    # (a child would otherwise mint its own timestamp).  Not for hand use.
    [string]$RunRoot = '',
    # 'boot<N>' label for the current child; empty on a plain single-boot run.
    [string]$BootLabel = ''
)

$ErrorActionPreference = 'Continue'
$REPO = 'D:\code\vllm-windows'
$DEV  = "$REPO\_dev"
$py = "$REPO\.venv\Scripts\python.exe"

# Serial CUDA execution.  The spawned server inherits it.  -Blocking 0 clears
# it so the run matches the original crash conditions exactly.
if ($Blocking -eq '1') {
    $env:CUDA_LAUNCH_BLOCKING = '1'
} else {
    Remove-Item Env:\CUDA_LAUNCH_BLOCKING -ErrorAction SilentlyContinue
}

# Keeps this run's bench logs out of `_bench_base` (the 170HX baseline set):
# a rejected/failed repro run must not be able to overwrite baseline evidence.
# Measured 2026-09-21 20:24 -- the previous version clobbered 4 baseline logs
# plus _bench_base\SUMMARY.md before anyone noticed the requests were 404ing.
# Inherited by the Start-Process'd server.  Off by default so a normal repro run
# keeps the original timing.
if ($IdCheck) {
    $env:DFLASH2_IDCHECK = '1'
    $env:DFLASH2_IDCHECK_EVERY = "$Every"
} else {
    Remove-Item Env:\DFLASH2_IDCHECK -ErrorAction SilentlyContinue
    Remove-Item Env:\DFLASH2_IDCHECK_EVERY -ErrorAction SilentlyContinue
}

# Requires _patch_draft_eager.py --apply.  -IdCheck without this cannot work:
# see the note on the -NoDraftGraph parameter.
if ($NoDraftGraph) {
    $env:DFLASH2_NO_DRAFT_CUDAGRAPH = '1'
    if ($IdCheck) {
        $probe = "$REPO\.venv\Lib\site-packages\vllm\v1\worker\gpu\spec_decode\dflash\speculator.py"
        if (-not (Select-String -Path $probe -Pattern 'draft-eager-exp' -Quiet)) {
            Write-Host "FATAL: -NoDraftGraph needs the override in dflash/speculator.py." -ForegroundColor Red
            Write-Host "Run:  & $py $DEV\probe\_patch_draft_eager.py --apply" -ForegroundColor Red
            exit 1
        }
    }
} else {
    Remove-Item Env:\DFLASH2_NO_DRAFT_CUDAGRAPH -ErrorAction SilentlyContinue
}

# Requires _clamp_exp.py --apply.  The guard only exists if both speculator
# copies carry the injected blocks, and a silent no-op would burn another
# 26-minute run, so refuse to start instead.
if ($Clamp) {
    $env:DFLASH2_CLAMP = '1'
    foreach ($rel in @('dflash2', 'dflash')) {
        $f = "$REPO\.venv\Lib\site-packages\vllm\v1\worker\gpu\spec_decode\$rel\speculator.py"
        if (-not (Select-String -Path $f -Pattern 'clamp-exp' -Quiet)) {
            Write-Host "FATAL: -Clamp needs the guard in $rel/speculator.py." -ForegroundColor Red
            Write-Host "Run:  & $py $DEV\probe\_clamp_exp.py --apply" -ForegroundColor Red
            exit 1
        }
    }
} else {
    Remove-Item Env:\DFLASH2_CLAMP -ErrorAction SilentlyContinue
}

# Keeps this run's bench logs out of `_bench_base` (the 170HX baseline set):
# a rejected/failed repro run must not be able to overwrite baseline evidence.
# Measured 2026-09-21 20:24 -- the previous version clobbered 4 baseline logs
# plus _bench_base\SUMMARY.md before anyone noticed the requests were 404ing.
#
# 2026-09-22: every run now gets its own timestamped directory.  Under -Repeat
# the pass dirs used to be fixed names ("_bench_repro\pass1".."passN"), so a NEW
# run silently overwrote the PREVIOUS run's evidence -- measured at 01:16, when
# that run's 4 passes destroyed the 23:34 run's SUMMARY.md files.  The 23:34 C8
# medITL (158 ms) now survives only in that run's serve log, as a 45.79 s
# elapsed_s.  Layout today: _bench_repro\run_<yyyyMMdd_HHmmss>[\pass<N>].
$benchRoot = "$DEV\out\_bench_repro"
if ($RunRoot) {
    $runRoot = $RunRoot
} else {
    $runStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $runRoot = "$benchRoot\run_$runStamp"
}
$env:VBENCH_OUT = $runRoot

# ------------------------------------------------------------- -Boots driver --
# The startup is the thing under test, so each boot must be a fresh pwsh + fresh
# python + fresh CUDA context.  Re-invoking this script is cleaner than growing
# a loop around the boot body: the body keeps its indentation, and the children
# cannot share process state by construction.
if ($Boots -gt 1) {
    $pwshExe = "$env:ProgramFiles\PowerShell\7\pwsh.exe"
    if (-not (Test-Path $pwshExe)) {
        $pwshExe = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
    }
    if (-not $pwshExe) {
        Write-Host "FATAL: -Boots needs pwsh 7 to re-invoke this script." -ForegroundColor Red
        exit 1
    }
    # Every parameter the caller passed, minus the three this driver owns.
    $childArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $MyInvocation.MyCommand.Path,
                   '-Boots', '1', '-RunRoot', $runRoot)
    foreach ($k in $PSBoundParameters.Keys) {
        if ($k -in @('Boots', 'RunRoot', 'BootLabel')) { continue }
        if ($PSBoundParameters[$k] -is [switch]) {
            if ($PSBoundParameters[$k].IsPresent) { $childArgs += "-$k" }
        } else {
            $childArgs += @("-$k", "$($PSBoundParameters[$k])")
        }
    }
    Write-Host "=== -Boots $($Boots): $Boots independent boots into $runRoot ===" -ForegroundColor Cyan
    Write-Host "    each boot = fresh pwsh + fresh python + fresh CUDA graph capture" -ForegroundColor Cyan
    # REPRO_PRINT_CHILD_ARGS=1 prints the child invocations instead of running
    # them -- the only way to check the parameter pass-through without paying
    # for (and interrupting) a real model load.  Added 2026-09-22.
    $printOnly = ($env:REPRO_PRINT_CHILD_ARGS -eq '1')
    if ($printOnly) {
        Write-Host "REPRO_PRINT_CHILD_ARGS=1: printing child invocations only, running nothing." -ForegroundColor Yellow
    }
    $rcs = @()
    for ($b = 1; $b -le $Boots; $b++) {
        Write-Host "`n############ boot $b / $Boots ############" -ForegroundColor Magenta
        if ($printOnly) {
            Write-Host "  child: $pwshExe $($childArgs -join ' ') -BootLabel boot$b" -ForegroundColor DarkGray
        } else {
            & $pwshExe @childArgs '-BootLabel' "boot$b"
            $rcs += $LASTEXITCODE
        }
        if ($b -lt $Boots) {
            # WDDM frees VRAM lazily (>90 s measured), and a boot started too
            # early dies with "Free memory ... less than GPU memory utilization"
            # -- which would read as a startup failure in the summary.
            Write-Host "waiting for VRAM to drain before boot $($b + 1) ..." -ForegroundColor DarkGray
            $dl = (Get-Date).AddSeconds(300)
            while ((Get-Date) -lt $dl) {
                $u = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null |
                     Select-Object -First 1
                if ($u -and [int]$u -lt 1500) { break }
                Start-Sleep -Seconds 5
            }
        }
    }
    $rcLine = if ($printOnly) { 'printed only, nothing ran' } else { "[$($rcs -join ', ')]" }
    Write-Host "`n=== -Boots $($Boots) done; per-boot stage rc = $rcLine ===" -ForegroundColor Cyan
    Write-Host "    rc=0 means the ladder ran AND every request was accepted; each boot's" -ForegroundColor Cyan
    Write-Host "    own crash/idcheck/clamp analysis is printed above. Run dir: $runRoot" -ForegroundColor Cyan
    exit
}

if ($Blocking -eq '1') {
    $mode = 'CUDA_LAUNCH_BLOCKING=1 (exact stack, may mask races)'
} else {
    $mode = 'blocking OFF (true crash config; stack will be the async report point)'
}
Write-Host "=== repro: tower + 71,680 + async + spec=$Spec, $mode ===" -ForegroundColor Cyan
Write-Host "passes: $Repeat   stages: [$Stages]   lens: [$Lens]" -ForegroundColor Cyan
if ($IdCheck) {
    Write-Host "idcheck: ON (every $Every steps) -- the serve log must carry one '[idcheck] ARMED' line;" -ForegroundColor Cyan
    Write-Host "         grep -c '\[idcheck\]' == 0 therefore means the probe was NOT armed." -ForegroundColor Cyan
    if (-not $NoDraftGraph) {
        Write-Host "!! -IdCheck without -NoDraftGraph: decode replays FULL CUDA graphs, so the probe" -ForegroundColor Red
        Write-Host "   will NOT run on decode steps. Add -NoDraftGraph (see _patch_draft_eager.py)." -ForegroundColor Red
    }
}
if ($NoDraftGraph) {
    Write-Host "drafter: EAGER (DFLASH2_NO_DRAFT_CUDAGRAPH=1) -- decode ids are now observable," -ForegroundColor Cyan
    Write-Host "         but decode is slower and the FULL-graph replay path is no longer exercised." -ForegroundColor Cyan
}
if ($Clamp) {
    Write-Host "clamp: ON (DFLASH2_CLAMP=1) -- the guard is captured INTO the drafter CUDA graph," -ForegroundColor Cyan
    Write-Host "       so unlike the Python probe it also runs on every replay. Watch for" -ForegroundColor Cyan
    Write-Host "       '[clamp] reads=N ... cand neg=.. ge=..' lines, and for whether a crash" -ForegroundColor Cyan
    Write-Host "       still happens at all." -ForegroundColor Cyan
    if ($NoDraftGraph) {
        Write-Host "!! -Clamp with -NoDraftGraph defeats the purpose: the guard exists precisely" -ForegroundColor Red
        Write-Host "   to instrument the REPLAY path. Drop -NoDraftGraph." -ForegroundColor Red
    }
}

# MUST launch under pwsh 7, not Windows PowerShell 5.1: 5.1's native-argument
# quoting eats the embedded double quotes, so
#   --default-chat-template-kwargs {"enable_thinking": false}
# reaches python as {enable_thinking: false} and api_server refuses to start
# with "invalid loads value". Measured 2026-09-21 20:16 (pid 3512, rc=2).
# 7.3+ passes the quotes through, which is also why running .\start_server.ps1
# directly in the VS Code terminal (pwsh 7.6.3) always worked.
$psExe = "$env:ProgramFiles\PowerShell\7\pwsh.exe"
if (-not (Test-Path $psExe)) {
    $psExe = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
}
if (-not $psExe) {
    Write-Host "FATAL: pwsh 7 not found. Windows PowerShell 5.1 mangles the JSON in" -ForegroundColor Red
    Write-Host "--default-chat-template-kwargs and the server will not start. Install pwsh 7 first." -ForegroundColor Red
    exit 1
}
Write-Host "launcher: $psExe"

# Remember which serve logs already exist, so the excerpt below can only ever
# come from THIS run (picking "whatever is newest" once analysed a stale log
# when the server failed to start).
$logBefore = @(Get-ChildItem "$DEV\out\logs\serve_*.log" -ErrorAction SilentlyContinue |
               Select-Object -ExpandProperty FullName)

$args_ = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "$REPO\start_server.ps1",
           '-Async', '-MaxLen', '71680', '-AllowUnstable')
if ($Spec -eq 'none') { $args_ += @('-Spec', 'none') }
$serve = Start-Process -PassThru -FilePath $psExe -ArgumentList $args_
Write-Host "server window started (pid $($serve.Id)); waiting for /health ..."

$deadline = (Get-Date).AddMinutes($WaitMin)
$healthy = $false
while ((Get-Date) -lt $deadline) {
    if ($serve.HasExited) {
        Write-Host "server exited during startup (rc=$($serve.ExitCode)) -- see its log" -ForegroundColor Red
        break
    }
    try {
        $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 5
        if ($r.StatusCode -eq 200) { $healthy = $true; break }
    } catch { Start-Sleep -Seconds 5 }
}

# This boot's evidence dir: a label dir when driven by -Boots, and pass<N> under
# it when -Repeat > 1.
$bootRoot = if ($BootLabel) { "$runRoot\$BootLabel" } else { $runRoot }
New-Item -ItemType Directory -Force -Path $bootRoot | Out-Null
$env:VBENCH_OUT = $bootRoot

if ($healthy) {
    Write-Host "healthy -- running stages [$Stages] (crash expected at C2/C4 of the cohort)" -ForegroundColor Cyan
    Write-Host "bench output -> $bootRoot" -ForegroundColor Cyan
    $rc = 0
    for ($pass = 1; $pass -le $Repeat; $pass++) {
        # Each pass gets its own log set: _bench_suite.py writes cohort_c1.log
        # etc. unconditionally, so a second pass would otherwise overwrite the
        # first pass's evidence (and a crash pass is THE evidence).
        $env:VBENCH_OUT = if ($Repeat -gt 1) { "$bootRoot\pass$pass" } else { $bootRoot }
        New-Item -ItemType Directory -Force -Path $env:VBENCH_OUT | Out-Null
        Write-Host "`n##### pass $pass / $Repeat -> $env:VBENCH_OUT #####" -ForegroundColor Magenta
        & $py "$DEV\bench\_bench_suite.py" --only $Stages --lens $Lens
        $rc = $LASTEXITCODE
        Write-Host "pass $pass stages rc=$rc"
        if ($rc -ne 0) {
            Write-Host "cohort output is NOT usable: the server rejected the requests, so this run" -ForegroundColor Red
            Write-Host "says nothing about the crash. Look at the '!!' lines above or" -ForegroundColor Red
            Write-Host "$($env:VBENCH_OUT)\cohort_c1.log" -ForegroundColor Red
            # A device-side assert kills the engine: the remaining passes would
            # only produce 500s, so stop here and keep this pass as the record.
            $alive = $false
            try {
                $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 5
                $alive = ($r.StatusCode -eq 200)
            } catch { $alive = $false }
            if (-not $alive) {
                Write-Host "server is dead after pass $pass -- stopping the ladder here" -ForegroundColor Red
                break
            }
            Write-Host "server still answers /health; continuing to the next pass" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "server never became healthy" -ForegroundColor Red
}

# Pull this run's serve log and show the region around the first CUDA error.
$log = Get-ChildItem "$DEV\out\logs\serve_*.log" -ErrorAction SilentlyContinue |
       Where-Object { $_.FullName -notin $logBefore } |
       Sort-Object LastWriteTime | Select-Object -Last 1
if (-not $log) {
    Write-Host "`n!! no new serve log was produced -- the server never got far enough. " -ForegroundColor Red
    Write-Host "Check the server window / the pwsh 7 guard message above." -ForegroundColor Red
}
if ($log) {
    Write-Host "`n=== crash excerpt from $($log.Name) ===" -ForegroundColor Yellow
    $lines = Get-Content $log.FullName
    $hit = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match 'device-side assert|CUDA error|Assertion.*failed') { $hit = $i; break }
    }
    if ($hit -ge 0) {
        # [idcheck] lines sit well above the first assert line, so widen the
        # window when the probe is armed.
        $back = if ($IdCheck) { 150 } else { 40 }
        $lo = [Math]::Max(0, $hit - $back)
        $hi = [Math]::Min($lines.Count - 1, $hit + 40)
        $lines[$lo..$hi]
        if ($Blocking -eq '1') {
            Write-Host "`n>>> the python frame immediately ABOVE the first error line is the TRUE faulting launch (blocking mode)" -ForegroundColor Green
        } else {
            Write-Host "`n>>> blocking is OFF: expect the stack to point at async_utils.py (report point), NOT the culprit. Paste it back anyway -- the batch shape and the 40 lines before it are the evidence." -ForegroundColor Green
        }
    } else {
        Write-Host "no CUDA error found in the log -- the crash did not reproduce." -ForegroundColor Yellow
        if ($Blocking -eq '1') {
            Write-Host "Blocking mode may be what hides it. Re-run with:  .\_repro_async_tower.ps1" -ForegroundColor Yellow
        } else {
            Write-Host "This WAS the true crash config with the era's stage order (stages=[$Stages])." -ForegroundColor Yellow
            Write-Host "If this is clean too, the crash is transient on the current tree, not a" -ForegroundColor Yellow
            Write-Host "property of the config -- and its crash-era company (a sick boot at 22:12" -ForegroundColor Yellow
            Write-Host "the same evening) points at the boot state, not at the workload." -ForegroundColor Yellow
        }
        $lines | Select-Object -Last 40
    }

    # Armed/not-armed must be decidable from THIS output alone: on 2026-09-21
    # 21:59 the run produced zero [idcheck] lines and there was no way to tell
    # a clean probe from an unarmed one (start_server.ps1's env dump dropped
    # DFLASH2_IDCHECK).  The probe now prints one '[idcheck] ARMED' line at
    # step 1, and the serve header prints `idcheck on/off`.
    $ic = @($lines | Where-Object { $_ -match '\[idcheck\]' })
    Write-Host "`n--- idcheck lines in this log: $($ic.Count) ---" -ForegroundColor Green
    if ($ic.Count -eq 0) {
        if ($IdCheck) {
            Write-Host "!! -IdCheck was passed but the server logged NO [idcheck] line: the probe was" -ForegroundColor Red
            Write-Host "   not armed. Check 'idcheck ...' in the serve-log header and the --- env --- block." -ForegroundColor Red
            Write-Host "   (probe installed? .\_idcheck_exp.py --status ; server started from THIS shell?)" -ForegroundColor Red
        } else {
            Write-Host "(-IdCheck was not passed, so nothing was instrumented -- expected.)" -ForegroundColor Yellow
        }
    } else {
        $ic | Select-Object -First 2
        if ($ic.Count -gt 2) { Write-Host '   ...' }
        $ic | Select-Object -Last 8
    }

    # The in-graph guard.  Unlike [idcheck], these lines come from propose(),
    # which runs in Python even when the draft is a FULL-graph replay -- so a
    # non-zero counter here is the first evidence that the replayed graph
    # really does feed an out-of-range id to the codebook gather.
    $cl = @($lines | Where-Object { $_ -match '\[clamp\]' })
    Write-Host "`n--- clamp lines in this log: $($cl.Count) ---" -ForegroundColor Green
    if ($cl.Count -eq 0) {
        if ($Clamp) {
            Write-Host "!! -Clamp was passed but the server logged NO [clamp] line. Either the guard" -ForegroundColor Red
            Write-Host "   never ran (check 'DFLASH2_CLAMP=1' in the --- env --- block) or it died" -ForegroundColor Red
            Write-Host "   before 50 propose() calls. '.\_clamp_exp.py --status' shows the injection." -ForegroundColor Red
        } else {
            Write-Host "(-Clamp was not passed, so no in-graph guard was installed -- expected.)" -ForegroundColor Yellow
        }
    } else {
        $cl | Select-Object -Last 6
        $lastClamp = $cl | Select-Object -Last 1
        if ($lastClamp -notmatch 'neg=0 ge=0 \| anchor neg=0 ge=0') {
            Write-Host ">>> CAUSE FOUND: the replayed drafter graph fed an out-of-range id into the" -ForegroundColor Green
            Write-Host "    [vocab, rank] codebook gather.  See the clause after '|' above." -ForegroundColor Green
        } else {
            Write-Host ">>> guard fired but never tripped: the ids reaching the gather were in range" -ForegroundColor Yellow
            Write-Host "    on every replayed step it covered. If the run still crashed, the bad index" -ForegroundColor Yellow
            Write-Host "    is produced somewhere else than candidate_ids/anchor_token_ids." -ForegroundColor Yellow
        }
    }
    Write-Host "`nfull log:     $($log.FullName)"
    Write-Host "bench output: $bootRoot"
    if ($BootLabel) {
        Write-Host "boot:         $BootLabel -- $(if ($healthy) { 'healthy' } else { 'NEVER BECAME HEALTHY' })" -ForegroundColor Cyan
    }
}

if (-not $serve.HasExited) {
    Write-Host "`nstopping server..." -ForegroundColor DarkGray
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.ExecutablePath -like '*\.venv\*' -or $_.CommandLine -like '*spawn_main*' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Stop-Process -Id $serve.Id -Force -ErrorAction SilentlyContinue
}
