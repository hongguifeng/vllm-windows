# Agent benchmarking runbook (SWE-bench Verified + mini-swe-agent + local vLLM)

Everything here was exercised in a real comparison run: two 3-bit Qwen3.8 Flash-Next
checkpoints (censored vs uncensored) across three strata, plus one aborted 27B arm. The
commands are the ones that actually ran, not idealised versions.

## 0. What is being measured, and why one number is never enough

Question shape: *does property X of a model change agent code-writing ability?*
Resolution rate on SWE-bench Verified is the observable. A single subset cannot answer it:

* A **saturated** stratum (easy) produces ~90% for both arms and tells you nothing except
  that the harness works.
* A **difficulty-labelled** stratum (`hard`) is cheap but inherits whatever the label
  encodes. A gap found there might be an artefact of how the set was chosen.
* A **structural** stratum (large codebases, many files in the reference patch, many
  FAIL_TO_PASS cases) is constructed independently of the labels, so a gap found on the
  labelled set can be tested for replication.

The last run demonstrated exactly why: a 23 pp gap on the labelled set did **not**
replicate on the structural set, and the pooled gap of 12 pp was not significant.

## 1. Environment as observed

* Windows host, WSL2, Docker for Linux containers. 16 CPU cores, 47 GB RAM.
* Two 64 GB GPUs. Convention: GPU0 serves arm A, GPU1 serves arm B.
* Windows-side repo: `d:\code\vllm-windows`, launcher scripts in the repo root.
* WSL-side bench root: `/home/hong/bench`, venv `/home/hong/bench/.venv` (Python 3.12).
* Dataset snapshot: `/home/hong/bench/swebench_verified.parquet` (also `verified_new.parquet`).
* Container images: `ghcr.io/epoch-research/swe-bench.eval.x86_64.<instance_id>:latest`.
* `wsl.exe` writes UTF-16 with NUL padding. **Always pipe output through `tr -d '\0'`**
  or every later grep/awk silently breaks.

## 2. Endpoint bring-up (Windows side)

Launcher scripts present in the repo root:
`start_qwen38_flash_next.ps1`, `start_qwen38_27b.ps1`, `start_qwen38_27b_170hx.ps1`,
`start_server.ps1`, `_start_vllm_service.ps1`, `stop_vllm.ps1`.

Non-negotiable details learned the hard way:

1. `start_server.ps1` refuses Windows PowerShell 5.1 because 5.1 strips the quotes around
   `--default-chat-template-kwargs '{"enable_thinking": false}'`. Use
   `C:\Program Files\PowerShell\7\pwsh.exe`.
2. `start_server.ps1` runs python in the foreground (`Invoke-Logged $py`), so the server's
   lifetime is tied to the calling process. If you launch it inside a tool call, the server
   dies when the call times out. Launch detached:
   `Start-Process -FilePath pwsh.exe -ArgumentList '-NoProfile','-File','<script>' -WindowStyle Hidden`
   or use `_start_vllm_service.ps1`, which detaches.
3. Flash-Next servers ran with `--max-num-seqs 4`, `--max-num-batched-tokens 2048`,
   `--gpu-memory-utilization 0.96`, and MTP/speculative decode `num_speculative_tokens: 3`.
   `--max-num-seqs 4` is the concurrency ceiling that shapes client worker counts.
4. Stopping by port is safe and does not disturb the other GPU:
   `pwsh -File stop_vllm.ps1 -Port <port>` then confirm with `nvidia-smi`.

Verification before any run — all four, in this order:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:<port>/health      # 200
curl -s http://127.0.0.1:<port>/v1/models | head -c 200                       # check "root"
curl -s --max-time 25 http://127.0.0.1:<port>/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<served name>","messages":[{"role":"user","content":"List files in /testbed using a tool."}], \
      "tools":[{"type":"function","function":{"name":"bash","description":"Run bash.", \
      "parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}], \
      "tool_choice":"auto","max_tokens":512, \
      "chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"medium"}}'
curl -s http://127.0.0.1:<port>/metrics | grep -E "^vllm:(num_requests_running|num_requests_waiting|num_preemptions_total|kv_cache_usage_perc|spec_decode_draft_acceptance_rate)\{"
```

The `/v1/models` check is the only reliable way to know **which checkpoint** is behind a
port: both Flash-Next checkpoints advertise `served_model_name = Qwen3.8-Flash-Next`.
Read the `root` field. Never identify arms by model name; identify them by port.

Also read the model's `chat_template.jinja` before running an arm:

* Which `reasoning_effort` values have instruction branches. In both checkpoints used here,
  `medium` has **no** branch, so `reasoning_instructions` renders as an empty string, i.e.
  medium gives the model no thinking-length cue at all.
* Whether prior thinking is re-rendered unconditionally. A template that echoes every
  assistant thinking block after the last user turn makes agentic use fragile: prompts grow,
  and some models then think until the token budget dies instead of emitting a tool call.
* Whether the template has a `tool_call_format` switch and what its default is. Default
  `xml` pairs with `--tool-call-parser qwen3_xml`; forcing `json` on such a template makes
  the model emit JSON-shaped calls that the XML parser turns into `'{}'`.

## 3. WSL-side environment

```bash
cd /home/hong/bench
uv venv --python 3.12 .venv                 # never system python, never bare pip
uv pip install mini-swe-agent swebench litellm pyarrow
.venv/bin/python -c "import minisweagent, swebench, litellm; print('ok')"
```

Deploy the kit into the bench root:

```bash
mkdir -p /home/hong/bench
cp /mnt/c/tmp/agent_bench/*.py /home/hong/bench/     # or from d:\code\vllm-windows\agent_bench
```

## 4. Build the strata

```bash
# easy, repo-quota sample, expected to saturate
.venv/bin/python make_subset.py --parquet swebench_verified.parquet --mode easy \
  --out subset_easy.json --seed 42 --images-out images_easy.txt

# difficulty-labelled hard set
.venv/bin/python make_subset.py --parquet swebench_verified.parquet --mode hard \
  --out subset_hard.json --n 26 --exclude subset_easy.json --images-out images_hard.txt

# structural set, independent of the labels
.venv/bin/python make_subset.py --parquet swebench_verified.parquet --mode structural \
  --out subset_struct.json --n 40 --exclude subset_easy.json --images-out images_struct.txt
```

`make_subset.py` prints difficulty mix, repo mix, and the medians of
`files touched / patch length / FAIL_TO_PASS count`, and reports overlap with the excluded
set. **Read those medians before trusting a stratum**: if the structural set ends up mostly
one repo, you are measuring that repo, not difficulty. In the run behind this document, S-40
was 47.5% django and that single repo produced 12 of 18 failures.

Pre-pull images (this is the step that takes hours, do it early):

```bash
xargs -a images_struct.txt -n1 docker pull
docker images | grep swe-bench.eval | wc -l
```

`while read` over an image list can miss the final line if the file has no trailing newline;
`xargs -a file` does not have that problem. Verify counts, don't assume.

## 5. Run one arm

`run_mini.py` is the runner. Parameters that matter:

| flag | meaning | value used here |
| --- | --- | --- |
| `--subset` | subset json built above | per stratum |
| `--output` | arm output dir, gets `preds.json` + per-instance trajectories | `run<ArmTag>` |
| `--api-base` | `http://localhost:<port>/v1` | per arm |
| `--model-name` | litellm string, must keep the `openai/` prefix | `openai/<served name>` |
| `--max-tokens` | per-call completion budget | 1536 (Flash-Next) |
| `--step-limit` | agent steps per episode | 60 |
| `--enable-thinking --reasoning-effort` | injected into `chat_template_kwargs` | `medium` |
| `--workers` | concurrent episodes | **tiered: off 3-4, medium 2, xhigh 2** |
| `--strip-thinking-history` | do not send previous thinking back | only for endpoints proven to need it |
| `--tool-call-format` | override template's tool format | usually leave empty |
| `--only` / `--redo-existing` | resume or force rerun | debugging |

Worker tiering is not performance tuning, it is memory safety. Do not run two concurrent
benchmark arms on one GPU; per-request KV footprint is large enough that co-locating arms
starves context. One arm per GPU.

Launch pattern (WSL background jobs need a `sleep` inside the same `wsl.exe` call, otherwise
the job is killed when the call returns):

```bash
wsl.exe -e bash -lc "cd /home/hong/bench; setsid nohup .venv/bin/python run_mini.py \
  --subset subset_struct.json --output /home/hong/bench/runS40A \
  --api-base http://localhost:9393/v1 --workers 2 --enable-thinking --reasoning-effort medium \
  --max-tokens 1536 --step-limit 60 >> /home/hong/bench/runS40A.sup.log 2>&1 < /dev/null & sleep 6"
```

Use `setsid nohup ... & sleep 6`, redirect to a log, and close stdin. To kill a specific run
without killing your own shell, use the bracket trick so `pkill` cannot match the command you
are typing: `pkill -f "[r]un_mini.py --output /home/hong/bench/runS40A"`.

Heredocs into WSL must use a **quoted** delimiter (`<<'SHEOF'`) so the shell does not expand
`$` inside the Python.

## 6. Supervision: why the supervisor lives on Windows

WSL restarts kill every background job inside WSL. A supervisor running inside WSL dies with
the jobs it is supposed to restart. So supervision goes on the Windows side, and only the
work itself lives in WSL.

```powershell
Start-Process -FilePath 'C:\Program Files\PowerShell\7\pwsh.exe' `
  -ArgumentList '-NoProfile','-File','d:\code\vllm-windows\agent_bench\supervise_arm.ps1' `
  -WindowStyle Hidden -PassThru | Select-Object -ExpandProperty Id
```

`supervise_arm.ps1` arguments: `-RunTag -Subset -ApiBase -ModelName -Target -Workers
-MaxTokens -StepLimit -Effort -EvalDataset -EvalWorkers -EvalTimeout -ExtraRunArgs
-MaxEmptyRestarts -PollSeconds -MaxPolls -LogFile -BenchDir -OutName`.

It polls `arm_state.py` (prints `"<preds> <empty>"` from `preds.json`), relaunches a dead
`run_mini`, waits for Docker to be ready after a WSL reboot, declares inference complete when
`preds >= Target` with at least one non-empty patch, then launches and babysits evaluation
until the report file appears.

The guard `-MaxEmptyRestarts` (default 2) exists because of a real incident. An arm whose
every episode yields an empty patch looks "complete but useless"; the original supervisor
deleted `preds.json` and restarted, forever, on an endpoint that was structurally incapable
of producing patches. Restarting that arm does not fix it. If the supervisor aborts with
`ABORT: all patches empty`, treat it as an endpoint/protocol finding and run the diagnostic
in section 8.

## 7. Evaluation

```bash
.venv/bin/python -m swebench.harness.run_evaluation \
  --dataset_name /home/hong/bench/subset_struct.json \
  --predictions_path /home/hong/bench/runS40A/preds.json \
  --run_id runS40A --max_workers 4 --timeout 2400 \
  --report_dir /home/hong/bench
```

* `--dataset_name` accepts a **path to a local subset json** ✓. Use that; it keeps the
  evaluated set identical to the inferred set and needs no network.
* `--max_workers` must be ≤ 75% of cores (16 cores → ≤ 12). 4 was used because each
  evaluation is a Docker container running a real test suite.
* `--model_name` does **not** exist. The harness derives both the log directory and the
  report filename from each prediction's `model_name_or_path`, so two endpoints advertising
  the same served name produce the same prefix and only `--run_id` separates them. Stamp a
  label when you want the filename itself to carry arm identity:

```bash
.venv/bin/python tag_preds.py --preds /home/hong/bench/runS40A/preds.json \
  --label armA/FlashNext-censored --in-place
```

Sanity-check the evaluation before believing the numbers. A report that lands minutes after
launch can be legitimate (the harness runs only the specified tests, ~20-40 s per instance)
but must be verified:

```bash
find /home/hong/bench/logs/run_evaluation/<run_id> -maxdepth 3 -type d | wc -l   # per-instance dirs
find /home/hong/bench/logs/run_evaluation/<run_id> -name report.json | wc -l
head -c 300 /home/hong/bench/logs/run_evaluation/<run_id>/*/<some_instance>/report.json
ls -l /home/hong/bench/logs/run_evaluation/<run_id>/*/<some django instance>/   # patch.diff, eval.sh, run_instance.log, test_output.txt must exist
```

Report structure is aggregate, not per instance: `total_instances`, `submitted_instances`,
`completed_instances`, `resolved_instances`, `unresolved_instances`,
`empty_patch_instances`, `ambiguous_failure_instances`, `infra_failure_instances`,
`error_instances`, plus `resolved_ids` / `unresolved_ids` / `empty_patch_ids` which partition
the instance set **without overlap**. Two rates come out of this and both must be reported:

* resolved / `completed_instances` → accuracy conditioned on having produced a patch;
* resolved / `total_instances` → end-to-end rate including episodes that never submitted.

Empty patches are skipped by the harness, which is why the two denominators differ.

## 8. Before believing a low score: is it capability or protocol?

A low resolution rate is only a capability result if the agent actually got to act. Two
failure shapes silently destroy episodes:

* `finish_reason=length` with `reasoning == completion == max_tokens`, no content, no tool
  call → mini records a format error; two in a row end the episode.
* `finish_reason=tool_calls` with parsed `arguments == '{}'` → mini records
  `Missing 'command' argument`, same path to episode death.

Either shape means the score measures the protocol, not the model. `diagnose_thinking.py`
settles it by replaying a real failing state and varying one thing at a time:

```bash
.venv/bin/python diagnose_thinking.py \
  --traj /home/hong/bench/runS40A/<instance>/<instance>.traj.json \
  --api-base http://localhost:9394/v1 --model <served name> \
  --caps 3072,6144,20000 --effort medium \
  --second-api-base http://localhost:9393/v1 --second-model <other served name>
```

Interpretation grid:

| observation | conclusion | action |
| --- | --- | --- |
| reasoning equals the cap at **every** cap tested | thinking never terminates in that state | raising `max_tokens` cannot help; fix the history or the template |
| stripping echoed reasoning turns `length`/`{}` into a real tool call | history thinking is the cause | document the protocol change; run a matched arm for the comparator |
| second endpoint terminates the same state naturally | first endpoint's template/decoding is at fault | do not compare those endpoints as equals |
| thinking fits under the cap yet arguments are still `'{}'` | budget is not the binding constraint | stop escalating the budget; fix the tool-call path |

Truncated thinking is **not returned** by the API (`reasoning_chars = 0`, also under
streaming), so you cannot decide "coherent long reasoning vs repetition loop" from the
response. If that distinction matters, bypass the reasoning parser: render the state to
tokens with `/v1/chat/completions/render`, then drive `/v1/completions` with
`tokens`/`prompt_token_ids` and explicit stop strings, and inspect the text.

```bash
.venv/bin/python status_hist.py --run /home/hong/bench/runS40A --run /home/hong/bench/runS40B
.venv/bin/python scan_format_errors.py --out /home/hong/bench/runS40A --limit 0
```

`status_hist.py` counts episode exit statuses. Keep the two notions of "empty patch" apart:
the trajectory-level check reads what the agent claimed in its `exit` message, while
`preds.json` records the `git diff` that actually exists; they can disagree.

`scan_format_errors.py` classifies every format error regardless of `--limit`, which only
gates printing. Example from the run behind this document:

```text
A  32 format errors -> thinking_filled_cap 29, tool_call_args_empty 2, no_tool_call_emitted 1
B  20 format errors -> thinking_filled_cap 14, tool_call_args_empty 3, no_tool_call 2, truncated_other 1
27B 42              -> thinking_filled_cap 31, tool_call_args_empty 10, no_tool_call 1
```

Two readings that matter. Even the arms accepted as valid were under protocol pressure: at
`max_tokens=1536` thinking occasionally filled the budget on both arms and killed episodes,
so a budget that looks adequate on average still fails on the tail. And error counts do not
track resolution counts: A had more format errors than B (32 vs 20) yet resolved more
instances (22 vs 20). Never infer ability from error counts.

Patch-presence pairing between arms, which separates "did not submit" from "submitted and
failed", comes from comparing `preds.json` entry by entry.

## 9. Analysis

Arm-to-arm comparison is paired, so use McNemar on discordant pairs, not a two-proportion
z-test. Exact two-sided p is `2 * P(X >= max(b, c))`, `X ~ Binomial(b+c, 1/2)`. Implementing
it with `min(b, c)` yields a large probability that gets clamped to 1.0, which makes every
comparison look null. `mcnemar_compare.py` implements the correct form and also reports the
sample size still needed.

```bash
.venv/bin/python mcnemar_compare.py \
  --pair easy32 /home/hong/bench/openai__Qwen3.8-Flash-Next.runT9393.json \
             /home/hong/bench/openai__Qwen3.8-Flash-Next.runT9394.json \
  --pair HARD26 /home/hong/bench/openai__Qwen3.8-Flash-Next.runH9393.json \
             /home/hong/bench/openai__Qwen3.8-Flash-Next.runH9394.json \
  --pair S40    /home/hong/bench/openai__Qwen3.8-Flash-Next.runS40A.json \
             /home/hong/bench/openai__Qwen3.8-Flash-Next.runS40B.json \
  --union HARD26,S40
```

Discordant strata may be pooled **only if their instance sets are disjoint**; the script
checks overlap and says so.

Report the behaviour columns next to the score, they are what makes a result interpretable:

| column | source |
| --- | --- |
| submitted with non-empty patch | `preds.json` |
| median LLM calls per episode | `*.traj.json` message count |
| average reasoning tokens per call | `usage.completion_tokens_details.reasoning_tokens` |
| total completion tokens | `usage.completion_tokens` summed |
| median prompt tokens at the last step | `usage.prompt_tokens` |

## 10. Decisions this run produced, and what they teach

* **A gap must replicate before you repeat it.** The labelled-hard stratum showed A ahead by
  23 pp; the independently built structural stratum showed 5 pp with p = 0.75; pooled over 66
  instances the gap was 12 pp with exact p = 0.115. Direction also flipped on the easy stratum
  at xhigh, where B was better by 6.2 pp.
* **Absence of significance is not absence of effect.** With discordant proportion 20/66 and
  a 12.1 pp gap, confirming it at α = 0.05 with 80% power needs ≈160 paired instances, about
  2.4× what existed here. Size the strata from the effect you care about, not from convenience.
* **Failure structure was informative even without an A/B difference.** S-40 failures
  concentrated in the largest repo (12 of 18 in django, 19 of 40 tasks) while small-codebase
  repos were near perfect, and thinking per call stayed flat at 126-131 tokens instead of
  scaling with task size. That points at retrieval/editing in large codebases, not reasoning.
* **One arm never produced data, and the cause was proven rather than assumed.** Thinking
  length tracked the token cap exactly at 3072, 6144, 20000, and 50000 on the same replayed
  state, while stripping echoed history gave 70 reasoning tokens and a correct tool call; the
  other endpoint terminated the same state at 1241 tokens. An earlier claim of a 1024-token
  hard budget for medium was wrong and is retracted: neither template contains any numeric
  budget, and `medium` has no instruction branch in either.
* **Self-traffic matters.** This pi session is served by port 9393, so arm A's throughput and
  wall-clock numbers are contaminated by the session's own requests. Resolution rates are
  unaffected; do not quote latency or tokens-per-second for that arm.

## 11. Traps encountered, each with its fix

| trap | fix |
| --- | --- |
| `wsl.exe` output is UTF-16 with NUL padding | `... \| tr -d '\0'` on every WSL call |
| WSL background job dies when the tool call returns | include `sleep 6` inside the same `wsl.exe -e bash -lc` call |
| WSL reboot kills all WSL jobs | run the supervisor on Windows, not in WSL |
| vLLM server dies with the tool call | launch detached (`Start-Process`, or `_start_vllm_service.ps1`) |
| Windows PowerShell 5.1 mangles `--default-chat-template-kwargs` | use `pwsh` 7 |
| `pkill -f run_mini.py` also matches the shell doing the killing | bracket trick `[r]un_mini.py` |
| unquoted heredoc lets the shell expand `$` inside Python | `<<'SHEOF'` |
| `while read` misses the last line of a file without a trailing newline | `xargs -a file` |
| both arms advertise the same `served_model_name` | identify by port; read `/v1/models` `root`; stamp `model_name_or_path` |
| McNemar implemented with `min(b,c)` | use `2 * P(X >= max(b,c))` |
| `preds.json` written only as a top-level file | state script must read `preds.json`, not per-instance patch files |
| all-empty `preds.json` triggers endless delete-and-restart | cap the restarts; treat all-empty as a finding |
| evaluation finishing suspiciously fast | verify `patch.diff`, `eval.sh`, `run_instance.log`, `test_output.txt` per instance |
| empty patches are skipped, so two denominators exist | report both conditioned and end-to-end rates |

## 12. Files in this kit

| file | role |
| --- | --- |
| `RUNBOOK.md` | this document |
| `run_mini.py` | runner; canonical copy, dead flags removed |
| `make_subset.py` | builds easy / hard / structural strata, prints medians and overlap |
| `arm_state.py` | prints `"<preds> <empty>"` for the supervisor |
| `supervise_arm.ps1` | Windows-side supervisor with an all-empty restart cap |
| `tag_preds.py` | stamps `model_name_or_path` so evaluation output is separable |
| `mcnemar_compare.py` | correct exact McNemar, pair mode and union mode, power estimate |
| `diagnose_thinking.py` | decides capability vs protocol before accepting a score |
| `status_hist.py` | episode exit-status histogram, claimed-patch emptiness, token use per arm |
| `scan_format_errors.py` | classifies format errors into the two protocol failure shapes |
| `RESULTS_flash_next_censored_vs_uncensored.md` | the comparison this kit was built for |

## 13. Checklist for testing a new model

Ordered; each step is a gate, and skipping a gate is how a bad number gets believed.

1. `/v1/models` → read `root`, confirm the checkpoint you think you are serving.
2. Health 200, then one tool-call probe with `tool_choice=auto` and `medium` thinking; the
   parsed arguments must be real, not `'{}'`.
3. Read `chat_template.jinja`: which efforts have instructions, how prior thinking is
   handled, what the default `tool_call_format` is. Compare against the comparator's
   template; a difference here is a protocol difference you must document, not tune away.
4. Check `/metrics` once at idle: running, waiting, preemptions, KV usage. Note KV token
   capacity if you intend to run long-context strata.
5. Decide `max_tokens` from a short ladder probe (a few hundred to a couple of thousand)
   rather than by copying another arm's value; templates that cap thinking differently
   need different budgets.
6. Run the strata you already have, with the same subset files, same step limit, same
   temperature. Only `max_tokens` and effort may legitimately differ, and differences must
   appear in the report.
7. Supervise from Windows with `supervise_arm.ps1`; give the arm a unique `-RunTag`.
8. Stamp `preds.json` with `tag_preds.py` before evaluation so the report filename carries
   the arm.
9. Evaluate, then verify per-instance artefacts exist (`patch.diff`, `eval.sh`,
   `run_instance.log`, `test_output.txt`).
10. Report both denominators, run `status_hist.py` and `scan_format_errors.py`, and only
    then run `mcnemar_compare.py` against the comparator arm on the same stratum.
11. A gap on one stratum is not a finding until it replicates on a stratum built by a
    different criterion. If it does not replicate, say that.

Keeping the reusable kit in the tree is fine; keeping per-run outputs, trajectories, and
ad-hoc probe scripts in it is not. If a benchmark result ends up in a pull request,
`AGENTS.md` applies: duplicate-work checks, no busywork PRs, named AI assistance, tests
run, and model evaluation results for anything that affects output.
