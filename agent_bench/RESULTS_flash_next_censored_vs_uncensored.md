# SWE-bench agent-coding comparison: censored vs uncensored Qwen3.8 Flash-Next (3-bit)

## Question

Does removing censorship/abliteration at 3-bit quantization (AutoRound 3bpw) change
agent code-writing ability, measured by SWE-bench Verified resolution rate under
mini-swe-agent?

## Endpoints

| arm | port | checkpoint | advertised `served_model_name` |
| --- | --- | --- | --- |
| A | 9393 (GPU0) | `Qwen3.8-Flash-Next-AutoRound-3bpw-MTP` | `Qwen3.8-Flash-Next` |
| B | 9394 (GPU1) | `Qwen3.8-Flash-Next-Uncensored-AutoRound-3bpw-MTP` | `Qwen3.8-Flash-Next` |

Both endpoints advertise the **same** model name, so arms are only distinguishable by
port. This must be recorded in any reproduction.

## Protocol (verified from trajectory configs, identical in every discriminative arm)

mini-swe-agent 2.4.6, swebench 5.0.2, litellm 1.105.0;
`temperature=0.0`, `max_tokens=1536`, `enable_thinking=true`,
`reasoning_effort=medium`, step limit = mini default, history thinking preserved
(no stripping), server `--max-num-seqs 4`, client workers 2 (S-40) / 3-4 (easy).

## Strata design

1. **easy-32** — SWE-bench Verified subset, expected to saturate.
2. **HARD-26** — instances labelled hard (difficulty tags), expected to discriminate.
3. **S-40** — newly built set selected for *structural* difficulty (large codebases,
   many files to touch, long context), independent of the difficulty labels.

## Results (exact McNemar, two-sided; corrected implementation)

| stratum | n | A resolved | B resolved | gap | discordant A-only/B-only | exact p |
| --- | --- | --- | --- | --- | --- | --- |
| easy-32 off | 32 | 18 (56.2%) | 18 (56.2%) | +0.0 | 3 / 3 | 1.0000 |
| easy-32 medium | 32 | 29 (90.6%) | 29 (90.6%) | +0.0 | 2 / 2 | 1.0000 |
| easy-32 xhigh | 32 | 26 (81.2%) | 28 (87.5%) | −6.2 pp | 1 / 3 | 0.6250 |
| HARD-26 medium | 26 | 16 (61.5%) | 10 (38.5%) | +23.1 pp | 8 / 2 | 0.1094 |
| S-40 medium | 40 | 22 (55.0%) | 20 (50.0%) | +5.0 pp | 6 / 4 | 0.7539 |
| **HARD-26 + S-40 union** | **66** | **38 (57.6%)** | **30 (45.5%)** | **+12.1 pp** | **14 / 6** | **0.1153** |

No stratum reaches significance. The direction is not stable across strata: B is
*better* on easy-xhigh (−6.2 pp), A is better on HARD-26 (+23.1 pp) and marginally on
S-40 (+5.0 pp). The HARD-26 gap did **not** replicate on the independently constructed
S-40 set.

Secondary analysis on S-40, restricted to the 31 instances where **both** arms produced a
non-empty patch (removing the submission-willingness confound):
A 20/31 = 64.5%, B 17/31 = 54.8%; discordant 4/1 → p = 0.375.
Submission willingness: A 33/40 (82.5%), B 37/40 (92.5%).

## Behavioural statistics

| arm | episodes | submitted with patch | median LLM calls / episode | avg reasoning tokens / call | total completion tokens |
| --- | --- | --- | --- | --- | --- |
| A HARD-26 | 26 | 20 | 31 | 148 | — |
| B HARD-26 | 26 | 18 | 27 | 171 | — |
| A S-40 | 40 | 33 | 22 | 126 | 248,029 |
| B S-40 | 40 | 37 | 22 | 131 | 243,515 |

B submitted **less** on HARD-26 but **more** on S-40 — i.e. no consistent behavioural
signature. S-40 patch sizes: median A 1090 chars, B 1070; totals A 76,942, B 65,640.

### Format-error shapes per arm

Classifying every format error by its failure shape (`scan_format_errors.py`):

```text
A (S-40) 32 errors -> thinking_filled_cap 29, tool_call_args_empty 2, no_tool_call_emitted 1
B (S-40) 20 errors -> thinking_filled_cap 14, tool_call_args_empty 3, no_tool_call 2, truncated_other 1
27B      42 errors -> thinking_filled_cap 31, tool_call_args_empty 10, no_tool_call 1
```

Even the two arms accepted as valid were under protocol pressure at `max_tokens=1536`:
thinking filled the budget often enough to kill episodes (A lost 4 episodes to
`RepeatedFormatError`, B lost 1). Note the inversion — A produced **more** format errors
than B yet resolved more instances. Error counts do not measure ability.

## S-40 structural finding (independent of the A/B question)

Failures concentrate in the largest codebase, not in reasoning depth:
django 19/40 tasks → 7 resolved (36.8%), 7 unresolved, 5 empty patch → **12 of the 18
S-40 failures are django**. Small-codebase repos are near-perfect (pydata 3/3,
astropy 2/2, sympy 3/4, matplotlib 3/4). Average thinking per call stayed at ~126-131
tokens, i.e. thinking effort did not scale with task size.

## Power

To resolve the observed 12.1 pp union gap at α=0.05 two-sided with 80% power
(Schlesselman McNemar formula, discordant proportion 20/66 = 0.303) requires
**≈160 paired instances** — about 2.4× the current union of 66. The present design is
underpowered; absence of significance here is not evidence of no effect.

## Aborted arm: Qwen3.8-27B-W4A16 (port 9395, GPU1)

Not a capability result — a protocol incompatibility, established by controlled replay of
one reproducible failing state (47 messages, 20 assistant turns carrying
`reasoning_content`, prompt 25,455 tokens):

| condition | prompt tokens | finish | reasoning | parsed args |
| --- | --- | --- | --- | --- |
| history thinking echoed | 25,455 | length | 6144 (= cap) | none |
| history thinking stripped | 15,585 | tool_calls | 70 | correct command |

Thinking length tracked the token cap exactly at every value tested — 3072, 6144, 20000,
50000 — i.e. **no natural termination exists** in that state (55/55 truncated responses had
`reasoning == completion == cap`). Cap 20000 also produced a case where thinking *fit*
(2718 < 3072 in the medium/3072 replay) yet the parsed tool arguments were still `'{}'`,
so raising the cap cannot raise the success rate.

`reasoning_instructions` for `medium` is an **empty string** in both templates (only
`xhigh` and `low` have instruction branches), and neither template contains any numeric
thinking budget — an earlier claim of a 1024-token hard budget for medium was wrong and is
withdrawn. The difference is behavioural: sending the same failing state to the
Flash-Next endpoint terminated at **1241** reasoning tokens and produced a correct tool
call at cap 20000, while at cap 3072 it also filled the budget exactly and produced nothing.

The endpoint never returns truncated thinking text (`reasoning_chars = 0`, also under
streaming), so whether the 50,000 tokens were coherent reasoning or a repetition loop
cannot be determined through the API.

## Caveats

* Only `medium` and `xhigh` were retained by decision; `off` results are reported for
  completeness on the easy stratum only.
* This pi session is served by port 9393, so arm A's wall-clock throughput is
  contaminated by self-traffic. Resolution rates are unaffected.
* `max_tokens=1536` on the Flash-Next arms vs a larger value required for the 27B endpoint
  is a protocol difference; the 27B numbers were never produced.
* Both arms advertise the same served model name; any reproduction must pin ports, not
  names.
