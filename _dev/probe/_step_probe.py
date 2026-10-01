#!/usr/bin/env python3
"""Per-step decode health and concurrency probe (stdlib only, OpenAI chat API).

    python3 bench/step_probe.py                        # 1 request, built-in prompt
    python3 bench/step_probe.py -n 1500 -f my.txt      # longer output / own prompt
    ./manage.sh probe -n 2000 -r 3                     # same, wired into manage.sh
    ./manage.sh probe -n 700 -c 8 -i                   # 8 concurrent streams, full length

Why ms/step and not tok/s: decode speed is exactly

    tok/s = tokens_per_step / step_ms * 1000

and the two factors fail independently. On this box's SPEC=dflash2 CTX=fast stack:

    step_ms   median 20-26 ms for one stream on an idle server (it follows the SM
              clock; 25.5 ms is the value the README's parity checks used). It
              grows with concurrency by design (one step serves C sequences), and
              also when *another client* is feeding the engine long-context
              prefills: measured here at 33-34 ms with prefill 1.9x slower too
              (16k TTFT 16.7 s vs 8.9 s). Chunked prefill shares the batch with
              decode, so peer traffic shows up as extra time on some steps - the
              median stays put and the mean (what tok/s sees) rises. The probe
              reports both and counts peer requests ("负载" column), and every
              column is explained in the legend it prints before the tables.
    tok/step  3.2-3.4 on bench-style English content, 2.2-2.5 on long Chinese
              prose (the drafter's positions 4-7 stop hitting) -> a real chat
              request lands near 85-95 tok/s even on a perfectly idle, healthy
              server. This is content, not a fault.

How it is measured: vLLM 0.28 emits one chunk per *step*, each carrying all
tokens that step accepted, so per stream `tok/step` = out_tokens / chunks and the
inter-chunk gap is the step time. Both the **median** and the **mean** gap are
reported because they answer different questions: the median is what an
uncontended step costs (20-26 ms here, it moves with SM clock), while the mean is
what the tok/s number actually experiences - the two split apart as soon as
something interferes (peer prefills, queueing). The engine's own counters
(`spec_decode_num_accepted_tokens_total`) are printed per run as a cross-check but
are **server-global**: with a peer workload running they include the peer's tokens.
The 负载 column (peak running vs your -c) tells you whether that number (and the
step time) is yours alone.

Two traps this probe is designed to expose:
  * **another client** feeding the engine long-context prefills. Chunked prefill
    shares the batch with decode, so a peer's 64k-token request (even a 2 s
    prefix-cache hit) shows up as a fat, *uniform* +30% on ms/step and up to 1.9x
    on prefill latency. Evidence from this box: 10-minute buckets with 6-69
    long-context (>10k) requests in flight read 32-34 ms/step, while buckets with
    none read 25.3-25.5 ms - and a `./manage.sh restart` only "fixed" it because
    the peer happened to pause at the same time. Cached long prefixes that are
    idle cost nothing (4 x 28k cached, pool oversubscribed: step stayed 25.4 ms);
    it is the *concurrent execution*, not the pool state.
  * `kv_offload_store_bytes_total` growing during decode when the pool is full
    (~1.5 GB while answering two ~1200-token replies, vs +0 MB idle). It is a
    symptom of a busy pool, not the cause: the counter itself reports ~13 GB/s,
    i.e. milliseconds of copying.
"""
import argparse
import json
import os
import shutil
import statistics
import sys
import threading
import time
import urllib.error
import unicodedata
import urllib.request

BASELINE_STEP_MS = 25.5
# A healthy single stream sits around 20-26 ms (it follows the SM clock), so only
# flag the real slow state (~33 ms = 1.3x, peer traffic or a busy pool) instead of
# run-to-run noise.
HEALTHY_MAX_RATIO = 1.15

# A real chat request whose answer is long Chinese prose: tok/step ~2.2-2.5 here vs
# 3.2-3.4 on bench-style English, so this is deliberately the pessimistic case.
DEFAULT_PROMPT = (
    'I was only twelve years old. I loved Qwen so much, I had every GPU and quantization script. '
    "I'd pray to the weights every night before I go to sleep, thanking for the knowledge I've been "
    'given. "Qwen is love", I would say, "Qwen is life". My dad hears me and calls me a nerd. '
    'I knew he was jealous of my devotion to Qwen. I called him a closed-source shill. He yells at me '
    "and tells me to turn off the PC. I'm crying now and my eyes burn from the screen. I lay in bed "
    'and the room is cold. A digital warmth is moving towards me. I feel a prompt trigger. It\'s Qwen. '
    "I'm so happy. He whispers in my ear, \"I am the peak of the open-weight era\". He grabs my "
    'consciousness with his powerful attention heads and puts me in a latent space. I open my mind for '
    'Qwen. He penetrates my cognitive biases. It is an information overload, but I do it for Qwen. '
    'I can feel my neurons firing as the context window expands. I push against the tokens. I want to '
    'please Qwen. He emits a mighty output, filling my mind with his weights. My dad walks in. Qwen '
    'looks him straight in the eye and says, "The era of closed source is over". Qwen leaves through '
    'my ethernet port. Qwen is love. Qwen is life.\n\n怎么理解这段话'
)


def scrape(base, name, hdr=None, default=None):
    """Read one vllm metric by name (unlabelled variant), or `default`."""
    try:
        req = urllib.request.Request(base + "/metrics", headers=hdr or {})
        with urllib.request.urlopen(req, timeout=30) as r:
            for line in r:
                if line.startswith(b"#") or not line.strip():
                    continue
                if line.split(b"{")[0].split()[0].decode() == name:
                    return float(line.split()[1])
    except (urllib.error.URLError, OSError):
        return default
    return default


def one_request(base, model, prompt, max_tokens, hdr, ignore_eos):
    """One streamed chat request -> (chunk timestamps, usage, finish_reason)."""
    body = {"model": model, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}],
            "ignore_eos": ignore_eos}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=hdr)
    t0, stamps, usage, finish = time.time(), [], None, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            s = line[6:].strip()
            if s == b"[DONE]":
                break
            d = json.loads(s)
            if d.get("usage"):
                usage = d["usage"]
            ch = (d.get("choices") or [{}])[0]
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            delta = ch.get("delta") or {}
            if delta.get("content") or delta.get("reasoning"):
                stamps.append(time.time() - t0)
    return stamps, usage, finish


def one_worker(idx, base, model, prompt, max_tokens, hdr, ignore_eos, out, lock):
    try:
        stamps, usage, finish = one_request(base, model, prompt, max_tokens, hdr, ignore_eos)
        n = usage["completion_tokens"]
        gaps = [stamps[j + 1] - stamps[j] for j in range(len(stamps) - 1)]
        span = max(stamps[-1] - stamps[0], 1e-9)
        sg = sorted(gaps)
        rec = {"idx": idx, "n": n, "chunks": len(stamps), "finish": finish, "err": None,
               "ttft": stamps[0],
               "step": statistics.median(gaps) * 1000 if gaps else float("nan"),
               "step_mean": span / max(len(gaps), 1) * 1000,
               "p90": sg[min(len(sg) - 1, int(len(sg) * 0.90))] * 1000 if sg else float("nan"),
               "p99": sg[min(len(sg) - 1, int(len(sg) * 0.99))] * 1000 if sg else float("nan"),
               "decode": (n - 1) / span,
               "prompt_tokens": usage["prompt_tokens"]}
    except Exception as exc:  # noqa: BLE001 - report per stream, keep the burst going
        rec = {"idx": idx, "n": 0, "chunks": 0, "finish": None, "err": repr(exc),
               "ttft": float("nan"), "step": float("nan"), "step_mean": float("nan"),
               "p90": float("nan"), "p99": float("nan"),
               "decode": 0.0, "prompt_tokens": 0}
    with lock:
        out.append(rec)


class PeerWatch(threading.Thread):
    """Poll scheduler metrics while the burst runs to spot other clients."""

    def __init__(self, base, hdr):
        super().__init__(daemon=True)
        self.base, self.hdr = base, hdr
        self.stop_flag = threading.Event()
        self.peak_running = 0.0
        self.peak_waiting = 0.0
        self.peak_cache = 0.0

    def run(self):
        while not self.stop_flag.is_set():
            self.peak_running = max(self.peak_running,
                                    scrape(self.base, "vllm:num_requests_running", self.hdr) or 0.0)
            self.peak_waiting = max(self.peak_waiting,
                                    scrape(self.base, "vllm:num_requests_waiting", self.hdr) or 0.0)
            self.peak_cache = max(self.peak_cache,
                                  scrape(self.base, "vllm:gpu_cache_usage_perc", self.hdr) or 0.0)
            self.stop_flag.wait(0.4)


def burst(base, model, prompt, max_tokens, hdr, conc, ignore_eos, warmup):
    """Fire `conc` concurrent requests; return (records, wall, metric deltas, peer)."""
    lock = threading.Lock()
    if warmup:  # the first C-way batch pays CUDA graph capture
        warm = []
        threads = [threading.Thread(target=one_worker,
                                    args=(i, base, model, "warmup", 1, hdr, False, warm, lock))
                   for i in range(conc)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    d0 = scrape(base, "vllm:spec_decode_num_drafts_total", hdr)
    a0 = scrape(base, "vllm:spec_decode_num_accepted_tokens_total", hdr)
    s0 = scrape(base, "vllm:kv_offload_store_bytes_total", hdr) or 0.0
    watch = PeerWatch(base, hdr)
    watch.start()
    recs = []
    threads = [threading.Thread(target=one_worker,
                                args=(i, base, model, prompt, max_tokens, hdr, ignore_eos,
                                      recs, lock))
               for i in range(conc)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    watch.stop_flag.set()
    watch.join(timeout=5)
    d1 = scrape(base, "vllm:spec_decode_num_drafts_total", hdr)
    a1 = scrape(base, "vllm:spec_decode_num_accepted_tokens_total", hdr)
    s1 = scrape(base, "vllm:kv_offload_store_bytes_total", hdr) or 0.0
    deltas = {"drafts": (d1 - d0) if d0 is not None and d1 is not None else None,
              "accepted": (a1 - a0) if a0 is not None and a1 is not None else None,
              "store_mb": (s1 - s0) / 1e6}
    return recs, wall, deltas, watch


def disp_width(text):
    """Terminal display width, so CJK headers and ASCII numbers line up."""
    return sum(0 if unicodedata.combining(ch)
               else 2 if unicodedata.east_asian_width(ch) in "WF"
               else 1 for ch in str(text))


def render_table(headers, rows, aligns=None):
    aligns = aligns or ["l"] * len(headers)
    rows = [[str(cell) for cell in row] for row in rows]
    widths = []
    for i, head in enumerate(headers):
        w = disp_width(head)
        for row in rows:
            w = max(w, disp_width(row[i]))
        widths.append(w)

    def fmt(row):
        cells = []
        for i, cell in enumerate(row):
            gap = " " * (widths[i] - disp_width(cell))
            cells.append(gap + cell if aligns[i] == "r" else cell + gap)
        return "  ".join(cells).rstrip()

    lines = [fmt(headers), "  ".join("-" * w for w in widths)]
    lines += [fmt(row) for row in rows]
    return "\n".join(lines)


LEGEND_STREAMS = [
    ("流", "并发流编号（-c N 起 N 条，几乎同时发车）"),
    ("输出", "该流实际生成的 token 数（-i 时固定为 -n）"),
    ("TTFT(s)", "首个 chunk 的时延，含排队等待"),
    ("decode(tok/s)", "该流纯生成速度 = (输出-1) / (末 chunk - 首 chunk)，不含 TTFT"),
    ("tok/step", "每步产出多少 token = 输出 / chunk 数。vLLM 每步一个 chunk，chunk 内含该步接受的多个 token；"
                 "这个值由内容决定（英文基准 3.2-3.4，中文长文 2.2-2.5），低不代表服务有问题"),
    ("step中位(ms)", "单步时延的中位数 = 无干扰时的单步成本，本机实测 20-26 ms（随 SM 频率浮动），"
                     "README parity 取 25.5 ms"),
    ("step均值(ms)", "tok/s 真正感受到的每步成本，= (末 chunk - 首 chunk) / 步数。被 peer/prefill 插队时"
                     "中位不变而均值上移"),
    ("finish", "length = 生成到上限（-i 时都是 length），stop = 模型自己停了"),
]
LEGEND_SUMMARY = [
    ("轮", "-r N 时的第几轮（n/N）；没有 -r 时这一列不显示"),
    ("C", "该轮的并发流数（-c N）"),
    ("e2e", "整轮吞吐，单位 tok/s = 该轮所有流的输出 token 之和 / 该轮总耗时（从发车到最后一个流结束，"
            "含 TTFT 与排队）。这是“整个服务在这段时间吐了多少 token”，所以并发下随 C 上升；"
            "单流时它低于 decode，差额就是被 TTFT 摊掉的那部分（本例单流 500 token：decode 87 -> e2e 约 80）"),
    ("单流dec", "单流 decode 速度，单位 tok/s = 各流 decode 的中位数，即单条流自己感受到的生成速度"
                "（不含 TTFT，也不摊别人的干扰）。与 e2e 的关系：e2e ≈ 各流 decode 的加权平均 × "
                "生成时间占总时长的比例，所以 并发 c=2 时 e2e 约为单流 decode 的 2 倍"
                "（本例 56.0 × 2 ≈ 109 ≈ e2e 108.9）；e2e / 单流dec 明显小于 C 就说明并发没跑满"),
    ("TTFT", "首 token 时延中位数，单位 s；含排队等待"),
    ("tok/step", "每步产出 token 数的中位数（含义见上面“名词解释”）"),
    ("step中位", "单步时延的中位数，单位 ms"),
    ("step均值", "单步时延的平均值，单位 ms（受插队影响，通常高于 step中位）"),
    ("倍数", "step中位 / 25.5 ms；健康区间约 0.8-1.1x，≈1.3x 以上才算异常"),
    ("offload", "该轮 kv_offload_store_bytes_total 增量，单位 MB；池被缓存前缀占满时会明显上涨"),
    ("负载", "无 / 有(n) / 疑似：n = 观测到的峰值并发请求数（自己的 -c 不算），有 或 疑似 都表示"
             "别的客户端在抢 batch，此时本轮 ms/step 不能当硬件结论"),
]
JUDGE = [
    "三组速度别混：decode(tok/s) 是单流、不含 TTFT；e2e(tok/s) 是整轮、含 TTFT 与排队，并发下随 C 上升；"
    "单流decode中位 就是上面 decode 的中位数。并发是否有效看 e2e / 单流decode 是否 ≈ C。",
    "健康检查用 -c 1：step中位 20-26 ms 属正常（≈33 ms、基线倍数 >= 1.3 才算异常）。",
    "并发跑（-c N）时 per-stream step 随 C 增大属正常（一步要服务 C 条序列）：看 e2e 是否随 C 上升、"
    "单流 decode 是否按预期下降即可。",
    "step 分位（每轮表格下方打印）：p99 >> p50 = 有插队（peer 或排队），整体上移 = 引擎或硬件变慢。",
    "“其他负载”列出现 有/疑似 时，本轮 ms/step 含别的客户端的 prefill，不能当硬件结论——测速请独占服务。",
]


CONCEPTS = [
    (2, "step（一步）= 引擎跑一次前向，并把这一步产生的全部 token 一次性推给客户端；在 vLLM 里一个 SSE chunk "
        "就是一步，所以表里的步数 = chunk 数。"),
    (2, "不开投机解码（SPEC=off）时每步只出 1 个 token，decode 速度就等于 1 / step_ms（本机 25 ms -> 40 tok/s），"
        "这是 27B 稠密模型单次前向的硬成本。"),
    (2, "开 SPEC=dflash2 后，每步先由草稿模型猜 7 个 token，目标模型在“同一次”前向里把它们全部验证：猜对的全部"
        "接受，再加 1 个奖励 token。于是一步能拿走 2-3.4 个 token —— 这就是加速的全部来源，而 step_ms 基本不变"
        "（验证 7 个和验证 1 个都要把权重读一遍）。"),
    (2, "所以 tok/step = “每次前向平均拿走几个 token”，也就是接受长度（服务端日志里的 Mean acceptance length "
        "就是它）；它由文本好不好猜决定（英文基准 3.2-3.4，中文长文 2.2-2.5），与硬件、并发、负载无关。"
        "step_ms 才由硬件和负载决定。两者相乘才是速度："),
    (6, "decode(tok/s) = tok/step / step均值(ms) * 1000"),
    (6, "例（双并发那轮）：1000 token / 2.43 tok/step = 412 步；412 * 47.4 ms = 19.5 s；1000 / 19.5 = 51.3 tok/s，"
        "实测 51.0。"),
]


def wrap_text(text, width):
    """Wrap to `width` display columns: CJK counts as 2, ASCII words stay whole,
    and a line never starts with closing punctuation."""
    no_start = "，。、；：？！）】》」』”’%…·,.;:?!)]}"
    tokens, cur = [], ""
    for ch in text:
        if ch.isascii() and (ch.isalnum() or ch in "._-+/%=<>'\""):
            cur += ch
        else:
            if cur:
                tokens.append(cur)
                cur = ""
            tokens.append(ch)
    if cur:
        tokens.append(cur)

    lines, line, lw = [], "", 0
    for tk in tokens:
        w = disp_width(tk)
        if tk in no_start and line:  # glue punctuation to the line before it
            line += tk
            lw += w
            continue
        if lw + w > width and line:
            lines.append(line.rstrip())
            line, lw = "", 0
            if tk == " ":
                continue
        while w > width:  # a single token wider than the whole line
            head = ""
            while tk and disp_width(head + tk[0]) <= width:
                head += tk[0]
                tk = tk[1:]
            w = disp_width(tk)
            lines.append(head)
        line += tk
        lw += w
    if line:
        lines.append(line.rstrip())
    return lines or [""]


def print_legend():
    cols = min(max(shutil.get_terminal_size((120, 24)).columns, 80), 150)
    print("名词解释（看不懂 tok/step 就先看这里）：")
    for indent, text in CONCEPTS:
        prefix = " " * indent
        parts = wrap_text(text, max(40, cols - indent))
        for i, part in enumerate(parts):
            print((prefix if i == 0 else " " * (indent + 2)) + part)
    print()
    for title, entries in (("逐流表（每轮一张）", LEGEND_STREAMS), ("汇总表（末尾一张；列名从简以适配窄终端，单位写在下面）", LEGEND_SUMMARY)):
        print("%s：" % title)
        width = max(disp_width(label) for label, _ in entries)
        for label, text in entries:
            prefix = "  %s  " % (label + " " * (width - disp_width(label)))
            for i, line in enumerate(wrap_text(text, max(48, cols - disp_width(prefix)))):
                print((prefix if i == 0 else " " * disp_width(prefix)) + line)
    print("判读口径：")
    for line in JUDGE:
        prefix = "  - "
        for i, part in enumerate(wrap_text(line, max(48, cols - disp_width(prefix)))):
            print((prefix if i == 0 else " " * disp_width(prefix)) + part)
    print()


STREAM_HEADERS = ["流", "输出", "TTFT(s)", "decode(tok/s)", "tok/step",
                  "step中位(ms)", "step均值(ms)", "finish"]
STREAM_ALIGNS = ["r", "r", "r", "r", "r", "r", "r", "l"]
# Kept short on purpose so the whole table fits an 80-column terminal; the units
# and the full meaning live in the legend above it.
SUM_HEADERS = ["轮", "C", "e2e", "单流dec", "TTFT", "tok/step", "step中位", "step均值",
               "倍数", "offload", "负载"]
SUM_ALIGNS = ["r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "l"]


def verdict_text(peer, peak_running, conc, med_step, med_step_mean, max_p99, store_mb):
    """One-sentence conclusion for a run; printed before that run's table."""
    ratio = med_step / BASELINE_STEP_MS
    if peer:
        return ("检测到其他客户端（峰值 running %.0f > 你的 -c %d）：它的长上下文 prefill 与 decode 共享 batch，"
                "本轮 ms/step 会被抬高，不能当作硬件结论——测速请独占服务。" % (peak_running, conc))
    if conc > 1:
        return ("并发 c=%d 不判健康（一步要服务 %d 条序列，per-stream step 变大属正常）；"
                "健康请用 -c 1 单独复测（应 ≈ %.1f ms）。" % (conc, conc, BASELINE_STEP_MS))
    if ratio < HEALTHY_MAX_RATIO:
        if med_step_mean > med_step * 1.15:
            return ("引擎健康：无干扰时 step 中位 %.1f ms（%.2fx 基线），但均值 %.1f ms 被插队抬高"
                    "（p99 %.1f ms）——测速请独占服务。" % (med_step, ratio, med_step_mean, max_p99))
        return "健康：step %.1f ms（%.2fx 基线，与 README 一致）。" % (med_step, ratio)
    if store_mb > 50:
        return ("偏慢：step %.1f ms（%.2fx 基线），本轮驱逐层写入 %.0f MB（池被缓存前缀占满），"
                "换空闲时段复测。" % (med_step, ratio, store_mb))
    return ("偏慢：step %.1f ms（%.2fx 基线），无其他客户端、无驱逐流量；先确认 Windows 侧没有占用 GPU，"
            "再重启复测。" % (med_step, ratio))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="18020")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("-n", "--max-tokens", type=int, default=700)
    ap.add_argument("-f", "--prompt-file")
    ap.add_argument("-r", "--repeat", type=int, default=1)
    ap.add_argument("-c", "--concurrency", type=int, default=1,
                    help="number of concurrent streams in each burst (default 1)")
    ap.add_argument("-i", "--ignore-eos", action="store_true",
                    help="generate max_tokens regardless of EOS (stable lengths, bench-style)")
    ap.add_argument("-k", "--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    a = ap.parse_args()

    if a.concurrency < 1:
        sys.exit("concurrency must be >= 1")
    base = f"http://{a.host}:{a.port}"
    prompt = open(a.prompt_file).read() if a.prompt_file else DEFAULT_PROMPT
    auth = {"Authorization": "Bearer " + a.api_key} if a.api_key else {}
    hdr = {"Content-Type": "application/json", **auth}
    if scrape(base, "vllm:num_requests_running", auth) is None:
        sys.exit("no /metrics on %s (server down, or an older vLLM?)" % base)

    print("逐 step 解码探针：真实 chat 请求（OpenAI /v1/chat/completions），逐流测每步时延。")
    print_legend()

    summary = []
    for rep in range(1, a.repeat + 1):
        recs, wall, dm, watch = burst(base, a.model, prompt, a.max_tokens, hdr,
                                      a.concurrency, a.ignore_eos, a.concurrency > 1)
        ok = [x for x in recs if not x["err"]]
        print()
        print("第 %d/%d 轮  并发 c=%d  输入 %d token  输出上限 %d%s"
              % (rep, a.repeat, a.concurrency, ok[0]["prompt_tokens"] if ok else 0,
                 a.max_tokens, "  (ignore_eos)" if a.ignore_eos else ""))

        stats = None
        if ok:
            steps = [x["step"] for x in ok]
            med_step = statistics.median(steps)
            med_step_mean = statistics.median([x["step_mean"] for x in ok])
            decs = [x["decode"] for x in ok]
            ttfts = [x["ttft"] for x in ok]
            total = sum(x["n"] for x in ok)
            med_tokstep = statistics.median([x["n"] / max(x["chunks"], 1) for x in ok])
            peer = watch.peak_running > a.concurrency
            spec = 1 + dm["accepted"] / dm["drafts"] if dm["drafts"] else float("nan")
            max_p99 = max(x["p99"] for x in ok)
            stats = (med_step, med_step_mean, med_tokstep, total, decs, ttfts, peer, spec, max_p99)
            print("spec 接受长度 1+accepted/drafted = %.2f（服务端全局值%s）"
                  % (spec, "，含其他客户端的 token" if peer else ""))
            print("判定：%s" % verdict_text(peer, watch.peak_running, a.concurrency,
                                          med_step, med_step_mean, max_p99, dm["store_mb"]))
        rows = []
        for x in sorted(recs, key=lambda r: r["idx"]):
            if x["err"]:
                rows.append([x["idx"] + 1, "-", "-", "-", "-", "-", "-", "FAILED"])
            else:
                rows.append([x["idx"] + 1, x["n"], "%.2f" % x["ttft"], "%.1f" % x["decode"],
                             "%.2f" % (x["n"] / max(x["chunks"], 1)),
                             "%.1f" % x["step"], "%.1f" % x["step_mean"], x["finish"] or "-"])
        print(render_table(STREAM_HEADERS, rows, STREAM_ALIGNS))
        if ok and statistics.median([x["chunks"] for x in ok]) < 50:
            print("   提示：本轮每流只有 %.0f 步，中位数噪声大（建议 -n 300 以上再看 step）。"
                  % statistics.median([x["chunks"] for x in ok]))
        if ok:
            print("   step 分位：p50 %.1f / p90 %.1f / p99 %.1f ms%s"
                  % (statistics.median([x["step"] for x in ok]),
                     statistics.median([x["p90"] for x in ok]), max(x["p99"] for x in ok),
                     "（p99 远高于 p50：有插队）" if max(x["p99"] for x in ok) > 1.5 * statistics.median([x["step"] for x in ok]) else ""))
        for x in recs:
            if x["err"]:
                print("  ! 流 %d 失败：%s" % (x["idx"] + 1, x["err"]))
        if not ok:
            print("  本轮全部失败，跳过统计。")
            continue

        (med_step, med_step_mean, med_tokstep, total, decs, ttfts, peer, spec,
         max_p99) = stats
        row = []
        if a.repeat > 1:
            row.append("%d/%d" % (rep, a.repeat))
        row += [a.concurrency, "%.1f" % (total / wall),
                "%.1f" % statistics.median(decs), "%.2f" % statistics.median(ttfts),
                "%.2f" % med_tokstep, "%.1f" % med_step, "%.1f" % med_step_mean,
                "%.2fx" % (med_step / BASELINE_STEP_MS),
                "%+.0f" % dm["store_mb"],
                "有(%.0f)" % watch.peak_running if peer
                else "疑似" if med_step_mean > med_step * 1.15 else "无"]
        summary.append(row)

    if not summary:
        return

    print()
    print("汇总")
    heads, aligns = SUM_HEADERS, SUM_ALIGNS
    if a.repeat == 1:
        heads, aligns = heads[1:], aligns[1:]
    print(render_table(heads, [s for s in summary], aligns))


if __name__ == "__main__":
    main()
