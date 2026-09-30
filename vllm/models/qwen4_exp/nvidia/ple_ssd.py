# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Asynchronous PLE row reads from the original safetensors checkpoint."""

import bisect
import copy
import ctypes
import json
import os
import struct
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    eager_break_during_capture,
)
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger

from .ngram_embedding import Qwen4ExpNGramEmbedding, Qwen4ExpPLEEmbedding

logger = init_logger(__name__)

_IS_WINDOWS = os.name == "nt"

# Win32 constants for opening a checkpoint without tripping over other holders.
_GENERIC_READ = 0x80000000
_FILE_SHARE_ALL = 0x00000007
_OPEN_EXISTING = 3
_FILE_FLAG_RANDOM_ACCESS = 0x10000000


def _open_read_fd(path: str) -> int:
    """Open a read-only file descriptor that coexists with other holders.

    The Windows CRT defaults to denying all sharing, so an open fails whenever an
    antivirus or an indexer holds the file. Ask Win32 for full sharing instead and
    wrap that handle as a CRT descriptor so positional reads work the same way.
    On POSIX this is just ``os.open``.
    """
    if not _IS_WINDOWS:
        return os.open(path, os.O_RDONLY)
    import ctypes
    import msvcrt

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = ctypes.c_void_p
    k32.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
    ]
    handle = k32.CreateFileW(
        path,
        _GENERIC_READ,
        _FILE_SHARE_ALL,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_RANDOM_ACCESS,
        None,
    )
    if handle in (None, ctypes.c_void_p(-1).value):
        raise OSError(ctypes.get_last_error(), f"CreateFileW failed: {path}")
    return msvcrt.open_osfhandle(handle, os.O_BINARY)


def _pread(fd: int, count: int, offset: int) -> bytes:
    """Positional read. Windows has no ``os.pread``, so seek and read together."""
    if hasattr(os, "pread"):
        return os.pread(fd, count, offset)
    os.lseek(fd, offset, os.SEEK_SET)
    return os.read(fd, count)


def ple_ssd_weights_iterator(files, use_tqdm, strategy, local_expert_ids=None):
    """Avoid mmap's commit charge for checkpoint files containing only PLE."""
    from vllm.model_executor.model_loader.weight_utils import (
        safetensors_weights_iterator,
    )

    regular_files = []
    for path in files:
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            if header_len > 100_000_000:
                raise ValueError(f"Invalid safetensors header: {path}")
            header = json.loads(f.read(header_len))
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        if tensors and all(".ngram_embedding.shard_" in k for k in tensors):
            for name, info in tensors.items():
                if info["dtype"] != "BF16":
                    raise ValueError("PLE SSD requires BF16 checkpoint tensors")
                yield (
                    name,
                    torch.empty(info["shape"], dtype=torch.bfloat16, device="meta"),
                )
        else:
            if any(".ngram_embedding.shard_" in k for k in tensors):
                raise ValueError("PLE SSD requires PLE tensors in dedicated shards")
            regular_files.append(path)
    yield from safetensors_weights_iterator(
        regular_files, use_tqdm, strategy, local_expert_ids
    )


def _percentile(values: list[int], q: float) -> int:
    """Nearest-rank percentile, used to describe per-lookup miss counts."""
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(int(q * (len(ordered) - 1)), len(ordered) - 1)]


def _ms_percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(q * (len(ordered) - 1)), len(ordered) - 1)]


TRACE_STEPS = 200
TRACE_MARKS = 8
TRACE_ENABLED = os.getenv("VLLM_PLE_SSD_TRACE", "0") == "1"

# The host measures the ids wait from its own clock, so whatever the main
# stream already queued ahead of the preserve point is invisible to it. This
# ring marks where each step's model dispatch begins, which puts that prefix on
# the same device clock the intervals use.
_prefix_free: deque[torch.cuda.Event] = deque()
_prefix_pending: deque[torch.cuda.Event] = deque()


def record_step_prefix() -> None:
    """Mark the start of a step's model work. Called only while tracing."""
    if not TRACE_ENABLED or torch.cuda.is_current_stream_capturing():
        return
    if _prefix_free:
        event = _prefix_free.popleft()
    else:
        event = torch.cuda.Event(enable_timing=True)
    event.record(torch.cuda.current_stream())
    _prefix_pending.append(event)
    while len(_prefix_pending) > TRACE_STEPS:
        _prefix_free.append(_prefix_pending.popleft())


class _TraceSlot:
    """One step's timeline marks, preallocated so tracing allocates nothing."""

    __slots__ = (
        "tokens",
        "events",
        "prefix",
        "h_ids_seen",
        "h_ids_pre_true",
        "h_ids_polls",
        "h_gate",
        "h_gate_end",
        "h_ids",
        "h_ids_end",
        "h_rows",
        "h_h2d",
        "h_pending",
        "h_pending_end",
    )

    def __init__(self, events: list[torch.cuda.Event]) -> None:
        self.tokens = 0
        self.events = events
        self.prefix = None
        for name in self.__slots__[3:]:
            setattr(self, name, 0.0)


class PLESSDNativeReader:
    """Bounded asynchronous reads; the native call releases the Python GIL.

    Linux uses direct AIO through a duplicated O_DIRECT descriptor. Windows has
    no AIO, so the helper reads via handles it opens itself and the descriptor the
    table already holds only serves as an identity key for that file.
    """

    def __init__(self, table, library: str, depth: int) -> None:
        if not 1 <= depth <= 4096 or not 1 <= table.row_bytes <= 4096:
            raise ValueError("PLE AIO requires depth and row bytes in [1, 4096]")
        self._lib = ctypes.CDLL(library, use_errno=True)
        self._lib.rows_open.argtypes = [ctypes.c_uint]
        self._lib.rows_open.restype = ctypes.c_void_p
        self._lib.rows_close.argtypes = [ctypes.c_void_p]
        self._lib.rows_close.restype = None
        self._lib.rows_read.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
        ]
        self._lib.rows_read.restype = ctypes.c_int
        self._reader = self._lib.rows_open(depth)
        self._fds: dict[int, int] = {}
        if not self._reader:
            raise OSError(ctypes.get_errno(), "PLE io_setup failed")
        try:
            if _IS_WINDOWS:
                self._lib.rows_bind.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_wchar_p,
                ]
                self._lib.rows_bind.restype = ctypes.c_int
                self._lib.rows_depth.argtypes = [ctypes.c_void_p]
                self._lib.rows_depth.restype = ctypes.c_int
                for filename, fd in table._fds.items():
                    status = self._lib.rows_bind(
                        self._reader, fd, table._paths[filename]
                    )
                    if status:
                        raise OSError(-status, f"PLE bind failed: {filename}")
                self._depth = self._lib.rows_depth(self._reader)
                self._parts = np.asarray(table._parts, dtype=np.int64)
            else:
                for fd in table._fds.values():
                    self._fds[fd] = os.open(
                        f"/proc/self/fd/{fd}", os.O_RDONLY | os.O_DIRECT
                    )
                self._depth = depth
                self._parts = np.asarray(table._parts, dtype=np.int64)
                self._parts[:, 2] = [self._fds[fd] for fd in self._parts[:, 2]]
            self.row_bytes = table.row_bytes
        except BaseException:
            self.close()
            raise

    def read(self, rows: list[int]) -> list[tuple[int, bytes]]:
        ids = np.asarray(rows, dtype=np.int64)
        parts = self._parts[np.searchsorted(self._parts[:, 0], ids, side="right") - 1]
        fds = np.ascontiguousarray(parts[:, 2], dtype=np.int32)
        offsets = np.ascontiguousarray(
            parts[:, 3] + (ids - parts[:, 0]) * self.row_bytes, dtype=np.uint64
        )
        out = np.empty((len(ids), self.row_bytes), dtype=np.uint8)
        status = self._lib.rows_read(
            self._reader,
            fds.ctypes.data,
            offsets.ctypes.data,
            len(ids),
            self.row_bytes,
            out.ctypes.data,
        )
        if status:
            raise OSError(-status, "PLE asynchronous SSD read failed")
        return [(row, out[i].tobytes()) for i, row in enumerate(rows)]

    def close(self) -> None:
        if self._reader:
            self._lib.rows_close(self._reader)
            self._reader = None
        for fd in self._fds.values():
            os.close(fd)
        self._fds = {}
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


class PLESSDTable:
    """Read immutable BF16 rows with bounded concurrent I/O and an LRU cache.

    Native AIO or disk workers release the GIL while reading. Neither the
    full table nor whole checkpoint shards are loaded into RAM.
    """

    def __init__(
        self,
        model_path: str,
        prefix: str,
        rows: int,
        dim: int,
        workers: int = 8,
        cache_mb: int = 128,
        native_library: str | None = None,
        io_depth: int = 256,
    ) -> None:
        if workers < 1 or cache_mb < 0:
            raise ValueError(
                "PLE SSD workers must be positive and cache_mb nonnegative"
            )
        self.row_bytes = dim * 2
        self.rows = rows
        self.workers = workers
        self.cache_limit = cache_mb * 1024 * 1024 // (self.row_bytes + 128)
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.hits = self.reads = 0
        # Window-scoped counters, cleared on every report so a logged percentage
        # describes the window beside it instead of the whole process. Hits count
        # distinct rows within a lookup; occurrences count rows as they are asked
        # for, duplicates included, which is what a consumer actually waits on.
        self.demanded_lookups = 0
        self.demanded_hit_rows = 0
        self.demanded_miss_rows = 0
        self.demanded_hit_occ = 0
        self.demanded_miss_occ = 0
        self.demanded_rows = 0
        self.demanded_all_hit = 0
        self.demanded_miss_hist: list[int] = []
        self.lookahead_lookups = 0
        self.lookahead_hit_rows = 0
        self.lookahead_miss_rows = 0
        self._condition = threading.Condition()
        self._reading = False
        self._waiting = 0
        self._fds: dict[str, int] = {}
        self._paths: dict[str, str] = {}
        self._parts: list[tuple[int, int, int, int]] = []
        root = Path(model_path)
        with open(root / "model.safetensors.index.json") as f:
            weight_map = json.load(f)["weight_map"]
        # HF and vLLM use different wrappers around the same decoder layers.
        suffix = prefix[prefix.index("layers.") :] + ".shard_"
        keys = [k for k in weight_map if suffix in k and k.endswith(".weight")]
        keys.sort(key=lambda k: int(k.rsplit("shard_", 1)[1].split(".")[0]))
        if not keys:
            raise ValueError(f"No checkpoint PLE shards match {prefix}")
        headers = {}
        start = 0
        try:
            for i, key in enumerate(keys):
                if int(key.rsplit("shard_", 1)[1].split(".")[0]) != i:
                    raise ValueError("PLE checkpoint shard indices must be contiguous")
                filename = weight_map[key]
                if filename not in headers:
                    path = root / filename
                    with open(path, "rb") as f:
                        header_len = struct.unpack("<Q", f.read(8))[0]
                        if header_len > 100_000_000:
                            raise ValueError(f"Invalid safetensors header: {path}")
                        headers[filename] = (
                            json.loads(f.read(header_len)),
                            8 + header_len,
                        )
                    fd = _open_read_fd(str(path))
                    self._fds[filename] = fd
                    self._paths[filename] = str(path)
                    if hasattr(os, "posix_fadvise"):
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
                header, data_start = headers[filename]
                info = header[key]
                shape = info["shape"]
                if info["dtype"] != "BF16" or len(shape) != 2 or shape[1] != dim:
                    raise ValueError(f"PLE SSD requires BF16 [rows, {dim}]: {key}")
                begin, end = info["data_offsets"]
                fd = self._fds[filename]
                if end - begin != shape[0] * self.row_bytes:
                    raise ValueError(f"Invalid PLE tensor byte range: {key}")
                if data_start + end > os.fstat(fd).st_size:
                    raise ValueError(f"Incomplete PLE checkpoint: {filename}")
                self._parts.append((start, start + shape[0], fd, data_start + begin))
                start += shape[0]
            if start != rows:
                raise ValueError(f"PLE table has {start} rows, expected {rows}")
        except BaseException:
            for fd in self._fds.values():
                os.close(fd)
            raise
        self._starts = [p[0] for p in self._parts]
        # Windows seek+read is stateful per descriptor, so workers need a lock
        # per file; POSIX pread needs none.
        self._locks = (
            {fd: threading.Lock() for fd in self._fds.values()}
            if _IS_WINDOWS
            else {}
        )
        self._native = None
        self._pool = None
        try:
            if native_library:
                self._native = PLESSDNativeReader(self, native_library, io_depth)
            else:
                self._pool = ThreadPoolExecutor(
                    workers, thread_name_prefix="ple-ssd-io"
                )
        except BaseException:
            for fd in self._fds.values():
                os.close(fd)
            raise

    def _read_rows(self, ids: list[int]) -> list[tuple[int, bytes]]:
        result = []
        for row in ids:
            start, _, fd, offset = self._parts[
                bisect.bisect_right(self._starts, row) - 1
            ]
            offset = offset + (row - start) * self.row_bytes
            if self._locks:
                with self._locks[fd]:
                    data = _pread(fd, self.row_bytes, offset)
            else:
                data = _pread(fd, self.row_bytes, offset)
            if len(data) != self.row_bytes:
                raise OSError(f"Short PLE SSD read for row {row}")
            result.append((row, data))
        return result

    def read(
        self, ids: np.ndarray, output: np.ndarray, *, prefetch: bool = False
    ) -> None:
        """Serialize cache/I/O access, giving demanded rows priority over read-ahead."""
        with self._condition:
            if not prefetch:
                self._waiting += 1
            try:
                while self._reading or (prefetch and self._waiting):
                    self._condition.wait()
                self._reading = True
            finally:
                if not prefetch:
                    self._waiting -= 1
        try:
            self._read(ids, output, prefetch)
        finally:
            with self._condition:
                self._reading = False
                self._condition.notify_all()

    def _read(
        self, ids: np.ndarray, output: np.ndarray, prefetch: bool
    ) -> None:
        """Fill a byte buffer in request order, preserving duplicate rows."""
        flat = ids.reshape(-1)
        if output.shape != (flat.size, self.row_bytes) or output.dtype != np.uint8:
            raise ValueError("PLE SSD output must be uint8 [number of IDs, row bytes]")
        unique, counts = np.unique(flat, return_counts=True)
        row_ids = unique.tolist()
        if row_ids and (row_ids[0] < 0 or row_ids[-1] >= self.rows):
            raise IndexError("PLE row ID outside checkpoint table")
        found = {}
        missing = []
        hit_occ = miss_occ = 0
        for row, count in zip(row_ids, counts.tolist()):
            cached = self.cache.get(row)
            if cached is None:
                missing.append(row)
                miss_occ += int(count)
            else:
                found[row] = cached
                self.cache.move_to_end(row)
                hit_occ += int(count)
        self.hits += len(found)
        self.reads += len(missing)
        if prefetch:
            self.lookahead_lookups += 1
            self.lookahead_hit_rows += len(found)
            self.lookahead_miss_rows += len(missing)
        else:
            self.demanded_lookups += 1
            self.demanded_hit_rows += len(found)
            self.demanded_miss_rows += len(missing)
            self.demanded_hit_occ += hit_occ
            self.demanded_miss_occ += miss_occ
            self.demanded_rows += flat.size
            if not missing:
                self.demanded_all_hit += 1
            self.demanded_miss_hist.append(len(missing))
        if missing:
            # Keep only workers tasks queued, even for a large prefill.
            active_workers = min(self.workers, len(missing))
            if len(missing) < 1024:
                active_workers = min(active_workers, 8)
            results: Iterable[list[tuple[int, bytes]]]
            if self._native is not None:
                results = [self._native.read(missing)]
            else:
                assert self._pool is not None
                chunks = [missing[i::active_workers] for i in range(active_workers)]
                results = self._pool.map(self._read_rows, chunks)
            for result in results:
                for row, data in result:
                    found[row] = data
                    if self.cache_limit:
                        self.cache[row] = data
                        if len(self.cache) > self.cache_limit:
                            self.cache.popitem(last=False)
        data = b"".join(found[row] for row in flat.tolist())
        output.reshape(-1)[:] = np.frombuffer(data, dtype=np.uint8)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        if self._native is not None:
            self._native.close()
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


class PLEPromptPrefetcher:
    """Warm a bounded prefix of known prompt rows while CUDA executes chunks."""

    def __init__(self, ngram: Qwen4ExpNGramEmbedding, limit: int) -> None:
        self.table = ngram.ngram_embedding._table
        self.ngram = copy.copy(ngram)
        self.ngram._modules = {}
        self.ngram._parameters = {}
        self.ngram._buffers = {k: v.detach().cpu() for k, v in ngram._buffers.items()}
        self.limit = min(limit, self.table.cache_limit // (2 * ngram.ngram_heads))
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="ple-prompt")
        self._future: Future | None = None
        self._stop = threading.Event()
        self._request_id: str | None = None

    def submit(self, request_id: str, tokens: list[int], start: int) -> None:
        if min(self.limit, len(tokens) - start) < 2048:
            return
        if self._future is not None:
            if not self._future.done():
                return
            self._future.result()
        history = min(start, self.ngram.ngram_size - 1)
        tokens = tokens[start - history : start + self.limit]
        self._request_id = request_id
        self._stop.clear()
        self._future = self._pool.submit(self._prefetch, tokens, history)

    @torch.inference_mode()
    def _prefetch(self, tokens: list[int], history: int) -> None:
        ngram = self.ngram
        context = [ngram.eos_token_id] * (ngram.ngram_size - 1 - history)
        context += tokens[:history]
        ids = ngram.compute_ngram_ids(
            torch.tensor(tokens[history:]),
            torch.tensor([0, len(tokens) - history]),
            torch.tensor([context]),
        ).numpy()
        for start in range(0, len(ids), 256):
            if self._stop.is_set():
                return
            batch = ids[start : start + 256]
            output = np.empty((batch.size, self.table.row_bytes), dtype=np.uint8)
            self.table.read(batch, output, prefetch=True)

    def cancel(self, request_id: str) -> None:
        if request_id == self._request_id:
            self._stop.set()

    def close(self) -> None:
        self._stop.set()
        self._pool.shutdown(wait=True)
        if self._future is not None:
            self._future.result()


class Qwen4ExpPLESSDEmbedding(Qwen4ExpPLEEmbedding):
    """Prefetch SSD rows while the preceding decoder layer executes on CUDA."""

    supports_prefetch: ClassVar[bool] = True

    def __init__(self, *args, **kwargs) -> None:
        config = get_current_vllm_config()
        if config.parallel_config.tensor_parallel_size != 1 or (
            config.parallel_config.data_parallel_size != 1
        ):
            raise ValueError("PLE SSD offload currently requires TP=DP=1")
        super().__init__(*args, **kwargs)
        if self.weight.dtype != torch.bfloat16:
            raise ValueError("PLE SSD offload currently requires BF16 embeddings")
        options = config.additional_config
        self._table = PLESSDTable(
            config.model_config.model,
            kwargs["prefix"],
            self.org_vocab_size,
            self.embedding_dim,
            int(options.get("ple_ssd_workers", 8)),
            int(options.get("ple_ssd_cache_mb", 128)),
            options.get("ple_ssd_native_library"),
            int(options.get("ple_ssd_io_depth", 256)),
        )
        self.prompt_prefetcher: PLEPromptPrefetcher | None = None
        self._device = torch.device("cuda", torch.cuda.current_device())
        shape = (kwargs["max_total_tokens"], kwargs["num_ngram_heads"])
        self._ids = torch.empty(shape, dtype=torch.int64, device="cpu", pin_memory=True)
        self._device_ids = torch.empty_like(self._ids, device=self._device)
        self._host = torch.empty(
            (*shape, self.embedding_dim),
            dtype=torch.bfloat16,
            device="cpu",
            pin_memory=True,
        )
        self._gpu = torch.empty_like(self._host, device=self._device)
        self._stream = torch.cuda.Stream(device=self._device)
        self._ids_ready = torch.cuda.Event()
        self._copy_ready = torch.cuda.Event()
        self._previous_use = torch.cuda.Event()
        self._previous_use.record()
        self._pending: Future | None = None
        self._coordinator = ThreadPoolExecutor(1, thread_name_prefix="ple-ssd-prefetch")
        self._stats = os.getenv("VLLM_PLE_SSD_STATS", "0") == "1"
        self._stat_wait = 0.0
        self._stat_ids = 0.0
        self._stat_read = 0.0
        self._stat_steps = 0
        self._stat_tokens = 0
        self._trace = os.getenv("VLLM_PLE_SSD_TRACE", "0") == "1"
        # Waiting on an event keeps the interpreter lock, so a coordinator that
        # blocks there stops the thread that is trying to submit the next step.
        # Polling yields the lock instead, at the cost of coarse wake-up timing.
        self._ids_poll = os.getenv("VLLM_PLE_SSD_IDS_POLL", "0") == "1"
        # Events are made once, eight per traced step, and read only when a
        # window closes, so timing never costs a synchronization on the path
        # being timed. Device marks pair up through torch's elapsed_time, which
        # compares one device clock against itself rather than against the host.
        self._trace_events: list[torch.cuda.Event] = []
        self._trace_pool: list[_TraceSlot] = []
        self._trace_active: list[_TraceSlot] = []
        self._trace_index = 0
        self._trace_slot: _TraceSlot | None = None
        if self._trace:
            self._trace_events = [
                torch.cuda.Event(enable_timing=True)
                for _ in range(TRACE_STEPS * TRACE_MARKS)
            ]
            self._trace_pool = [
                _TraceSlot(self._trace_events[i * TRACE_MARKS : (i + 1) * TRACE_MARKS])
                for i in range(TRACE_STEPS)
            ]
            logger.info("PLE SSD timeline tracing on, %d steps per window", TRACE_STEPS)
        if self._ids_poll:
            logger.info("PLE SSD ids wait polls instead of blocking on the event")
        self._loaded_starts: set[int] = set()
        logger.info(
            "PLE SSD: %.2f GiB on disk, %d I/O workers, %d MiB row cache",
            self.org_vocab_size * self._table.row_bytes / 2**30,
            self._table.workers,
            int(options.get("ple_ssd_cache_mb", 128)),
        )
        if self._table._native is not None:
            logger.info(
                "PLE SSD uses native %s, queue depth %d (asked %d)",
                "Windows overlapped reads" if _IS_WINDOWS else "Linux AIO",
                self._table._native._depth,
                int(options.get("ple_ssd_io_depth", 256)),
            )

    def allocate_embedding_weight(self, num_embeddings, embedding_dim, dtype):
        return torch.empty(num_embeddings, embedding_dim, dtype=dtype, device="meta")

    def weight_loader(self, param, loaded_weight, checkpoint_start=None) -> None:
        if checkpoint_start is None or loaded_weight.dtype != torch.bfloat16:
            raise ValueError("PLE SSD requires checkpoint-split BF16 weights")
        self._loaded_starts.add(checkpoint_start)

    def _read_and_copy(self, tokens: int) -> None:
        t0 = time.perf_counter()
        polls = 0
        slot = self._trace_slot
        pre_true = 1 if self._ids_ready.query() else 0
        if self._ids_poll:
            while not self._ids_ready.query():
                polls += 1
                time.sleep(0.0002)
        else:
            self._ids_ready.synchronize()
        self._stat_ids += time.perf_counter() - t0
        if slot is not None:
            slot.h_ids = t0
            slot.h_ids_end = time.perf_counter()
            slot.h_ids_polls = polls
            slot.h_ids_pre_true = pre_true
        t0 = time.perf_counter()
        self._table.read(
            self._ids[:tokens].numpy(),
            self._host[:tokens]
            .view(torch.uint8)
            .numpy()
            .reshape(-1, self._table.row_bytes),
        )
        self._stat_read += time.perf_counter() - t0
        if slot is not None:
            slot.h_rows = time.perf_counter()
        self._stat_tokens += tokens
        self._copy_to_gpu(tokens)

    def _copy_to_gpu(self, tokens: int) -> None:
        slot = self._trace_slot
        if slot is not None:
            slot.h_h2d = time.perf_counter()
        with torch.cuda.device(self._device), torch.cuda.stream(self._stream):
            self._gpu[:tokens].copy_(self._host[:tokens], non_blocking=True)
            self._copy_ready.record(self._stream)
            if slot is not None:
                slot.events[6].record(self._stream)

    @partial(eager_break_during_capture, always=True)
    def start_prefetch(self, hidden_states, ngram_ids) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("PLE SSD requires eager or breakable CUDA graphs")
        if self._pending is not None:
            raise RuntimeError("Previous PLE SSD lookup has not been consumed")
        if len(self._loaded_starts) != len(self._table._parts):
            raise RuntimeError("PLE checkpoint shards have not all been loaded")
        tokens = ngram_ids.shape[0]
        if tokens > self._ids.shape[0]:
            raise ValueError("PLE SSD prefetch exceeds configured token capacity")
        if BreakableCUDAGraphCapture.current() is not None:
            # Captured ID-producing kernels have not executed yet. Actual
            # lookups start on replay, when those kernels have run.
            return
        # Host staging must outlive H2D; device staging must outlive its consumer.
        slot: _TraceSlot | None = None
        if self._trace:
            slot = self._trace_pool[self._trace_index]
            self._trace_index = (self._trace_index + 1) % TRACE_STEPS
            slot.tokens = tokens
            slot.prefix = _prefix_pending.pop() if _prefix_pending else None
            self._trace_active.append(slot)
            self._trace_slot = slot
            slot.h_gate = time.perf_counter()
        self._copy_ready.synchronize()
        if slot is not None:
            slot.h_gate_end = time.perf_counter()
        self._stream.wait_event(self._previous_use)
        # Preserve IDs before the next graph segment reuses their pool slot.
        self._device_ids[:tokens].copy_(ngram_ids)
        if slot is not None:
            slot.events[0].record(torch.cuda.current_stream())
        self._stream.wait_stream(torch.cuda.current_stream())
        if slot is not None:
            slot.events[1].record(self._stream)
            slot.events[2].record(self._stream)
        with torch.cuda.stream(self._stream):
            if slot is not None:
                slot.events[3].record(self._stream)
            self._ids[:tokens].copy_(self._device_ids[:tokens], non_blocking=True)
            self._ids_ready.record(self._stream)
            if slot is not None:
                slot.events[4].record(self._stream)
                slot.events[5].record(self._stream)
        self._pending = self._coordinator.submit(self._read_and_copy, tokens)

    def _flush_trace(self) -> None:
        """Report the traced window stratified by step shape, then free the slots.

        Device intervals come from elapsed_time between event pairs, so they
        compare one device clock against itself; host intervals come from one
        monotonic host clock. The two are compared by totals rather than bridged,
        because there is no cheap way to map a device clock onto a host clock.
        """
        if not self._trace_active:
            return
        torch.cuda.synchronize(self._device)
        names = ("p->w", "w->j", "j->s", "copy", "s->e", "e->h", "h->c")
        pairs = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7))
        groups: dict[int, list[_TraceSlot]] = {}
        for slot in self._trace_active:
            groups.setdefault(slot.tokens, []).append(slot)
        for tokens in sorted(groups):
            slots = groups[tokens]
            device: dict[str, list[float]] = {name: [] for name in names}
            device["chain"] = []
            device["prefix"] = []
            host: dict[str, list[float]] = {
                "gate": [],
                "ids": [],
                "rows": [],
                "pending": [],
            }
            for slot in slots:
                events = slot.events
                total = 0.0
                for name, (a, b) in zip(names, pairs):
                    value = events[a].elapsed_time(events[b])
                    device[name].append(value)
                    total += value
                device["chain"].append(total)
                if slot.prefix is not None:
                    device["prefix"].append(slot.prefix.elapsed_time(events[0]))
                host["gate"].append(slot.h_gate_end - slot.h_gate)
                host["ids"].append(slot.h_ids_end - slot.h_ids)
                host["rows"].append(slot.h_rows - slot.h_ids_end)
                host["pending"].append(slot.h_pending_end - slot.h_pending)
            parts = [
                f"{name} {1e3 * _ms_percentile(device[name], 0.5):.2f}/"
                f"{1e3 * _ms_percentile(device[name], 0.95):.2f}"
                for name in names
            ]
            logger.info(
                "PLE trace device (tokens=%d, n=%d, p50/p95 us): %s",
                tokens,
                len(slots),
                ", ".join(parts),
            )
            logger.info(
                "PLE trace chain (tokens=%d): p50 %.2f p95 %.2f us; host "
                "gate %.2f/%.2f ids %.2f/%.2f rows %.2f/%.2f "
                "pending %.2f/%.2f us",
                tokens,
                1e3 * _ms_percentile(device["chain"], 0.5),
                1e3 * _ms_percentile(device["chain"], 0.95),
                1e6 * _ms_percentile(host["gate"], 0.5),
                1e6 * _ms_percentile(host["gate"], 0.95),
                1e6 * _ms_percentile(host["ids"], 0.5),
                1e6 * _ms_percentile(host["ids"], 0.95),
                1e6 * _ms_percentile(host["rows"], 0.5),
                1e6 * _ms_percentile(host["rows"], 0.95),
                1e6 * _ms_percentile(host["pending"], 0.5),
                1e6 * _ms_percentile(host["pending"], 0.95),
            )
            logger.info(
                "PLE trace ids (tokens=%d): wait p50 %.2f p95 %.2f us, "
                "polls p50 %.1f, us per poll %.2f, ready before waiting in "
                "%d of %d steps, seen true by main thread in %d of %d steps",
                tokens,
                1e6 * _ms_percentile(host["ids"], 0.5),
                1e6 * _ms_percentile(host["ids"], 0.95),
                _percentile([s.h_ids_polls for s in slots], 0.5),
                1e6
                * _ms_percentile(host["ids"], 0.5)
                / max(_percentile([s.h_ids_polls for s in slots], 0.5), 1.0),
                sum(1 for s in slots if s.h_ids_pre_true > 0.0),
                len(slots),
                sum(1 for s in slots if s.h_ids_seen > 0.0),
                len(slots),
            )
            logger.info(
                "PLE trace prefix (tokens=%d): model dispatch to preserve "
                "p50 %.2f p95 %.2f us over %d steps",
                tokens,
                1e3 * _ms_percentile(device["prefix"], 0.5),
                1e3 * _ms_percentile(device["prefix"], 0.95),
                len(device["prefix"]),
            )
            for slot in slots:
                if slot.prefix is not None:
                    _prefix_free.append(slot.prefix)
                    slot.prefix = None
        self._trace_active = []

    @partial(eager_break_during_capture, always=True)
    def _finalize_prefetch(self, output: torch.Tensor) -> None:
        if BreakableCUDAGraphCapture.current() is not None:
            output.zero_()
            return
        if self._pending is None:
            raise RuntimeError("PLE SSD lookup was not prefetched")
        t0 = time.perf_counter()
        self._pending.result()
        self._stat_wait += time.perf_counter() - t0
        slot = self._trace_slot
        if slot is not None:
            slot.h_pending = t0
            slot.h_pending_end = time.perf_counter()
            slot.h_ids_seen = (
                time.perf_counter() if self._ids_ready.query() else 0.0
            )
            slot.events[7].record(torch.cuda.current_stream())
        self._pending = None
        self._trace_slot = None
        self._stat_steps += 1
        if self._trace and self._stat_steps % TRACE_STEPS == 0:
            self._flush_trace()
        if self._stats and self._stat_steps % 200 == 0:
            table = self._table
            logger.info(
                "PLE SSD stats over %d steps: wait %.2f ms/step, "
                "row assembly %.2f ms/step, ids wait %.2f ms/step",
                self._stat_steps,
                1e3 * self._stat_wait / self._stat_steps,
                1e3 * self._stat_read / self._stat_steps,
                1e3 * self._stat_ids / self._stat_steps,
            )
            rows = table.demanded_hit_rows + table.demanded_miss_rows
            occ = table.demanded_hit_occ + table.demanded_miss_occ
            lookups = table.demanded_lookups
            logger.info(
                "PLE decode window: %d lookups, %.1f rows per lookup, "
                "%.1f%% unique hits, %.1f%% row hits, %.1f%% all-hit "
                "lookups, miss p50/p95/p99 %d/%d/%d, "
                "%d lookahead lookups",
                lookups,
                table.demanded_rows / lookups if lookups else 0.0,
                100 * table.demanded_hit_rows / rows if rows else 0.0,
                100 * table.demanded_hit_occ / occ if occ else 0.0,
                100 * table.demanded_all_hit / lookups if lookups else 0.0,
                _percentile(table.demanded_miss_hist, 0.5),
                _percentile(table.demanded_miss_hist, 0.95),
                _percentile(table.demanded_miss_hist, 0.99),
                table.lookahead_lookups,
            )
            self._stat_wait = 0.0
            self._stat_read = 0.0
            self._stat_ids = 0.0
            self._stat_steps = 0
            table.hits = table.reads = 0
            table.demanded_lookups = 0
            table.demanded_hit_rows = 0
            table.demanded_miss_rows = 0
            table.demanded_hit_occ = 0
            table.demanded_miss_occ = 0
            table.demanded_rows = 0
            table.demanded_all_hit = 0
            table.demanded_miss_hist = []
            table.lookahead_lookups = 0
            table.lookahead_hit_rows = 0
            table.lookahead_miss_rows = 0
        torch.cuda.current_stream().wait_event(self._copy_ready)
        output.copy_(self._gpu[: output.shape[0]].flatten(-2))
        self._previous_use.record()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = torch.empty(
            (hidden_states.shape[0], self._gpu.shape[1] * self.embedding_dim),
            dtype=self.weight.dtype,
            device=hidden_states.device,
        )
        self._finalize_prefetch(output)
        return output

    def close(self) -> None:
        self._coordinator.shutdown(wait=True)
        try:
            if self.prompt_prefetcher is not None:
                self.prompt_prefetcher.close()
            self._stream.synchronize()
        finally:
            self._table.close()
