/* SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 * Windows asynchronous reads for exact BF16 rows in a safetensors checkpoint.
 *
 * Same row-at-a-time page reads as ple_ssd_io.c: every row is served by one
 * page-aligned read of round_up(row_in_page + row_bytes, 4096), so unaligned rows
 * never need unaligned I/O, and the native call runs without the Python GIL.
 *
 * Requests are tracked with one auto-reset event each and reaped with
 * WaitForMultipleObjects, because associating a file handle with an existing I/O
 * completion port is refused on this machine with ERROR_INVALID_PARAMETER. The
 * reap is pipelined: as soon as one slot frees up it takes the next row, so the
 * queue stays full instead of draining once per batch.
 *
 * WaitForMultipleObjects accepts at most MAXIMUM_WAIT_OBJECTS handles, which caps
 * the queue depth here at 64; the Linux helper's io_depth of 256 is not
 * reproducible with this pattern. rows_depth reports what was actually used.
 *
 * A handle opened with FILE_FLAG_NO_BUFFERING cannot read past the end of the
 * file, so rows whose page crosses EOF are read exactly, buffered, straight into
 * the caller's output. Those are the last rows of the last shard.
 */
#include <windows.h>

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PAGE 4096
#define SLOT 8192
#define WAIT_MS 60000
#ifndef EREMOTEIO
#define EREMOTEIO 124   /* the read returned nothing at all */
#endif

struct bound {
  struct bound* next;
  int fd;
  HANDLE direct;    /* FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS */
  HANDLE buffered;  /* FILE_FLAG_RANDOM_ACCESS */
  long long size;
};

struct win_req {
  OVERLAPPED ov;
  struct bound* b;
  unsigned char* dst;  /* where the row's bytes end up */
  unsigned char* slot; /* page buffer, NULL for exact buffered reads */
  unsigned need;       /* bytes that must arrive */
  unsigned ordinal;    /* index into the caller's request list */
  int exact;           /* 1 when read straight into dst, buffered */
  int outstanding;
};

struct reader {
  unsigned depth;
  unsigned char* buf;
  struct win_req* reqs;
  HANDLE* events;
  struct bound* bounds;
  int dead;
  unsigned fault_slot;
  unsigned fault_ordinal;
  int fault_exact;
};

__declspec(dllexport) int rows_depth(struct reader* r) {
  return r ? (int)r->depth : -1;
}

__declspec(dllexport) int rows_fault(struct reader* r, unsigned* slot,
                                     unsigned* ordinal, int* exact) {
  if (!r) return -1;
  *slot = r->fault_slot;
  *ordinal = r->fault_ordinal;
  *exact = r->fault_exact;
  return 0;
}

__declspec(dllexport) void rows_close(struct reader* r) {
  if (!r) return;
  for (struct bound* b = r->bounds; b;) {
    struct bound* next = b->next;
    if (b->direct) CloseHandle(b->direct);
    if (b->buffered) CloseHandle(b->buffered);
    free(b);
    b = next;
  }
  if (r->events) {
    for (unsigned j = 0; j < r->depth; ++j) {
      if (r->events[j]) CloseHandle(r->events[j]);
    }
    free(r->events);
  }
  if (r->buf) VirtualFree(r->buf, 0, MEM_RELEASE);
  free(r->reqs);
  free(r);
}

__declspec(dllexport) struct reader* rows_open(unsigned depth) {
  if (!depth) {
    errno = EINVAL;
    return NULL;
  }
  if (depth > MAXIMUM_WAIT_OBJECTS) depth = MAXIMUM_WAIT_OBJECTS;
  struct reader* r = calloc(1, sizeof(*r));
  if (!r) {
    errno = ENOMEM;
    return NULL;
  }
  r->depth = depth;
  r->reqs = calloc(depth, sizeof(*r->reqs));
  r->events = calloc(depth, sizeof(*r->events));
  if (!r->reqs || !r->events) {
    rows_close(r);
    errno = ENOMEM;
    return NULL;
  }
  /* VirtualAlloc hands back 64 KiB granules, so every slot slice is page aligned
   * by construction, which is what NO_BUFFERING requires. */
  r->buf = VirtualAlloc(NULL, (size_t)depth * SLOT, MEM_COMMIT | MEM_RESERVE,
                        PAGE_READWRITE);
  if (!r->buf) {
    rows_close(r);
    errno = ENOMEM;
    return NULL;
  }
  for (unsigned j = 0; j < depth; ++j) {
    r->events[j] = CreateEventW(NULL, FALSE, FALSE, NULL);
    if (!r->events[j]) {
      int err = (int)GetLastError();
      rows_close(r);
      errno = err;
      return NULL;
    }
  }
  return r;
}

/* Associate one checkpoint fd with the file this reader opens itself. */
__declspec(dllexport) int rows_bind(struct reader* r, int fd,
                                    const wchar_t* path) {
  if (!r || r->dead) {
    errno = EBADF;
    return -EBADF;
  }
  for (struct bound* b = r->bounds; b; b = b->next) {
    if (b->fd == fd) return 0;
  }
  struct bound* b = calloc(1, sizeof(*b));
  if (!b) {
    errno = ENOMEM;
    return -ENOMEM;
  }
  b->fd = fd;
  DWORD share = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
  b->direct = CreateFileW(path, GENERIC_READ, share, NULL, OPEN_EXISTING,
                          FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS, NULL);
  if (b->direct == INVALID_HANDLE_VALUE) {
    int err = (int)GetLastError();
    free(b);
    errno = err;
    return -1000000 - err;
  }
  b->buffered = CreateFileW(path, GENERIC_READ, share, NULL, OPEN_EXISTING,
                            FILE_FLAG_RANDOM_ACCESS, NULL);
  if (b->buffered == INVALID_HANDLE_VALUE) {
    int err = (int)GetLastError();
    CloseHandle(b->direct);
    free(b);
    errno = err;
    return -2000000 - err;
  }
  LARGE_INTEGER size;
  if (!GetFileSizeEx(b->direct, &size)) {
    int err = (int)GetLastError();
    CloseHandle(b->direct);
    CloseHandle(b->buffered);
    free(b);
    errno = err;
    return -3000000 - err;
  }
  b->size = size.QuadPart;
  b->next = r->bounds;
  r->bounds = b;
  return 0;
}

static struct bound* find_bound(struct reader* r, int fd) {
  for (struct bound* b = r->bounds; b; b = b->next) {
    if (b->fd == fd) return b;
  }
  return NULL;
}

static void kill_reader(struct reader* r) {
  for (struct bound* b = r->bounds; b; b = b->next) {
    CancelIoEx(b->direct, NULL);
    CancelIoEx(b->buffered, NULL);
  }
  r->dead = 1;   /* mirrors io_destroy: the helper is unusable afterwards */
}

/* Queue one row onto a slot. Returns 1 when queued, 0 with error left in place. */
static int start_req(struct reader* r, unsigned slot, int fd, uint64_t offset,
                     unsigned row_bytes, unsigned char* out, unsigned ordinal) {
  struct bound* b = find_bound(r, fd);
  if (!b) {
    errno = EBADF;
    return 0;
  }
  struct win_req* req = &r->reqs[slot];
  memset(req, 0, sizeof(*req));
  req->b = b;
  req->ordinal = ordinal;
  req->dst = out + (size_t)ordinal * row_bytes;
  req->ov.hEvent = r->events[slot];
  unsigned delta = (unsigned)(offset & (PAGE - 1));
  long long page = (long long)(offset & ~(uint64_t)(PAGE - 1));
  if (page + (long long)(delta + row_bytes) > b->size) {
    /* The page would run past the end of the file, which an unbuffered handle
     * cannot deliver. Read exactly the row, buffered, straight into dst. */
    req->exact = 1;
    req->slot = NULL;
    req->need = row_bytes;
    req->ov.Offset = (DWORD)(offset & 0xFFFFFFFF);
    req->ov.OffsetHigh = (DWORD)((offset >> 32) & 0xFFFFFFFF);
    if (!ReadFile(b->buffered, req->dst, row_bytes, NULL, &req->ov)) {
      int err = (int)GetLastError();
      if (err != ERROR_IO_PENDING) {
        errno = err;
        return 0;
      }
    }
    req->outstanding = 1;
    return 1;
  }
  req->exact = 0;
  req->slot = r->buf + (size_t)slot * SLOT;
  req->need = delta + row_bytes;
  req->ov.Offset = (DWORD)(page & 0xFFFFFFFF);
  req->ov.OffsetHigh = (DWORD)((page >> 32) & 0xFFFFFFFF);
  unsigned need = (delta + row_bytes + PAGE - 1) & ~(PAGE - 1);
  if (!ReadFile(b->direct, req->slot, need, NULL, &req->ov)) {
    int err = (int)GetLastError();
    if (err != ERROR_IO_PENDING) {
      errno = err;
      return 0;
    }
  }
  req->outstanding = 1;
  return 1;
}

__declspec(dllexport) int rows_read(struct reader* r, const int* fds,
                                    const uint64_t* offsets, unsigned count,
                                    unsigned row_bytes, unsigned char* out) {
  if (!r || r->dead) return -EBADF;
  if (!row_bytes || row_bytes > PAGE) return -EINVAL;
  if (!count) return 0;

  unsigned submitted = 0;
  unsigned harvested = 0;
  int error = 0;
  for (unsigned slot = 0; slot < r->depth && submitted < count; ++slot) {
    if (!start_req(r, slot, fds[submitted], offsets[submitted], row_bytes, out,
                   submitted)) {
      error = -errno;
      goto failed;
    }
    ++submitted;
  }
  while (harvested < count && !error) {
    DWORD w = WaitForMultipleObjects(r->depth, r->events, FALSE, WAIT_MS);
    if (w == WAIT_TIMEOUT) {
      error = -ETIMEDOUT;
      goto failed;
    }
    if (w == WAIT_FAILED || w >= WAIT_OBJECT_0 + r->depth) {
      error = -(int)GetLastError();
      goto failed;
    }
    unsigned slot = (unsigned)(w - WAIT_OBJECT_0);
    struct win_req* req = &r->reqs[slot];
    if (!req->outstanding) continue;   /* event fired without a live request */
    DWORD got = 0;
    HANDLE handle = req->exact ? req->b->buffered : req->b->direct;
    if (!GetOverlappedResult(handle, &req->ov, &got, FALSE)) {
      r->fault_slot = slot;
      r->fault_ordinal = req->ordinal;
      r->fault_exact = req->exact;
      error = -(int)GetLastError();
      goto failed;
    }
    if (got < req->need) {
      r->fault_slot = slot;
      r->fault_ordinal = req->ordinal;
      r->fault_exact = req->exact;
      error = got ? -EIO : -EREMOTEIO;
      goto failed;
    }
    if (req->exact) {
      if (got != row_bytes) {
        error = -EIO;
        goto failed;
      }
    } else {
      unsigned delta = (unsigned)(offsets[req->ordinal] & (PAGE - 1));
      memcpy(req->dst, req->slot + delta, row_bytes);
    }
    req->outstanding = 0;
    ++harvested;
    if (submitted < count) {
      if (!start_req(r, slot, fds[submitted], offsets[submitted], row_bytes, out,
                     submitted)) {
        error = -errno;
        goto failed;
      }
      ++submitted;
    }
  }
  return error;
failed:
  kill_reader(r);
  return error;
}
