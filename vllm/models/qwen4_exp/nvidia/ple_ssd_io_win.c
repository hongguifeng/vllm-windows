/* SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 * Windows asynchronous reads for exact BF16 rows in a safetensors checkpoint.
 *
 * Same row-at-a-time page reads as ple_ssd_io.c: every row is served by one
 * page-aligned read of round_up(row_in_page + row_bytes, 4096), so unaligned rows
 * never need unaligned I/O, and the native call runs without the Python GIL.
 *
 * Requests are submitted to one I/O completion port. Unlike
 * WaitForMultipleObjects, an IOCP can reap more than MAXIMUM_WAIT_OBJECTS
 * completions, so the requested queue depth is preserved up to the reader's
 * allocation limit. rows_depth reports the actual depth.
 *
 * Buffered reads are the default. Define PLE_WINDOWS_DIRECT only for A/B tests.
 * Rows whose page crosses EOF are read exactly, buffered, into the output.
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
  HANDLE iocp;
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
  if (r->iocp) CloseHandle(r->iocp);
  if (r->buf) VirtualFree(r->buf, 0, MEM_RELEASE);
  free(r->reqs);
  free(r);
}

__declspec(dllexport) struct reader* rows_open(unsigned depth) {
  if (!depth || depth > 4096) {
    errno = EINVAL;
    return NULL;
  }
  struct reader* r = calloc(1, sizeof(*r));
  if (!r) {
    errno = ENOMEM;
    return NULL;
  }
  r->depth = depth;
  r->reqs = calloc(depth, sizeof(*r->reqs));
  if (!r->reqs) {
    rows_close(r);
    errno = ENOMEM;
    return NULL;
  }
  r->iocp = CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, 0, 0);
  if (!r->iocp) {
    rows_close(r);
    errno = (int)GetLastError();
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
#ifdef PLE_WINDOWS_DIRECT
  b->direct = CreateFileW(path, GENERIC_READ, share, NULL, OPEN_EXISTING,
                          FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS |
                              FILE_FLAG_OVERLAPPED,
                          NULL);
  if (b->direct == INVALID_HANDLE_VALUE) {
    int err = (int)GetLastError();
    free(b);
    errno = err;
    return -1000000 - err;
  }
#else
  b->direct = NULL;
#endif
  b->buffered = CreateFileW(path, GENERIC_READ, share, NULL, OPEN_EXISTING,
                            FILE_FLAG_RANDOM_ACCESS | FILE_FLAG_OVERLAPPED,
                            NULL);
  if (b->buffered == INVALID_HANDLE_VALUE) {
    int err = (int)GetLastError();
    if (b->direct) CloseHandle(b->direct);
    free(b);
    errno = err;
    return -2000000 - err;
  }
  LARGE_INTEGER size;
  HANDLE size_handle = b->direct ? b->direct : b->buffered;
  if (!GetFileSizeEx(size_handle, &size)) {
    int err = (int)GetLastError();
    if (b->direct) CloseHandle(b->direct);
    CloseHandle(b->buffered);
    free(b);
    errno = err;
    return -3000000 - err;
  }
  b->size = size.QuadPart;
  if (b->direct && !CreateIoCompletionPort(b->direct, r->iocp,
                                             (ULONG_PTR)b, 0)) {
    int err = (int)GetLastError();
    CloseHandle(b->direct);
    CloseHandle(b->buffered);
    free(b);
    errno = err;
    return -4000000 - err;
  }
  if (!CreateIoCompletionPort(b->buffered, r->iocp, (ULONG_PTR)b, 0)) {
    int err = (int)GetLastError();
    if (b->direct) CloseHandle(b->direct);
    CloseHandle(b->buffered);
    free(b);
    errno = err;
    return -5000000 - err;
  }
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
    if (b->direct) CancelIoEx(b->direct, NULL);
    CancelIoEx(b->buffered, NULL);
  }
  /* IOCP status is published when the packet is dequeued. Reap cancellation
   * packets before freeing any request buffers or OVERLAPPED structures. */
  unsigned pending = 0;
  for (unsigned slot = 0; slot < r->depth; ++slot) {
    pending += r->reqs[slot].outstanding != 0;
  }
  while (pending) {
    DWORD got = 0;
    ULONG_PTR key = 0;
    OVERLAPPED* ov = NULL;
    GetQueuedCompletionStatus(r->iocp, &got, &key, &ov, INFINITE);
    if (!ov) continue;
    struct win_req* req = (struct win_req*)ov;
    unsigned slot = (unsigned)(req - r->reqs);
    if (slot >= r->depth || !req->outstanding) continue;
    req->outstanding = 0;
    --pending;
  }
  r->dead = 1;   /* the helper is unusable afterwards */
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
  req->ov.hEvent = NULL;
  unsigned delta = (unsigned)(offset & (PAGE - 1));
  long long page = (long long)(offset & ~(uint64_t)(PAGE - 1));
  /* Keep the same page-shaped requests for buffered/direct comparisons. */
  unsigned rounded = (delta + row_bytes + PAGE - 1) & ~(PAGE - 1);
  if (page + (long long)rounded > b->size) {
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
#ifdef PLE_WINDOWS_DIRECT
  HANDLE read_handle = b->direct;
#else
  HANDLE read_handle = b->buffered;
#endif
  if (!ReadFile(read_handle, req->slot, rounded, NULL, &req->ov)) {
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
    DWORD got = 0;
    ULONG_PTR key = 0;
    OVERLAPPED* ov = NULL;
    BOOL ok = GetQueuedCompletionStatus(r->iocp, &got, &key, &ov, WAIT_MS);
    if (!ok && !ov) {
      int err = (int)GetLastError();
      error = err == WAIT_TIMEOUT ? -ETIMEDOUT : -err;
      goto failed;
    }
    if (!ov) {
      error = -EIO;
      goto failed;
    }
    struct win_req* req = (struct win_req*)ov;
    unsigned slot = (unsigned)(req - r->reqs);
    if (slot >= r->depth || !req->outstanding) continue;
    req->outstanding = 0;
    if (!ok) {
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
