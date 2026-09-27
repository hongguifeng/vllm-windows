// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
/* Read-only Linux AIO for exact BF16 rows in a safetensors checkpoint. */
#define _GNU_SOURCE
#include <errno.h>
#include <linux/aio_abi.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

struct reader {
  aio_context_t ctx;
  unsigned depth;
  unsigned char* buf;
  struct iocb *cbs, **ptrs;
  struct io_event* events;
};
void rows_close(struct reader* r) {
  if (!r) return;
  if (r->ctx) syscall(__NR_io_destroy, r->ctx);
  free(r->buf);
  free(r->cbs);
  free(r->ptrs);
  free(r->events);
  free(r);
}
struct reader* rows_open(unsigned depth) {
  if (!depth || depth > 4096) {
    errno = EINVAL;
    return NULL;
  }
  struct reader* r = calloc(1, sizeof(*r));
  if (!r) return NULL;
  r->depth = depth;
  r->cbs = calloc(depth, sizeof(*r->cbs));
  r->ptrs = calloc(depth, sizeof(*r->ptrs));
  r->events = calloc(depth, sizeof(*r->events));
  int err = posix_memalign((void**)&r->buf, 4096, (size_t)depth * 8192);
  if (err || !r->cbs || !r->ptrs || !r->events) {
    rows_close(r);
    errno = err ? err : ENOMEM;
    return NULL;
  }
  if (syscall(__NR_io_setup, depth, &r->ctx) < 0) {
    err = errno;
    rows_close(r);
    errno = err;
    return NULL;
  }
  return r;
}
int rows_read(struct reader* r, const int* fds, const uint64_t* offsets,
              unsigned count, unsigned row_bytes, unsigned char* out) {
  if (!r->ctx) return -EBADF;
  if (!row_bytes || row_bytes > 4096) return -EINVAL;
  int error = 0;
  for (unsigned base = 0; base < count && !error; base += r->depth) {
    unsigned batch = count - base < r->depth ? count - base : r->depth;
    memset(r->cbs, 0, batch * sizeof(*r->cbs));
    for (unsigned j = 0; j < batch; ++j) {
      struct iocb* cb = &r->cbs[j];
      cb->aio_data = j;
      cb->aio_lio_opcode = IOCB_CMD_PREAD;
      cb->aio_fildes = fds[base + j];
      cb->aio_buf = (uint64_t)(uintptr_t)(r->buf + (size_t)j * 8192);
      cb->aio_offset = offsets[base + j] & ~(uint64_t)4095;
      cb->aio_nbytes = ((offsets[base + j] & 4095) + row_bytes + 4095) & ~4095;
      r->ptrs[j] = cb;
    }
    unsigned submitted = 0, complete = 0;
    while (complete < batch) {
      if (submitted < batch) {
        long rc = syscall(__NR_io_submit, r->ctx, batch - submitted,
                          r->ptrs + submitted);
        if (rc > 0)
          submitted += rc;
        else if (rc < 0 && errno == EINTR)
          continue;
        else if (rc <= 0 &&
                 (rc == 0 || errno != EAGAIN || submitted == complete)) {
          error = rc < 0 ? -errno : -EIO;
          goto failed;
        }
      }
      long n = syscall(__NR_io_getevents, r->ctx, 1, submitted - complete,
                       r->events, NULL);
      if (n < 0) {
        if (errno == EINTR) continue;
        error = -errno;
        goto failed;
      }
      for (long i = 0; i < n; ++i) {
        struct io_event* e = &r->events[i];
        unsigned j = e->data;
        size_t delta = offsets[base + j] & 4095;
        if (e->res < (int64_t)(delta + row_bytes))
          error = e->res < 0 ? e->res : -EIO;
        else
          memcpy(out + (size_t)(base + j) * row_bytes,
                 r->buf + (size_t)j * 8192 + delta, row_bytes);
      }
      complete += n;
    }
  }
  return error;
failed:
  /* io_destroy waits for outstanding requests before buffers can be reused. */
  syscall(__NR_io_destroy, r->ctx);
  r->ctx = 0;
  return error;
}
