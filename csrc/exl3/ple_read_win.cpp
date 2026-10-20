// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Windows host reader for selected rows of the compressed PLE table.
//
// Same interface and failure semantics as the POSIX ple_read.cpp
// (pread/preadv + a thread pool). Rows are read one at a time with
// _lseeki64/_read on the calling thread: functional first, and the IOCP
// reader in vllm/models/qwen4_exp/nvidia/ple_ssd_io_win.c is the fast path to
// build on for the SSD placement.

#include "ple.h"

#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <utility>
#include <vector>

#include <torch/custom_class.h>
#include <torch/library.h>

namespace {

struct Failure {
  int64_t index;
  int64_t error;  // errno, or 0 for end of file
  int64_t offset;
};

struct Table {
  int fd;
  int64_t base;
  int64_t row_bytes;
  int64_t owner_lo;
  int64_t owner_hi;
  const int64_t* head_offsets;
  const int64_t* head_ends;  // offsets[h] + vocab_sizes[h]
  int64_t heads;
  int64_t retries;
};

// The arguments both entry points share; returns the heads' ends.
std::vector<int64_t> check_table(int64_t base, int64_t row_bytes,
                                 int64_t owner_lo, int64_t owner_hi,
                                 const torch::Tensor& head_offsets,
                                 const torch::Tensor& head_vocab_sizes,
                                 int64_t retries) {
  TORCH_CHECK(
      head_offsets.device().is_cpu() && head_vocab_sizes.device().is_cpu() &&
          head_offsets.scalar_type() == torch::kInt64 &&
          head_vocab_sizes.scalar_type() == torch::kInt64 &&
          head_offsets.dim() == 1 && head_offsets.sizes() == head_vocab_sizes.sizes() &&
          head_offsets.is_contiguous() && head_vocab_sizes.is_contiguous(),
      "PLE heads must be matching 1-D CPU int64 tensors");
  TORCH_CHECK(row_bytes > 0 && base >= 0 && 0 <= owner_lo &&
                  owner_lo <= owner_hi && retries >= 0,
              "Invalid PLE read arguments");
  TORCH_CHECK(owner_hi <= (INT64_MAX - base) / row_bytes,
              "PLE file offset exceeds signed 64-bit range");
  const int64_t heads = head_offsets.numel();
  const int64_t* offsets = head_offsets.data_ptr<int64_t>();
  const int64_t* sizes = head_vocab_sizes.data_ptr<int64_t>();
  std::vector<int64_t> ends(heads);
  for (int64_t h = 0; h < heads; ++h) ends[h] = offsets[h] + sizes[h];
  return ends;
}

// Inside the owner range and inside its head's vocabulary (padding rows after
// a head belong to nobody).
bool owns(const Table& t, int64_t row) {
  if (row < t.owner_lo || row >= t.owner_hi) return false;
  const int64_t* end = t.head_offsets + t.heads;
  const int64_t* after = std::upper_bound(t.head_offsets, end, row);
  if (after == t.head_offsets) return false;
  return row < t.head_ends[after - t.head_offsets - 1];
}

// Partial reads continue; any other error or end of file is returned with the
// offset reached. Returns -1 on success.
int64_t read_exact(int fd, uint8_t* dest, int64_t length, int64_t offset,
                   int64_t* reached) {
  int64_t done = 0;
  while (done < length) {
    if (_lseeki64(fd, offset + done, SEEK_SET) < 0) {
      *reached = offset + done;
      return errno;
    }
    const unsigned chunk =
        static_cast<unsigned>(std::min<int64_t>(length - done, 1u << 30));
    const int got = _read(fd, dest + done, chunk);
    if (got < 0) {
      if (errno == EINTR) continue;
      *reached = offset + done;
      return errno;
    }
    if (got == 0) {
      *reached = offset + done;
      return 0;
    }
    done += got;
  }
  return -1;
}

// The same bounded retries and backoff (1 ms, doubling) as the POSIX reader.
bool read_row(const Table& t, uint8_t* dest, int64_t row, Failure* failure) {
  const int64_t offset = t.base + row * t.row_bytes;
  int64_t reached = offset;
  int64_t error = -1;
  for (int64_t attempt = 0; attempt <= t.retries; ++attempt) {
    error = read_exact(t.fd, dest, t.row_bytes, offset, &reached);
    if (error < 0) return true;
    if (attempt == t.retries) break;
    const int64_t ms = int64_t{1} << std::min<int64_t>(attempt, 10);
    ::Sleep(static_cast<DWORD>(ms));
  }
  failure->error = error;
  failure->offset = reached;
  return false;
}

void poison(uint8_t* dest, int64_t row_bytes) { std::memset(dest, 0, row_bytes); }

// Reads ids[0:count) of the table into dest, row i at byte i * row_bytes.
std::vector<Failure> read_rows(const Table& t, const int64_t* ids,
                               int64_t count, uint8_t* dest) {
  std::vector<Failure> failures;
  for (int64_t i = 0; i < count; ++i) {
    const int64_t id = ids[i];
    uint8_t* row_dest = dest + i * t.row_bytes;
    if (!owns(t, id)) {
      poison(row_dest, t.row_bytes);
      continue;
    }
    Failure failure{i, 0, 0};
    if (!read_row(t, row_dest, id, &failure)) {
      poison(row_dest, t.row_bytes);
      failures.push_back(failure);
    }
  }
  return failures;
}

torch::Tensor failures_tensor(const std::vector<Failure>& failures,
                              const torch::TensorOptions& options) {
  auto result = torch::empty(
      {static_cast<int64_t>(failures.size()), 3}, options.dtype(torch::kInt64));
  auto* out = result.data_ptr<int64_t>();
  for (size_t i = 0; i < failures.size(); ++i) {
    out[i * 3 + 0] = failures[i].index;
    out[i * 3 + 1] = failures[i].error;
    out[i * 3 + 2] = failures[i].offset;
  }
  return result;
}

// A PleHostReader reads ids[0:count) of rows at the table into rows at offset
// zero, as the staging buffers of one PLE step. Reads run on the calling
// thread; take() reports each batch once, in queue order:
// [sequence, count, status, failed rows, first failed index, its row id,
//  its errno, byte offset reached, read ns].
class PleHostReader : public torch::CustomClassHolder {
 public:
  PleHostReader(int64_t fd, torch::Tensor ids, torch::Tensor rows, int64_t base,
                int64_t row_bytes, int64_t owner_lo, int64_t owner_hi,
                torch::Tensor head_offsets, torch::Tensor head_vocab_sizes,
                int64_t retries, int64_t max_threads, int64_t slots)
      : ids_(std::move(ids)),
        rows_(std::move(rows)),
        base_(base),
        row_bytes_(row_bytes),
        owner_lo_(owner_lo),
        owner_hi_(owner_hi),
        head_offsets_(std::move(head_offsets)),
        head_vocab_sizes_(std::move(head_vocab_sizes)),
        retries_(retries),
        slots_(slots) {
    TORCH_CHECK(fd >= 0, "PleHostReader: fd must be a valid descriptor");
    TORCH_CHECK(max_threads > 0 && slots > 0,
                "PleHostReader: threads and slots must be positive");
    TORCH_CHECK(ids_.device().is_cpu() && rows_.device().is_cpu(),
                "PleHostReader: staging buffers must be CPU tensors");
    TORCH_CHECK(ids_.scalar_type() == torch::kInt64 && ids_.dim() == 1 &&
                    ids_.is_contiguous(),
                "PleHostReader: ids must be a contiguous 1-D int64 CPU tensor");
    TORCH_CHECK(rows_.scalar_type() == torch::kInt16 && rows_.dim() == 2 &&
                    rows_.is_contiguous(),
                "PleHostReader: rows must be a contiguous 2-D int16 CPU tensor");
    ends_ = check_table(base_, row_bytes_, owner_lo_, owner_hi_, head_offsets_,
                        head_vocab_sizes_, retries_);
    fd_ = _dup(static_cast<int>(fd));
    TORCH_CHECK(fd_ >= 0, "PleHostReader: _dup failed: ", errno);
    (void)_setmode(fd_, _O_BINARY);
  }

  ~PleHostReader() override {
    if (fd_ >= 0) _close(fd_);
  }

  void enqueue(int64_t count) { read_batch(count); }

  void run(int64_t count) { read_batch(count); }

  torch::Tensor take() {
    const int64_t n = static_cast<int64_t>(done_.size());
    auto result = torch::empty({n, 9}, ids_.options().dtype(torch::kInt64));
    auto* out = result.data_ptr<int64_t>();
    int64_t i = 0;
    while (!done_.empty()) {
      const auto& entry = done_.front();
      std::copy(entry.begin(), entry.end(), out + i * 9);
      done_.pop_front();
      ++i;
    }
    return result;
  }

 private:
  void read_batch(int64_t count) {
    TORCH_CHECK(count >= 0 && count <= ids_.numel() && count <= rows_.size(0),
                "PleHostReader: count ", count, " exceeds the staging buffers");
    TORCH_CHECK(static_cast<int64_t>(done_.size()) < slots_,
                "PleHostReader: every slot is taken; take() first");
    const auto started = std::chrono::steady_clock::now();
    const Table table{fd_,
                      base_,
                      row_bytes_,
                      owner_lo_,
                      owner_hi_,
                      head_offsets_.data_ptr<int64_t>(),
                      ends_.data(),
                      heads(),
                      retries_};
    auto failures =
        read_rows(table, ids_.data_ptr<int64_t>(), count,
                  reinterpret_cast<uint8_t*>(rows_.data_ptr<int16_t>()));
    const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now() - started)
                           .count();
    std::array<int64_t, 9> entry{
        sequence_++,
        count,
        0,
        static_cast<int64_t>(failures.size()),
        failures.empty() ? -1 : failures.front().index,
        failures.empty() ? -1
                         : ids_.data_ptr<int64_t>()[failures.front().index],
        failures.empty() ? 0 : failures.front().error,
        failures.empty() ? 0 : failures.front().offset,
        ns};
    done_.push_back(entry);
  }

  int64_t heads() const { return head_offsets_.numel(); }

  torch::Tensor ids_;
  torch::Tensor rows_;
  torch::Tensor head_offsets_;
  torch::Tensor head_vocab_sizes_;
  std::vector<int64_t> ends_;
  int fd_ = -1;
  int64_t base_;
  int64_t row_bytes_;
  int64_t owner_lo_;
  int64_t owner_hi_;
  int64_t retries_;
  int64_t slots_;
  int64_t sequence_ = 0;
  std::deque<std::array<int64_t, 9>> done_;
};

}  // namespace

torch::Tensor ple_read_rows(int64_t fd, const torch::Tensor& ids,
                            torch::Tensor& dest, int64_t count, int64_t base,
                            int64_t row_bytes, int64_t owner_lo,
                            int64_t owner_hi, const torch::Tensor& head_offsets,
                            const torch::Tensor& head_vocab_sizes,
                            int64_t retries, int64_t max_threads) {
  TORCH_CHECK(fd >= 0, "ple_read_rows: fd must be a valid descriptor");
  TORCH_CHECK(ids.device().is_cpu() && ids.scalar_type() == torch::kInt64 &&
                  ids.dim() == 1 && ids.is_contiguous(),
              "ple_read_rows: ids must be a contiguous 1-D int64 CPU tensor");
  TORCH_CHECK(dest.device().is_cpu() && dest.scalar_type() == torch::kInt16 &&
                  dest.dim() == 2 && dest.is_contiguous(),
              "ple_read_rows: dest must be a contiguous 2-D int16 CPU tensor");
  TORCH_CHECK(count >= 0 && count <= ids.numel() && count <= dest.size(0),
              "ple_read_rows: count exceeds the buffers");
  TORCH_CHECK(dest.size(1) * dest.element_size() >= row_bytes,
              "ple_read_rows: dest rows are shorter than row_bytes");
  TORCH_CHECK(max_threads > 0, "ple_read_rows: max_threads must be positive");

  auto ends = check_table(base, row_bytes, owner_lo, owner_hi, head_offsets,
                          head_vocab_sizes, retries);
  const Table table{static_cast<int>(fd),
                    base,
                    row_bytes,
                    owner_lo,
                    owner_hi,
                    head_offsets.data_ptr<int64_t>(),
                    ends.data(),
                    head_offsets.numel(),
                    retries};
  auto failures =
      read_rows(table, ids.data_ptr<int64_t>(), count,
                reinterpret_cast<uint8_t*>(dest.data_ptr<int16_t>()));
  return failures_tensor(failures, ids.options());
}

void register_ple_host_reader(torch::Library& m) {
  m.class_<PleHostReader>("PleHostReader")
      .def(torch::init<int64_t, torch::Tensor, torch::Tensor, int64_t, int64_t,
                       int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t,
                       int64_t, int64_t>())
      .def("enqueue", &PleHostReader::enqueue)
      .def("run", &PleHostReader::run)
      .def("take", &PleHostReader::take);
}
