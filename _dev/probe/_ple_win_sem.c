/* What does the deployed PLE/SSD Windows reader actually rely on?
 *
 * Three questions, all CPU-only:
 *   1. With the flags in rows_bind (NO_BUFFERING|RANDOM_ACCESS, no
 *      FILE_FLAG_OVERLAPPED), does ReadFile(..., &ov) complete synchronously,
 *      return ERROR_IO_PENDING, or signal ov.hEvent? The current harvest loop
 *      blocks on that event, so the answer decides whether the design is sound
 *      or is leaning on environment-specific behaviour.
 *   2. Does the exact-read branch trigger where it should? The branch test uses
 *      the logical row end, while the unbuffered request is rounded up to a page,
 *      so a row ending exactly at EOF can be routed to an unbuffered read that
 *      runs past EOF.
 *   3. Depth clamp and teardown: does a forced I/O failure followed immediately
 *      by rows_close ever fault?
 *
 * Usage: ple_win_sem.exe <path-to-ple_ssd_io_win.dll>
 */
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <io.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PAGE 4096u

struct reader;
typedef struct reader* (*fn_open)(unsigned);
typedef int (*fn_bind)(struct reader*, int, const wchar_t*);
typedef int (*fn_read)(struct reader*, const int*, const uint64_t*, unsigned,
                       unsigned, unsigned char*);
typedef unsigned (*fn_depth)(struct reader*);
typedef void (*fn_close)(struct reader*);

static fn_open p_open;
static fn_bind p_bind;
static fn_read p_read;
static fn_depth p_depth;
static fn_close p_close;

static unsigned failures;

static void fill_pattern(unsigned char* buf, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        buf[i] = (unsigned char)(i * 7 + 13);
}

/* A CRT fd is only a lookup key for the reader, which opens its own handles. */
static int crt_fd(const wchar_t* wpath)
{
    HANDLE h = CreateFileW(wpath, GENERIC_READ,
                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) {
        printf("  CreateFileW for fd failed %lu\n", GetLastError());
        return -1;
    }
    return _open_osfhandle((intptr_t)h, 0);
}

static void make_file(const wchar_t* wpath, size_t size)
{
    unsigned char* buf = malloc(size ? size : 1);
    fill_pattern(buf, size);
    FILE* f = _wfopen(wpath, L"wb");
    fwrite(buf, 1, size, f);
    fclose(f);
    free(buf);
}

/* ---- 1: what does ReadFile do with these flags? ---- */
static void probe_flags(const wchar_t* wpath, DWORD extra)
{
    DWORD flags = FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS | extra;
    HANDLE h = CreateFileW(wpath, GENERIC_READ,
                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           NULL, OPEN_EXISTING, flags, NULL);
    printf("  flags %sNO_BUFFERING|RANDOM_ACCESS%s -> ",
           extra == FILE_FLAG_OVERLAPPED ? "OVERLAPPED|" : "",
           extra == FILE_FLAG_OVERLAPPED ? "" : " (as shipped)");
    if (h == INVALID_HANDLE_VALUE) {
        printf("open failed %lu\n", GetLastError());
        failures++;
        return;
    }
    unsigned char* buf = VirtualAlloc(NULL, 2 * PAGE, MEM_COMMIT | MEM_RESERVE,
                                      PAGE_READWRITE);
    if (buf == NULL) {
        printf("alloc failed %lu\n", GetLastError());
        CloseHandle(h);
        return;
    }
    memset(buf, 0, 2 * PAGE);

    HANDLE ev = CreateEventW(NULL, FALSE, FALSE, NULL);
    OVERLAPPED ov;
    memset(&ov, 0, sizeof(ov));
    ov.hEvent = ev;
    ov.Offset = 0;
    ov.OffsetHigh = 0;

    BOOL ok = ReadFile(h, buf, 2 * PAGE, NULL, &ov);
    DWORD err = GetLastError();
    DWORD now = WaitForSingleObject(ev, 0);
    DWORD later = now == WAIT_OBJECT_0 ? WAIT_OBJECT_0 : WaitForSingleObject(ev, 200);
    DWORD got = 0;
    BOOL ovres = GetOverlappedResult(h, &ov, &got, FALSE);
    DWORD reserr = GetLastError();

    printf("ReadFile=%d err=%lu event_now=%s event_200ms=%s GetOverlappedResult=%d "
           "bytes=%lu err=%lu\n",
           ok, err,
           now == WAIT_OBJECT_0 ? "signaled" : now == WAIT_TIMEOUT ? "unsig" : "failed",
           later == WAIT_OBJECT_0 ? "signaled" : "unsig",
           ovres, got, reserr);

    /* The production harvest loop waits on this event, so an unsignalled event
     * after a "successful" ReadFile means every row would time out. */
    if (ok && err == 0 && now != WAIT_OBJECT_0 && later != WAIT_OBJECT_0) {
        printf("    -> completed synchronously and the event is NEVER signalled: "
               "the harvest loop cannot work with these flags\n");
        failures++;
    }
    CloseHandle(ev);
    VirtualFree(buf, 0, MEM_RELEASE);
    CloseHandle(h);
}

/* Each check gets its own reader: one failure marks the reader dead, so sharing
 * one would turn every later case into a spurious EBADF cascade. */
static int check_row(const wchar_t* wpath, uint64_t offset, unsigned row_bytes,
                     size_t file_size, const char* label)
{
    struct reader* r = p_open(64);
    int fd = crt_fd(wpath);
    if (p_bind(r, fd, wpath) != 0) {
        printf("    FAIL %s bind\n", label);
        failures++;
        p_close(r);
        return -1;
    }
    unsigned char out[PAGE];
    memset(out, 0xFF, sizeof(out));
    const int fds[1] = { fd };
    const uint64_t offs[1] = { offset };
    int rc = p_read(r, fds, offs, 1, row_bytes, out);
    int reported = rc;
    if (rc != 0) {
        printf("    FAIL %s off=%llu row=%u rc=%d\n", label,
               (unsigned long long)offset, row_bytes, rc);
        failures++;
    } else {
        for (unsigned i = 0; i < row_bytes; ++i) {
            unsigned char want = (unsigned char)((offset + i) * 7 + 13);
            if (out[i] != want) {
                printf("    FAIL %s off=%llu row=%u byte %u got %u want %u\n",
                       label, (unsigned long long)offset, row_bytes, i, out[i], want);
                failures++;
                reported = -1;
                break;
            }
        }
    }
    p_close(r);
    return reported;
}

static void matrix(const wchar_t* wpath, size_t file_size)
{
    const unsigned rows[] = { 320, 512, 1024, 4096 };
    for (unsigned ri = 0; ri < 4; ++ri) {
        unsigned rb = rows[ri];
        if (rb >= file_size) continue;
        /* Only offsets whose row actually fits. The interesting ones are a row
         * ending exactly at EOF, and a row ending at EOF whose page-rounded
         * request runs past EOF -- that needs a file size that is not a whole
         * number of pages, which is what safetensors tensor ends look like. */
        uint64_t ends_at_eof[] = { file_size - rb, file_size - rb - 1,
                                   file_size - rb - 2 };
        for (unsigned oi = 0; oi < 3; ++oi)
            check_row(wpath, ends_at_eof[oi], rb, file_size, "row-end-at-EOF");
        if (file_size > 4096 + rb)
            check_row(wpath, 4095, rb, file_size, "unaligned-delta");
        if (file_size > 4096 + rb)
            check_row(wpath, 4096, rb, file_size, "page-aligned");
        /* A row whose page extends past EOF while the row itself does not. */
        uint64_t o = (uint64_t)(file_size - PAGE / 2 - 1);
        if (o + rb <= file_size)
            check_row(wpath, o, rb, file_size, "page-extends-past-EOF");
        /* A row that sits in the last partial page, unaligned inside it. */
        size_t tail_bytes = file_size % PAGE;
        if (tail_bytes != 0 && file_size > rb + tail_bytes + 1) {
            uint64_t tail = (uint64_t)(file_size - rb - tail_bytes - 1);
            if (tail + rb <= file_size)
                check_row(wpath, tail, rb, file_size, "row-in-last-partial-page");
        }
    }
}

/* ---- 3: depth clamp, and teardown after a forced failure ---- */
static void depth_and_teardown(const wchar_t* wpath, size_t file_size)
{
    const unsigned want[] = { 256, 64, 1 };
    for (unsigned i = 0; i < 3; ++i) {
        struct reader* r = p_open(want[i]);
        unsigned got = p_depth(r);
        unsigned expect = want[i] > 64 ? 64 : want[i];
        printf("  depth request %u -> reported %u  %s\n", want[i], got,
               got == expect ? "OK" : "WRONG");
        if (got != expect) failures++;
        p_close(r);
    }

    struct reader* r = p_open(64);
    int fd = crt_fd(wpath);
    if (p_bind(r, fd, wpath) != 0) { printf("    bind failed\n"); failures++; }
    unsigned char out[PAGE];
    const int fds[1] = { fd };
    uint64_t bad = file_size + PAGE;   /* one page past the end: must fail */
    const uint64_t offs[1] = { bad };
    int rc = p_read(r, fds, offs, 1, 320, out);
    printf("  forced failure rc=%d (expected non-zero), then close immediately\n", rc);
    p_close(r);
    if (rc == 0) { printf("    forced failure did not fail\n"); failures++; }

    for (int i = 0; i < 500; ++i) {
        struct reader* rr = p_open(64);
        int f2 = crt_fd(wpath);
        p_bind(rr, f2, wpath);
        const int fds2[1] = { f2 };
        uint64_t o = (uint64_t)(file_size - 320);
        const uint64_t off2[1] = { o };
        int rc2 = p_read(rr, fds2, off2, 1, 320, out);
        if (rc2 != 0) { printf("    cycle %d rc=%d\n", i, rc2); failures++; }
        p_close(rr);
    }
    printf("  500 open/read/close cycles survived\n");
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("usage: ple_win_sem.exe <ple_ssd_io_win.dll>\n");
        return 1;
    }
    HMODULE dll = LoadLibraryA(argv[1]);
    if (!dll) {
        printf("LoadLibrary failed %lu\n", GetLastError());
        return 1;
    }
    p_open = (fn_open)GetProcAddress(dll, "rows_open");
    p_bind = (fn_bind)GetProcAddress(dll, "rows_bind");
    p_read = (fn_read)GetProcAddress(dll, "rows_read");
    p_depth = (fn_depth)GetProcAddress(dll, "rows_depth");
    p_close = (fn_close)GetProcAddress(dll, "rows_close");
    if (!p_open || !p_bind || !p_read || !p_depth || !p_close) {
        printf("missing exports: open=%p bind=%p read=%p depth=%p close=%p\n",
               (void*)p_open, (void*)p_bind, (void*)p_read, (void*)p_depth,
               (void*)p_close);
        return 1;
    }

    wchar_t wpath[260];
    mbstowcs(wpath, "ple_sem_probe.bin", 260);
    const size_t sizes[] = { PAGE, 2 * PAGE, 3 * PAGE, 5 * PAGE,
                             PAGE + 37, 3 * PAGE + 19, 5 * PAGE + 1,
                             7 * PAGE + 320 };
    for (unsigned s = 0; s < 8; ++s) make_file(wpath, sizes[s]);

    printf("== 1: ReadFile semantics with the shipped flags\n");
    make_file(wpath, 4 * PAGE);
    probe_flags(wpath, 0);
    probe_flags(wpath, FILE_FLAG_OVERLAPPED);

    printf("\n== 2: row correctness against a buffered expectation\n");
    for (unsigned s = 0; s < 8; ++s) {
        make_file(wpath, sizes[s]);
        printf("  file %zu bytes (last page %zu bytes)\n", sizes[s],
               sizes[s] % PAGE);
        matrix(wpath, sizes[s]);
    }

    printf("\n== 3: depth clamp and teardown\n");
    make_file(wpath, 8 * PAGE);
    depth_and_teardown(wpath, 8 * PAGE);

    printf("\nfailures: %u\n", failures);
    return failures ? 1 : 0;
}
