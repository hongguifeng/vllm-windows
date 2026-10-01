/* Windows unbuffered-IO shim, callable from Python.
 *
 * Purpose 1: settle whether the 87s are a ctypes marshalling artifact -- the
 * ReadFile here is issued by compiled C inside the same python.exe process.
 * Purpose 2: become the actual PLE/SSD read path on Windows, mirroring the
 * container's ple_ssd_io.so loaded through ctypes.
 *
 * Build (from _dev/bin/_nbshim_build.ps1):
 *   cl /nologo /O2 /LD /Fe:nbshim.dll _nbshim.c
 */
#include <windows.h>
#include <stdio.h>

__declspec(dllexport) void *nb_open(const wchar_t *path, DWORD access,
                                    DWORD share, DWORD flags)
{
    HANDLE h = CreateFileW(path, access, share, NULL, OPEN_EXISTING, flags, NULL);
    if (h == INVALID_HANDLE_VALUE) {
        SetLastError(GetLastError());
        return NULL;
    }
    return h;
}

__declspec(dllexport) int nb_close(void *h)
{
    return CloseHandle((HANDLE)h) ? 1 : 0;
}

/* Positional read on a 512-byte aligned offset into a 512-byte aligned buffer.
 * Returns bytes transferred, or -1 with the Win32 error left in place. */
__declspec(dllexport) long long nb_read(void *h, long long offset,
                                        DWORD count, void *buf)
{
    LARGE_INTEGER li;
    li.QuadPart = offset;
    if (!SetFilePointerEx((HANDLE)h, li, NULL, FILE_BEGIN))
        return -1;
    DWORD got = 0;
    if (!ReadFile((HANDLE)h, buf, count, &got, NULL))
        return -1;
    return (long long)got;
}

/* Same, but through an OVERLAPPED so many can be outstanding at once. */
__declspec(dllexport) long long nb_read_ov(void *h, long long offset,
                                           DWORD count, void *buf,
                                           LPOVERLAPPED ov)
{
    ov->Offset = (DWORD)(offset & 0xFFFFFFFF);
    ov->OffsetHigh = (DWORD)((offset >> 32) & 0xFFFFFFFF);
    ov->Internal = 0;
    ov->InternalHigh = 0;
    ov->hEvent = NULL;
    if (!ReadFile((HANDLE)h, buf, count, NULL, ov))
        return -1;
    return 0;
}

__declspec(dllexport) long long nb_size(void *h)
{
    LARGE_INTEGER sz;
    if (!GetFileSizeEx((HANDLE)h, &sz))
        return -1;
    return sz.QuadPart;
}

/* Positional write, buffered. Returns bytes written or -1. */
__declspec(dllexport) long long nb_write(void *h, long long offset,
                                         DWORD count, const void *buf)
{
    LARGE_INTEGER li;
    li.QuadPart = offset;
    if (!SetFilePointerEx((HANDLE)h, li, NULL, FILE_BEGIN))
        return -1;
    DWORD put = 0;
    if (!WriteFile((HANDLE)h, buf, count, &put, NULL))
        return -1;
    return (long long)put;
}

__declspec(dllexport) int nb_flush(void *h)
{
    return FlushFileBuffers((HANDLE)h) ? 1 : 0;
}

__declspec(dllexport) DWORD nb_last_error(void)
{
    return GetLastError();
}

/* ---- completion-port batch path ----
 * The container's PLE reader is fully asynchronous (io_submit / io_getevents),
 * so the Windows equivalent is IOCP. Each request is a struct whose first
 * member is the OVERLAPPED, so a completion's OVERLAPPED pointer maps straight
 * back to the request and its key. The system posts the completion packet; we
 * never post it ourselves.
 */
typedef struct {
    OVERLAPPED ov;      /* must stay first: completions cast back to this */
    void      *buf;
    ULONG      key;
    DWORD      count;
    long long  offset;
} NB_REQ;

__declspec(dllexport) void *nb_iocp_new(ULONG concurrency)
{
    return CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, 0, concurrency);
}

__declspec(dllexport) int nb_iocp_bind(void *iocp, void *file, ULONG key)
{
    return CreateIoCompletionPort((HANDLE)file, (HANDLE)iocp, key, 0) ? 1 : 0;
}

/* Queue one positional read. Returns 1 on success, 0 on failure.
 * ERROR_IO_PENDING counts as success: the packet arrives when it completes. */
__declspec(dllexport) int nb_iocp_submit(void *iocp, void *file, long long offset,
                                         DWORD count, void *buf, ULONG key)
{
    NB_REQ *r = (NB_REQ *)calloc(1, sizeof(NB_REQ));
    if (!r) { SetLastError(ERROR_NOT_ENOUGH_MEMORY); return 0; }
    r->key = key;
    r->buf = buf;
    r->count = count;
    r->offset = offset;
    r->ov.Offset = (DWORD)(offset & 0xFFFFFFFF);
    r->ov.OffsetHigh = (DWORD)((offset >> 32) & 0xFFFFFFFF);
    r->ov.hEvent = NULL;
    if (!ReadFile((HANDLE)file, buf, count, NULL, &r->ov)) {
        DWORD e = GetLastError();
        if (e != ERROR_IO_PENDING) {
            free(r);
            SetLastError(e);
            return 0;
        }
    }
    return 1;
}

/* Fetch one completion: 1 = success (key, bytes, request token),
 * -1 = the I/O itself failed (token still identifies which request),
 * 0 = timeout or port error. */
__declspec(dllexport) int nb_iocp_wait(void *iocp, DWORD ms, ULONG *key,
                                       DWORD *bytes, void **token)
{
    ULONG k = 0;
    DWORD n = 0;
    LPOVERLAPPED ov = NULL;
    BOOL ok = GetQueuedCompletionStatus((HANDLE)iocp, &n, &k, &ov, ms);
    if (ok) {
        if (!ov) { *key = k; *bytes = n; *token = NULL; return 1; }
        NB_REQ *r = (NB_REQ *)ov;
        *key = r->key;
        *bytes = n;
        *token = r;
        return 1;
    }
    DWORD e = GetLastError();
    if (ov) {
        NB_REQ *r = (NB_REQ *)ov;
        *key = r->key;
        *bytes = 0;
        *token = r;
        SetLastError(e);
        return -1;
    }
    SetLastError(e);
    *key = 0;
    *bytes = 0;
    *token = NULL;
    return 0;
}

/* Release the request token after its completion was reported. */
__declspec(dllexport) void nb_iocp_release(void *token)
{
    if (token) free((NB_REQ *)token);
}

__declspec(dllexport) int nb_iocp_close(void *iocp)
{
    return CloseHandle((HANDLE)iocp) ? 1 : 0;
}
