/* Unbuffered I/O straight from C, so nothing about ctypes bindings is in play.
 *
 * Usage: nbtest.exe <path>
 * Prints the Win32 error for a sector-aligned NO_BUFFERING read and write.
 */
#include <windows.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("usage: nbtest.exe <path>\n");
        return 1;
    }
    const char *path = argv[1];

    HANDLE h = CreateFileA(path, GENERIC_READ,
                           FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                           OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS, NULL);
    if (h == INVALID_HANDLE_VALUE) {
        printf("CreateFile(read, NO_BUFFERING): error %lu\n", GetLastError());
        return 1;
    }
    printf("CreateFile(read, NO_BUFFERING): handle ok\n");

    LONG off = 4096;
    LONG cur = SetFilePointer(h, off, NULL, FILE_BEGIN);
    printf("SetFilePointer(4096): %ld\n", cur);

    void *buf = VirtualAlloc(NULL, 8192, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    DWORD got = 0;
    BOOL ok = ReadFile(h, buf, 4096, &got, NULL);
    printf("ReadFile(4096 @4096, 4096-aligned buf): ok=%d bytes=%lu error=%lu\n",
           ok, got, ok ? 0UL : GetLastError());

    /* 512-byte alignment variant */
    SetFilePointer(h, 512, NULL, FILE_BEGIN);
    got = 0;
    ok = ReadFile(h, buf, 512, &got, NULL);
    printf("ReadFile(512 @512): ok=%d bytes=%lu error=%lu\n",
           ok, got, ok ? 0UL : GetLastError());
    CloseHandle(h);

    h = CreateFileA(path, GENERIC_WRITE,
                    FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                    OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, NULL);
    if (h == INVALID_HANDLE_VALUE) {
        printf("CreateFile(write, NO_BUFFERING): error %lu\n", GetLastError());
        VirtualFree(buf, 0, MEM_RELEASE);
        return 1;
    }
    SetFilePointer(h, 4096, NULL, FILE_BEGIN);
    got = 0;
    ok = WriteFile(h, buf, 4096, &got, NULL);
    printf("WriteFile(4096 @4096): ok=%d bytes=%lu error=%lu\n",
           ok, got, ok ? 0UL : GetLastError());
    CloseHandle(h);

    VirtualFree(buf, 0, MEM_RELEASE);
    return 0;
}
