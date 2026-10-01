# Third caller: P/Invoke from PowerShell/C# (real IL, not ctypes).
#
# If this succeeds while ctypes fails, the rejection is specific to how Python
# forms the call rather than to the process or the volume.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File _dev\bin\_nbtest_csharp.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File _dev\bin\_nbtest_csharp.ps1 -Path 'D:\other\file'

param(
    [string]$Path = 'D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP\model-00001-of-00011.safetensors'
)

$source = @'
using System;
using System.Runtime.InteropServices;

namespace NbProbe
{
    public static class Nb
    {
        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        static extern IntPtr CreateFileW(string name, uint access, uint share,
            IntPtr sa, uint disp, uint flags, IntPtr templ);

        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool SetFilePointerEx(IntPtr h, long off,
            out long moved, uint origin);

        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool ReadFile(IntPtr h, IntPtr buf, uint count,
            out uint read, IntPtr overlapped);

        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool CloseHandle(IntPtr h);

        public static string Probe(string path)
        {
            // GENERIC_READ, share read+write+delete,
            // NO_BUFFERING | RANDOM_ACCESS
            IntPtr h = CreateFileW(path, 0x80000000, 0x7, IntPtr.Zero, 3,
                0x40000000 | 0x10000000, IntPtr.Zero);
            if (h == new IntPtr(-1)) return "CreateFile: " + Marshal.GetLastWin32Error();
            try
            {
                long moved;
                if (!SetFilePointerEx(h, 4096, out moved, 0))
                    return "Seek: " + Marshal.GetLastWin32Error();
                IntPtr buf = Marshal.AllocHGlobal(8192);
                try
                {
                    uint got;
                    bool ok = ReadFile(h, buf, 4096, out got, IntPtr.Zero);
                    return ok ? ("ok, " + got + " bytes")
                              : ("ReadFile: " + Marshal.GetLastWin32Error());
                }
                finally { Marshal.FreeHGlobal(buf); }
            }
            finally { CloseHandle(h); }
        }
    }
}
'@

Add-Type -TypeDefinition $source -Language CSharp
Write-Host "path: $Path" -ForegroundColor Cyan
Write-Host ("C#/P-Invoke NO_BUFFERING read @4096 -> " `
    + [NbProbe.Nb]::Probe($Path)) -ForegroundColor Yellow
