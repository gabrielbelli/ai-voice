$ErrorActionPreference='Continue'
Write-Output "=== MY SESSION ==="
Write-Output ("ssh session id: " + (Get-Process -Id $PID).SessionId)
Write-Output "=== QUERY SESSION (all) ==="
query session 2>&1
Write-Output "=== QUERY USER ==="
query user 2>&1
Write-Output "=== LOGONUI / LOCK PROCESSES ==="
Get-Process -Name LogonUI,LockApp -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,SessionId | Format-Table -Auto | Out-String
Write-Output "=== CONSOLE SESSION PROCESSES (explorer/dwm sessionid) ==="
Get-Process -Name explorer,dwm -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,SessionId | Format-Table -Auto | Out-String
Write-Output "=== WIN32 API: GetLastInputInfo + foreground window, from THIS (ssh) session ==="
$sig = @"
using System;
using System.Runtime.InteropServices;
using System.Text;
public class W {
  [StructLayout(LayoutKind.Sequential)] public struct LASTINPUTINFO { public uint cbSize; public uint dwTime; }
  [DllImport("user32.dll")] public static extern bool GetLastInputInfo(ref LASTINPUTINFO p);
  [DllImport("kernel32.dll")] public static extern uint GetTickCount();
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowTextW(IntPtr h, StringBuilder s, int n);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
  [DllImport("kernel32.dll")] public static extern uint WTSGetActiveConsoleSessionId();
}
"@
try {
  Add-Type -TypeDefinition $sig -ErrorAction Stop
  $li = New-Object W+LASTINPUTINFO; $li.cbSize = 8
  $ok = [W]::GetLastInputInfo([ref]$li)
  Write-Output ("GetLastInputInfo ok=$ok  idle_ms=" + ([W]::GetTickCount() - $li.dwTime))
  $h = [W]::GetForegroundWindow()
  Write-Output ("GetForegroundWindow handle=" + $h)
  if ($h -ne [IntPtr]::Zero) {
    $p = 0; [void][W]::GetWindowThreadProcessId($h, [ref]$p)
    $r = New-Object W+RECT; [void][W]::GetWindowRect($h, [ref]$r)
    $sb = New-Object System.Text.StringBuilder 512; [void][W]::GetWindowTextW($h,$sb,512)
    Write-Output ("  fg pid=$p name=" + (Get-Process -Id $p -EA SilentlyContinue).ProcessName + " rect=$($r.L),$($r.T),$($r.R),$($r.B) title='" + $sb.ToString() + "'")
  }
  Write-Output ("WTSGetActiveConsoleSessionId = " + [W]::WTSGetActiveConsoleSessionId())
} catch { Write-Output ("ADDTYPE FAIL: " + $_.Exception.Message) }
Write-Output "=== C# COMPILER AVAILABLE (Add-Type worked?) ==="
Write-Output ("csc path: " + (Get-ChildItem 'C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe' -EA SilentlyContinue).FullName)
Write-Output "=== DISPLAY / POWER ==="
powercfg /requests 2>&1 | Select-Object -First 40
