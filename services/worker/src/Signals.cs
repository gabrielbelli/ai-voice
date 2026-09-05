// Signal acquisition for the ai-voice GPU worker.
//
// EVERY THRESHOLD AND EVERY CHOICE OF SIGNAL IN THIS FILE COMES FROM A
// MEASUREMENT TAKEN ON THE TARGET MACHINE (spring, RTX 3070, driver 610.47,
// Windows 11 Pro 10.0.26200) ON 2026-09-05. The probe scripts that produced
// them are in ../probe and re-run unchanged. Numbers quoted in comments are
// from those runs, not from documentation.
//
// Written to C# 5. WHY: the only C# compiler on a stock Windows 11 is
// C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe, which is the
// pre-Roslyn compiler. Measured on spring: present, and no .NET SDK is
// installed, so there is no Roslyn to fall back on. No string interpolation,
// no null-conditional operator, no nameof.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using System.ServiceProcess;
using System.Text;
using System.Threading;
using Microsoft.Win32;

namespace AiVoice.Worker
{
    // ---------------------------------------------------------------- GPU ---


    /// Streams nvidia-smi at 1 Hz from ONE long-lived child process.
    ///
    /// WHY A STREAM RATHER THAN REPEATED ONE-SHOTS. Measured on spring: a single
    /// `nvidia-smi --query-gpu=...` invocation costs 39-44 ms of wall time (probe
    /// p4, section B, five runs). Polling that at 1 Hz would be tolerable but at
    /// 4 Hz it is 16 per cent of a core spent on process creation alone, on a
    /// machine whose whole point is that we must not disturb it. `-l 1` costs one
    /// process for the lifetime of the agent and was measured (probe p4, A1) to
    /// emit one line per second, line-buffered, through a redirected stdout.
    ///
    /// WHY NOT `-l 1 -c N`. Measured (probe p4, A3): combining -c with --query-gpu
    /// makes nvidia-smi reject --query-gpu itself with
    ///   "ERROR: Option --query-gpu=utilization.gpu is not recognized"
    /// which is a misleading message for an argument that is perfectly valid on
    /// its own. -c belongs to `dmon`/`pmon`. Use -l alone and count lines here.
    public class GpuMonitor : IDisposable
    {
        const string Fields =
            "utilization.gpu,utilization.memory,utilization.encoder,utilization.decoder," +
            "clocks.mem,clocks.sm,pstate,power.draw,memory.used";

        readonly object _lock = new object();
        readonly Queue<GpuSample> _recent = new Queue<GpuSample>();
        readonly int _keep;
        Process _proc;
        Thread _reader;
        volatile bool _stop;
        GpuSample _latest = new GpuSample();

        public GpuMonitor(int keepSamples)
        {
            _keep = keepSamples;
            _latest.PState = "?";
        }

        public string ExePath = "nvidia-smi";

        public void Start()
        {
            var psi = new ProcessStartInfo(ExePath,
                "--query-gpu=" + Fields + " --format=csv,noheader,nounits -l 1");
            psi.UseShellExecute = false;
            psi.RedirectStandardOutput = true;
            psi.RedirectStandardError = true;
            psi.CreateNoWindow = true;
            _proc = Process.Start(psi);
            _reader = new Thread(ReadLoop);
            _reader.IsBackground = true;
            _reader.Start();
        }

        void ReadLoop()
        {
            try
            {
                while (!_stop)
                {
                    string line = _proc.StandardOutput.ReadLine();
                    if (line == null) break;
                    GpuSample s = Parse(line);
                    if (!s.Valid) continue;
                    lock (_lock)
                    {
                        _latest = s;
                        _recent.Enqueue(s);
                        while (_recent.Count > _keep) _recent.Dequeue();
                    }
                }
            }
            catch (Exception) { /* process died; Latest goes stale and Healthy() reports it */ }
        }

        static int PInt(string s)
        {
            int v;
            // "[N/A]" and "[Not Supported]" both appear in nvidia-smi output; treat as -1.
            if (int.TryParse(s.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out v)) return v;
            return -1;
        }

        static double PDbl(string s)
        {
            double v;
            if (double.TryParse(s.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out v)) return v;
            return -1;
        }

        static GpuSample Parse(string line)
        {
            var s = new GpuSample();
            s.At = DateTime.UtcNow;
            s.PState = "?";
            string[] f = line.Split(',');
            if (f.Length < 9) return s;
            s.UtilGpu = PInt(f[0]);
            s.UtilMem = PInt(f[1]);
            s.UtilEncoder = PInt(f[2]);
            s.UtilDecoder = PInt(f[3]);
            s.ClockMemMhz = PInt(f[4]);
            s.ClockSmMhz = PInt(f[5]);
            s.PState = f[6].Trim();
            s.PowerWatts = PDbl(f[7]);
            s.MemUsedMiB = PInt(f[8]);
            s.Valid = s.UtilGpu >= 0 && s.ClockMemMhz >= 0;
            return s;
        }

        public GpuSample Latest { get { lock (_lock) { return _latest; } } }

        public GpuSample[] Recent { get { lock (_lock) { return _recent.ToArray(); } } }

        /// True while the stream is producing fresh samples. A dead nvidia-smi must
        /// mean "assume busy", never "assume idle" - see Policy.
        public bool Healthy
        {
            get
            {
                GpuSample l = Latest;
                return l.Valid && (DateTime.UtcNow - l.At) < TimeSpan.FromSeconds(10);
            }
        }

        public void Dispose()
        {
            _stop = true;
            try { if (_proc != null && !_proc.HasExited) _proc.Kill(); }
            catch (Exception) { }
            try { if (_proc != null) _proc.Dispose(); }
            catch (Exception) { }
        }
    }

    // ------------------------------------------------ per-process GPU state ---


    /// Per-process GPU memory, from the Windows "GPU Process Memory" performance
    /// counter set.
    ///
    /// WHY THIS EXISTS AT ALL. On this machine nvidia-smi reports used_memory as
    /// [N/A] for EVERY process (probe p1, "COMPUTE APPS": fifteen processes, all
    /// [N/A]), and `nvidia-smi pmon` reports "-" for sm/mem/enc/dec per process.
    /// That is the WDDM driver model, not a fault, and it removes the obvious way
    /// to notice a game that is paused but still resident. The OS counters are not
    /// subject to that limitation: probe p2 read dwm at 172.4 MiB, CamoStudio at
    /// 112.3 MiB, explorer at 40.3 MiB, and the adapter total at 423.4 MiB against
    /// nvidia-smi's own 417 MiB for the same instant - a 1.5 per cent
    /// disagreement, which is two different accountings of the same truth rather
    /// than two different truths.
    ///
    /// These counters are a documented OS API read through PDH. Nothing is
    /// injected, no process is opened for write, no graphics API is hooked. That
    /// matters here specifically: Riot Vanguard (vgk, kernel-mode, measured
    /// Running with StartType System) is resident on this machine at all times.
    ///
    /// "Dedicated Usage" is an instantaneous gauge, so ONE sample is meaningful.
    /// Contrast "GPU Engine\Utilization Percentage", which is a rate and needs two
    /// samples an interval apart - measured cost of a single Get-Counter call on
    /// that set was 1069-1851 ms (probe p7), which is why utilisation attribution
    /// runs on a slow cadence and never gates a decision on its own.
    public class GpuProcessMemory
    {
        readonly Dictionary<int, string> _nameCache = new Dictionary<int, string>();

        public List<ProcessGpuUse> Read()
        {
            var outp = new List<ProcessGpuUse>();
            PerformanceCounterCategory cat;
            try { cat = new PerformanceCounterCategory("GPU Process Memory"); }
            catch (Exception) { return outp; }

            string[] instances;
            try { instances = cat.GetInstanceNames(); }
            catch (Exception) { return outp; }

            // Sum per pid: one process can appear under several adapter LUIDs.
            var byPid = new Dictionary<int, double>();
            foreach (string inst in instances)
            {
                int pid = PidFromInstance(inst);
                if (pid <= 0) continue;
                try
                {
                    using (var c = new PerformanceCounter("GPU Process Memory", "Dedicated Usage", inst, true))
                    {
                        double mib = c.RawValue / 1048576.0;
                        if (!byPid.ContainsKey(pid)) byPid[pid] = 0;
                        byPid[pid] += mib;
                    }
                }
                catch (Exception) { /* instance vanished between enumerate and read */ }
            }

            foreach (var kv in byPid)
            {
                var u = new ProcessGpuUse();
                u.Pid = kv.Key;
                u.DedicatedMiB = kv.Value;
                u.Name = NameOf(kv.Key);
                outp.Add(u);
            }
            outp.Sort(delegate(ProcessGpuUse a, ProcessGpuUse b) { return b.DedicatedMiB.CompareTo(a.DedicatedMiB); });
            return outp;
        }

        /// Instance names look like
        ///   pid_1752_luid_0x00000000_0x0000fbc9_phys_0
        /// for GPU Process Memory and
        ///   pid_5640_luid_0x00000000_0x0000fbc9_phys_0_eng_0_engtype_3d
        /// for GPU Engine. Both measured verbatim in probe p2.
        public static int PidFromInstance(string inst)
        {
            if (inst == null || !inst.StartsWith("pid_")) return -1;
            int end = inst.IndexOf('_', 4);
            if (end < 0) return -1;
            int pid;
            if (int.TryParse(inst.Substring(4, end - 4), out pid)) return pid;
            return -1;
        }

        string NameOf(int pid)
        {
            string n;
            if (_nameCache.TryGetValue(pid, out n)) return n;
            try { n = Process.GetProcessById(pid).ProcessName; }
            catch (Exception) { n = "(gone)"; }
            if (_nameCache.Count > 512) _nameCache.Clear();
            _nameCache[pid] = n;
            return n;
        }
    }

    /// Utilisation split by engine type, summed across processes.
    /// Measured at idle (probe p2): 3d 0.95, copy 0.03, videodecode 0.00,
    /// videoencode 0.00. The split is the whole reason this class exists: it is
    /// what separates "a browser is decoding video", which lands on videodecode
    /// and leaves the shaders free, from "something is rendering", which lands on
    /// 3d. Utilisation alone cannot tell those apart and would yield to a YouTube
    /// tab for no reason.
    public class GpuEngineUtil
    {
        readonly Dictionary<string, PerformanceCounter> _counters = new Dictionary<string, PerformanceCounter>();
        DateTime _refreshed = DateTime.MinValue;

        public Dictionary<string, double> ReadByEngineType()
        {
            var sums = new Dictionary<string, double>();
            // Rebuild the instance list periodically: processes come and go, and a
            // stale PerformanceCounter throws on read rather than returning zero.
            if ((DateTime.UtcNow - _refreshed) > TimeSpan.FromSeconds(30)) Refresh();

            foreach (var kv in _counters)
            {
                string eng = EngineTypeOf(kv.Key);
                double v;
                try { v = kv.Value.NextValue(); }
                catch (Exception) { continue; }
                if (!sums.ContainsKey(eng)) sums[eng] = 0;
                sums[eng] += v;
            }
            return sums;
        }

        void Refresh()
        {
            foreach (var c in _counters.Values) { try { c.Dispose(); } catch (Exception) { } }
            _counters.Clear();
            try
            {
                var cat = new PerformanceCounterCategory("GPU Engine");
                foreach (string inst in cat.GetInstanceNames())
                {
                    try
                    {
                        var c = new PerformanceCounter("GPU Engine", "Utilization Percentage", inst, true);
                        c.NextValue(); // prime: a rate counter's first read is always 0
                        _counters[inst] = c;
                    }
                    catch (Exception) { }
                }
            }
            catch (Exception) { }
            _refreshed = DateTime.UtcNow;
        }

        public static string EngineTypeOf(string inst)
        {
            int i = inst.IndexOf("engtype_");
            if (i < 0) return "unknown";
            return inst.Substring(i + 8);
        }
    }

    // ------------------------------------------------------- Windows session ---


    public static class Win
    {
        [StructLayout(LayoutKind.Sequential)]
        struct LASTINPUTINFO { public uint cbSize; public uint dwTime; }

        [StructLayout(LayoutKind.Sequential)]
        public struct RECT { public int Left, Top, Right, Bottom; }

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        struct MONITORINFO
        {
            public int cbSize;
            public RECT rcMonitor;
            public RECT rcWork;
            public uint dwFlags;
        }

        [DllImport("user32.dll")] static extern bool GetLastInputInfo(ref LASTINPUTINFO plii);
        [DllImport("kernel32.dll")] static extern uint GetTickCount();
        [DllImport("user32.dll")] static extern IntPtr GetForegroundWindow();
        [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
        [DllImport("user32.dll")] static extern bool GetWindowRect(IntPtr h, out RECT r);
        [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern int GetWindowTextW(IntPtr h, StringBuilder s, int n);
        [DllImport("user32.dll")] static extern IntPtr MonitorFromWindow(IntPtr h, uint flags);
        [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern bool GetMonitorInfoW(IntPtr hMonitor, ref MONITORINFO mi);
        [DllImport("kernel32.dll")] static extern uint WTSGetActiveConsoleSessionId();
        [DllImport("kernel32.dll")] static extern bool ProcessIdToSessionId(uint pid, out uint sid);

        const uint MONITOR_DEFAULTTONEAREST = 2;

        /// Collects everything the OS will tell us about the human at the keyboard.
        ///
        /// READ THIS BEFORE TRUSTING InputIdleSeconds OR ForegroundPid. These two
        /// are PER SESSION, and that is the single most important structural fact
        /// measured during this investigation. Probe p3 on spring:
        ///
        ///   ssh session id            : 0
        ///   query session             : console / htcga / id 1 / Active
        ///   WTSGetActiveConsoleSessionId: 1
        ///   GetForegroundWindow()     : 0        (session 0 has no desktop)
        ///   GetLastInputInfo()        : 620953 ms (session 0's own input, meaningless)
        ///
        /// So an agent started over SSH, or installed as a Windows service, sits in
        /// session 0 and is BLIND to the user it exists to get out of the way of. It
        /// would see a foreground window handle of zero and ten minutes of "idle"
        /// while the user was mid-match. The agent MUST run inside the interactive
        /// session, launched from the user's Run key or Startup folder. See
        /// RunningInConsoleSession, which the policy treats as a hard precondition.
        public static SessionSignals Read()
        {
            var s = new SessionSignals();
            s.ConsoleSessionId = WTSGetActiveConsoleSessionId();
            uint own;
            s.OwnSessionId = ProcessIdToSessionId((uint)Process.GetCurrentProcess().Id, out own) ? own : 0xFFFFFFFF;
            s.RunningInConsoleSession = (s.OwnSessionId == s.ConsoleSessionId);

            // Lock detection. LogonUI.exe exists only while the secure desktop is
            // up. Measured unlocked on spring (probe p3): no LogonUI process at all,
            // dwm and explorer both in session 1. A locked machine is the safest
            // possible time to run, so this is worth detecting even though the tray
            // app also gets SystemEvents.SessionSwitch, which is instant.
            s.Locked = false;
            try
            {
                foreach (var p in Process.GetProcessesByName("LogonUI"))
                {
                    if (p.SessionId == s.ConsoleSessionId) { s.Locked = true; }
                    p.Dispose();
                }
            }
            catch (Exception) { }

            s.InputIdleSeconds = -1;
            if (s.RunningInConsoleSession)
            {
                var li = new LASTINPUTINFO();
                li.cbSize = (uint)Marshal.SizeOf(typeof(LASTINPUTINFO));
                if (GetLastInputInfo(ref li))
                {
                    // Both are unsigned 32-bit millisecond counters that wrap every
                    // 49.7 days; the subtraction is correct across the wrap only if
                    // it is done in unsigned arithmetic, which this is.
                    s.InputIdleSeconds = (int)((GetTickCount() - li.dwTime) / 1000);
                }
            }

            s.ForegroundPid = -1;
            s.ForegroundProcess = "";
            s.ForegroundTitle = "";
            IntPtr h = GetForegroundWindow();
            if (h != IntPtr.Zero)
            {
                uint fpid;
                GetWindowThreadProcessId(h, out fpid);
                s.ForegroundPid = (int)fpid;
                try { s.ForegroundProcess = Process.GetProcessById((int)fpid).ProcessName; }
                catch (Exception) { s.ForegroundProcess = "(gone)"; }
                var sb = new StringBuilder(512);
                GetWindowTextW(h, sb, sb.Capacity);
                s.ForegroundTitle = sb.ToString();
                s.ForegroundIsFullScreen = IsFullScreen(h);
            }
            return s;
        }

        /// Covers its whole monitor, work area included.
        ///
        /// WHY THIS AND NOT A FULLSCREEN HOOK. Exclusive fullscreen can be detected
        /// properly with IDXGIOutput or the Shell's fullscreen notification, but both
        /// mean loading graphics interfaces or registering shell hooks inside a
        /// process that lives on a machine running kernel-mode anti-cheat. Comparing
        /// a window rectangle to a monitor rectangle uses only user32 read calls that
        /// every window manager and screen reader makes, catches borderless-window
        /// mode (which is what most current games actually use), and cannot be
        /// mistaken for instrumentation. It is deliberately the weaker technique.
        public static bool IsFullScreen(IntPtr h)
        {
            RECT w;
            if (!GetWindowRect(h, out w)) return false;
            IntPtr mon = MonitorFromWindow(h, MONITOR_DEFAULTTONEAREST);
            if (mon == IntPtr.Zero) return false;
            var mi = new MONITORINFO();
            mi.cbSize = Marshal.SizeOf(typeof(MONITORINFO));
            if (!GetMonitorInfoW(mon, ref mi)) return false;
            RECT m = mi.rcMonitor;
            return w.Left <= m.Left && w.Top <= m.Top && w.Right >= m.Right && w.Bottom >= m.Bottom;
        }
    }

    // -------------------------------------------------------------- launchers ---


    /// Reads what the launchers already publish about themselves.
    ///
    /// WHY THIS IS THE STRONGEST SIGNAL IN THE SYSTEM. Steam writes the appid of
    /// the game it is starting to HKCU\Software\Valve\Steam\RunningAppID BEFORE the
    /// game process has initialised its renderer, so this fires during the launcher
    /// splash - seconds ahead of any GPU-side evidence. Measured on spring (probe
    /// p6) with no game running: RunningAppID = 0, six app subkeys, all with
    /// Running = 0, and Counter-Strike 2 (730), Rainbow Six Siege Test Server
    /// (623990), Ready or Not and Ghost Recon Breakpoint present.
    ///
    /// It is also the signal that survives a paused game. A game sitting at a pause
    /// menu draws almost nothing - the naive "utilisation below N per cent" test
    /// calls that idle and hands its VRAM to a job - but RunningAppID stays set for
    /// as long as the process lives.
    ///
    /// Valorant does not use Steam. It is covered by the vgc service instead:
    /// measured on spring, vgk (the kernel driver) is Running with StartType System
    /// and is therefore useless as a signal because it is ALWAYS running, whereas
    /// vgc (the user-mode service) was measured Stopped with StartType Manual and
    /// starts only with the game. vgc is the one to watch. Reading a service's
    /// status is a plain SCM query and is not something an anti-cheat can mistake
    /// for tampering.
    public static class Launchers
    {
        public static LauncherSignals Read(string[] gameProcessNames)
        {
            var s = new LauncherSignals();
            s.SteamRunningAppId = 0;
            s.SteamRunningAppName = "";

            try
            {
                using (RegistryKey k = Registry.CurrentUser.OpenSubKey(@"Software\Valve\Steam"))
                {
                    if (k != null)
                    {
                        object v = k.GetValue("RunningAppID");
                        if (v != null) s.SteamRunningAppId = Convert.ToInt32(v);
                    }
                }
                if (s.SteamRunningAppId != 0)
                {
                    using (RegistryKey a = Registry.CurrentUser.OpenSubKey(
                        @"Software\Valve\Steam\Apps\" + s.SteamRunningAppId.ToString(CultureInfo.InvariantCulture)))
                    {
                        if (a != null)
                        {
                            object n = a.GetValue("Name");
                            if (n != null) s.SteamRunningAppName = n.ToString();
                        }
                    }
                }
                else
                {
                    // Belt and braces: RunningAppID is cleared on a crash in some
                    // Steam builds, but the per-app Running flag is written
                    // separately. Cheap because there were six subkeys, not six
                    // hundred (measured, probe p6).
                    using (RegistryKey apps = Registry.CurrentUser.OpenSubKey(@"Software\Valve\Steam\Apps"))
                    {
                        if (apps != null)
                        {
                            foreach (string sub in apps.GetSubKeyNames())
                            {
                                using (RegistryKey a = apps.OpenSubKey(sub))
                                {
                                    if (a == null) continue;
                                    object r = a.GetValue("Running");
                                    if (r != null && Convert.ToInt32(r) == 1)
                                    {
                                        int id;
                                        if (int.TryParse(sub, out id)) s.SteamRunningAppId = id;
                                        object n = a.GetValue("Name");
                                        if (n != null) s.SteamRunningAppName = n.ToString();
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            catch (Exception) { }

            try
            {
                using (var sc = new ServiceController("vgc"))
                {
                    s.ValorantAntiCheatActive = (sc.Status == ServiceControllerStatus.Running ||
                                                 sc.Status == ServiceControllerStatus.StartPending);
                }
            }
            catch (Exception) { s.ValorantAntiCheatActive = false; }

            foreach (string want in gameProcessNames)
            {
                string w = want.Trim();
                if (w.Length == 0) continue;
                try
                {
                    Process[] found = Process.GetProcessesByName(w);
                    if (found.Length > 0) s.GameProcesses.Add(w);
                    foreach (var p in found) p.Dispose();
                }
                catch (Exception) { }
            }
            return s;
        }
    }
}
