// The loop, and the thing it supervises.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace AiVoice.Worker
{
    /// Runs the GPU job as a child process inside a Windows job object.
    ///
    /// WHY A JOB OBJECT AND NOT Process.Kill(). The payload this exists for is
    /// Chatterbox on torch, which spawns worker processes and CUDA helper threads.
    /// Killing the parent alone orphans those, and an orphaned CUDA context keeps
    /// its VRAM allocated - which on an 8 GiB card that the user is about to want
    /// back for a game is exactly the failure we are trying to prevent. A job
    /// object with KILL_ON_JOB_CLOSE takes the whole tree down together, and takes
    /// it down even if the agent itself is killed, because the handle closes when
    /// the agent's process object is torn down by the kernel.
    public class JobRunner : IDisposable
    {
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
        static extern IntPtr CreateJobObjectW(IntPtr sec, string name);
        [DllImport("kernel32.dll")]
        static extern bool SetInformationJobObject(IntPtr job, int infoClass, IntPtr info, uint len);
        [DllImport("kernel32.dll")]
        static extern bool AssignProcessToJobObject(IntPtr job, IntPtr proc);
        [DllImport("kernel32.dll")]
        static extern bool CloseHandle(IntPtr h);

        [StructLayout(LayoutKind.Sequential)]
        struct JOBOBJECT_BASIC_LIMIT_INFORMATION
        {
            public long PerProcessUserTimeLimit, PerJobUserTimeLimit;
            public uint LimitFlags;
            public UIntPtr MinimumWorkingSetSize, MaximumWorkingSetSize;
            public uint ActiveProcessLimit;
            public UIntPtr Affinity;
            public uint PriorityClass, SchedulingClass;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct IO_COUNTERS { public ulong r, w, o, rt, wt, ot; }

        [StructLayout(LayoutKind.Sequential)]
        struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        {
            public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
            public IO_COUNTERS IoInfo;
            public UIntPtr ProcessMemoryLimit, JobMemoryLimit, PeakProcessMemoryUsed, PeakJobMemoryUsed;
        }

        const int JobObjectExtendedLimitInformation = 9;
        const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000;

        IntPtr _job = IntPtr.Zero;
        Process _proc;
        readonly Config _c;
        readonly Action<string> _log;

        public JobRunner(Config c, Action<string> log) { _c = c; _log = log; }

        public bool Running
        {
            get
            {
                try { return _proc != null && !_proc.HasExited; }
                catch (Exception) { return false; }
            }
        }

        public int Pid { get { try { return _proc == null ? -1 : _proc.Id; } catch (Exception) { return -1; } } }

        public bool Start()
        {
            if (Running) return true;
            if (string.IsNullOrEmpty(_c.JobCommand))
            {
                _log("no JobCommand configured; running in detection-only mode");
                return false;
            }
            try
            {
                _job = CreateJobObjectW(IntPtr.Zero, null);
                var info = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                int len = Marshal.SizeOf(typeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION));
                IntPtr p = Marshal.AllocHGlobal(len);
                Marshal.StructureToPtr(info, p, false);
                SetInformationJobObject(_job, JobObjectExtendedLimitInformation, p, (uint)len);
                Marshal.FreeHGlobal(p);

                var psi = new ProcessStartInfo(_c.JobCommand, _c.JobArguments);
                psi.UseShellExecute = false;
                psi.CreateNoWindow = true;
                if (!string.IsNullOrEmpty(_c.JobWorkingDir)) psi.WorkingDirectory = _c.JobWorkingDir;
                _proc = Process.Start(psi);
                AssignProcessToJobObject(_job, _proc.Handle);
                _log("started job pid " + _proc.Id.ToString(CultureInfo.InvariantCulture));
                return true;
            }
            catch (Exception ex)
            {
                _log("failed to start job: " + ex.Message);
                return false;
            }
        }

        /// Ask, wait the grace period, then take it.
        public void Stop(string why)
        {
            if (!Running) return;
            _log("yielding the GPU (" + why + "), grace " +
                 _c.YieldGraceSeconds.ToString(CultureInfo.InvariantCulture) + "s");
            try
            {
                // CloseMainWindow is the polite request. A console job has no window,
                // so this usually does nothing and the grace period simply elapses -
                // which is why the hard kill below is not optional. A cooperative
                // runtime should watch its own stdin or a sentinel file instead; that
                // contract is documented in the README rather than assumed here.
                _proc.CloseMainWindow();
            }
            catch (Exception) { }
            try
            {
                if (!_proc.WaitForExit(_c.YieldGraceSeconds * 1000))
                {
                    _log("grace expired; killing the job tree");
                    KillTree();
                }
            }
            catch (Exception) { KillTree(); }
        }

        void KillTree()
        {
            // Closing the job handle kills every process in it. That is the whole
            // point of the job object: no child survives to hold VRAM.
            try { if (_job != IntPtr.Zero) { CloseHandle(_job); _job = IntPtr.Zero; } }
            catch (Exception) { }
            try { if (_proc != null && !_proc.HasExited) _proc.Kill(); }
            catch (Exception) { }
        }

        public void Dispose() { KillTree(); }
    }

    /// Owns the samplers, the policy and the job, and steps them on a timer.
    public class Agent : IDisposable
    {
        readonly Config _c;
        readonly GpuMonitor _gpu;
        readonly GpuProcessMemory _vram = new GpuProcessMemory();
        readonly GpuEngineUtil _engines = new GpuEngineUtil();
        readonly Policy _policy;
        readonly JobRunner _job;
        readonly object _lock = new object();

        Thread _fast, _slow;
        volatile bool _stop;
        Snapshot _snap = new Snapshot();
        List<ProcessGpuUse> _lastVram = new List<ProcessGpuUse>();
        Dictionary<string, double> _lastEngines = new Dictionary<string, double>();
        DateTime _countersAt = DateTime.MinValue;
        int _lastCounterCostMs = -1;

        public Mode Mode = Mode.Auto;
        public event Action<WorkerState, string> StateChanged;
        public Action<string> Log = delegate(string s) { };

        public Agent(Config c)
        {
            _c = c;
            _gpu = new GpuMonitor(120);
            _policy = new Policy(c);
            _job = new JobRunner(c, delegate(string s) { Log(s); });
        }

        public Policy Policy { get { return _policy; } }
        public Snapshot Current { get { lock (_lock) { return _snap; } } }
        public bool JobRunning { get { return _job.Running; } }
        public int CounterCostMs { get { lock (_lock) { return _lastCounterCostMs; } } }

        public void Start()
        {
            _gpu.Start();
            _fast = new Thread(FastLoop); _fast.IsBackground = true; _fast.Start();
            _slow = new Thread(SlowLoop); _slow.IsBackground = true; _slow.Start();
        }

        void SlowLoop()
        {
            while (!_stop)
            {
                try
                {
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    List<ProcessGpuUse> all = _vram.Read();
                    var foreign = new List<ProcessGpuUse>();
                    foreach (var u in all)
                    {
                        if (u.DedicatedMiB < _c.ForeignVramBusyMiB) continue;
                        bool allowed = false;
                        foreach (string a in _c.VramAllowlist)
                            if (string.Equals(a.Trim(), u.Name, StringComparison.OrdinalIgnoreCase)) { allowed = true; break; }
                        // Never yield to our own job: it is the thing we are deciding
                        // about, and counting its VRAM as evidence of a user would make
                        // the policy oscillate the moment a job allocated anything.
                        if (u.Pid == _job.Pid) allowed = true;
                        if (!allowed) foreign.Add(u);
                    }
                    Dictionary<string, double> eng = _engines.ReadByEngineType();
                    lock (_lock)
                    {
                        _lastVram = foreign;
                        _lastEngines = eng;
                        _countersAt = DateTime.UtcNow;
                        _lastCounterCostMs = (int)sw.ElapsedMilliseconds;
                    }
                }
                catch (Exception ex) { Log("counter sample failed: " + ex.Message); }
                Sleep(_c.CounterPollMs);
            }
        }

        void FastLoop()
        {
            WorkerState prev = WorkerState.Blocked;
            while (!_stop)
            {
                Snapshot s = null;
                try
                {
                    s = new Snapshot();
                    s.At = DateTime.UtcNow;
                    s.Gpu = _gpu.Latest;
                    s.GpuHealthy = _gpu.Healthy;
                    s.Session = Win.Read();
                    s.Launchers = Launchers.Read(_c.GameProcessNames);
                    lock (_lock)
                    {
                        s.ForeignVram = _lastVram;
                        s.CountersFresh = (DateTime.UtcNow - _countersAt) < TimeSpan.FromSeconds(30);
                        double d;
                        s.Util3d = _lastEngines.TryGetValue("3d", out d) ? d : 0;
                        s.UtilVideoDecode = _lastEngines.TryGetValue("videodecode", out d) ? d : 0;
                        s.UtilVideoEncode = _lastEngines.TryGetValue("videoencode", out d) ? d : 0;
                    }

                    Verdict v = _policy.Evaluate(s);
                    lock (_lock) { _snap = s; }

                    bool can = _policy.CanRun(Mode);
                    if (can && !_job.Running && !string.IsNullOrEmpty(_c.JobCommand)) _job.Start();
                    if (!can && _job.Running) _job.Stop(Mode == Mode.Off ? "switched off" : v.ReasonText);

                    if (_policy.State != prev)
                    {
                        prev = _policy.State;
                        var h = StateChanged;
                        if (h != null) h(_policy.State, v.ReasonText);
                    }
                }
                catch (Exception ex) { Log("sample failed: " + ex.Message); }
                Sleep(_c.FastPollMs);
            }
        }

        void Sleep(int ms)
        {
            // Chopped so Stop() is responsive without an event handle.
            int left = ms;
            while (left > 0 && !_stop) { int n = Math.Min(200, left); Thread.Sleep(n); left -= n; }
        }

        public string StatusJson()
        {
            Snapshot s = Current;
            var sb = new StringBuilder();
            sb.Append("{");
            sb.Append(Json.P("timestamp", Json.Esc(s.At.ToString("o", CultureInfo.InvariantCulture)))).Append(",");
            sb.Append(Json.P("mode", Json.Esc(Mode.ToString()))).Append(",");
            sb.Append(Json.P("state", Json.Esc(Policy.Describe(_policy.State)))).Append(",");
            sb.Append(Json.P("can_run", _policy.CanRun(Mode) ? "true" : "false")).Append(",");
            sb.Append(Json.P("job_running", _job.Running ? "true" : "false")).Append(",");
            sb.Append(Json.P("seconds_until_available", Json.Num(_policy.SecondsUntilAvailable))).Append(",");
            sb.Append(Json.P("reason", Json.Esc(_policy.Last.ReasonText))).Append(",");
            sb.Append(Json.P("veto", _policy.Last.IsVeto ? "true" : "false")).Append(",");

            sb.Append("\"gpu\":");
            if (s.Gpu != null && s.Gpu.Valid)
            {
                sb.Append(Json.Obj(
                    Json.P("healthy", s.GpuHealthy ? "true" : "false"),
                    Json.P("utilisation_pct", Json.Num(s.Gpu.UtilGpu)),
                    Json.P("memory_utilisation_pct", Json.Num(s.Gpu.UtilMem)),
                    Json.P("encoder_pct", Json.Num(s.Gpu.UtilEncoder)),
                    Json.P("decoder_pct", Json.Num(s.Gpu.UtilDecoder)),
                    Json.P("memory_clock_mhz", Json.Num(s.Gpu.ClockMemMhz)),
                    Json.P("sm_clock_mhz", Json.Num(s.Gpu.ClockSmMhz)),
                    Json.P("pstate", Json.Esc(s.Gpu.PState)),
                    Json.P("power_watts", Json.Num(s.Gpu.PowerWatts)),
                    Json.P("memory_used_mib", Json.Num(s.Gpu.MemUsedMiB))));
            }
            else sb.Append("null");
            sb.Append(",");

            sb.Append("\"engines\":").Append(Json.Obj(
                Json.P("three_d_pct", Json.Num(s.Util3d)),
                Json.P("video_decode_pct", Json.Num(s.UtilVideoDecode)),
                Json.P("video_encode_pct", Json.Num(s.UtilVideoEncode)),
                Json.P("fresh", s.CountersFresh ? "true" : "false"),
                Json.P("sample_cost_ms", Json.Num(CounterCostMs)))).Append(",");

            sb.Append("\"session\":");
            if (s.Session != null)
            {
                sb.Append(Json.Obj(
                    Json.P("own_session", Json.Num(s.Session.OwnSessionId)),
                    Json.P("console_session", Json.Num(s.Session.ConsoleSessionId)),
                    Json.P("in_console_session", s.Session.RunningInConsoleSession ? "true" : "false"),
                    Json.P("locked", s.Session.Locked ? "true" : "false"),
                    Json.P("input_idle_seconds", Json.Num(s.Session.InputIdleSeconds)),
                    Json.P("foreground_process", Json.Esc(s.Session.ForegroundProcess)),
                    Json.P("foreground_fullscreen", s.Session.ForegroundIsFullScreen ? "true" : "false")));
            }
            else sb.Append("null");
            sb.Append(",");

            sb.Append("\"launchers\":");
            if (s.Launchers != null)
            {
                sb.Append(Json.Obj(
                    Json.P("steam_running_appid", Json.Num(s.Launchers.SteamRunningAppId)),
                    Json.P("steam_running_app", Json.Esc(s.Launchers.SteamRunningAppName)),
                    Json.P("vgc_running", s.Launchers.ValorantAntiCheatActive ? "true" : "false"),
                    Json.P("game_processes", "[" + string.Join(",",
                        s.Launchers.GameProcesses.Select(Json.Esc).ToArray()) + "]")));
            }
            else sb.Append("null");
            sb.Append(",");

            sb.Append("\"foreign_vram\":[");
            for (int i = 0; i < s.ForeignVram.Count; i++)
            {
                if (i > 0) sb.Append(",");
                sb.Append(Json.Obj(
                    Json.P("pid", Json.Num(s.ForeignVram[i].Pid)),
                    Json.P("name", Json.Esc(s.ForeignVram[i].Name)),
                    Json.P("dedicated_mib", Json.Num(s.ForeignVram[i].DedicatedMiB))));
            }
            sb.Append("]}");
            return sb.ToString();
        }

        /// Written next to the agent so anything on the box - a scheduled task, the
        /// gateway's own poller, a human with `type` - can read the worker's mind
        /// without an open port. WHY NOT AN HTTP ENDPOINT: a listening socket on a
        /// machine running kernel-mode anti-cheat is a conversation nobody wants to
        /// have, and the status is one small document that changes once a second.
        public void WriteState()
        {
            try
            {
                string dir = Path.GetDirectoryName(_c.StatePath);
                if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
                string tmp = _c.StatePath + ".tmp";
                File.WriteAllText(tmp, StatusJson(), new UTF8Encoding(false));
                if (File.Exists(_c.StatePath)) File.Delete(_c.StatePath);
                File.Move(tmp, _c.StatePath);   // atomic-enough rename; no torn reads
            }
            catch (Exception) { }
        }

        public void Dispose()
        {
            _stop = true;
            try { _job.Stop("agent shutting down"); } catch (Exception) { }
            try { _job.Dispose(); } catch (Exception) { }
            try { _gpu.Dispose(); } catch (Exception) { }
        }
    }
}
