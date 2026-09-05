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

        // The measurement this proof of concept exists to produce. LastYieldMs is
        // the wall time from the policy saying yield to the child process being
        // gone - which is when the driver gets the CUDA context and the weights
        // back. LastYieldWasKill says whether the polite stdin stage was enough or
        // whether the job object had to take it, because "it yielded in 2.1 s"
        // means something different if 2.0 s of that was the grace timer.
        public int LastYieldMs = -1;
        public bool LastYieldWasKill;
        public int Yields;
        public int Starts;

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
                // stdin is the yield channel, and it does double duty.
                //
                // 1. We write a single line, "YIELD", when the policy says the user
                //    wants the GPU back. runner.py has a thread blocked on stdin
                //    that sets a threading.Event; the chunk loop checks it BEFORE
                //    starting the next generate(), which is the only place it can
                //    be checked because generate() has no interruption point.
                //
                // 2. When the agent dies for any reason - killed, crashed, logged
                //    out - the kernel closes this pipe and the child's read returns
                //    EOF. That is a dead man's switch for free: the runner exits on
                //    its own instead of becoming an orphan holding a CUDA context
                //    on an 8 GiB card. The job object already covers the ordinary
                //    case; this covers it a second way, at the cost of one flag.
                psi.RedirectStandardInput = true;
                if (!string.IsNullOrEmpty(_c.JobWorkingDir)) psi.WorkingDirectory = _c.JobWorkingDir;
                _proc = Process.Start(psi);
                AssignProcessToJobObject(_job, _proc.Handle);
                Starts++;
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
            var sw = Stopwatch.StartNew();
            Yields++;
            LastYieldWasKill = false;
            _log("yielding the GPU (" + why + "), grace " +
                 _c.YieldGraceSeconds.ToString(CultureInfo.InvariantCulture) + "s");
            try
            {
                // The polite stage, and it has to be something the child can
                // actually hear. This used to be CloseMainWindow(), which does
                // nothing at all to a console process with no window - so the
                // "cooperative" stage was a no-op and the grace period simply
                // elapsed before the kill every single time. Measuring a yield
                // with that in place would have measured the grace timer and
                // nothing else.
                //
                // One line on stdin, then close the pipe. Closing matters: it is
                // the EOF that tells a runner which is not watching for YIELD, or
                // one wedged inside generate(), that the agent has finished with
                // it.
                _proc.StandardInput.WriteLine("YIELD");
                _proc.StandardInput.Flush();
                _proc.StandardInput.Close();
            }
            catch (Exception) { }
            try
            {
                if (!_proc.WaitForExit(_c.YieldGraceSeconds * 1000))
                {
                    _log("grace expired; killing the job tree");
                    LastYieldWasKill = true;
                    KillTree();
                }
            }
            catch (Exception) { LastYieldWasKill = true; KillTree(); }
            LastYieldMs = (int)sw.ElapsedMilliseconds;
            _log("gpu released after " + LastYieldMs.ToString(CultureInfo.InvariantCulture) +
                 " ms (" + (LastYieldWasKill ? "killed" : "exited on request") + ")");
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

        /// Non-null while runtime/provision.ps1 is downloading, carrying its own
        /// progress line. The tray shows this instead of a state, in its own
        /// colour, because a grey icon during a 5.6 GiB first run reads as broken.
        public string ProvisionStatus
        {
            get
            {
                try
                {
                    if (!File.Exists(_c.ProvisionStatusPath)) return null;
                    string t = File.ReadAllText(_c.ProvisionStatusPath).Trim();
                    return t.Length == 0 ? null : t;
                }
                catch (Exception) { return null; }
            }
        }

        public Agent(Config c)
        {
            _c = c;
            _gpu = new GpuMonitor(120);
            _policy = new Policy(c);
            _job = new JobRunner(c, delegate(string s) { Log(s); Append(s); });
            try { Mode = (Mode)Enum.Parse(typeof(Mode), c.StartMode, true); }
            catch (Exception) { Mode = Mode.Auto; }
        }

        public Policy Policy { get { return _policy; } }
        public Snapshot Current { get { lock (_lock) { return _snap; } } }
        public bool JobRunning { get { return _job.Running; } }
        public int CounterCostMs { get { lock (_lock) { return _lastCounterCostMs; } } }
        public int LastYieldMs { get { return _job.LastYieldMs; } }
        public bool LastYieldWasKill { get { return _job.LastYieldWasKill; } }
        public int Yields { get { return _job.Yields; } }
        public int JobStarts { get { return _job.Starts; } }

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
                    // Raw, unfiltered. The threshold, the allowlist and the
                    // own-job exemption used to be applied here, which put three
                    // consequential judgements outside the module that holds the
                    // decisions and outside anything a test could reach. They live
                    // in Policy.ForeignVram now; this loop just reads the counter.
                    List<ProcessGpuUse> all = _vram.Read();
                    Dictionary<string, double> eng = _engines.ReadByEngineType();
                    lock (_lock)
                    {
                        _lastVram = all;
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
                        s.GpuProcesses = _lastVram;
                        s.CountersFresh = (DateTime.UtcNow - _countersAt) < TimeSpan.FromSeconds(30);
                        double d;
                        s.Util3d = _lastEngines.TryGetValue("3d", out d) ? d : 0;
                        s.UtilVideoDecode = _lastEngines.TryGetValue("videodecode", out d) ? d : 0;
                        s.UtilVideoEncode = _lastEngines.TryGetValue("videoencode", out d) ? d : 0;
                    }

                    // Our own child is never evidence of a user. Read here rather
                    // than in the policy because the policy must stay free of
                    // anything that touches a process handle.
                    s.OwnJobPid = _job.Pid;

                    Verdict v = _policy.Evaluate(s);
                    lock (_lock) { _snap = s; }

                    bool can = _policy.CanRun(Mode);
                    if (can && !_job.Running && !string.IsNullOrEmpty(_c.JobCommand)) _job.Start();
                    if (!can && _job.Running)
                    {
                        // Off the sampling thread, deliberately.
                        //
                        // Stop() waits up to YieldGraceSeconds for the child to go.
                        // Called inline it blocked this loop for that whole time, so
                        // no sample was taken, the tray showed stale state and
                        // state.json stopped updating during exactly the seconds
                        // whose latency this proof of concept exists to measure.
                        // The restart gate is !_job.Running, so a Stop still in
                        // flight cannot race a Start: the child is only gone once
                        // the handle is closed, and until then Running stays true.
                        string why = Mode == Mode.Off ? "switched off" : v.ReasonText;
                        ThreadPool.QueueUserWorkItem(delegate(object ignored)
                        {
                            try { _job.Stop(why); } catch (Exception) { }
                        });
                    }

                    if (_policy.State != prev)
                    {
                        prev = _policy.State;
                        // Every transition, with its reason and the mode it
                        // happened under. This is the whole evidence base for the
                        // false-idle question, so it is written even in Off mode -
                        // a week of Off-mode transitions is what shows the detector
                        // is trustworthy before it is ever allowed to take the GPU.
                        Append(string.Format(CultureInfo.InvariantCulture,
                            "state {0} mode={1}{2} reason={3}",
                            Policy.Describe(_policy.State), Mode,
                            _policy.IsOverriding(Mode) ? " OVERRIDDEN" : "",
                            v.ReasonText));
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
            // Always-on runs jobs regardless of the verdict. It must still PUBLISH
            // the verdict it is overriding, otherwise switching to Always-on
            // silently destroys the false-idle evidence this trial is collecting.
            // "overriding" true means: a job is running, and Auto would have
            // stopped it.
            sb.Append(Json.P("overriding", _policy.IsOverriding(Mode) ? "true" : "false")).Append(",");
            sb.Append(Json.P("detector_state", Json.Esc(Policy.Describe(_policy.State)))).Append(",");
            sb.Append(Json.P("job_starts", Json.Num(_job.Starts))).Append(",");
            sb.Append(Json.P("yields", Json.Num(_job.Yields))).Append(",");
            sb.Append(Json.P("last_yield_ms", Json.Num(_job.LastYieldMs))).Append(",");
            sb.Append(Json.P("last_yield_was_kill", _job.LastYieldWasKill ? "true" : "false")).Append(",");

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

            // Only the processes the POLICY judged foreign, not the raw counter
            // read. The raw list is fifteen shell processes on an idle desktop and
            // burying the one that matters in it would defeat the point of the
            // file, which is that a human can read it and see why.
            List<ProcessGpuUse> foreign = _policy.ForeignVram(s);
            sb.Append("\"foreign_vram\":[");
            for (int i = 0; i < foreign.Count; i++)
            {
                if (i > 0) sb.Append(",");
                sb.Append(Json.Obj(
                    Json.P("pid", Json.Num(foreign[i].Pid)),
                    Json.P("name", Json.Esc(foreign[i].Name)),
                    Json.P("dedicated_mib", Json.Num(foreign[i].DedicatedMiB))));
            }
            sb.Append("]}");
            return sb.ToString();
        }

        /// Append one timestamped line to worker.log.
        void Append(string line)
        {
            try
            {
                string dir = Path.GetDirectoryName(_c.LogPath);
                if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
                File.AppendAllText(_c.LogPath,
                    DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture) + " " + line +
                    Environment.NewLine, new UTF8Encoding(false));
            }
            catch (Exception) { }
        }

        /// Persist the tray mode back into worker.ini, rewriting the StartMode line
        /// in place if it is there and appending it if it is not. Deliberately
        /// leaves every other line, comment and blank alone: this file is one a
        /// human edits, and a settings writer that reformats somebody's comments
        /// out of existence is a settings writer they stop using.
        public void SaveMode()
        {
            try
            {
                if (string.IsNullOrEmpty(_c.ConfigPath)) return;
                var lines = new List<string>();
                bool replaced = false;
                if (File.Exists(_c.ConfigPath))
                {
                    foreach (string raw in File.ReadAllLines(_c.ConfigPath))
                    {
                        string t = raw.Trim();
                        if (!t.StartsWith("#") && !t.StartsWith(";") &&
                            t.ToLowerInvariant().Replace(" ", "").StartsWith("startmode="))
                        {
                            lines.Add("StartMode = " + Mode);
                            replaced = true;
                        }
                        else lines.Add(raw);
                    }
                }
                if (!replaced) lines.Add("StartMode = " + Mode);
                File.WriteAllLines(_c.ConfigPath, lines.ToArray(), new UTF8Encoding(false));
                Append("mode set to " + Mode);
            }
            catch (Exception) { }
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
