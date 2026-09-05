// The decision.
//
// THE SHAPE OF THE PROBLEM. "Utilisation below N per cent" is wrong in both
// directions and both were confirmed by measurement rather than assumed:
//
//   Wrong low.  A paused game draws nothing. It still owns its VRAM and expects
//               to resume in one frame. nvidia-smi cannot see that on this
//               machine at all - used_memory is [N/A] for every process, measured
//               (probe p1), because of WDDM. The OS performance counters can, and
//               do (probe p2), so the veto is built on those.
//
//   Wrong high. A browser decoding video raises utilisation without touching the
//               shaders. The engine-type split separates the two: measured at
//               idle, 3d 0.95 and videodecode 0.00, and video playback moves the
//               second and not the first.
//
//   Wrong late. utilisation.gpu is a driver-side average over a sampling window,
//               and on an idle desktop it never reads below 5 per cent anyway
//               (92 of 92 samples were 5 or 6, probe p5). A threshold under 5 is
//               unreachable on this machine; a threshold over 6 is noise-bound.
//
// THE ANSWER IS NOT A BETTER THRESHOLD, IT IS A DIFFERENT CLASS OF SIGNAL.
// Ranked by how early they fire:
//
//   1. Steam's RunningAppID     set before the game renders its first frame
//   2. the vgc service           starts with Valorant, before the game window
//   3. a game process exists     survives pause, alt-tab and minimise
//   4. foreign VRAM held         survives pause; invisible to nvidia-smi here
//   5. memory clock / pstate     the driver's own answer to "is this real work"
//   6. power draw                idle 33.7-35.1 W against 200+ W under load
//   7. utilisation               last, weakest, advisory only
//
// Signals 1-4 are vetoes: any one of them alone means yield, immediately, with no
// confirmation window, because each of them means "a game exists" rather than "a
// game is currently drawing", and that is the question that actually matters.
// Signals 5-7 vote, and need three consecutive seconds to carry.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;

namespace AiVoice.Worker
{
    public enum Mode { Auto, AlwaysOn, Off }

    public enum WorkerState
    {
        Available,   // the GPU is ours to use
        Busy,        // the user, or something of theirs, has it
        Draining,    // we are giving it back; a job is being wound down
        Blocked      // we cannot judge safely, so we behave as if busy
    }

    public class Snapshot
    {
        public DateTime At;
        public GpuSample Gpu;
        public bool GpuHealthy;
        public SessionSignals Session;
        public LauncherSignals Launchers;
        public List<ProcessGpuUse> ForeignVram = new List<ProcessGpuUse>();
        public double Util3d;
        public double UtilVideoDecode;
        public double UtilVideoEncode;
        public bool CountersFresh;
    }

    public class Verdict
    {
        public bool WantsGpu;               // something other than us wants it now
        public List<string> Reasons = new List<string>();
        public bool IsVeto;                 // fired on a tier-1 signal, so no confirmation window

        public string ReasonText
        {
            get { return Reasons.Count == 0 ? "clear" : string.Join("; ", Reasons.ToArray()); }
        }
    }

    public class Policy
    {
        readonly Config _c;
        int _busyStreak;
        DateTime _lastWanted = DateTime.MinValue;
        WorkerState _state = WorkerState.Blocked;

        public Policy(Config c)
        {
            _c = c;
            // Start Blocked, not Available. An agent that has not yet seen a single
            // sample knows nothing, and "knows nothing" must never mean "help
            // yourself to the GPU".
            _lastWanted = DateTime.UtcNow;
        }

        public WorkerState State { get { return _state; } }
        public Verdict Last = new Verdict();

        public DateTime ClearSince { get { return _lastWanted; } }

        public int SecondsUntilAvailable
        {
            get
            {
                double s = _c.ClearCooldownSeconds - (DateTime.UtcNow - _lastWanted).TotalSeconds;
                return s <= 0 ? 0 : (int)Math.Ceiling(s);
            }
        }

        public Verdict Evaluate(Snapshot s)
        {
            var v = new Verdict();

            // --- TIER 0: can we judge at all? --------------------------------
            //
            // Fail closed. Every one of these means the agent has lost the ability
            // to see the user coming, and the correct response to blindness is to
            // stop, not to carry on and hope.

            if (!s.GpuHealthy)
            {
                v.WantsGpu = true; v.IsVeto = true;
                v.Reasons.Add("nvidia-smi stream is stale or dead");
            }

            if (s.Session != null && !s.Session.RunningInConsoleSession)
            {
                // Measured on spring (probe p3): over SSH the agent lands in session
                // 0 while the user is on the console in session 1, and from there
                // GetForegroundWindow() returns 0 and GetLastInputInfo() reports the
                // SSH session's own idle time - 620953 ms, which is a lie about the
                // user. An agent in the wrong session cannot see the person it is
                // supposed to yield to, so it must not claim the GPU.
                v.WantsGpu = true; v.IsVeto = true;
                v.Reasons.Add(string.Format(CultureInfo.InvariantCulture,
                    "agent is in session {0}, console is session {1}: cannot observe the user",
                    s.Session.OwnSessionId, s.Session.ConsoleSessionId));
            }

            // --- TIER 1: vetoes. A game EXISTS. -------------------------------

            if (s.Launchers != null)
            {
                if (s.Launchers.SteamRunningAppId != 0)
                {
                    v.WantsGpu = true; v.IsVeto = true;
                    string name = string.IsNullOrEmpty(s.Launchers.SteamRunningAppName)
                        ? "appid " + s.Launchers.SteamRunningAppId.ToString(CultureInfo.InvariantCulture)
                        : s.Launchers.SteamRunningAppName;
                    v.Reasons.Add("Steam is running " + name);
                }
                if (s.Launchers.ValorantAntiCheatActive)
                {
                    v.WantsGpu = true; v.IsVeto = true;
                    v.Reasons.Add("Vanguard user-mode service vgc is running");
                }
                if (s.Launchers.GameProcesses.Count > 0)
                {
                    v.WantsGpu = true; v.IsVeto = true;
                    v.Reasons.Add("game process present: " + string.Join(", ", s.Launchers.GameProcesses.ToArray()));
                }
            }

            if (s.ForeignVram != null && s.ForeignVram.Count > 0)
            {
                ProcessGpuUse top = s.ForeignVram[0];
                v.WantsGpu = true; v.IsVeto = true;
                v.Reasons.Add(string.Format(CultureInfo.InvariantCulture,
                    "{0} (pid {1}) holds {2:N0} MiB of GPU memory", top.Name, top.Pid, top.DedicatedMiB));
            }

            if (s.Session != null && s.Session.ForegroundIsFullScreen && !s.Session.Locked)
            {
                // A full-screen window is not proof of a game, but on a desktop it is
                // proof that the user is doing one thing with their whole screen, and
                // the cost of being wrong here is a job rather than a match.
                v.WantsGpu = true; v.IsVeto = true;
                v.Reasons.Add("full-screen foreground window: " + s.Session.ForegroundProcess);
            }

            // --- TIER 2: load votes. Something is DRAWING. --------------------

            var load = new List<string>();
            if (s.Gpu != null && s.Gpu.Valid)
            {
                if (s.Gpu.ClockMemMhz > _c.MemClockBusyMhz)
                    load.Add(string.Format(CultureInfo.InvariantCulture,
                        "memory clock {0} MHz over the {1} MHz trip point",
                        s.Gpu.ClockMemMhz, _c.MemClockBusyMhz));

                if (!_c.IdlePStates.Contains(s.Gpu.PState))
                    load.Add("performance state " + s.Gpu.PState);

                if (s.Gpu.PowerWatts > _c.PowerBusyWatts)
                    load.Add(string.Format(CultureInfo.InvariantCulture,
                        "power draw {0:N1} W", s.Gpu.PowerWatts));

                if (s.Gpu.UtilGpu > _c.UtilGpuBusyPct)
                    load.Add(string.Format(CultureInfo.InvariantCulture,
                        "utilisation {0}%", s.Gpu.UtilGpu));
            }

            // Deliberately NOT a load vote: utilisation.decoder and the videodecode
            // engine. Video playback uses a fixed-function block, not the shaders,
            // and a Chatterbox job and a YouTube tab can share this card. Yielding to
            // a video would make the worker useless on a desktop that is nearly always
            // playing something. Measured at idle: decoder 0 per cent throughout, so
            // this is a decision about what to ignore, not about noise.
            if (s.CountersFresh && s.Util3d > _c.Util3dBusyPct)
                load.Add(string.Format(CultureInfo.InvariantCulture, "3D engine at {0:N0}%", s.Util3d));

            if (load.Count > 0)
            {
                _busyStreak++;
                if (_busyStreak >= _c.BusyConfirmSamples)
                {
                    v.WantsGpu = true;
                    foreach (string r in load) v.Reasons.Add(r);
                }
            }
            else
            {
                _busyStreak = 0;
            }
            if (v.IsVeto) _busyStreak = 0;

            // --- state machine ------------------------------------------------

            if (v.WantsGpu) _lastWanted = DateTime.UtcNow;

            if (v.WantsGpu)
                _state = WorkerState.Busy;
            else if ((DateTime.UtcNow - _lastWanted).TotalSeconds >= _c.ClearCooldownSeconds)
                _state = WorkerState.Available;
            else
                _state = WorkerState.Draining;   // clear, but inside the cooldown

            Last = v;
            return v;
        }

        /// The single question the rest of the program asks.
        public bool CanRun(Mode mode)
        {
            if (mode == Mode.Off) return false;
            if (mode == Mode.AlwaysOn) return true;
            return _state == WorkerState.Available;
        }

        public static string Describe(WorkerState st)
        {
            switch (st)
            {
                case WorkerState.Available: return "available";
                case WorkerState.Busy: return "busy";
                case WorkerState.Draining: return "cooling down";
                default: return "blocked";
            }
        }
    }
}
