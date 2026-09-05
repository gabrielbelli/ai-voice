// What the policy must decide, asserted against recorded samples.
//
// These tests drive the SAME Policy.Evaluate the tray runs. There is no second
// implementation and no mock: the only difference is that the snapshots come from
// a CSV instead of from nvidia-smi and the Windows performance counters. That is
// the whole reason Model.cs was split out of Signals.cs - this binary links
// Model, Config, Policy and Replay and touches no Windows API, no GPU and no
// registry, so it runs on any machine that can compile C#.
//
// Every test is named after the mistake it stops, because a test named after the
// function it calls tells you nothing when it goes red at two in the morning.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace AiVoice.Worker
{
    static class Tests
    {
        static int _failed, _passed;
        static string _fixtures;

        static void Main(string[] args)
        {
            _fixtures = args.Length > 0 ? args[0]
                : Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "fixtures");

            IdleDesktopEventuallyRunsJobs();
            IdleDesktopNeverLooksBusy();
            Session0RefusesForEverAndSaysWhyItCannotTell();
            SteamVetoesBeforeTheGpuMoves();
            PausedGameIsCaughtByVramAlone();
            VideoPlaybackDoesNotYield();
            Tier2NeedsExactlyThreeConsecutiveSamples();
            ValorantIsCaughtByVgcNotVgk();
            DeadSmiIsBlockedNotBusy();
            CooldownIsNinetySecondsNotEightyNine();
            AltTabOutOfAGameNeverBecomesAvailable();
            OurOwnJobIsNotEvidenceOfAUser();
            WithoutTheOwnPidExemptionItWouldOscillate();
            LockedScreenIsNotAFullScreenGame();
            EveryProcessOnTheRealIdleDesktopIsHarmless();
            ThresholdsClearTheMeasuredIdleCeiling();

            Console.WriteLine();
            Console.WriteLine("{0} passed, {1} failed", _passed, _failed);
            Environment.Exit(_failed == 0 ? 0 : 1);
        }

        // ------------------------------------------------------------ helpers ---

        static List<Snapshot> Load(string name)
        {
            return Replay.Load(Path.Combine(_fixtures, name));
        }

        /// Replay a fixture and return the state after every row.
        static List<WorkerState> Run(string name, out Policy p, int ownJobPid)
        {
            p = new Policy(new Config());
            var states = new List<WorkerState>();
            foreach (Snapshot s in Load(name))
            {
                s.OwnJobPid = ownJobPid;
                p.Evaluate(s);
                states.Add(p.State);
            }
            return states;
        }

        static List<WorkerState> Run(string name, out Policy p) { return Run(name, out p, -1); }

        static void Ok(bool cond, string what)
        {
            if (cond) { _passed++; Console.WriteLine("  pass  " + what); }
            else { _failed++; Console.WriteLine("  FAIL  " + what); }
        }

        static void Case(string name) { Console.WriteLine(); Console.WriteLine(name); }

        static int FirstIndexOf(List<WorkerState> st, WorkerState want)
        {
            for (int i = 0; i < st.Count; i++) if (st[i] == want) return i;
            return -1;
        }

        // -------------------------------------------------------------- tests ---

        /// The baseline claim: on a real idle desktop the worker eventually works.
        /// If this fails the whole project is pointless, because the machine never
        /// offers the GPU at all.
        static void IdleDesktopEventuallyRunsJobs()
        {
            Case("a genuinely idle desktop ends up available");
            Policy p;
            List<WorkerState> st = Run("idle_desktop.csv", out p);
            Ok(st[st.Count - 1] == WorkerState.Available,
                "150 s of measured idle ends Available (ended " +
                Policy.Describe(st[st.Count - 1]) + ")");
            Ok(p.Last.Reasons.Count == 0, "and gives no reason to yield");
            Ok(p.CanRun(Mode.Auto), "so Auto will take a job");
        }

        /// The measured idle baseline must not trip a single tier-2 vote. This is
        /// the test that fails if somebody "tidies up" UtilGpuBusyPct to 5, which
        /// looks reasonable and is unreachable: 150 of 150 measured samples read 5
        /// or 6 per cent on an idle desktop.
        static void IdleDesktopNeverLooksBusy()
        {
            Case("measured idle never reads as load");
            Policy p;
            List<WorkerState> st = Run("idle_desktop.csv", out p);
            Ok(FirstIndexOf(st, WorkerState.Busy) < 0,
                "no sample in the 150-sample baseline is Busy");
        }

        /// Fail-closed, and the reason has to name the actual problem. A worker
        /// that says "busy" when it means "I am blind" sends the user looking for
        /// a game that is not running.
        static void Session0RefusesForEverAndSaysWhyItCannotTell()
        {
            Case("an agent in session 0 refuses to run, for ever");
            Policy p;
            List<WorkerState> st = Run("session0_blocked.csv", out p);
            Ok(st.TrueForAll(delegate(WorkerState s) { return s == WorkerState.Blocked; }),
                "every one of 150 samples is Blocked");
            Ok(!p.CanRun(Mode.Auto), "Auto will not run");
            Ok(p.Last.ReasonText.Contains("session 0") && p.Last.ReasonText.Contains("cannot observe"),
                "and the reason names the session, not a game: " + p.Last.ReasonText);
        }

        /// The single most important latency claim in the design. Steam writes
        /// RunningAppID before the game renders its first frame, so the veto must
        /// fire on a sample whose GPU columns are still at the idle values.
        static void SteamVetoesBeforeTheGpuMoves()
        {
            Case("Steam vetoes on the first sample, before the GPU moves");
            List<Snapshot> rows = Load("steam_launch.csv");
            var p = new Policy(new Config());
            int firstBusy = -1;
            for (int i = 0; i < rows.Count; i++)
            {
                p.Evaluate(rows[i]);
                if (p.State == WorkerState.Busy && firstBusy < 0) firstBusy = i;
            }
            Ok(firstBusy == 100, "Busy on row 100, the row the appid appears (got " + firstBusy + ")");
            Ok(rows[100].Gpu.PState == "P5" && rows[100].Gpu.ClockMemMhz == 810,
                "and that row's GPU is still idle: P5 at 810 MHz");
            Ok(rows[100].Gpu.UtilGpu <= 6, "and utilisation is still 5-6%, so no load signal could have fired");
        }

        /// The case the naive policy gets wrong in the expensive direction, and the
        /// reason the OS counters are read at all: nvidia-smi reports used_memory
        /// as [N/A] per process on this machine, so this is invisible to it.
        static void PausedGameIsCaughtByVramAlone()
        {
            Case("a paused game holding VRAM is caught with no other signal");
            List<Snapshot> rows = Load("paused_game_vram.csv");
            var p = new Policy(new Config());
            int firstBusy = -1;
            for (int i = 0; i < rows.Count; i++)
            {
                p.Evaluate(rows[i]);
                if (p.State == WorkerState.Busy && firstBusy < 0) firstBusy = i;
            }
            Ok(firstBusy == 100, "Busy on row 100 (got " + firstBusy + ")");
            Ok(rows[100].Launchers.SteamRunningAppId == 0, "with no Steam appid at all");
            Ok(rows[100].Gpu.PState == "P5", "and the GPU at idle P5, drawing nothing");
            Ok(p.Last.ReasonText.Contains("TheFinals") && p.Last.ReasonText.Contains("MiB"),
                "and the reason names the process and its VRAM: " + p.Last.ReasonText);
        }

        /// Deliberate: a desktop is nearly always playing something, and a worker
        /// that yields to a YouTube tab is a worker that never runs.
        static void VideoPlaybackDoesNotYield()
        {
            Case("video playback does not take the GPU away");
            Policy p;
            List<WorkerState> st = Run("video_playback.csv", out p);
            Ok(FirstIndexOf(st, WorkerState.Busy) < 0, "no sample is Busy despite the decoder at 41%");
            Ok(st[st.Count - 1] == WorkerState.Available, "and it ends Available");
        }

        /// Off-by-one on the confirmation window is the classic way this goes
        /// wrong: two samples is jumpy, four is a second of the user's frames.
        static void Tier2NeedsExactlyThreeConsecutiveSamples()
        {
            Case("a tier-2 load vote needs exactly three consecutive samples");
            Policy p;
            List<WorkerState> st = Run("tier2_load_only.csv", out p);
            int firstBusy = FirstIndexOf(st, WorkerState.Busy);
            Ok(firstBusy == 62, "load starts at row 60, Busy at row 62 (got " + firstBusy + ")");
        }

        /// vgk is Running/System at all times on this machine and is therefore
        /// useless. vgc is Stopped/Manual and starts with the game. Both measured.
        static void ValorantIsCaughtByVgcNotVgk()
        {
            Case("Valorant is caught by the vgc service");
            Policy p;
            List<WorkerState> st = Run("valorant_vgc.csv", out p);
            int firstBusy = FirstIndexOf(st, WorkerState.Busy);
            Ok(firstBusy == 80, "Busy on the row vgc starts (got " + firstBusy + ")");
            Ok(p.Last.ReasonText.Contains("vgc"), "and says so: " + p.Last.ReasonText);
        }

        /// Blind is not the same as busy, and the distinction is what tells the
        /// user whether to go looking for a problem.
        static void DeadSmiIsBlockedNotBusy()
        {
            Case("a dead nvidia-smi stream blocks rather than blaming a game");
            Policy p;
            List<WorkerState> st = Run("smi_dead.csv", out p);
            Ok(st.TrueForAll(delegate(WorkerState s) { return s == WorkerState.Blocked; }),
                "every sample is Blocked, never Busy");
            Ok(!p.CanRun(Mode.Auto), "and Auto will not run");
        }

        /// The asymmetry is the safety property: instant to yield, slow to return.
        /// A test that only checked "eventually available" would pass with a
        /// one-second cooldown, which is the setting that costs somebody a match.
        static void CooldownIsNinetySecondsNotEightyNine()
        {
            Case("returning to available takes the full 90 s of continuous clear");
            var cfg = new Config();
            Policy p;
            List<WorkerState> st = Run("cooldown_90s.csv", out p);
            // Rows 0-29 carry the Steam appid; row 29 is the last busy sample.
            Ok(st[29] == WorkerState.Busy, "row 29 is the last Busy row");
            Ok(st[30] == WorkerState.Draining, "row 30 is Draining, not Available");
            int firstAvail = FirstIndexOf(st, WorkerState.Available);
            Ok(firstAvail == 29 + cfg.ClearCooldownSeconds,
                "Available at row " + (29 + cfg.ClearCooldownSeconds) + " exactly (got " + firstAvail + ")");
            Ok(st[firstAvail - 1] == WorkerState.Draining, "and the row before it is still Draining");
        }

        /// Alt-tabbing to a browser mid-match drops the full-screen flag and lets
        /// the GPU fall back towards idle. The game is still there. This is the
        /// fixture that fails if somebody rebuilds the policy on utilisation.
        static void AltTabOutOfAGameNeverBecomesAvailable()
        {
            Case("alt-tabbing out of a running game never frees the GPU");
            Policy p;
            List<WorkerState> st = Run("alt_tab_midgame.csv", out p);
            Ok(FirstIndexOf(st, WorkerState.Available) < 0,
                "no sample in 120 s is Available");
            Ok(st.TrueForAll(delegate(WorkerState s) { return s == WorkerState.Busy; }),
                "all 120 are Busy, including the 60 after the GPU fell back to idle");
        }

        /// Without this the policy oscillates: allocate, see our own VRAM, yield,
        /// free it, see nothing, allocate. For ever.
        static void OurOwnJobIsNotEvidenceOfAUser()
        {
            Case("our own job's 3.2 GiB is not a user");
            Policy p;
            List<WorkerState> st = Run("own_job_vram.csv", out p, 31337);
            Ok(st[st.Count - 1] == WorkerState.Available,
                "with our job at pid 31337 the card is still ours");
        }

        /// The other half of the same test. If the exemption silently stopped
        /// working, the test above would keep passing on a policy that had simply
        /// stopped reading VRAM at all.
        static void WithoutTheOwnPidExemptionItWouldOscillate()
        {
            Case("the same 3.2 GiB held by anything else is a veto");
            Policy p;
            List<WorkerState> st = Run("own_job_vram.csv", out p, -1);
            Ok(st.TrueForAll(delegate(WorkerState s) { return s == WorkerState.Busy; }),
                "every sample is Busy when the pid is not ours");
        }

        /// A locked machine is the safest possible moment to run: the user is
        /// demonstrably not there. The full-screen veto must not fire behind the
        /// lock screen, which is itself a full-screen window.
        static void LockedScreenIsNotAFullScreenGame()
        {
            Case("a locked screen is not a full-screen game");
            Policy p;
            List<WorkerState> st = Run("locked_fullscreen.csv", out p);
            Ok(st[st.Count - 1] == WorkerState.Available,
                "LogonUI full-screen behind a lock does not veto");
        }

        /// The allowlist and the 512 MiB trip point, against the real measured
        /// process list rather than against a story about it. Fifteen processes
        /// were holding GPU memory on spring's idle desktop; not one of them may
        /// read as a game.
        static void EveryProcessOnTheRealIdleDesktopIsHarmless()
        {
            Case("no process on the real idle desktop trips the VRAM veto");
            var cfg = new Config();
            var p = new Policy(cfg);
            var s = new Snapshot();
            s.At = DateTime.UtcNow;
            s.OwnJobPid = -1;

            string path = Path.Combine(_fixtures, "_recorded_vram_spring.csv");
            string[] lines = File.ReadAllLines(path);
            double biggest = 0; string biggestName = "";
            for (int i = 1; i < lines.Length; i++)
            {
                string[] f = Replay.SplitCsv(lines[i]);
                if (f.Length < 3) continue;
                var u = new ProcessGpuUse();
                u.Pid = int.Parse(f[0], CultureInfo.InvariantCulture);
                u.Name = f[1];
                u.DedicatedMiB = double.Parse(f[2], CultureInfo.InvariantCulture);
                s.GpuProcesses.Add(u);
                if (u.DedicatedMiB > biggest) { biggest = u.DedicatedMiB; biggestName = u.Name; }
            }

            Ok(s.GpuProcesses.Count == 15, "read all 15 measured processes");
            List<ProcessGpuUse> foreign = p.ForeignVram(s);
            Ok(foreign.Count == 0, "none of them is foreign; got " +
                (foreign.Count == 0 ? "none" : foreign[0].Name));
            Ok(biggest < cfg.ForeignVramBusyMiB,
                "the largest measured consumer (" + biggestName + " at " +
                biggest.ToString("N1", CultureInfo.InvariantCulture) +
                " MiB) is under the " + cfg.ForeignVramBusyMiB.ToString("N0", CultureInfo.InvariantCulture) +
                " MiB trip point");
        }

        /// Guards the defaults themselves against a well-meaning edit. Each of
        /// these is a margin over a measured idle ceiling, and the measurement is
        /// in tests/fixtures/_recorded_idle_spring.csv where anybody can check it.
        static void ThresholdsClearTheMeasuredIdleCeiling()
        {
            Case("every threshold clears the measured idle ceiling");
            var cfg = new Config();
            List<Snapshot> idle = Load("idle_desktop.csv");

            int maxUtil = idle.Max(delegate(Snapshot s) { return s.Gpu.UtilGpu; });
            double maxPower = idle.Max(delegate(Snapshot s) { return s.Gpu.PowerWatts; });
            int maxClk = idle.Max(delegate(Snapshot s) { return s.Gpu.ClockMemMhz; });
            double max3d = idle.Max(delegate(Snapshot s) { return s.Util3d; });

            Ok(cfg.UtilGpuBusyPct > maxUtil,
                "utilisation trip " + cfg.UtilGpuBusyPct + "% clears measured idle max " + maxUtil + "%");
            Ok(cfg.PowerBusyWatts > maxPower,
                "power trip " + cfg.PowerBusyWatts + " W clears measured idle max " +
                maxPower.ToString("N2", CultureInfo.InvariantCulture) + " W");
            Ok(cfg.MemClockBusyMhz > maxClk,
                "memory clock trip " + cfg.MemClockBusyMhz + " MHz clears measured idle " + maxClk + " MHz");
            Ok(cfg.Util3dBusyPct > max3d,
                "3D engine trip " + cfg.Util3dBusyPct + "% clears measured idle " +
                max3d.ToString("N2", CultureInfo.InvariantCulture) + "%");
            Ok(Array.IndexOf(cfg.IdlePStates, "P5") >= 0,
                "P5 is treated as idle, which is what all 150 measured samples were");
        }
    }
}
