// Configuration and its defaults.
//
// EVERY DEFAULT BELOW IS A MEASUREMENT OR A MARGIN ON ONE. The measurements were
// taken on spring (RTX 3070, driver 610.47, Windows 11 Pro 10.0.26200) on
// 2026-09-05, with the desktop up and the machine untouched: dwm, explorer,
// ShellHost, SearchHost, StartMenuExperienceHost, two msedgewebview2, Raycast,
// steamwebhelper, CamoStudio and Riot Vanguard's vgtray all resident on the GPU.
// That is a realistic floor, not a clean-room one, which is the point.
//
// The 92-sample, 90-second idle baseline (probe p5) was:
//
//   utilisation.gpu   min 5     max 6      mean 5.13   p95 6
//   utilisation.mem   min 3     max 4      mean 3.73   p95 4
//   encoder/decoder   0 throughout
//   clocks.mem        810 MHz, ZERO variance across all 92 samples
//   pstate            P5 for all 92 samples
//   power.draw        min 33.68 W  max 35.13 W  mean 34.59 W   (range 1.45 W)
//   memory.used       416 MiB, zero variance
//
// Note what that rules out: "utilisation.gpu below 5 per cent" can never be
// satisfied on this machine. The naive threshold is not merely imprecise here, it
// is unreachable. The clock, the pstate and the power figure all separate far
// more cleanly than utilisation does, and that is why the policy leans on them.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;

namespace AiVoice.Worker
{
    public class Config
    {
        // -- GPU load thresholds -------------------------------------------------

        /// Measured idle memory clock: 810 MHz, unchanging across 92 samples.
        /// The GDDR6 on a 3070 sits here whenever nothing is rendering and steps to
        /// several thousand MHz the moment a 3D application takes the adapter. Any
        /// value above this floor by a wide margin is real work, so the threshold is
        /// set well clear of it rather than just above it.
        public int MemClockBusyMhz = 1500;

        // A WARNING ABOUT THIS SIGNAL, found by reading what else runs on spring.
        //
        // The machine runs C:\gpu-clocklock.ps1 on a loop, which applies
        // `nvidia-smi -lgc 1800` and `-pl 270` every ten minutes (the user's own
        // script; see GAB-578 for why it re-applies rather than firing once).
        // That LOCKS THE CORE CLOCK, and it is why clocks.sm reads exactly 1800 in
        // all 150 samples of the idle baseline. The core clock is therefore a
        // constant on this machine and worthless as a load signal - which is why
        // the policy reads clocks.MEM instead, and clocks.mem is not what -lgc
        // pins. Measured idle memory clock is 810 MHz with zero variance.
        //
        // If anyone ever adds `-lmc` to that script, this signal dies silently:
        // the memory clock would then be a constant too and MemClockBusyMhz would
        // never fire again. The tier-1 vetoes would carry on working, so the
        // failure would be a quiet loss of sensitivity rather than a visible
        // break. Worth knowing before debugging it from scratch.

        /// P5 for 100 per cent of the idle baseline. P0 and P2 are the performance
        /// states; P8 is deeper idle. Treating anything faster than P5 as load gives
        /// a signal that moves before utilisation does, because the driver raises the
        /// pstate in anticipation of work rather than in response to it.
        public string[] IdlePStates = new string[] { "P5", "P8", "P12" };

        /// Idle power measured 33.68-35.13 W. Gaming load on a 3070 is an order of
        /// magnitude above that. 70 W leaves 35 W of headroom over the measured idle
        /// ceiling - roughly double the entire idle draw - so it cannot be tripped by
        /// desktop compositing, and still fires long before a game reaches frame rate.
        public double PowerBusyWatts = 70.0;

        /// Utilisation is the weakest signal here and is used only as one of several
        /// load votes. Idle p95 was 6 per cent, so 25 is four times the measured
        /// noise floor.
        public int UtilGpuBusyPct = 25;

        /// Sum of GPU Engine "3d" utilisation across processes. Measured 0.95 at idle.
        public double Util3dBusyPct = 20.0;

        // -- The paused-game veto ------------------------------------------------

        /// Dedicated GPU memory, in MiB, held by any process outside the allowlist
        /// that means "a game is resident, whether or not it is currently drawing".
        ///
        /// The largest non-allowlisted consumer measured at idle was CamoStudio at
        /// 112.3 MiB and the largest of all was dwm at 172.4 MiB (probe p2). 512 MiB
        /// is three times the largest thing measured on an idle desktop and far below
        /// what any game holds. This is the check that catches the case the naive
        /// policy gets wrong in the expensive direction: a game paused at a menu draws
        /// nothing but still owns its VRAM and expects to resume instantly.
        public double ForeignVramBusyMiB = 512.0;

        /// Processes permitted to hold GPU memory without being read as a game. Every
        /// name here was observed resident on spring's idle desktop (probes p1, p2).
        public string[] VramAllowlist = new string[] {
            "dwm", "explorer", "csrss", "ShellHost", "SearchHost",
            "StartMenuExperienceHost", "ShellExperienceHost", "ApplicationFrameHost",
            "SystemSettings", "CrossDeviceResume", "TextInputHost", "SearchApp",
            // pid 4, the kernel. Measured holding 4.0 MiB on spring's idle desktop
            // 2026-09-05. It was missing from this list, which was harmless only
            // because 4 MiB is three orders of magnitude under the trip point.
            "System",
            "msedgewebview2", "Raycast", "steamwebhelper", "steam", "steamservice",
            "CamoStudio", "vgtray", "LogonUI", "sihost", "ctfmon"
        };

        /// Process names whose presence alone means yield. Derived from what is
        /// installed on spring (probe p6: Steam library plus C:\Riot Games).
        public string[] GameProcessNames = new string[] {
            "cs2", "RainbowSix", "RainbowSixGame", "ReadyOrNot", "GRB",
            "VALORANT-Win64-Shipping", "VALORANT", "LeagueOfLegends"
        };

        // -- Timing --------------------------------------------------------------

        /// How many consecutive 1 Hz GPU samples must show load before yielding on
        /// GPU evidence alone. Three seconds. Short because the cost of being wrong
        /// in this direction is one abandoned job; long enough that a single frame of
        /// a window animation does not trip it.
        public int BusyConfirmSamples = 3;

        /// How long every signal must stay clear before returning to Available.
        ///
        /// This is the asymmetry that makes the whole design safe. Ninety seconds is
        /// longer than a level load, longer than alt-tabbing to a browser mid-match,
        /// and longer than the gap between two rounds. A false "busy" costs one job.
        /// A false "idle" costs the user their game, so the cooldown is deliberately
        /// far longer than the detection window.
        public int ClearCooldownSeconds = 90;

        /// Poll interval for the cheap Win32 and registry signals.
        public int FastPollMs = 1000;

        /// Poll interval for the performance counters.
        ///
        /// Two seconds, not five. The counters are the ONLY way an unknown game -
        /// one with no Steam appid and no name in GameProcessNames - is detected,
        /// via the VRAM veto, so this interval is that path's entire detection
        /// latency. Read in-process through PDH the whole set costs 98 ms
        /// (probe p7), against Get-Counter's measured 1069-1851 ms for the same
        /// data, which is the reason this program is not a PowerShell script.
        /// Paying three extra seconds of the user's frame time to save 98 ms of
        /// one core every two seconds is the wrong way round; at 2000 ms the
        /// counters cost under 5 per cent of one core and the unknown-game path
        /// drops from a ~7 s worst case to ~3 s.
        public int CounterPollMs = 2000;

        /// Grace period given to a running job when the policy says yield, before
        /// the job object is closed and the whole tree dies.
        ///
        /// Two seconds, and the kill is the NORMAL path rather than the exception.
        /// The reason is a hard limit rather than a preference: Chatterbox's
        /// generate() has no interruption point inside it (services/tts-long's
        /// synth.py says so in its own docstring, and DELETE /jobs/{id} documents
        /// the same limit for the local CPU worker). So the child cannot abort
        /// mid-chunk. What it can do is decline to start another chunk and drop
        /// the one in flight, which is what the YIELD line on stdin asks for.
        ///
        /// Waiting politely for a chunk to finish would cost a bounded but
        /// unmeasured number of seconds of the user's frame time, and the stated
        /// constraint is that the user has full priority. The cooperative stage
        /// exists only to save a chunk that is already computed and one write away
        /// from being delivered; anything slower than that gets killed. Killing
        /// mid-CUDA-kernel is safe by construction - process teardown returns the
        /// context and the allocation to the driver, and a chunk only becomes real
        /// when the whole array is delivered, so there is no partial write to
        /// corrupt.
        public int YieldGraceSeconds = 2;

        // -- Job runtime ---------------------------------------------------------

        /// Command the agent runs when it has the GPU. Empty means detection only,
        /// which is the state this proof of concept ships in.
        public string JobCommand = "";
        public string JobArguments = "";
        public string JobWorkingDir = "";

        /// The tray mode, persisted. Auto | AlwaysOn | Off.
        public string StartMode = "Auto";

        /// Where the agent appends one line per state change. This is the file the
        /// week-long false-idle trial is read out of, so it is plain text with a
        /// timestamp and a reason on every line and nothing that needs a parser.
        public string LogPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "ai-voice-worker", "worker.log");

        /// Written by runtime/provision.ps1 while it downloads. One line of plain
        /// text, e.g. "torch, 1.4 of 2.4 GB". Present means setup is in progress;
        /// absent means it is not. A file rather than a pipe because provisioning
        /// is a separate script the user can run on its own.
        public string ProvisionStatusPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "ai-voice-worker", "runtime", "provision.status");

        public string ConfigPath = "";

        public string StatePath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "ai-voice-worker", "state.json");

        // -- Loading -------------------------------------------------------------

        /// Deliberately an INI, not JSON. There is no JSON parser in the .NET
        /// Framework subset available to the legacy csc.exe without dragging in
        /// System.Web.Extensions, and a settings file a human edits by hand at 2am
        /// should not be able to fail on a trailing comma.
        public static Config Load(string path)
        {
            var c = new Config();
            c.ConfigPath = path == null ? "" : path;
            if (path == null || !File.Exists(path)) return c;
            foreach (string raw in File.ReadAllLines(path))
            {
                string line = raw.Trim();
                if (line.Length == 0 || line.StartsWith("#") || line.StartsWith(";")) continue;
                int eq = line.IndexOf('=');
                if (eq <= 0) continue;
                string k = line.Substring(0, eq).Trim().ToLowerInvariant();
                string v = line.Substring(eq + 1).Trim();
                try { Apply(c, k, v); }
                catch (Exception) { /* a bad line must not stop the agent starting */ }
            }
            return c;
        }

        static void Apply(Config c, string k, string v)
        {
            switch (k)
            {
                case "memclockbusymhz": c.MemClockBusyMhz = int.Parse(v, CultureInfo.InvariantCulture); break;
                case "powerbusywatts": c.PowerBusyWatts = double.Parse(v, CultureInfo.InvariantCulture); break;
                case "utilgpubusypct": c.UtilGpuBusyPct = int.Parse(v, CultureInfo.InvariantCulture); break;
                case "util3dbusypct": c.Util3dBusyPct = double.Parse(v, CultureInfo.InvariantCulture); break;
                case "foreignvrambusymib": c.ForeignVramBusyMiB = double.Parse(v, CultureInfo.InvariantCulture); break;
                case "busyconfirmsamples": c.BusyConfirmSamples = int.Parse(v, CultureInfo.InvariantCulture); break;
                case "clearcooldownseconds": c.ClearCooldownSeconds = int.Parse(v, CultureInfo.InvariantCulture); break;
                case "fastpollms": c.FastPollMs = int.Parse(v, CultureInfo.InvariantCulture); break;
                case "counterpollms": c.CounterPollMs = int.Parse(v, CultureInfo.InvariantCulture); break;
                case "yieldgraceseconds": c.YieldGraceSeconds = int.Parse(v, CultureInfo.InvariantCulture); break;
                case "jobcommand": c.JobCommand = v; break;
                case "jobarguments": c.JobArguments = v; break;
                case "jobworkingdir": c.JobWorkingDir = v; break;
                case "gameprocessnames": c.GameProcessNames = v.Split(','); break;
                case "vramallowlist": c.VramAllowlist = v.Split(','); break;
                case "idlepstates": c.IdlePStates = v.Split(','); break;
                case "startmode": c.StartMode = v; break;
                case "logpath": c.LogPath = v; break;
                case "statepath": c.StatePath = v; break;
                case "provisionstatuspath": c.ProvisionStatusPath = v; break;
            }
        }
    }

    /// A JSON writer sized to this program's needs and no larger.
    /// WHY hand-rolled: the agent must build with the in-box csc.exe against the
    /// in-box reference assemblies, with no NuGet restore and no network. Pulling
    /// System.Web.Extensions in for one status object is a worse trade than
    /// thirty lines of escaping.
    public static class Json
    {
        public static string Esc(string s)
        {
            if (s == null) return "null";
            var sb = new StringBuilder("\"");
            foreach (char ch in s)
            {
                if (ch == '"') sb.Append("\\\"");
                else if (ch == '\\') sb.Append("\\\\");
                else if (ch == '\n') sb.Append("\\n");
                else if (ch == '\r') sb.Append("\\r");
                else if (ch == '\t') sb.Append("\\t");
                else if (ch < 0x20) sb.Append("\\u").Append(((int)ch).ToString("x4"));
                else sb.Append(ch);
            }
            return sb.Append('"').ToString();
        }

        public static string Num(double d)
        {
            return d.ToString("0.###", CultureInfo.InvariantCulture);
        }

        public static string Obj(params string[] pairs)
        {
            return "{" + string.Join(",", pairs) + "}";
        }

        public static string P(string key, string rawValue)
        {
            return Esc(key) + ":" + rawValue;
        }
    }
}
