// Turning a recorded CSV back into Snapshots.
//
// WHY THE FIXTURE FORMAT AND THE CALIBRATION FORMAT ARE THE SAME FORMAT. This
// reads exactly what `ai-voice-worker.exe --calibrate game.csv 20` writes. That
// is the whole point: when the user runs the agent through twenty minutes of a
// real match, the CSV it produces drops straight into tests/fixtures/ and becomes
// a regression test, with no conversion step and nobody transcribing numbers.
//
// The busy side of every tier-2 threshold is currently unmeasured - read-only
// probing on somebody's gaming PC cannot generate GPU load - so the busy fixtures
// in tests/fixtures/ are SYNTHETIC and are named to say so. Replacing them with a
// recorded match, or with a run of runtime/loadgen.py, is the point at which the
// thresholds stop being reasoned and start being measured.
//
// NOTE ON THE VRAM COLUMNS. The recorded row carries the top RAW GPU-memory
// consumer - name, pid and MiB, before any allowlist is applied - rather than a
// pre-filtered "foreign" figure. That is what lets a replay exercise
// Policy.ForeignVram itself: a fixture whose top consumer is dwm at 169 MiB must
// come back clear because dwm is allowlisted, and a fixture whose top consumer is
// cs2 at 3000 MiB must veto. Recording the post-filter answer would have made
// those two rows indistinguishable and the test worthless.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;

namespace AiVoice.Worker
{
    public static class Replay
    {
        public const string Header =
            "iso_time,state,util_gpu,util_mem,enc,dec,mem_clk_mhz,sm_clk_mhz,pstate,power_w," +
            "mem_used_mib,eng_3d,eng_decode,eng_encode,gpu_healthy,counters_fresh," +
            "own_session,console_session,locked,input_idle_s,fullscreen,fg_process," +
            "steam_appid,steam_appname,vgc,game_procs,vram_top_pid,vram_top_name,vram_top_mib,reason";

        public static List<Snapshot> Load(string path)
        {
            var rows = new List<Snapshot>();
            string[] lines = File.ReadAllLines(path);
            if (lines.Length == 0) return rows;

            string[] cols = SplitCsv(lines[0]);
            var idx = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < cols.Length; i++) idx[cols[i].Trim()] = i;

            for (int li = 1; li < lines.Length; li++)
            {
                if (lines[li].Trim().Length == 0) continue;
                string[] f = SplitCsv(lines[li]);
                rows.Add(Row(f, idx));
            }
            return rows;
        }

        static Snapshot Row(string[] f, Dictionary<string, int> idx)
        {
            var s = new Snapshot();
            s.At = Dt(Get(f, idx, "iso_time"));

            var g = new GpuSample();
            g.At = s.At;
            g.UtilGpu = I(Get(f, idx, "util_gpu"));
            g.UtilMem = I(Get(f, idx, "util_mem"));
            g.UtilEncoder = I(Get(f, idx, "enc"));
            g.UtilDecoder = I(Get(f, idx, "dec"));
            g.ClockMemMhz = I(Get(f, idx, "mem_clk_mhz"));
            g.ClockSmMhz = I(Get(f, idx, "sm_clk_mhz"));
            g.PState = Get(f, idx, "pstate");
            g.PowerWatts = D(Get(f, idx, "power_w"));
            g.MemUsedMiB = I(Get(f, idx, "mem_used_mib"));
            g.Valid = g.PState.Length > 0;
            s.Gpu = g;

            s.Util3d = D(Get(f, idx, "eng_3d"));
            s.UtilVideoDecode = D(Get(f, idx, "eng_decode"));
            s.UtilVideoEncode = D(Get(f, idx, "eng_encode"));

            // Default true when the column is absent, so a trace recorded by an
            // older build still replays as "the stream was fine".
            s.GpuHealthy = B(Get(f, idx, "gpu_healthy"), true);
            s.CountersFresh = B(Get(f, idx, "counters_fresh"), true);

            var ss = new SessionSignals();
            ss.OwnSessionId = (uint)I(Get(f, idx, "own_session"));
            ss.ConsoleSessionId = (uint)I(Get(f, idx, "console_session"));
            ss.RunningInConsoleSession = ss.OwnSessionId == ss.ConsoleSessionId;
            ss.Locked = B(Get(f, idx, "locked"), false);
            ss.InputIdleSeconds = I(Get(f, idx, "input_idle_s"));
            ss.ForegroundIsFullScreen = B(Get(f, idx, "fullscreen"), false);
            ss.ForegroundProcess = Get(f, idx, "fg_process");
            ss.ForegroundTitle = "";
            s.Session = ss;

            var l = new LauncherSignals();
            l.SteamRunningAppId = I(Get(f, idx, "steam_appid"));
            l.SteamRunningAppName = Get(f, idx, "steam_appname");
            l.ValorantAntiCheatActive = B(Get(f, idx, "vgc"), false);
            string gp = Get(f, idx, "game_procs");
            if (gp.Length > 0)
                foreach (string one in gp.Split(' '))
                    if (one.Trim().Length > 0) l.GameProcesses.Add(one.Trim());
            s.Launchers = l;

            string vn = Get(f, idx, "vram_top_name");
            double vm = D(Get(f, idx, "vram_top_mib"));
            if (vn.Length > 0 && vm > 0)
            {
                var u = new ProcessGpuUse();
                u.Pid = I(Get(f, idx, "vram_top_pid"));
                u.Name = vn;
                u.DedicatedMiB = vm;
                s.GpuProcesses.Add(u);
            }
            return s;
        }

        static string Get(string[] f, Dictionary<string, int> idx, string name)
        {
            int i;
            if (!idx.TryGetValue(name, out i)) return "";
            if (i >= f.Length) return "";
            return f[i] == null ? "" : f[i].Trim();
        }

        static int I(string s)
        {
            int v;
            return int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out v) ? v : 0;
        }

        static double D(string s)
        {
            double v;
            return double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out v) ? v : 0;
        }

        static bool B(string s, bool dflt)
        {
            if (s.Length == 0) return dflt;
            return s == "1" || string.Equals(s, "true", StringComparison.OrdinalIgnoreCase);
        }

        static DateTime Dt(string s)
        {
            DateTime v;
            if (DateTime.TryParse(s, CultureInfo.InvariantCulture,
                DateTimeStyles.AdjustToUniversal | DateTimeStyles.AssumeUniversal, out v)) return v;
            return DateTime.MinValue;
        }

        /// Minimal CSV, matching what Program.Csv writes: commas separate, double
        /// quotes wrap a field containing a comma, and "" is a literal quote.
        public static string[] SplitCsv(string line)
        {
            var outp = new List<string>();
            var sb = new StringBuilder();
            bool q = false;
            for (int i = 0; i < line.Length; i++)
            {
                char c = line[i];
                if (q)
                {
                    if (c == '"')
                    {
                        if (i + 1 < line.Length && line[i + 1] == '"') { sb.Append('"'); i++; }
                        else q = false;
                    }
                    else sb.Append(c);
                }
                else if (c == '"') q = true;
                else if (c == ',') { outp.Add(sb.ToString()); sb.Length = 0; }
                else sb.Append(c);
            }
            outp.Add(sb.ToString());
            return outp.ToArray();
        }
    }
}
