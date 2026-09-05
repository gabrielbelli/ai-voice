// Entry point.
//
// Modes exist so the two claims this proof of concept has to prove can each be
// checked without a tray icon and without a human at the console:
//
//   --once       print one full signal snapshot as JSON and exit
//   --watch N    print one line per second for N seconds and exit
//   --calibrate  append every signal to a CSV for N minutes and exit
//   --tray       the actual product (default)
//
// --watch and --calibrate exit on their own. That is not decoration: it is what
// makes the thing safe to drive over SSH against somebody's gaming PC without
// leaving a process behind.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows.Forms;

namespace AiVoice.Worker
{
    static class Program
    {
        [DllImport("kernel32.dll")] static extern bool AttachConsole(int pid);
        const int ATTACH_PARENT_PROCESS = -1;

        /// The binary is linked as a Windows subsystem executable so that --tray
        /// does not flash a console window on the user's desktop every login. The
        /// cost is that the CLI modes start with no stdout, so they borrow the
        /// console of whatever shell launched them. Reopening the standard handles
        /// afterwards is required: .NET captured the (invalid) originals during
        /// static initialisation, before AttachConsole ran.
        static void BorrowParentConsole()
        {
            try
            {
                if (!AttachConsole(ATTACH_PARENT_PROCESS)) return;
                var so = new StreamWriter(Console.OpenStandardOutput());
                so.AutoFlush = true;
                Console.SetOut(so);
                var se = new StreamWriter(Console.OpenStandardError());
                se.AutoFlush = true;
                Console.SetError(se);
            }
            catch (Exception) { }
        }

        [STAThread]
        static int Main(string[] args)
        {
            string configPath = Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory, "worker.ini");
            var rest = new List<string>();
            string mode = "--tray";
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "--config" && i + 1 < args.Length) { configPath = args[++i]; continue; }
                if (args[i].StartsWith("--") && mode == "--tray" && args[i] != "--tray") { mode = args[i]; continue; }
                rest.Add(args[i]);
            }

            Config cfg = Config.Load(configPath);

            if (mode != "--tray") BorrowParentConsole();

            if (mode == "--once") return Once(cfg);
            if (mode == "--watch") return Watch(cfg, rest.Count > 0 ? int.Parse(rest[0], CultureInfo.InvariantCulture) : 20);
            if (mode == "--calibrate") return Calibrate(cfg, rest.Count > 0 ? rest[0] : "calibration.csv",
                                                        rest.Count > 1 ? int.Parse(rest[1], CultureInfo.InvariantCulture) : 10);
            if (mode == "--help" || mode == "-h") { Usage(); return 0; }
            return Tray(cfg);
        }

        static void Usage()
        {
            Console.WriteLine("ai-voice GPU worker");
            Console.WriteLine("  --once                    one JSON snapshot, then exit");
            Console.WriteLine("  --watch [seconds]         one line per second, then exit (default 20)");
            Console.WriteLine("  --calibrate [csv] [mins]  log every signal to CSV, then exit (default 10)");
            Console.WriteLine("  --tray                    tray icon (default)");
            Console.WriteLine("  --config <path>           settings file (default worker.ini beside the exe)");
        }

        static Agent Spin(Config cfg, int settleSeconds)
        {
            var a = new Agent(cfg);
            a.Log = delegate(string s) { Console.Error.WriteLine("[agent] " + s); };
            a.Start();
            // The GPU stream emits at 1 Hz and the engine counters are rate counters
            // needing two reads, so nothing is trustworthy for the first few seconds.
            Thread.Sleep(settleSeconds * 1000);
            return a;
        }

        static int Once(Config cfg)
        {
            using (Agent a = Spin(cfg, 9))
            {
                Console.WriteLine(a.StatusJson());
                return a.Policy.CanRun(a.Mode) ? 0 : 1;
            }
        }

        static int Watch(Config cfg, int seconds)
        {
            using (Agent a = Spin(cfg, 9))
            {
                Console.WriteLine("{0,-8} {1,-13} {2,-5} {3,-6} {4,-7} {5,-7} {6,-6} {7,-6} {8}",
                    "time", "state", "util", "power", "memclk", "pstate", "eng3d", "vram", "reason");
                for (int i = 0; i < seconds; i++)
                {
                    Snapshot s = a.Current;
                    GpuSample g = s.Gpu;
                    Console.WriteLine("{0,-8} {1,-13} {2,-5} {3,-6} {4,-7} {5,-7} {6,-6} {7,-6} {8}",
                        DateTime.Now.ToString("HH:mm:ss", CultureInfo.InvariantCulture),
                        Policy.Describe(a.Policy.State),
                        g == null ? "-" : g.UtilGpu.ToString(CultureInfo.InvariantCulture),
                        g == null ? "-" : g.PowerWatts.ToString("0.0", CultureInfo.InvariantCulture),
                        g == null ? "-" : g.ClockMemMhz.ToString(CultureInfo.InvariantCulture),
                        g == null ? "-" : g.PState,
                        s.CountersFresh ? s.Util3d.ToString("0.0", CultureInfo.InvariantCulture) : "-",
                        s.ForeignVram.Count.ToString(CultureInfo.InvariantCulture),
                        a.Policy.Last.ReasonText);
                    Thread.Sleep(1000);
                }
                return 0;
            }
        }

        /// The honest answer to a question this investigation could not settle from
        /// a read-only SSH session: what the signals look like WITH A GAME RUNNING.
        ///
        /// Nothing may be installed on spring and nothing may be started that
        /// outlives a command, so no load could be generated to measure the busy side
        /// of every threshold. The idle side is measured to four significant figures
        /// (probe p5, 92 samples); the busy side is inferred. This mode closes that
        /// gap by having the user run it once while they play, which turns the
        /// remaining assumption into a measurement without anyone guessing.
        static int Calibrate(Config cfg, string csv, int minutes)
        {
            using (Agent a = Spin(cfg, 3))
            {
                bool fresh = !File.Exists(csv);
                using (var w = new StreamWriter(csv, true, new UTF8Encoding(false)))
                {
                    if (fresh)
                        w.WriteLine("iso_time,state,util_gpu,util_mem,enc,dec,mem_clk_mhz,sm_clk_mhz," +
                                    "pstate,power_w,mem_used_mib,eng_3d,eng_decode,eng_encode," +
                                    "locked,input_idle_s,fullscreen,fg_process,steam_appid,vgc," +
                                    "foreign_vram_top_mib,foreign_vram_top_name,reason");
                    int n = minutes * 60;
                    for (int i = 0; i < n; i++)
                    {
                        Snapshot s = a.Current;
                        GpuSample g = s.Gpu;
                        ProcessGpuUse top = s.ForeignVram.Count > 0 ? s.ForeignVram[0] : null;
                        w.WriteLine(string.Join(",", new string[] {
                            DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture),
                            Policy.Describe(a.Policy.State),
                            g == null ? "" : g.UtilGpu.ToString(CultureInfo.InvariantCulture),
                            g == null ? "" : g.UtilMem.ToString(CultureInfo.InvariantCulture),
                            g == null ? "" : g.UtilEncoder.ToString(CultureInfo.InvariantCulture),
                            g == null ? "" : g.UtilDecoder.ToString(CultureInfo.InvariantCulture),
                            g == null ? "" : g.ClockMemMhz.ToString(CultureInfo.InvariantCulture),
                            g == null ? "" : g.ClockSmMhz.ToString(CultureInfo.InvariantCulture),
                            g == null ? "" : g.PState,
                            g == null ? "" : g.PowerWatts.ToString("0.00", CultureInfo.InvariantCulture),
                            g == null ? "" : g.MemUsedMiB.ToString(CultureInfo.InvariantCulture),
                            s.Util3d.ToString("0.00", CultureInfo.InvariantCulture),
                            s.UtilVideoDecode.ToString("0.00", CultureInfo.InvariantCulture),
                            s.UtilVideoEncode.ToString("0.00", CultureInfo.InvariantCulture),
                            s.Session == null ? "" : (s.Session.Locked ? "1" : "0"),
                            s.Session == null ? "" : s.Session.InputIdleSeconds.ToString(CultureInfo.InvariantCulture),
                            s.Session == null ? "" : (s.Session.ForegroundIsFullScreen ? "1" : "0"),
                            s.Session == null ? "" : Csv(s.Session.ForegroundProcess),
                            s.Launchers == null ? "" : s.Launchers.SteamRunningAppId.ToString(CultureInfo.InvariantCulture),
                            s.Launchers == null ? "" : (s.Launchers.ValorantAntiCheatActive ? "1" : "0"),
                            top == null ? "0" : top.DedicatedMiB.ToString("0", CultureInfo.InvariantCulture),
                            top == null ? "" : Csv(top.Name),
                            Csv(a.Policy.Last.ReasonText)
                        }));
                        w.Flush();   // survive a hard reboot mid-game, which is the point
                        Thread.Sleep(1000);
                    }
                }
                Console.Error.WriteLine("[agent] wrote " + csv);
                return 0;
            }
        }

        static string Csv(string s)
        {
            if (s == null) return "";
            if (s.IndexOfAny(new char[] { ',', '"', '\n' }) < 0) return s;
            return "\"" + s.Replace("\"", "\"\"") + "\"";
        }

        static int Tray(Config cfg)
        {
            var a = new Agent(cfg);
            a.Start();
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new TrayApp(a));
            return 0;
        }
    }
}
