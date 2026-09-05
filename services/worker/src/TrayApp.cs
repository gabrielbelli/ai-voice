// The status bar icon.
//
// Three modes, as asked for: Auto, Always on, Off. Auto is the interesting one
// and everything above exists to make it trustworthy; Always on and Off are there
// because a user who does not trust Auto needs somewhere to go that is not
// "uninstall it".

using System;
using System.Drawing;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows.Forms;
using Microsoft.Win32;

namespace AiVoice.Worker
{
    public class TrayApp : ApplicationContext
    {
        [DllImport("user32.dll", SetLastError = true)]
        static extern bool DestroyIcon(IntPtr handle);

        readonly Agent _agent;
        readonly NotifyIcon _icon;
        readonly ToolStripMenuItem _status, _detail, _auto, _always, _off;
        readonly Timer _tick;
        Icon _current;

        public TrayApp(Agent agent)
        {
            _agent = agent;

            _status = new ToolStripMenuItem("starting...");
            _status.Enabled = false;
            _detail = new ToolStripMenuItem("");
            _detail.Enabled = false;

            _auto = new ToolStripMenuItem("Auto", null, delegate { SetMode(Mode.Auto); });
            _always = new ToolStripMenuItem("Always on", null, delegate { SetMode(Mode.AlwaysOn); });
            _off = new ToolStripMenuItem("Off", null, delegate { SetMode(Mode.Off); });

            var menu = new ContextMenuStrip();
            menu.Items.Add(_status);
            menu.Items.Add(_detail);
            menu.Items.Add(new ToolStripSeparator());
            menu.Items.Add(_auto);
            menu.Items.Add(_always);
            menu.Items.Add(_off);
            menu.Items.Add(new ToolStripSeparator());
            menu.Items.Add(new ToolStripMenuItem("Copy diagnostics", null, delegate
            {
                try { Clipboard.SetText(_agent.StatusJson()); } catch (Exception) { }
            }));
            menu.Items.Add(new ToolStripMenuItem("Exit", null, delegate { Quit(); }));

            _icon = new NotifyIcon();
            _icon.ContextMenuStrip = menu;
            _icon.Visible = true;
            SetIcon(WorkerState.Blocked);

            _agent.StateChanged += OnStateChanged;

            // Lock and unlock arrive as events rather than being polled for. A locked
            // machine is the safest possible moment to run - the user is demonstrably
            // not at the keyboard - so it is worth reacting to in the same tick rather
            // than up to a second later.
            SystemEvents.SessionSwitch += delegate(object s, SessionSwitchEventArgs e)
            {
                if (e.Reason == SessionSwitchReason.SessionLock ||
                    e.Reason == SessionSwitchReason.SessionUnlock) Refresh();
            };

            _tick = new Timer();
            _tick.Interval = 1000;
            _tick.Tick += delegate { Refresh(); };
            _tick.Start();
        }

        void SetMode(Mode m)
        {
            _agent.Mode = m;
            Refresh();
        }

        void OnStateChanged(WorkerState st, string why)
        {
            // Marshal onto the UI thread: StateChanged is raised from the agent's
            // sampling thread and NotifyIcon is not thread-safe.
            try
            {
                if (_icon.ContextMenuStrip.InvokeRequired)
                {
                    _icon.ContextMenuStrip.BeginInvoke(new Action(delegate { Refresh(); }));
                    return;
                }
            }
            catch (Exception) { }
            Refresh();
        }

        void Refresh()
        {
            try
            {
                Snapshot s = _agent.Current;
                WorkerState st = _agent.Policy.State;
                bool can = _agent.Policy.CanRun(_agent.Mode);

                _auto.Checked = _agent.Mode == Mode.Auto;
                _always.Checked = _agent.Mode == Mode.AlwaysOn;
                _off.Checked = _agent.Mode == Mode.Off;

                string headline;
                if (_agent.Mode == Mode.Off) headline = "Off";
                else if (_agent.Mode == Mode.AlwaysOn) headline = _agent.JobRunning ? "Always on: working" : "Always on: idle";
                else if (st == WorkerState.Available) headline = _agent.JobRunning ? "Working" : "Ready for jobs";
                else if (st == WorkerState.Draining)
                    headline = "Cooling down: " + _agent.Policy.SecondsUntilAvailable.ToString(CultureInfo.InvariantCulture) + "s";
                else if (st == WorkerState.Blocked) headline = "Blocked";
                else headline = "Yielded to you";

                _status.Text = headline;
                _detail.Text = Trim(_agent.Policy.Last.ReasonText, 60);

                string tip = headline;
                if (s != null && s.Gpu != null && s.Gpu.Valid)
                {
                    tip += string.Format(CultureInfo.InvariantCulture,
                        "\n{0}% GPU, {1} W, {2}, mem clk {3} MHz",
                        s.Gpu.UtilGpu, s.Gpu.PowerWatts.ToString("0", CultureInfo.InvariantCulture),
                        s.Gpu.PState, s.Gpu.ClockMemMhz);
                }
                // NotifyIcon.Text is capped at 63 characters by the shell and throws
                // above that, which is a silly way to crash a tray app.
                _icon.Text = Trim(tip, 62);

                SetIcon(_agent.Mode == Mode.Off ? WorkerState.Blocked : st);
                _agent.WriteState();
            }
            catch (Exception) { }
        }

        static string Trim(string s, int n)
        {
            if (s == null) return "";
            return s.Length <= n ? s : s.Substring(0, n - 1) + "…";
        }

        /// The icon is drawn rather than shipped, so the deliverable stays a single
        /// file with nothing beside it. Colour carries the state: green available,
        /// amber cooling down, red yielded, grey blocked or off.
        void SetIcon(WorkerState st)
        {
            Color c;
            if (st == WorkerState.Available) c = Color.FromArgb(46, 160, 67);
            else if (st == WorkerState.Draining) c = Color.FromArgb(210, 153, 34);
            else if (st == WorkerState.Busy) c = Color.FromArgb(218, 54, 51);
            else c = Color.FromArgb(125, 133, 144);

            using (var bmp = new Bitmap(16, 16))
            using (var g = Graphics.FromImage(bmp))
            {
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
                g.Clear(Color.Transparent);
                using (var b = new SolidBrush(c)) g.FillEllipse(b, 1, 1, 14, 14);
                using (var p = new Pen(Color.FromArgb(90, 0, 0, 0))) g.DrawEllipse(p, 1, 1, 14, 14);
                IntPtr h = bmp.GetHicon();
                Icon fresh = (Icon)Icon.FromHandle(h).Clone();
                _icon.Icon = fresh;
                if (_current != null) _current.Dispose();
                _current = fresh;
                // GetHicon allocates an icon handle the GC knows nothing about. This
                // runs once a second forever, so leaking it would exhaust the desktop
                // heap in about a day.
                DestroyIcon(h);
            }
        }

        void Quit()
        {
            try { _tick.Stop(); } catch (Exception) { }
            try { _icon.Visible = false; _icon.Dispose(); } catch (Exception) { }
            try { _agent.Dispose(); } catch (Exception) { }
            ExitThread();
        }
    }
}
