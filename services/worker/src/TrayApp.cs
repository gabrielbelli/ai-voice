// The status bar icon.
//
// Three modes, as asked for: Auto, Always on, Off. Auto is the interesting one
// and everything above exists to make it trustworthy; Always on and Off are there
// because a user who does not trust Auto needs somewhere to go that is not
// "uninstall it".
//
// THE "WHY" LINE IS THE FEATURE, NOT DECORATION. The one question this proof of
// concept has to answer is whether the user trusts the detector, and trust comes
// from the thing being able to say "dwm (pid 1752) holds 169 MiB" or "agent is in
// session 0, console is session 1: cannot observe the user" rather than just
// going red. The menu's one-line summary is capped so it stays a menu; the Why?
// submenu carries every reason in full, because a truncated reason is a reason
// the user cannot check.
//
// DELIBERATELY ABSENT: balloon notifications. A toast over a ranked match is
// exactly the wrong thing and it is also the behaviour that makes people
// uninstall software.

using System;
using System.Collections.Generic;
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
        readonly ToolStripMenuItem _status, _detail, _why, _auto, _always, _off, _counters;
        readonly Timer _tick;
        Icon _current;

        public TrayApp(Agent agent)
        {
            _agent = agent;

            _status = new ToolStripMenuItem("starting...");
            _status.Enabled = false;
            _detail = new ToolStripMenuItem("");
            _detail.Enabled = false;
            _why = new ToolStripMenuItem("Why?");
            _counters = new ToolStripMenuItem("");
            _counters.Enabled = false;

            _auto = new ToolStripMenuItem("Auto", null, delegate { SetMode(Mode.Auto); });
            _always = new ToolStripMenuItem("Always on", null, delegate { SetMode(Mode.AlwaysOn); });
            _off = new ToolStripMenuItem("Off", null, delegate { SetMode(Mode.Off); });

            var menu = new ContextMenuStrip();
            menu.Items.Add(_status);
            menu.Items.Add(_detail);
            menu.Items.Add(_why);
            menu.Items.Add(new ToolStripSeparator());
            menu.Items.Add(_auto);
            menu.Items.Add(_always);
            menu.Items.Add(_off);
            menu.Items.Add(new ToolStripSeparator());
            menu.Items.Add(_counters);
            menu.Items.Add(new ToolStripMenuItem("Copy diagnostics", null, delegate
            {
                try { Clipboard.SetText(_agent.StatusJson()); } catch (Exception) { }
            }));
            menu.Items.Add(new ToolStripMenuItem("Exit", null, delegate { Quit(); }));

            _icon = new NotifyIcon();
            _icon.ContextMenuStrip = menu;
            _icon.Visible = true;
            SetIcon(WorkerState.Blocked, false, false);

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
            // Persisted immediately. A setting that silently resets at the next
            // login is a setting the user stops trusting, and this one is the
            // control they reach for when they do not trust something already.
            _agent.SaveMode();
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
                bool overriding = _agent.Policy.IsOverriding(_agent.Mode);
                string provisioning = _agent.ProvisionStatus;

                _auto.Checked = _agent.Mode == Mode.Auto;
                _always.Checked = _agent.Mode == Mode.AlwaysOn;
                _off.Checked = _agent.Mode == Mode.Off;

                string headline;
                if (provisioning != null)
                {
                    // First run pulls about 5.6 GiB. A grey icon for twenty minutes
                    // reads as broken and gets killed at 4 GB, so setup gets its
                    // own colour and its own headline and says how far along it is.
                    headline = "Setting up: " + provisioning;
                }
                else if (_agent.Mode == Mode.Off)
                {
                    // Off still samples and still logs. "Switched off by the user"
                    // and "gone" are materially different states, and during a
                    // trust-building trial the off-mode trace is exactly the
                    // evidence that shows the detector is harmless before it ever
                    // claims the GPU. So the headline says Off and the detail line
                    // below still reports what the detector would have decided.
                    headline = "Off - still watching";
                }
                else if (overriding)
                {
                    headline = "Always on: using your GPU anyway";
                }
                else if (_agent.Mode == Mode.AlwaysOn)
                {
                    headline = _agent.JobRunning ? "Always on: working" : "Always on: idle";
                }
                else if (st == WorkerState.Available) headline = _agent.JobRunning ? "Working" : "Ready for jobs";
                else if (st == WorkerState.Draining)
                    headline = "Cooling down: " + _agent.Policy.SecondsUntilAvailable.ToString(CultureInfo.InvariantCulture) + "s";
                else if (st == WorkerState.Blocked) headline = "Blocked";
                else headline = "Yielded to you";

                _status.Text = headline;

                // In Off and in Always-on the detector is not in charge, so the
                // detail line is prefixed to say whose opinion it is. Publishing
                // the overridden verdict is the whole point: it is how a week in
                // Off mode produces a false-idle record.
                string reason = _agent.Policy.Last.ReasonText;
                string prefix = "";
                if (_agent.Mode == Mode.Off) prefix = "detector says: ";
                else if (overriding) prefix = "overriding: ";
                _detail.Text = Trim(prefix + reason, 60);

                RebuildWhy(_agent.Policy.Last.Reasons, st, overriding);

                _counters.Text = string.Format(CultureInfo.InvariantCulture,
                    "{0} started, {1} yielded{2}",
                    _agent.JobStarts, _agent.Yields,
                    _agent.LastYieldMs >= 0
                        ? ", last " + _agent.LastYieldMs.ToString(CultureInfo.InvariantCulture) + " ms" +
                          (_agent.LastYieldWasKill ? " (killed)" : "")
                        : "");

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

                SetIcon(st, overriding, provisioning != null);
                _agent.WriteState();
            }
            catch (Exception) { }
        }

        /// The full reason list, one item per line, untruncated.
        void RebuildWhy(List<string> reasons, WorkerState st, bool overriding)
        {
            _why.DropDownItems.Clear();
            if (reasons.Count == 0)
            {
                var none = new ToolStripMenuItem(
                    st == WorkerState.Available
                        ? "Nothing is using the GPU."
                        : "Nothing is using the GPU; waiting out the cooldown.");
                none.Enabled = false;
                _why.DropDownItems.Add(none);
            }
            else
            {
                foreach (string r in reasons)
                {
                    var it = new ToolStripMenuItem(r);
                    it.Enabled = false;
                    _why.DropDownItems.Add(it);
                }
            }
            if (overriding)
            {
                _why.DropDownItems.Add(new ToolStripSeparator());
                var note = new ToolStripMenuItem("Always on is set, so this is being ignored.");
                note.Enabled = false;
                _why.DropDownItems.Add(note);
            }
        }

        static string Trim(string s, int n)
        {
            if (s == null) return "";
            return s.Length <= n ? s : s.Substring(0, n - 1) + "…";
        }

        /// The icon is drawn rather than shipped, so the deliverable stays a single
        /// file with nothing beside it. Colour carries the state: green available,
        /// amber cooling down or setting up, red yielded, grey blocked or off.
        ///
        /// The one shape distinction is deliberate: Always-on running against a
        /// busy verdict draws a HOLLOW RING rather than a filled dot, so "I am
        /// using your GPU while you game" is never invisible. A user who has
        /// overridden the detector should be able to see that they have, without
        /// opening a menu.
        void SetIcon(WorkerState st, bool overriding, bool provisioning)
        {
            Color c;
            if (provisioning) c = Color.FromArgb(88, 166, 255);          // blue: setting up
            else if (_agent.Mode == Mode.Off) c = Color.FromArgb(125, 133, 144);
            else if (st == WorkerState.Available) c = Color.FromArgb(46, 160, 67);
            else if (st == WorkerState.Draining) c = Color.FromArgb(210, 153, 34);
            else if (st == WorkerState.Busy) c = Color.FromArgb(218, 54, 51);
            else c = Color.FromArgb(125, 133, 144);

            using (var bmp = new Bitmap(16, 16))
            using (var g = Graphics.FromImage(bmp))
            {
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
                g.Clear(Color.Transparent);
                if (overriding)
                {
                    using (var pen = new Pen(c, 3f)) g.DrawEllipse(pen, 2, 2, 12, 12);
                }
                else
                {
                    using (var b = new SolidBrush(c)) g.FillEllipse(b, 1, 1, 14, 14);
                    using (var p = new Pen(Color.FromArgb(90, 0, 0, 0))) g.DrawEllipse(p, 1, 1, 14, 14);
                }
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
