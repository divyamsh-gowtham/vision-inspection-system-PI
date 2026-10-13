#!/usr/bin/env python3
"""
ProInspect kiosk host.

A fullscreen, chrome-less WebKit window that renders the HMI. Uses the
WebKit2GTK bindings already present on the JetPack image, so the appliance
needs no browser install and none of Chromium's memory footprint.

The UI is HTML so the same pages can later be served to the LAN dashboard.

The page cannot touch the system on its own, so it posts messages to this host
over a WebKit script-message channel and the host runs them against a fixed
allowlist. Nothing outside ACTIONS can be invoked from the page.

    proinspect/kiosk.py [--url URL] [--windowed] [--hide-cursor] [--zoom 1.3]
"""

import argparse
import json
import os
import pathlib
import socket
import subprocess
import sys

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("WebKit2", "4.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, GLib, Gtk, WebKit2  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_PAGE = HERE / "ui" / "index.html"
VERSION = (HERE / "VERSION").read_text().strip() if (HERE / "VERSION").exists() else "0.0.0"

# A blank screen on a factory floor is indistinguishable from a dead machine,
# so a page that dies gets reloaded rather than left black.
RELOAD_DELAY_SECONDS = 3

SYSTEMCTL = "/usr/bin/systemctl"

# The complete set of system actions the HMI may request. The page sends a key;
# it never sends a command. Anything not in here is logged and dropped.
#
# polkit grants these three to an active local seat session, which the kiosk is
# (Seat=seat0, Active=yes, Remote=no), so no sudo rule or polkit override is
# needed. Verified with pkcheck against the kiosk's own pid.
ACTIONS = {
    "sleep":    [SYSTEMCTL, "suspend"],
    "reboot":   [SYSTEMCTL, "reboot"],
    "shutdown": [SYSTEMCTL, "poweroff"],
}

# "restart the application" is deliberately NOT a systemctl call: managing units
# needs an interactive polkit auth the panel can't give. The unit is
# Restart=always, so exiting is all it takes - systemd brings the HMI straight
# back. Same outcome, no privileges.


def run_privileged(argv):
    """
    Run a power action.

    An active seat session is often allowed these by polkit directly, so try
    unprivileged first and only fall back to the sudoers allowlist. Returns
    (ok, detail) rather than raising: a failed shutdown must not kill the HMI.
    """
    for attempt in (argv, ["sudo", "-n"] + argv):
        try:
            p = subprocess.run(attempt, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"{attempt[0]}: {exc}"
        if p.returncode == 0:
            return True, " ".join(attempt)
        last = (p.stderr or p.stdout or "").strip()
    return False, last or "command failed"


class Kiosk(Gtk.Window):
    def __init__(self, url, windowed=False, hide_cursor=False, zoom=1.0):
        super().__init__(title="ProInspect")
        self.url = url
        self.hide_cursor = hide_cursor
        self.zoom = zoom
        self.connect("destroy", Gtk.main_quit)
        self.connect("key-press-event", self._on_key)

        # Channel the page uses to reach the host: window.webkit.messageHandlers.proinspect
        #
        # Held on self deliberately. The WebView keeps the manager alive on the C
        # side, but the Python wrapper carrying this signal connection does not,
        # so a local would be collected and messages would silently stop arriving.
        self.ucm = WebKit2.UserContentManager()
        self.ucm.register_script_message_handler("proinspect")
        self.ucm.connect("script-message-received::proinspect", self._on_message)

        self.webview = WebKit2.WebView.new_with_user_content_manager(self.ucm)
        self._configure_webview()
        self.add(self.webview)

        self.webview.connect("load-changed", self._on_load_changed)
        self.webview.connect("load-failed", self._on_load_failed)
        # A web process crash must not take the HMI down with it.
        self.webview.connect("web-process-terminated", self._on_crash)

        if not windowed:
            self.set_decorated(False)
            # There is no window manager in this session, so fullscreen() - which
            # is a request to a WM - does nothing, and the window would keep its
            # default size. Size it to the monitor explicitly instead. Getting
            # this wrong silently clips the page against the screen edge.
            self._fit_to_screen()
            self.fullscreen()          # honoured if a WM ever is present
            self.set_keep_above(True)

        self._paint_background()
        # The pages are laid out at native size; a 1768x828 touch panel wants
        # everything bigger. Zoom scales layout, text and touch targets alike.
        if abs(self.zoom - 1.0) > 0.01:
            self.webview.set_zoom_level(self.zoom)
        self.webview.load_uri(self.url)

    # ---------------------------------------------------------------- webview
    def _fit_to_screen(self):
        """Match the window to the monitor, since no WM will do it for us."""
        display = Gdk.Display.get_default()
        if display is None:
            return
        mon = None
        if hasattr(display, "get_primary_monitor"):
            mon = display.get_primary_monitor()
        if mon is None and display.get_n_monitors() > 0:
            mon = display.get_monitor(0)
        if mon is None:
            return
        g = mon.get_geometry()
        print(f"[kiosk] monitor {g.width}x{g.height} at {g.x},{g.y}", flush=True)
        self.set_default_size(g.width, g.height)
        self.move(g.x, g.y)
        self.resize(g.width, g.height)

    def _configure_webview(self):
        s = self.webview.get_settings()
        s.set_enable_developer_extras(False)
        s.set_enable_write_console_messages_to_stdout(True)
        s.set_enable_page_cache(False)
        s.set_enable_tabs_to_links(False)
        s.set_enable_back_forward_navigation_gestures(False)
        # No user-facing browser affordances on an appliance.
        s.set_javascript_can_open_windows_automatically(False)
        s.set_enable_html5_database(True)
        s.set_enable_html5_local_storage(True)
        # The HMI is loaded from file:// and fetches its own JSON from beside
        # itself. A file:// origin is opaque, so without this every fetch fails
        # CORS. Safe here: the only pages loaded are the ones we ship.
        s.set_allow_file_access_from_file_urls(True)
        s.set_allow_universal_access_from_file_urls(True)

    def _paint_background(self):
        """Match the page ground so no flash of a different colour precedes paint."""
        rgba = Gdk.RGBA()
        rgba.parse("#EDF1F4")
        self.webview.set_background_color(rgba)

    def _js(self, script):
        if hasattr(self.webview, "evaluate_javascript"):
            self.webview.evaluate_javascript(script, -1, None, None, None, None, None)
        else:
            self.webview.run_javascript(script, None, None, None)

    # ---------------------------------------------------------------- bridge
    def _on_message(self, _ucm, result):
        try:
            raw = result.get_js_value().to_string()
            msg = json.loads(raw)
        except Exception as exc:                      # malformed page payload
            print(f"[kiosk] bad message: {exc}", file=sys.stderr)
            return

        action = msg.get("action")
        if action == "idle":
            self._go_idle()
            return
        if action == "restart":
            print("[kiosk] restarting application (systemd will respawn)", flush=True)
            GLib.timeout_add(150, lambda: (Gtk.main_quit(), False)[1])
            return
        if action not in ACTIONS:
            print(f"[kiosk] rejected unknown action: {action!r}", file=sys.stderr)
            self._report(action, False, "not permitted")
            return

        print(f"[kiosk] action: {action}", flush=True)
        ok, detail = run_privileged(ACTIONS[action])
        if ok:
            print(f"[kiosk] {action} via {detail}", flush=True)
        else:
            print(f"[kiosk] {action} FAILED: {detail}", file=sys.stderr)
            self._report(action, False, detail)

    def _report(self, action, ok, detail):
        """Tell the page an action didn't take, so it stops saying 'shutting down'."""
        payload = json.dumps({"action": action, "ok": ok, "detail": detail})
        self._js(f"window.onHostResult && window.onHostResult({payload});")

    def _go_idle(self):
        """
        Standby: blank the panel via DPMS. Any key or mouse movement wakes it.
        Needs no privileges, and leaves the inspection service running.
        """
        try:
            subprocess.run(["xset", "dpms", "force", "off"], timeout=5,
                           capture_output=True)
            print("[kiosk] display blanked (idle)", flush=True)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"[kiosk] idle failed: {exc}", file=sys.stderr)
            self._report("idle", False, str(exc))

    # ------------------------------------------------------------------ load
    def _on_load_changed(self, _webview, event):
        if event == WebKit2.LoadEvent.FINISHED:
            unit = socket.gethostname()
            js = (f"window.setUnit && window.setUnit({unit!r});"
                  f"window.setVersion && window.setVersion({VERSION!r});")
            if self.hide_cursor:
                js += "document.documentElement.classList.add('nocursor');"
            self._js(js)
            # Only blank the pointer when asked. A touchscreen-only panel wants it
            # gone; anything driven by a mouse needs to see where it is pointing.
            if self.hide_cursor:
                window = self.get_window()
                if window is not None:
                    window.set_cursor(Gdk.Cursor.new_for_display(Gdk.Display.get_default(),
                                                                 Gdk.CursorType.BLANK_CURSOR))

    def _on_load_failed(self, _webview, _event, failing_uri, error):
        print(f"[kiosk] load failed: {failing_uri}: {error.message}", file=sys.stderr)
        self._schedule_reload()
        return True  # suppress WebKit's own error page

    def _on_crash(self, _webview, reason):
        print(f"[kiosk] web process terminated: {reason}", file=sys.stderr)
        self._schedule_reload()

    def _schedule_reload(self):
        GLib.timeout_add_seconds(RELOAD_DELAY_SECONDS,
                                 lambda: (self.webview.load_uri(self.url), False)[1])

    def _on_key(self, _widget, event):
        """Ctrl+Shift+Q is the deliberate service exit. Nothing else quits."""
        ctrl = event.state & Gdk.ModifierType.CONTROL_MASK
        shift = event.state & Gdk.ModifierType.SHIFT_MASK
        if ctrl and shift and event.keyval in (Gdk.KEY_Q, Gdk.KEY_q):
            print("[kiosk] service exit requested", file=sys.stderr)
            Gtk.main_quit()
            return True
        return False


def main():
    ap = argparse.ArgumentParser(description="ProInspect kiosk host")
    ap.add_argument("--url", default=DEFAULT_PAGE.as_uri(),
                    help="page to display (default: bundled HMI)")
    ap.add_argument("--windowed", action="store_true",
                    help="run in a window instead of fullscreen (development)")
    ap.add_argument("--hide-cursor", action="store_true",
                    help="blank the mouse pointer (touchscreen-only panels)")
    ap.add_argument("--zoom", type=float, default=1.3,
                    help="page zoom factor for the panel (default 1.3)")
    args = ap.parse_args()

    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        print("[kiosk] no DISPLAY or WAYLAND_DISPLAY set", file=sys.stderr)
        return 1

    print(f"[kiosk] ProInspect v{VERSION} -> {args.url}", flush=True)
    Kiosk(args.url, windowed=args.windowed, hide_cursor=args.hide_cursor, zoom=args.zoom).show_all()
    Gtk.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
