"""browser.navigate_wait_until / browser.navigate_settle_ms: browser_navigate on a CDP
session sends ``navigate`` with ``waitUntil`` straight to the agent-browser daemon socket
instead of the CLI ``open`` (which always waits for the full ``load`` event).

A real AF_UNIX fake daemon records every request line, so the tests check what goes over
the wire: what is sent, how often, and when the CLI ``open`` is used instead."""

import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time

import pytest

import tools.browser_tool as bt
from tools import browser_tool_cdp as bt_cdp
from tools import browser_tool_cloud as bt_cloud
from tools import browser_tool_lifecycle as bt_lifecycle
from tools import browser_tool_session as bt_session

TASK = "nav-task"
CDP_URL = "ws://kernel.invalid/devtools/browser/abc"
SESSION_NAME = "cdp_fastnav01"

_NAV_CACHES = (
    ("_navigate_wait_until_resolved", "_cached_navigate_wait_until"),
    ("_navigate_settle_ms_resolved", "_cached_navigate_settle_ms"),
)


class FakeDaemon:
    """agent-browser daemon stand-in: newline-delimited JSON on ``<dir>/default.sock``.
    ``handler(request)`` returns a response dict, ``"close"`` (hang up without answering),
    ``"garbage"`` (non-JSON line) or ``"hang"`` (never answer)."""

    def __init__(self, socket_dir, handler):
        self.path = os.path.join(socket_dir, "default.sock")
        self.handler = handler
        self.requests = []
        self._release = threading.Event()
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self.path)
        self._server.listen(8)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while True:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        with conn:
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = conn.recv(65536)
                if not chunk:
                    return  # liveness probe: connect + close
                buf += chunk
            request = json.loads(buf)
            self.requests.append(request)
            reply = self.handler(request)
            if reply == "hang":
                self._release.wait(10)
            elif reply == "garbage":
                conn.sendall(b"not json\n")
            elif reply != "close":
                conn.sendall((json.dumps(reply) + "\n").encode())

    def actions(self):
        return [r.get("action") for r in self.requests]

    def stop(self):
        self._release.set()
        self._server.close()
        try:
            os.unlink(self.path)
        except OSError:
            pass


def nav_ok(request):
    if request["action"] == "navigate":
        return {"id": request["id"], "success": True,
                "data": {"url": request["url"], "title": "Daemon Title"}}
    if request["action"] == "url":
        return {"id": request["id"], "success": True, "data": {"url": "https://shop.example/item"}}
    return {"id": request["id"], "success": True, "data": {"state": request.get("state")}}


@pytest.fixture
def env(monkeypatch):
    """CDP session for TASK, a short socket root, a recording CLI stub and config control."""
    root = tempfile.mkdtemp(prefix="hfn", dir="/tmp" if sys.platform == "darwin" else None)
    socket_dir = os.path.join(root, f"agent-browser-{SESSION_NAME}")
    os.makedirs(socket_dir)
    session = {"session_name": SESSION_NAME, "bb_session_id": None, "cdp_url": CDP_URL,
               "features": {"cdp_override": True}, "_first_nav": False}
    state = {"config": {}, "cli": [], "daemons": [], "cli_hook": None}

    def fake_cli(task_id, command, args=None, timeout=None, **_kw):
        state["cli"].append((command, list(args or [])))
        if state["cli_hook"] is not None:
            hooked = state["cli_hook"](command, list(args or []))
            if hooked is not None:
                return hooked
        if command == "open":
            return {"success": True, "data": {"url": args[0], "title": "CLI Title"}}
        if command == "snapshot":
            return {"success": True, "data": {"snapshot": "- heading \"Hi\"", "refs": {"e1": {}}}}
        return {"success": True, "data": {}}

    def start_daemon(handler=nav_ok):
        daemon = FakeDaemon(socket_dir, handler)
        state["daemons"].append(daemon)
        return daemon

    for flag, cache in _NAV_CACHES:
        monkeypatch.setattr(bt, flag, False, raising=False)
        monkeypatch.setattr(bt, cache, None, raising=False)
    monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {"browser": dict(state["config"])})
    monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: root)
    monkeypatch.setattr(bt_lifecycle, "_write_owner_pid", lambda *_a: None)
    monkeypatch.setattr(bt_cdp, "_ensure_cdp_supervisor", lambda _t: None)
    monkeypatch.setattr(bt_cdp, "_stop_cdp_supervisor", lambda _t: None)
    monkeypatch.setattr(bt_session, "_get_session_info", lambda task_id=None: bt._active_sessions[task_id])
    monkeypatch.setattr(bt_session, "_run_browser_command", fake_cli)
    monkeypatch.setattr(bt_cloud, "_is_local_backend", lambda: False)
    monkeypatch.setattr(bt_cloud, "_allow_private_urls", lambda: False)
    monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(bt, "_navigation_session_key", lambda task_id, _url: task_id)
    monkeypatch.setattr(bt, "_maybe_start_recording", lambda _k: None)
    monkeypatch.setattr(bt, "check_website_access", lambda _url: None)
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
    monkeypatch.setattr(bt, "_is_safe_url", lambda url: "169.254." not in url and "10.0.0." not in url)
    bt._active_sessions[TASK] = session
    state.update(root=root, socket_dir=socket_dir, session=session, start_daemon=start_daemon)
    yield state
    for daemon in state["daemons"]:
        daemon.stop()
    bt._active_sessions.clear()
    bt._session_last_activity.clear()
    bt._last_active_session_key.clear()
    shutil.rmtree(root, ignore_errors=True)


def navigate(url="https://shop.example/item"):
    return json.loads(bt.browser_navigate(url, task_id=TASK))


def cli_opens(state):
    return [args for command, args in state["cli"] if command == "open"]


class TestNavigateConfig:
    def test_defaults_keep_full_load_and_no_settle(self, env):
        assert bt.get_navigate_wait_until() == "load"
        assert bt.get_navigate_settle_ms() == 0

    @pytest.mark.parametrize("raw, expected", [
        ("load", "load"),
        ("domcontentloaded", "domcontentloaded"),
        ("networkidle", "networkidle"),
        (" DOMContentLoaded ", "domcontentloaded"),
        ("none", "load"),
        ("fast", "load"),
        (None, "load"),
        (5, "load"),
        (["domcontentloaded"], "load"),
    ])
    def test_wait_until_accepts_known_values_else_load(self, env, raw, expected):
        env["config"]["navigate_wait_until"] = raw
        assert bt.get_navigate_wait_until() == expected

    @pytest.mark.parametrize("raw, expected", [
        (None, 0), (0, 0), (1500, 1500), ("2000", 2000), (-5, 0),
        (99_999, 10_000), ("abc", 0), ([1], 0),
    ])
    def test_settle_ms_is_clamped_or_defaulted(self, env, raw, expected):
        env["config"]["navigate_settle_ms"] = raw
        assert bt.get_navigate_settle_ms() == expected

    def test_cleanup_all_browsers_resets_the_caches(self, env):
        env["config"]["navigate_wait_until"] = "domcontentloaded"
        env["config"]["navigate_settle_ms"] = 1500
        assert bt.get_navigate_wait_until() == "domcontentloaded"
        assert bt.get_navigate_settle_ms() == 1500
        bt._active_sessions.clear()
        bt_lifecycle.cleanup_all_browsers()
        env["config"].clear()
        assert bt.get_navigate_wait_until() == "load"
        assert bt.get_navigate_settle_ms() == 0


class TestFullLoadDefault:
    def test_default_uses_cli_open_and_never_the_socket(self, env):
        daemon = env["start_daemon"]()
        result = navigate()
        assert result["success"] is True
        assert result["title"] == "CLI Title"
        assert cli_opens(env) == [["https://shop.example/item"]]
        assert daemon.requests == []

    def test_local_session_keeps_cli_open_even_when_configured(self, env):
        env["config"]["navigate_wait_until"] = "domcontentloaded"
        env["session"]["cdp_url"] = None
        daemon = env["start_daemon"]()
        assert navigate()["title"] == "CLI Title"
        assert cli_opens(env) == [["https://shop.example/item"]]
        assert daemon.requests == []

    def test_lightpanda_session_keeps_cli_open_for_its_chrome_fallback(self, env):
        env["config"]["navigate_wait_until"] = "domcontentloaded"
        env["session"]["features"] = {"local": True, "lightpanda": True}
        daemon = env["start_daemon"]()
        assert navigate()["title"] == "CLI Title"
        assert cli_opens(env) == [["https://shop.example/item"]]
        assert daemon.requests == []


class TestSocketNavigate:
    @pytest.mark.parametrize("wait_until", ["domcontentloaded", "networkidle"])
    def test_sends_one_navigate_with_wait_until_to_running_daemon(self, env, wait_until):
        env["config"]["navigate_wait_until"] = wait_until
        daemon = env["start_daemon"]()
        result = navigate()
        assert result["success"] is True
        assert result["title"] == "Daemon Title"
        assert result["url"] == "https://shop.example/item"
        assert "snapshot" in result  # auto-snapshot still follows
        assert len(daemon.requests) == 1
        request = daemon.requests[0]
        assert request["action"] == "navigate"
        assert request["url"] == "https://shop.example/item"
        assert request["waitUntil"] == wait_until
        assert isinstance(request["id"], str) and request["id"]
        assert cli_opens(env) == []

    def test_bare_host_gets_https_like_cli_open(self, env):
        env["config"]["navigate_wait_until"] = "domcontentloaded"
        daemon = env["start_daemon"]()
        navigate("shop.example/item")
        assert daemon.requests[0]["url"] == "https://shop.example/item"

    def test_cold_daemon_is_started_with_open_about_blank_first(self, env):
        env["config"]["navigate_wait_until"] = "domcontentloaded"

        def start_on_warmup(command, args):
            if command == "open" and args == ["about:blank"]:
                env["start_daemon"]()
            return None

        env["cli_hook"] = start_on_warmup
        result = navigate()
        assert result["title"] == "Daemon Title"
        assert cli_opens(env) == [["about:blank"]]
        assert env["daemons"][0].actions() == ["navigate"]

    def test_failed_warmup_is_reported_without_navigating(self, env):
        env["config"]["navigate_wait_until"] = "domcontentloaded"
        env["cli_hook"] = lambda command, args: (
            {"success": False, "error": "Failed to connect to CDP"} if args == ["about:blank"] else None)
        result = navigate()
        assert result["success"] is False
        assert "Failed to connect to CDP" in result["error"]
        assert cli_opens(env) == [["about:blank"]]

    def test_socket_unreachable_before_send_falls_back_to_cli_open(self, env):
        env["config"]["navigate_wait_until"] = "domcontentloaded"
        # Warm-up "succeeds" but no daemon socket ever appears: nothing can have been sent.
        result = navigate()
        assert result["success"] is True
        assert result["title"] == "CLI Title"
        assert cli_opens(env) == [["about:blank"], ["https://shop.example/item"]]

    def test_interrupt_stops_before_anything_is_sent(self, env, monkeypatch):
        env["config"]["navigate_wait_until"] = "domcontentloaded"
        daemon = env["start_daemon"]()
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: True)
        result = navigate()
        assert result["success"] is False
        assert result["error"] == "Interrupted"
        assert daemon.requests == []
        assert cli_opens(env) == []


class TestNoRetryAfterSend:
    @pytest.mark.parametrize("reply, error_part", [
        ("close", "closed the connection"),
        ("garbage", "Navigation failed"),
        ({"success": False, "error": "net::ERR_NAME_NOT_RESOLVED"}, "ERR_NAME_NOT_RESOLVED"),
    ])
    def test_failure_after_send_is_reported_once_never_retried(self, env, reply, error_part):
        env["config"]["navigate_wait_until"] = "domcontentloaded"
        daemon = env["start_daemon"](lambda request: reply)
        result = navigate()
        assert result["success"] is False
        assert error_part in result["error"]
        assert daemon.actions() == ["navigate"]
        assert cli_opens(env) == []

    def test_timeout_after_send_is_not_retried_and_recovers_the_session(self, env):
        env["config"]["navigate_wait_until"] = "domcontentloaded"
        daemon = env["start_daemon"](lambda request: "hang")
        started = time.monotonic()
        result = bt_session._run_navigate(TASK, "https://shop.example/item", timeout=0.5)
        assert time.monotonic() - started < 5
        assert result == {"success": False, "error": "Command timed out after 0.5 seconds"}
        assert daemon.actions() == ["navigate"]
        assert env["cli"] == []
        # Same recovery as a CLI timeout: the stuck CDP client generation is replaced.
        assert bt._active_sessions[TASK]["session_name"] != SESSION_NAME
        assert bt._active_sessions[TASK]["cdp_url"] == CDP_URL


class TestSettle:
    def test_zero_settle_sends_only_the_navigate(self, env):
        env["config"]["navigate_wait_until"] = "domcontentloaded"
        daemon = env["start_daemon"]()
        navigate()
        assert daemon.actions() == ["navigate"]

    def test_settle_asks_daemon_for_network_idle_capped_at_the_config(self, env):
        env["config"].update(navigate_wait_until="domcontentloaded", navigate_settle_ms=99_999)
        daemon = env["start_daemon"]()
        assert navigate()["success"] is True
        assert daemon.actions() == ["navigate", "waitforloadstate", "url"]
        settle = daemon.requests[1]
        assert settle["state"] == "networkidle"
        assert settle["timeout"] == 10_000

    def test_settle_timeout_reported_by_daemon_still_succeeds(self, env):
        env["config"].update(navigate_wait_until="domcontentloaded", navigate_settle_ms=1500)

        def handler(request):
            if request["action"] == "waitforloadstate":
                return {"id": request["id"], "success": False,
                        "error": "Timeout waiting for load state: networkidle"}
            return nav_ok(request)

        env["start_daemon"](handler)
        result = navigate()
        assert result["success"] is True
        assert result["title"] == "Daemon Title"

    def test_settle_is_bounded_client_side_when_daemon_never_answers(self, env, monkeypatch):
        env["config"].update(navigate_wait_until="domcontentloaded", navigate_settle_ms=200)
        monkeypatch.setattr(bt_session, "_SETTLE_GRACE_S", 0.3)

        def handler(request):
            return "hang" if request["action"] == "waitforloadstate" else nav_ok(request)

        daemon = env["start_daemon"](handler)
        started = time.monotonic()
        result = bt_session._run_navigate(TASK, "https://shop.example/item", timeout=30)
        assert time.monotonic() - started < 3
        assert result["success"] is True
        assert daemon.actions() == ["navigate", "waitforloadstate", "url"]
        assert bt._active_sessions[TASK]["session_name"] == SESSION_NAME  # a slow settle is not a failure


class TestPolicyStillEnforced:
    def test_daemon_action_policy_denial_is_returned_not_bypassed(self, env):
        env["config"]["navigate_wait_until"] = "domcontentloaded"
        daemon = env["start_daemon"](lambda request: {
            "id": request["id"], "success": False,
            "error": "Action 'navigate' denied by policy: navigate not allowed"})
        result = navigate()
        assert result["success"] is False
        assert "denied by policy" in result["error"]
        assert daemon.actions() == ["navigate"]
        assert cli_opens(env) == []

    @pytest.mark.parametrize("url", ["http://169.254.169.254/latest/meta-data", "http://10.0.0.5/admin"])
    def test_blocked_url_never_reaches_the_daemon(self, env, url):
        env["config"]["navigate_wait_until"] = "domcontentloaded"
        daemon = env["start_daemon"]()
        result = navigate(url)
        assert result["success"] is False
        assert result["error"].startswith("Blocked")
        assert daemon.requests == []
        assert env["cli"] == []

    def test_redirect_to_private_address_is_blocked_after_socket_navigate(self, env):
        env["config"]["navigate_wait_until"] = "domcontentloaded"
        env["start_daemon"](lambda request: {
            "id": request["id"], "success": True,
            "data": {"url": "http://10.0.0.5/internal", "title": "Internal"}})
        result = navigate()
        assert result["success"] is False
        assert "redirect landed on a private/internal address" in result["error"]
        assert cli_opens(env) == [["about:blank"]]  # page moved off the internal content

    def test_redirect_during_settle_is_checked_against_the_current_url(self, env):
        env["config"].update(navigate_wait_until="domcontentloaded", navigate_settle_ms=1500)

        def handler(request):
            if request["action"] == "url":
                return {"id": request["id"], "success": True, "data": {"url": "http://10.0.0.5/internal"}}
            return nav_ok(request)

        daemon = env["start_daemon"](handler)
        result = navigate()
        assert result["success"] is False
        assert "redirect landed on a private/internal address" in result["error"]
        assert daemon.actions() == ["navigate", "waitforloadstate", "url"]
        assert [c for c, _ in env["cli"]] == ["open"]  # about:blank only, no snapshot of the internal page

    def test_unreadable_url_after_settle_fails_closed(self, env):
        env["config"].update(navigate_wait_until="domcontentloaded", navigate_settle_ms=1500)

        def handler(request):
            if request["action"] == "url":
                return {"id": request["id"], "success": False, "error": "Browser not launched"}
            return nav_ok(request)

        env["start_daemon"](handler)
        result = navigate()
        assert result["success"] is False
        assert "could not confirm the page URL" in result["error"]
        assert not [c for c, _ in env["cli"] if c == "snapshot"]


class TestUnsupportedSocketPlatform:
    def test_no_unix_sockets_keeps_cli_open(self, env, monkeypatch):
        env["config"]["navigate_wait_until"] = "domcontentloaded"
        daemon = env["start_daemon"]()
        monkeypatch.delattr(socket, "AF_UNIX")
        result = navigate()
        assert result["title"] == "CLI Title"
        assert cli_opens(env) == [["https://shop.example/item"]]
        assert daemon.requests == []
