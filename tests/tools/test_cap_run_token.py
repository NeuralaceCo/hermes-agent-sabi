"""Cap (A126 §4): a routine run's per-run credential reaches cap-cli's env for THAT run only.

The gateway binds ``body.cap_run_token`` of one ``/v1/runs`` run into a ContextVar; the spawn
surfaces that can run cap-cli (foreground terminal, background processes, execute_code) put it in
the child env as ``CAP_RUN_TOKEN``. Everything else — a chat turn, a cron, a later run on the same
thread, the shared bash snapshot, a session kernel spawned for another credential, the post-turn
background review — must never see it.
"""

import contextvars
import json
import os
import sys
import threading
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from gateway.session_context import (
    CAP_RUN_TOKEN_ENV,
    apply_cap_run_token,
    cap_run_token,
    clear_session_vars,
    reset_session_vars,
    set_session_vars,
)


@contextmanager
def _bound(**kwargs):
    tokens = set_session_vars(**{"platform": "api_server", **kwargs})
    try:
        yield
    finally:
        clear_session_vars(tokens)


def _fresh(fn, *args):
    """Run *fn* in a brand-new context (a fresh task / thread that never bound a session)."""
    return contextvars.Context().run(fn, *args)


# --- session_context -------------------------------------------------------------------------

def test_bound_token_is_visible_and_cleared_after_the_run():
    def body():
        with _bound(cap_run_token="tok-a"):
            inside = cap_run_token()
        return inside, cap_run_token()

    assert _fresh(body) == ("tok-a", "")


def test_a_bind_without_a_token_masks_a_previous_runs_token_on_the_same_context():
    # A thread-pool worker's context persists across work items; the next bind (a chat turn,
    # another run) must not inherit the previous run's credential even if clear was skipped.
    def body():
        set_session_vars(cap_run_token="tok-a")
        set_session_vars(session_id="chat-turn")
        return cap_run_token()

    assert _fresh(body) == ""


def test_reset_and_never_bound_mean_no_token():
    def body():
        set_session_vars(cap_run_token="tok-a")
        reset_session_vars()
        return cap_run_token()

    assert _fresh(body) == ""
    assert _fresh(cap_run_token) == ""


def test_non_string_token_is_ignored():
    def body():
        with _bound(cap_run_token={"x": 1}):
            return cap_run_token()

    assert _fresh(body) == ""


def test_apply_sets_only_the_bound_token_and_strips_inherited_values():
    def body():
        with _bound(cap_run_token="tok-a"):
            inside = apply_cap_run_token({CAP_RUN_TOKEN_ENV: "stale", "PATH": "/bin"})
        outside = apply_cap_run_token({CAP_RUN_TOKEN_ENV: "stale", "PATH": "/bin"})
        return inside, outside

    inside, outside = _fresh(body)
    assert inside == {CAP_RUN_TOKEN_ENV: "tok-a", "PATH": "/bin"}
    assert outside == {"PATH": "/bin"}


def test_token_is_not_a_bridged_session_var():
    # Not in _VAR_MAP: it must not ride every child env (browser daemon, MCP, cron scripts)
    # nor fall back to os.environ through get_session_env.
    from gateway.session_context import _VAR_MAP, get_session_env

    assert CAP_RUN_TOKEN_ENV not in _VAR_MAP

    def body():
        with _bound(cap_run_token="tok-a"):
            return get_session_env(CAP_RUN_TOKEN_ENV, "")

    with patch.dict(os.environ, {CAP_RUN_TOKEN_ENV: ""}):
        assert _fresh(body) == ""


def test_concurrent_runs_each_see_only_their_own_token():
    barrier = threading.Barrier(2)
    seen = {}

    def run(name):
        with _bound(cap_run_token=f"tok-{name}"):
            barrier.wait(5)
            seen[name] = cap_run_token()
            barrier.wait(5)

    threads = [threading.Thread(target=contextvars.Context().run, args=(run, n)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert seen == {"a": "tok-a", "b": "tok-b"}


# --- spawn surfaces --------------------------------------------------------------------------

def test_background_process_env_carries_only_the_runs_token():
    from tools.process_registry import ProcessRegistry

    def body():
        with _bound(cap_run_token="tok-a"):
            inside = ProcessRegistry._spawn_env({})
        return inside, ProcessRegistry._spawn_env({})

    with patch.dict(os.environ, {CAP_RUN_TOKEN_ENV: "ambient"}):
        inside, outside = _fresh(body)
    assert inside[CAP_RUN_TOKEN_ENV] == "tok-a"
    assert CAP_RUN_TOKEN_ENV not in outside


def test_execute_code_child_env_carries_only_the_runs_token(tmp_path):
    from tools.code_execution_env import _build_child_env

    def build():
        return _build_child_env(rpc_endpoint="x", rpc_token="y", tmpdir=str(tmp_path),
                                child_python=sys.executable)

    def body():
        with _bound(cap_run_token="tok-a"):
            inside = build()
        return inside, build()

    with patch.dict(os.environ, {CAP_RUN_TOKEN_ENV: "ambient"}):
        inside, outside = _fresh(body)
    assert inside[CAP_RUN_TOKEN_ENV] == "tok-a"
    assert CAP_RUN_TOKEN_ENV not in outside


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_foreground_terminal_sees_the_token_in_run_only_and_snapshot_never_keeps_it(tmp_path):
    from tools.environments.local import LocalEnvironment

    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    env.init_session()

    def run_as(**bind):
        out = {}

        def worker():
            with _bound(**bind):
                # A command that tries to persist it must not leak either.
                out["r"] = env.execute(f'export {CAP_RUN_TOKEN_ENV}="${{{CAP_RUN_TOKEN_ENV}}}"; '
                                       f'echo "[${CAP_RUN_TOKEN_ENV}]"')

        t = threading.Thread(target=contextvars.Context().run, args=(worker,))
        t.start()
        t.join(30)
        return out["r"].get("output", "")

    try:
        with patch.dict(os.environ, {CAP_RUN_TOKEN_ENV: "ambient"}):
            in_run = run_as(session_id="run-1", cap_run_token="tok-a")
            chat_turn = run_as(session_id="chat-1")
            other_run = run_as(session_id="run-2", cap_run_token="tok-b")
        assert "[tok-a]" in in_run, in_run
        assert "[]" in chat_turn and "tok-a" not in chat_turn and "ambient" not in chat_turn, chat_turn
        assert "[tok-b]" in other_run and "tok-a" not in other_run, other_run
        if os.path.exists(env._snapshot_path):
            with open(env._snapshot_path) as f:
                snap = f.read()
            assert CAP_RUN_TOKEN_ENV not in snap and "tok-a" not in snap and "tok-b" not in snap
    finally:
        env.cleanup()


def test_snapshot_exclusion_names_the_token():
    import re

    from tools.environments.base_session_env import (
        _SNAPSHOT_EXCLUDED_ENV_REGEX,
        _export_dump_excluding_session_vars,
    )

    assert re.search(_SNAPSHOT_EXCLUDED_ENV_REGEX, f'declare -x {CAP_RUN_TOKEN_ENV}="x"')
    assert not re.search(_SNAPSHOT_EXCLUDED_ENV_REGEX, f'declare -x {CAP_RUN_TOKEN_ENV}_OTHER="x"')
    assert f" {CAP_RUN_TOKEN_ENV} " in _export_dump_excluding_session_vars('"$t"')


# --- execute_code session kernel -------------------------------------------------------------

@contextmanager
def _kernel():
    from tools.code_kernel import shutdown_all_kernels

    config = {"mode": "strict", "kernel_mode": "session", "timeout": 30}
    shutdown_all_kernels()
    with patch.dict(os.environ, {"TERMINAL_ENV": "local", CAP_RUN_TOKEN_ENV: "ambient"}), \
            patch("tools.code_execution_tool._load_config", return_value=config):
        try:
            yield
        finally:
            shutdown_all_kernels()


def _cell(session_key, token=None):
    """One execute_code cell in conversation *session_key* (same owner every call), printing
    the env cap-cli would inherit from ``subprocess.run(["cap-cli", ...])`` in the script."""
    from tools.approval_context import reset_current_session_key, set_current_session_key
    from tools.code_execution_tool import execute_code

    def body():
        key = set_current_session_key(session_key)
        try:
            # platform "" = a trusted local surface (the api_server approval gate is not under test).
            with _bound(platform="", **({"cap_run_token": token} if token else {})):
                return json.loads(execute_code(
                    "import os, subprocess, sys\n"
                    "print('[' + subprocess.run([sys.executable, '-c', "
                    "'import os; print(os.environ.get(\"CAP_RUN_TOKEN\", \"\"), end=\"\")'], "
                    "capture_output=True, text=True).stdout + ']')",
                    task_id="turn"))
        finally:
            reset_current_session_key(key)

    return _fresh(body)


def test_session_kernel_never_serves_a_cell_with_another_credential():
    with _kernel():
        run_a = _cell("conv", "tok-a")
        run_a_again = _cell("conv", "tok-a")
        chat = _cell("conv")
        run_b = _cell("conv", "tok-b")
    for result in (run_a, run_a_again, chat, run_b):
        assert result["status"] == "success", result
    assert "[tok-a]" in run_a["output"]
    assert "[tok-a]" in run_a_again["output"] and run_a_again["kernel"]["reused"] is True
    assert "[]" in chat["output"] and chat["kernel"]["reused"] is False
    assert "[tok-b]" in run_b["output"] and run_b["kernel"]["reused"] is False


def test_kernel_key_holds_a_fingerprint_never_the_token():
    from tools.code_kernel import _KERNELS

    with _kernel():
        _cell("conv", "tok-secret-value")
        keys = list(_KERNELS)
    assert keys and all("tok-secret-value" not in repr(k) for k in keys)


# --- post-turn background review -------------------------------------------------------------

def test_background_review_thread_runs_without_the_runs_token():
    import agent.background_review as br
    from run_agent import AIAgent

    seen, done = {}, threading.Event()

    def target():
        seen["token"] = cap_run_token()
        done.set()

    agent = MagicMock()
    with patch.object(br, "prepare_background_review_run", return_value=object()), \
            patch.object(br, "spawn_background_review_thread", return_value=(target, "prompt")):
        def body():
            with _bound(cap_run_token="tok-a"):
                AIAgent._spawn_background_review_now(agent, [], review_skills=True)
                assert done.wait(5)
                return cap_run_token()

        parent_after_spawn = _fresh(body)
    assert seen["token"] == ""
    assert parent_after_spawn == "tok-a"  # the run itself keeps its credential
