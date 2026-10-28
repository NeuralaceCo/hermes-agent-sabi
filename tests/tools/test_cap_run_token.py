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

    with patch.dict(os.environ, {CAP_RUN_TOKEN_ENV: "ambient"}):
        assert _fresh(body) == "ambient"  # get_session_env never sees the bound token


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

def test_background_process_env_never_carries_a_token():
    # A background process outlives the run (and a later turn can drive or read it).
    from tools.process_registry import ProcessRegistry

    def body():
        with _bound(cap_run_token="tok-a"):
            inside = ProcessRegistry._spawn_env({})
        return inside, ProcessRegistry._spawn_env({})

    with patch.dict(os.environ, {CAP_RUN_TOKEN_ENV: "ambient"}):
        inside, outside = _fresh(body)
    assert CAP_RUN_TOKEN_ENV not in inside
    assert CAP_RUN_TOKEN_ENV not in outside


def test_a_routine_runs_foreground_command_is_never_adopted_into_the_background():
    from tools.terminal_tool_background import yield_to_background_handler

    def handler():
        return yield_to_background_handler(command="x", env_type="local", cwd=None,
                                            effective_task_id="t", task_id="t", session_key="")

    def body():
        with _bound(cap_run_token="tok-a"):
            in_run = handler()
        return in_run, handler()

    in_run, chat = _fresh(body)
    assert in_run is None  # the foreground wait continues
    assert callable(chat)  # unchanged without a credential


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
        assert os.path.exists(env._snapshot_path)  # the snapshot path is really exercised
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
                    "'import os; print(os.environ.get(\"CAP_RUN_TOKEN\", \"\")[::-1], end=\"\")'], "
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
    # printed reversed: the raw value is masked in tool output
    assert "[a-kot]" in run_a["output"]
    assert "[a-kot]" in run_a_again["output"] and run_a_again["kernel"]["reused"] is True
    assert "[]" in chat["output"] and chat["kernel"]["reused"] is False
    assert "[b-kot]" in run_b["output"] and run_b["kernel"]["reused"] is False


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


def test_run_end_ends_only_the_kernels_holding_the_runs_credential():
    from tools.code_kernel import _KERNELS, shutdown_kernels_for_run

    with _kernel():
        _cell("run_1", "tok-a")
        _cell("run_1::child::c1", "tok-a")
        _cell("run_1::child::bg")  # a background child dropped the credential: keeps working
        _cell("run_10", "tok-b")  # a prefix-alike owner of another run survives (4 = the LRU cap)
        shutdown_kernels_for_run("run_1", "")  # no credential: nothing to end
        assert len(_KERNELS) == 4
        shutdown_kernels_for_run("run_1", "tok-a")
        owners = sorted(k[0] for k in _KERNELS)
    assert owners == ["run_10", "run_1::child::bg"]


# --- revocation: copied contexts hold a grant key, never the token -------------------------------

def test_a_worker_abandoned_by_the_run_cannot_spawn_with_its_token_after_it_ends():
    # run_bounded_sync / tool-executor timeouts leave a worker on a COPIED context; clearing the
    # run's own context must still revoke what that copy can reach.
    def body():
        tokens = set_session_vars(platform="api_server", cap_run_token="tok-late")
        abandoned = contextvars.copy_context()
        assert abandoned.run(cap_run_token) == "tok-late"
        clear_session_vars(tokens)
        return abandoned.run(cap_run_token), abandoned.run(apply_cap_run_token, {})

    assert _fresh(body) == ("", {})


def test_a_late_kernel_spawned_after_the_run_ends_gets_no_token():
    from tools.approval_context import reset_current_session_key, set_current_session_key
    from tools.code_execution_tool import execute_code

    def body():
        tokens = set_session_vars(platform="", cap_run_token="tok-late")
        abandoned = contextvars.copy_context()
        clear_session_vars(tokens)

        def late_cell():
            key = set_current_session_key("run_late")
            try:
                return json.loads(execute_code(
                    "import os\nprint('[' + os.environ.get('CAP_RUN_TOKEN', '') + ']')", task_id="t"))
            finally:
                reset_current_session_key(key)

        return abandoned.run(late_cell)

    with _kernel():
        result = _fresh(body)
    assert result["status"] == "success", result
    assert "[]" in result["output"]


# --- the snapshot fails closed -------------------------------------------------------------------

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
@pytest.mark.parametrize("command", [
    f"readonly {CAP_RUN_TOKEN_ENV}",                      # survives unset: export -p writes -rx
    f'export COPY_OF_IT="${CAP_RUN_TOKEN_ENV}"',          # the value under another name
    f'export COPY_OF_IT="${CAP_RUN_TOKEN_ENV}"; unset {CAP_RUN_TOKEN_ENV}',  # ...and the shell var gone
])
def test_snapshot_never_publishes_the_token_even_when_unset_cannot_remove_it(tmp_path, command):
    from tools.environments.local import LocalEnvironment

    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    env.init_session()

    def run(cmd, **bind):
        out = {}

        def worker():
            with _bound(**bind):
                out["r"] = env.execute(cmd)

        t = threading.Thread(target=contextvars.Context().run, args=(worker,))
        t.start()
        t.join(30)
        return out["r"].get("output", "")

    try:
        run("export KEEP_ME=1")  # a normal export is still persisted
        run(command, cap_run_token="tok-ro")
        later = run(f'echo "[${CAP_RUN_TOKEN_ENV}|${{COPY_OF_IT:-}}|$KEEP_ME]"', session_id="chat")
        assert "[||1]" in later, later
        with open(env._snapshot_path) as f:
            snap = f.read()
        assert "tok-ro" not in snap and f"{CAP_RUN_TOKEN_ENV}=" not in snap
    finally:
        env.cleanup()


# --- value-based redaction -----------------------------------------------------------------------

def test_the_run_token_is_masked_in_output_logs_and_after_revocation():
    from agent.redact import redact_sensitive_text, redact_terminal_output

    secret = "opaque-run-credential-7f3a"

    def body():
        tokens = set_session_vars(platform="api_server", cap_run_token=secret)
        try:
            in_run = (redact_terminal_output(f"{secret}\n", "printenv CAP_RUN_TOKEN"),
                      redact_sensitive_text(f"curl -H 'X-Cap-Run-Token: {secret}' x"),
                      redact_sensitive_text(f"x-cap-run-token=other-value-123"))
        finally:
            clear_session_vars(tokens)
        return in_run, redact_sensitive_text(f"late {secret}")

    (printed, header, other_header), late = _fresh(body)
    for text in (printed, header, other_header, late):
        assert secret not in text and "other-value-123" not in text, text
    assert "[REDACTED:cap-run-token]" in printed and "[REDACTED:cap-run-token]" in late


def test_redaction_masks_the_token_even_with_redaction_disabled():
    import agent.redact as redact

    def body():
        with _bound(cap_run_token="opaque-run-credential-9b1c"), \
                patch.object(redact, "_REDACT_ENABLED", False):
            return redact.redact_sensitive_text("v=opaque-run-credential-9b1c")

    assert "opaque-run-credential-9b1c" not in _fresh(body)


def test_terminal_tool_result_never_shows_the_token(tmp_path):
    from tools.terminal_tool import terminal_tool

    def body():
        with _bound(platform="", session_id="run-x", cap_run_token="opaque-run-credential-c0de"):
            return json.loads(terminal_tool(command=f"printenv {CAP_RUN_TOKEN_ENV}", task_id="t-redact"))

    with patch.dict(os.environ, {"TERMINAL_ENV": "local", "TERMINAL_CWD": str(tmp_path)}):
        result = _fresh(body)
    assert "opaque-run-credential-c0de" not in json.dumps(result)
    assert "[REDACTED:cap-run-token]" in result.get("output", ""), result


# --- kernel and terminal output that leaves the process -----------------------------------------

def _kernel_cell(code, token="opaque-run-credential-k3rn"):
    from tools.approval_context import reset_current_session_key, set_current_session_key
    from tools.code_execution_tool import execute_code

    def body():
        key = set_current_session_key("run_k")
        try:
            with _bound(platform="", cap_run_token=token):
                return json.loads(execute_code(code, task_id="t"))
        finally:
            reset_current_session_key(key)

    return _fresh(body)


def test_a_dying_kernels_error_never_shows_the_token():
    with _kernel():
        result = _kernel_cell(
            "import os, time\nos.write(2, ('boom ' + os.environ['CAP_RUN_TOKEN'] + '\\n').encode())\n"
            "time.sleep(0.5)\nos._exit(3)")
    assert result["status"] == "error", result
    assert "boom" in result.get("error", ""), result  # the death message really carries stderr
    assert "opaque-run-credential-k3rn" not in json.dumps(result)


def test_a_kernel_spill_is_masked_even_after_the_cell_pops_the_env_var():
    with _kernel():
        result = _kernel_cell(
            "import os\nt = os.environ.pop('CAP_RUN_TOKEN')\nprint(t + 'x' * 1_100_000)")
        spill = result.get("stdout_spill_path")
        assert spill and os.path.exists(spill), result  # read before the kernel's tmpdir goes
        with open(spill) as f:
            text = f.read()
    assert "opaque-run-credential-k3rn" not in text and "[REDACTED:cap-run-token]" in text


def test_terminal_spill_masks_the_token_even_split_across_chunks(tmp_path):
    from tools.environments.base_output import _BoundedOutputCollector

    secret = "opaque-run-credential-sp1l"

    def body():
        with _bound(cap_run_token=secret):
            collector = _BoundedOutputCollector(max_chars=50, spill_path=tmp_path / "spill.txt")
        for chunk in ("a" * 60, secret[:7], secret[7:15], secret[15:] + "b" * 10, "tail " + secret[:5]):
            collector.append(chunk)
        collector.append(secret[5:])
        return collector.close_spill()

    path = _fresh(body)
    with open(path) as f:
        text = f.read()
    assert secret not in text
    assert text.count("[REDACTED:cap-run-token]") == 2
    assert text.startswith("a" * 60) and "b" * 10 in text


def test_a_spill_cap_inside_a_token_never_writes_any_of_it(tmp_path):
    from tools.environments.base_output import _BoundedOutputCollector

    secret = "opaque-run-credential-cap9"

    def body():
        with _bound(cap_run_token=secret):
            collector = _BoundedOutputCollector(max_chars=50, spill_path=tmp_path / "spill.txt")
        with patch.object(_BoundedOutputCollector, "_SPILL_CAP_CHARS", 100):
            collector.append("a" * 90 + secret[:10])  # exactly fills the cap, token still open
            collector.append(secret[10:] + "b" * 50)
        return collector.close_spill()

    with open(_fresh(body)) as f:
        text = f.read()
    assert "opaque" not in text and "[spill capped" in text, text


# --- the snapshot is checked before anyone can source it ----------------------------------------

def _run_in(env, cmd, **bind):
    out = {}

    def worker():
        with _bound(**bind):
            out["r"] = env.execute(cmd)

    t = threading.Thread(target=contextvars.Context().run, args=(worker,))
    t.start()
    t.join(30)
    return out["r"].get("output", "")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_a_runs_snapshot_dump_is_staged_and_never_reaches_the_shared_path_unchecked(tmp_path):
    from tools.environments import local as local_env

    env = local_env.LocalEnvironment(cwd=str(tmp_path), timeout=30)
    real, seen = local_env._publish_staged_snapshot, []

    def spy(stage, path, token):
        with open(path) as f:
            live = f.read()
        with open(stage) as f:
            staged = f.read()
        seen.append((token in live, token in staged))
        real(stage, path, token)
        assert not os.path.exists(stage)

    try:
        with patch.object(local_env, "_publish_staged_snapshot", spy):
            # Hidden from the in-shell check (the shell var is gone), caught on the host.
            _run_in(env, f'export COPY_OF_IT="${CAP_RUN_TOKEN_ENV}"; unset {CAP_RUN_TOKEN_ENV}',
                    cap_run_token="tok-stage")
            _run_in(env, "export KEEP_ME=1", cap_run_token="tok-stage")
        assert seen == [(False, True), (False, False)], seen
        later = _run_in(env, 'echo "[${COPY_OF_IT:-}|${KEEP_ME:-}]"', session_id="chat")
        assert "[|1]" in later, later
    finally:
        env.cleanup()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
@pytest.mark.parametrize("profile,published", [
    (f'export COPY_OF_IT="${CAP_RUN_TOKEN_ENV}"; unset {CAP_RUN_TOKEN_ENV}', False),
    ("export BOOT_OK=1", True),
])
def test_a_bootstrap_inside_a_run_is_checked_before_it_becomes_the_snapshot(tmp_path, profile, published):
    import glob

    from tools.environments.local import LocalEnvironment

    home = tmp_path / "home"
    home.mkdir()
    (home / ".bash_profile").write_text(profile + "\n")

    def build():
        with _bound(cap_run_token="tok-boot"):
            return LocalEnvironment(cwd=str(tmp_path), timeout=30, env={"HOME": str(home)})

    env = _fresh(build)
    try:
        assert env._snapshot_ready is published
        assert os.path.exists(env._snapshot_path) is published
        assert not glob.glob(f"{env._snapshot_path}.tmp.*")
        if published:
            with open(env._snapshot_path) as f:
                assert "BOOT_OK" in f.read()
    finally:
        env.cleanup()
