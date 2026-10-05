"""delegate_task(background=true) on stateless API-server sessions.

Previously async_delivery_supported()=False forced SYNCHRONOUS execution for
every background dispatch on the API server, blocking the whole turn. Now
that background completions can wake the originating session via the
/v1/chat/completions self-post (gateway/wake.py), a session-continuable
turn (raw session id bound as the api_server chat_id) dispatches async; only
session-id-less one-shot requests keep the sync fallback.

The wake target must be captured from the request-scoped chat_id binding,
NOT from HERMES_SESSION_ID: constructing a child agent calls
set_current_session_id(child.session_id), clobbering the HERMES_SESSION_ID
ContextVar and os.environ with the subagent's internal id before the
dispatch code reads it — the fake child build below reproduces that clobber.
"""

import json
import time
from unittest.mock import MagicMock

import pytest

from gateway.session_context import set_session_vars
from tools.process_registry import process_registry


@pytest.fixture(autouse=True)
def _clean_queue_and_context(monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    while not process_registry.completion_queue.empty():
        try:
            process_registry.completion_queue.get_nowait()
        except Exception:
            break
    yield
    # Restore ContextVars to the pristine "never set" sentinel rather than
    # clear_session_vars()'s explicit-"" state, which would mask env vars for
    # unrelated tests running later in the same worker.
    import gateway.session_context as sc

    for var in sc._VAR_MAP.values():
        var.set(sc._UNSET)
    sc._SESSION_ASYNC_DELIVERY.set(sc._UNSET)
    # set_current_session_id (invoked by the clobber-reproducing fake child
    # build) writes os.environ directly — scrub it so it can't leak into
    # other test modules.
    import os

    os.environ.pop("HERMES_SESSION_ID", None)
    while not process_registry.completion_queue.empty():
        try:
            process_registry.completion_queue.get_nowait()
        except Exception:
            break


def _drain_one(timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not process_registry.completion_queue.empty():
            return process_registry.completion_queue.get_nowait()
        time.sleep(0.02)
    return None


def _fake_parent():
    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "sess"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None
    return parent


def _patch_delegate(monkeypatch):
    import tools.delegate_tool as dt

    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"
    fake_child._subagent_id = "s1"

    def fast_child(task_index, goal, child=None, parent_agent=None, **kw):
        return {
            "task_index": 0, "status": "completed", "summary": f"done: {goal}",
            "api_calls": 1, "duration_seconds": 0.1, "model": "m",
            "exit_reason": "completed",
        }

    creds = {
        "model": "m", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    }
    def clobbering_build_child(**kw):
        # Reproduce what the real _build_child_agent -> AIAgent -> agent_init
        # path does: it synchronizes the child's internal session id into the
        # HERMES_SESSION_ID ContextVar + os.environ, clobbering the spawner's
        # id ~milliseconds before delegate_tool dispatches the batch.
        from gateway.session_context import set_current_session_id

        set_current_session_id("20260715_child1")
        return fake_child

    monkeypatch.setattr(dt, "_build_child_agent", clobbering_build_child)
    monkeypatch.setattr(dt, "_run_single_child", fast_child)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: creds)
    return dt


def test_apiserver_session_with_id_dispatches_background(monkeypatch):
    """async_delivery=False + a raw session id (HERMES_SESSION_ID) →
    background dispatch (the completion wakes the session via the
    api_server self-post), NOT the forced-sync fallback."""
    dt = _patch_delegate(monkeypatch)
    monkeypatch.setenv("HERMES_SESSION_ID", "raw-sid-7")
    set_session_vars(
        platform="api_server",
        chat_id="raw-sid-7",
        session_key="raw-sid-7",
        session_id="raw-sid-7",
        async_delivery=False,
    )

    out = dt.delegate_task(
        goal="bg on api_server", context="ctx",
        background=True, parent_agent=_fake_parent(),
    )
    parsed = json.loads(out)
    assert parsed["status"] == "dispatched", parsed
    assert parsed["mode"] == "background"

    evt = _drain_one()
    assert evt is not None
    assert evt["type"] == "async_delegation"
    # The raw session id is stamped so the gateway drain can self-post the
    # wake to the REAL session (session_key alone is the raw id here, which
    # carries no parseable routing metadata). Crucially this is the SPAWNER's
    # id, not the subagent-internal id the child build clobbered
    # HERMES_SESSION_ID with (see clobbering_build_child).
    assert evt["origin_session_id"] == "raw-sid-7"


# ---------------------------------------------------------------------------
# _current_origin_session_id — the clobber-proof origin capture helper
# ---------------------------------------------------------------------------


def test_apiserver_session_without_id_stays_synchronous(monkeypatch):
    """No session id to wake → keep the sync fallback (a detached result
    would never re-enter any conversation)."""
    dt = _patch_delegate(monkeypatch)
    set_session_vars(
        platform="api_server",
        chat_id="",
        session_key="",
        session_id="",
        async_delivery=False,
    )

    out = dt.delegate_task(
        goal="one-shot", context="ctx",
        background=True, parent_agent=_fake_parent(),
    )
    parsed = json.loads(out)
    assert parsed.get("status") != "dispatched", parsed
    assert "SYNCHRONOUSLY" in parsed.get("note", "")
    assert process_registry.completion_queue.empty()


# ---------------------------------------------------------------------------
# delegation.sync_when_async_unsupported — opt back into the sync fallback
# ---------------------------------------------------------------------------


def _bind_apiserver(sid="raw-sid-7", async_delivery=False):
    set_session_vars(
        platform="api_server", chat_id=sid, session_key=sid, session_id=sid,
        async_delivery=async_delivery,
    )


def _set_delegation_cfg(monkeypatch, cfg):
    import tools.delegate_tool as dt

    monkeypatch.setattr(dt, "_load_config", lambda: cfg)


def test_sync_key_off_keeps_background_wake(monkeypatch):
    """Key explicitly off → unchanged: background dispatch + self-post wake sid."""
    from tools.delegate_tool_dispatch import _resolve_async_wake_sid

    monkeypatch.delenv("DELEGATION_SYNC_WHEN_ASYNC_UNSUPPORTED", raising=False)
    _set_delegation_cfg(monkeypatch, {"sync_when_async_unsupported": False})
    _bind_apiserver()
    assert _resolve_async_wake_sid("raw-sid-7") == "raw-sid-7"

    dt = _patch_delegate(monkeypatch)
    monkeypatch.setenv("HERMES_SESSION_ID", "raw-sid-7")
    parsed = json.loads(dt.delegate_task(
        goal="bg on api_server", context="ctx", background=True, parent_agent=_fake_parent(),
    ))
    assert parsed["status"] == "dispatched", parsed
    assert parsed["mode"] == "background"
    evt = _drain_one()
    assert evt is not None and evt["origin_session_id"] == "raw-sid-7"


def test_sync_key_on_apiserver_with_session_id_runs_synchronously(monkeypatch):
    """Key on + async_delivery=False + bound session id → None (sync fallback): the
    children's results come back inline in the same tool result, nothing is queued."""
    from tools.delegate_tool_dispatch import _resolve_async_wake_sid

    _set_delegation_cfg(monkeypatch, {"sync_when_async_unsupported": True})
    _bind_apiserver()
    assert _resolve_async_wake_sid("raw-sid-7") is None

    dt = _patch_delegate(monkeypatch)
    monkeypatch.setenv("HERMES_SESSION_ID", "raw-sid-7")
    parsed = json.loads(dt.delegate_task(
        tasks=[{"goal": "first sync task"}, {"goal": "second sync task"}, {"goal": "third sync task"}],
        context="ctx", background=True, parent_agent=_fake_parent(),
    ))
    assert parsed.get("status") != "dispatched", parsed
    assert "SYNCHRONOUSLY" in parsed.get("note", "")
    assert sorted(r["summary"] for r in parsed["results"]) == ["done: first sync task", "done: second sync task", "done: third sync task"]
    assert process_registry.completion_queue.empty()


def test_sync_key_env_fallback(monkeypatch):
    """DELEGATION_SYNC_WHEN_ASYNC_UNSUPPORTED applies when the config key is unset."""
    from tools.delegate_tool_dispatch import _resolve_async_wake_sid

    _set_delegation_cfg(monkeypatch, {})
    _bind_apiserver()
    monkeypatch.setenv("DELEGATION_SYNC_WHEN_ASYNC_UNSUPPORTED", "true")
    assert _resolve_async_wake_sid("raw-sid-7") is None
    monkeypatch.setenv("DELEGATION_SYNC_WHEN_ASYNC_UNSUPPORTED", "false")
    assert _resolve_async_wake_sid("raw-sid-7") == "raw-sid-7"


def test_sync_key_on_async_supported_unchanged(monkeypatch):
    """Key on but the session CAN receive detached completions → unchanged
    ("" = normal async routing, no sync fallback)."""
    from tools.delegate_tool_dispatch import _resolve_async_wake_sid

    _set_delegation_cfg(monkeypatch, {"sync_when_async_unsupported": True})
    _bind_apiserver(async_delivery=True)
    assert _resolve_async_wake_sid("raw-sid-7") == ""
