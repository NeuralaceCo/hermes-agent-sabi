"""Inline agent-loop tools must emit ``post_tool_call`` (observer plugins).

``session_search``/``todo``/``memory``/… are dispatched by the sequential
executor through ``_run_agent_tool_execution_middleware`` with an ``_execute``
closure that calls the tool function directly — nothing went through
``handle_function_call``, so ``post_tool_call`` never fired for them. The
Langfuse plugin opens a tool span on ``pre_tool_call`` and only closes it with
the result on ``post_tool_call``; without it the span is flush-closed at trace
finish with ``output=null`` (2151/2162 prod ``Tool: session_search`` spans,
2026-08-27). The middleware now emits for exactly that inline set.
"""

import json
from unittest.mock import MagicMock

import pytest

import agent.tool_executor as te
from tests.run_agent.test_tool_activity_heartbeat import _make_agent  # noqa: F401 (fixture-style helper)


@pytest.fixture()
def emitted(monkeypatch):
    calls: list = []
    monkeypatch.setattr(te, "_emit_terminal_post_tool_call", lambda agent, **kw: calls.append(kw))
    return calls


def _run(monkeypatch, name, execute):
    agent = _make_agent(monkeypatch)
    agent._tool_guardrails = MagicMock(before_call=lambda n, a: MagicMock(allows_execution=True))
    return te._run_agent_tool_execution_middleware(
        agent,
        function_name=name,
        function_args={"query": "x"},
        effective_task_id="task",
        tool_call_id="tc1",
        execute=execute,
        display_index=1,
    )


def test_session_search_emits_post_tool_call_with_result(monkeypatch, emitted):
    payload = json.dumps({"success": True, "results": []})
    managed = _run(monkeypatch, "session_search", lambda args: payload)
    assert managed.result == payload
    assert len(emitted) == 1
    kw = emitted[0]
    assert kw["function_name"] == "session_search"
    assert kw["result"] == payload
    assert kw["tool_call_id"] == "tc1"
    assert kw["function_args"] == {"query": "x"}
    assert kw["duration_ms"] >= 0


@pytest.mark.parametrize("name", sorted(te._INLINE_TOOLS_WITHOUT_POST_HOOK))
def test_every_inline_tool_emits_once(monkeypatch, emitted, name):
    _run(monkeypatch, name, lambda args: "ok")
    assert [kw["function_name"] for kw in emitted] == [name]


def test_generic_tool_is_left_to_its_own_path(monkeypatch, emitted):
    # Generic tools go through handle_function_call, which emits itself —
    # the middleware must not double-fire for them.
    _run(monkeypatch, "terminal", lambda args: "ok")
    assert emitted == []


def test_no_emit_when_execute_raises(monkeypatch, emitted):
    with pytest.raises(RuntimeError):
        _run(monkeypatch, "session_search", lambda args: (_ for _ in ()).throw(RuntimeError("boom")))
    assert emitted == []
