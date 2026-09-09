"""Two small regressions around the sequential tool-completion path:

1. ``session_search`` (an inline agent-loop tool dispatched without going
   through ``handle_function_call``) must get exactly one terminal
   ``post_tool_call`` out of the sequential publish path — Langfuse and other
   observer plugins open a tool span on ``pre_tool_call`` and only close it
   with a result on ``post_tool_call``; without it the span flush-closes at
   trace end with no output. ``_publish_sequential_result`` owns this via
   ``ref.emit_post`` for every non-blocked, non-timed-out result.
2. ``_commit_tool_result``'s ``tool.completed`` projection must carry the
   (redacted) tool args as its 4th positional, not ``None`` — display/UI
   consumers and observer plugins both need the args the call actually ran
   with to render/label the completed card.
"""

import json
from unittest.mock import MagicMock

from agent.tool_executor import (
    _ManagedToolResult,
    _ToolCallRef,
    _commit_tool_result,
    _publish_sequential_result,
)
from tests.run_agent.test_tool_activity_heartbeat import _make_agent  # reusable AIAgent-like stub
from tools.budget_config import BudgetConfig


def test_session_search_emits_exactly_one_terminal_post_tool_call(monkeypatch):
    import agent.tool_executor as te

    emitted: list = []
    monkeypatch.setattr(
        te, "_emit_terminal_post_tool_call", lambda agent, **kw: emitted.append(kw)
    )

    agent = _make_agent(monkeypatch)
    messages: list = []
    ref = _ToolCallRef(
        name="session_search", args={"query": "x"}, task_id="t", call_id="tc1", trace=[],
    )
    payload = json.dumps({"success": True, "results": []})
    managed = _ManagedToolResult(
        result=payload, args=ref.args, middleware_trace=[], blocked=False, dispatched=True,
    )

    ok = _publish_sequential_result(
        agent, messages, ref, managed, tool_duration=0.01, index=1, budget=BudgetConfig(),
    )

    assert ok is True
    assert len(emitted) == 1
    kw = emitted[0]
    assert kw["function_name"] == "session_search"
    assert kw["result"] == payload
    assert kw["tool_call_id"] == "tc1"
    assert kw["function_args"] == {"query": "x"}


def test_commit_tool_result_tool_completed_carries_redacted_args(monkeypatch):
    agent = _make_agent(monkeypatch)
    agent.tool_progress_callback = MagicMock()

    secret_text = "sk-" + "a" * 20
    ref = _ToolCallRef(
        name="browser_type", args={"ref": "e1", "text": secret_text}, task_id="t", call_id="tc2", trace=[],
    )
    messages: list = []

    committed = _commit_tool_result(
        agent, messages, ref, "ok",
        budget=BudgetConfig(), tool_duration=0.02, is_error=False, blocked=False,
        effect_disposition=None, observed=False,
    )

    assert committed is not None
    agent.tool_progress_callback.assert_called_once()
    call = agent.tool_progress_callback.call_args
    assert call.args[0] == "tool.completed"
    assert call.args[1] == "browser_type"
    display_args = call.args[3]
    assert display_args is not None
    assert display_args["ref"] == "e1"
    assert display_args["text"] != secret_text  # browser_type text is redacted for display


def test_commit_tool_result_falls_back_to_raw_args_on_redaction_failure(monkeypatch):
    """A broken/unregistered redactor must never drop args off the callback."""
    import agent.tool_executor as te

    def _boom(name, args):
        raise RuntimeError("redactor exploded")

    monkeypatch.setattr(te, "_redact_tool_args_for_display", _boom)

    agent = _make_agent(monkeypatch)
    agent.tool_progress_callback = MagicMock()

    ref = _ToolCallRef(name="terminal", args={"command": "ls"}, task_id="t", call_id="tc3", trace=[])
    messages: list = []

    committed = te._commit_tool_result(
        agent, messages, ref, "ok",
        budget=BudgetConfig(), tool_duration=0.02, is_error=False, blocked=False,
        effect_disposition=None, observed=False,
    )

    assert committed is not None
    call = agent.tool_progress_callback.call_args
    assert call.args[3] == {"command": "ls"}
