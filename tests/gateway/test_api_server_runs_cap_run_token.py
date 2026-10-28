"""Cap (A126 §4): /v1/runs binds body.cap_run_token for that run only, and never echoes it."""

import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_runs import _RunLaunch, _cap_run_token


def _app(adapter):
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    return app


def _agent(capture):
    agent = MagicMock()

    def _run(user_message=None, conversation_history=None, task_id=None):
        from gateway.session_context import cap_run_token
        from tools.environments.local import _make_run_env

        capture(user_message, cap_run_token(), _make_run_env({}).get("CAP_RUN_TOKEN"))
        return {"final_response": "done"}

    agent.run_conversation.side_effect = _run
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    return agent


async def _finish(cli, run_id):
    for _ in range(100):
        status = await (await cli.get(f"/v1/runs/{run_id}")).json()
        if status["status"] in {"completed", "failed", "cancelled"}:
            return status
        await asyncio.sleep(0.02)
    raise AssertionError(f"run {run_id} never finished: {status}")


@pytest.mark.asyncio
async def test_run_token_reaches_only_its_own_run():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    seen = {}
    with patch.object(adapter, "_create_agent",
                      side_effect=lambda **_: _agent(lambda m, t, e: seen.__setitem__(m, (t, e)))):
        async with TestClient(TestServer(_app(adapter))) as cli:
            for body in ({"input": "routine", "session_id": "s", "cap_run_token": "tok-secret"},
                         {"input": "chat", "session_id": "s"},
                         {"input": "junk", "session_id": "s", "cap_run_token": {"a": 1}}):
                resp = await cli.post("/v1/runs", json=body)
                assert resp.status == 202
                text = await resp.text()
                status = await _finish(cli, (await resp.json())["run_id"])
                assert status["status"] == "completed", status
                assert "tok-secret" not in text and "tok-secret" not in str(status)
    assert seen == {"routine": ("tok-secret", "tok-secret"), "chat": ("", None), "junk": ("", None)}


@pytest.mark.asyncio
async def test_concurrent_runs_never_see_each_others_token():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    barrier = threading.Barrier(2)
    seen = {}

    def capture(message, token, env_token):
        barrier.wait(5)  # both runs are in flight at once
        seen[message] = (token, env_token)

    with patch.object(adapter, "_create_agent", side_effect=lambda **_: _agent(capture)):
        async with TestClient(TestServer(_app(adapter))) as cli:
            a, b = await asyncio.gather(
                cli.post("/v1/runs", json={"input": "a", "session_id": "s", "cap_run_token": "tok-a"}),
                cli.post("/v1/runs", json={"input": "b", "session_id": "s", "cap_run_token": "tok-b"}))
            for resp in (a, b):
                await _finish(cli, (await resp.json())["run_id"])
    assert seen == {"a": ("tok-a", "tok-a"), "b": ("tok-b", "tok-b")}


def test_body_token_validation():
    assert _cap_run_token({"cap_run_token": " tok "}) == "tok"
    assert _cap_run_token({"cap_run_token": "x" * 4097}) == ""
    assert _cap_run_token({"cap_run_token": "a\nb"}) == ""
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.Ab-_c9"
    assert _cap_run_token({"cap_run_token": jwt}) == jwt
    assert _cap_run_token({"cap_run_token": "x" * 4096}) == "x" * 4096
    # Anything export -p would escape (so a literal snapshot match could miss it) is dropped.
    for unsafe in ("a b", "tok$x", 'a"b', "a'b", "a\\b", "a`b", "tök", "a\tb"):
        assert _cap_run_token({"cap_run_token": unsafe}) == "", unsafe
    assert _cap_run_token({"cap_run_token": 5}) == ""
    assert _cap_run_token({}) == ""
    assert _cap_run_token(None) == ""


def test_launch_repr_never_shows_the_token():
    launch = _RunLaunch(*([None] * 12), cap_run_token="tok-secret")
    assert "tok-secret" not in repr(launch)


@pytest.mark.asyncio
async def test_run_end_shuts_its_kernels_only_when_it_had_a_token():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    ended = []
    with patch.object(adapter, "_create_agent", side_effect=lambda **_: _agent(lambda *a: None)), \
            patch("tools.code_kernel.shutdown_kernels_for_run", side_effect=lambda *a: ended.append(a)):
        async with TestClient(TestServer(_app(adapter))) as cli:
            with_token = await cli.post("/v1/runs", json={"input": "r", "cap_run_token": "tok"})
            run_id = (await with_token.json())["run_id"]
            await _finish(cli, run_id)
            await _finish(cli, (await (await cli.post("/v1/runs", json={"input": "c"})).json())["run_id"])
    # owner = the run's approval session key + its credential; a tokenless run is untouched
    assert ended == [(run_id, "tok")]


@pytest.mark.asyncio
async def test_idempotent_replay_ignores_a_fresh_token(tmp_path):
    from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._run_idempotency_store.close()
    adapter._run_idempotency_store = RunIdempotencyStore(str(tmp_path / "idem.db"))
    with patch.object(adapter, "_create_agent", side_effect=lambda **_: _agent(lambda *a: None)):
        async with TestClient(TestServer(_app(adapter))) as cli:
            headers = {"Idempotency-Key": "retry-1"}
            first = await cli.post("/v1/runs", json={"input": "r", "cap_run_token": "tok-1"}, headers=headers)
            second = await cli.post("/v1/runs", json={"input": "r", "cap_run_token": "tok-2"}, headers=headers)
            assert first.status == second.status == 202
            assert (await first.json())["run_id"] == (await second.json())["run_id"]
            changed = await cli.post("/v1/runs", json={"input": "other", "cap_run_token": "tok-3"}, headers=headers)
            assert changed.status == 409
            await _finish(cli, (await first.json())["run_id"])
