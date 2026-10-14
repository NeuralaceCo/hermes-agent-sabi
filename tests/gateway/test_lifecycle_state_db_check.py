"""An unclean gateway death must trigger a state.db integrity check.

Regression for the 2026-08-31 incident. ``state.db`` was corrupt from
2026-08-26 evening (a SIGKILL landed on a gateway mid-WAL-checkpoint during a
``--replace`` restart storm), but nothing checked the file. The damage sat in
old, rarely-read session rows for 3.5 days until a Desktop read tripped over
it on 2026-08-30 17:15 and surfaced as "Session not found".

``record_startup`` already detects the unclean exit and logs "SIGKILL / OOM /
VM death" — it just never looked at the database that death may have torn.
The check is gated on the unclean exit precisely because it costs ~2s on a
500MB store; a clean boot must not pay it.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path

import pytest

import gateway.lifecycle_ledger as ledger
from gateway.lifecycle_ledger import (
    check_state_db_integrity,
    get_lifecycle_sentinel_path,
    record_startup,
)

_DEAD_PID = 2 ** 22 + 12345  # beyond default pid_max; never alive


def _write_sentinel(home: Path, phase: str = "running") -> None:
    path = get_lifecycle_sentinel_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "phase": phase,
            "pid": _DEAD_PID,
            "start_time": 1000.0,
            "started_at": "2026-08-26T23:56:45+00:00",
        }),
        encoding="utf-8",
    )


def _make_state_db(home: Path, *, corrupt: bool) -> Path:
    """Build a real SQLite file, optionally with a genuinely torn b-tree page."""
    path = home / "state.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany(
        "INSERT INTO sessions (v) VALUES (?)", [(f"row-{i}" * 40,) for i in range(4000)]
    )
    conn.commit()
    conn.close()
    if corrupt:
        with open(path, "r+b") as handle:
            handle.seek(4096 * 6)
            handle.write(b"\xEF" * 4096)
    return path


def _exit_diag_records(home: Path) -> list:
    log = home / "logs" / "gateway-exit-diag.log"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


# ── the checker itself ──────────────────────────────────────────────────────


def test_checker_passes_a_healthy_store(tmp_path: Path) -> None:
    _make_state_db(tmp_path, corrupt=False)
    assert check_state_db_integrity(home=tmp_path) == "ok"


def test_checker_reports_a_torn_btree_page(tmp_path: Path) -> None:
    _make_state_db(tmp_path, corrupt=True)
    verdict = check_state_db_integrity(home=tmp_path)
    assert verdict != "ok"
    assert "btreeInitPage" in verdict or "malformed" in verdict.lower()


def test_checker_tolerates_a_missing_store(tmp_path: Path) -> None:
    assert check_state_db_integrity(home=tmp_path) == "absent"


# ── wiring into the unclean-exit path ───────────────────────────────────────


def test_unclean_exit_records_the_corruption_verdict(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_STARTUP_DB_CHECK", "blocking")
    _make_state_db(tmp_path, corrupt=True)
    _write_sentinel(tmp_path)

    evidence = record_startup(home=tmp_path)

    assert evidence is not None
    assert evidence["state_db_integrity"] != "ok"
    record = _exit_diag_records(tmp_path)[0]
    assert record["state_db_integrity"] != "ok"


def test_unclean_exit_on_a_healthy_store_records_ok(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_STARTUP_DB_CHECK", "blocking")
    _make_state_db(tmp_path, corrupt=False)
    _write_sentinel(tmp_path)

    evidence = record_startup(home=tmp_path)

    assert evidence is not None
    assert evidence["state_db_integrity"] == "ok"


def test_clean_exit_does_not_pay_for_the_check(tmp_path: Path, monkeypatch) -> None:
    """A clean boot must not scan the store — that is the whole cost gate."""
    _make_state_db(tmp_path, corrupt=True)
    _write_sentinel(tmp_path, phase="exited")

    called = []
    monkeypatch.setattr(
        ledger, "check_state_db_integrity", lambda **kw: called.append(1) or "ok"
    )
    record_startup(home=tmp_path)

    assert not called, "integrity check ran on a clean boot"


# ── HERMES_STARTUP_DB_CHECK: when the check runs ────────────────────────────


@pytest.mark.parametrize("mode", ["blocking", "background"])
def test_new_life_is_claimed_before_the_check_runs(tmp_path: Path, monkeypatch, mode: str) -> None:
    """A boot killed mid-check must not leave the PREVIOUS life's "running"
    sentinel behind, or the next boot pays the full check again."""
    monkeypatch.setenv("HERMES_STARTUP_DB_CHECK", mode)
    _make_state_db(tmp_path, corrupt=False)
    _write_sentinel(tmp_path)
    seen = []

    def fake_check(**kw):
        seen.append(json.loads(get_lifecycle_sentinel_path(tmp_path).read_text())["pid"])
        return "ok"

    monkeypatch.setattr(ledger, "check_state_db_integrity", fake_check)
    record_startup(home=tmp_path)
    if ledger._integrity_thread is not None:
        ledger._integrity_thread.join(timeout=5)

    assert seen == [os.getpid()]


def test_background_check_does_not_block_startup(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_STARTUP_DB_CHECK", "background")
    _write_sentinel(tmp_path)
    release = threading.Event()
    monkeypatch.setattr(
        ledger, "check_state_db_integrity", lambda **kw: release.wait(5) and "*** in database main ***"
    )

    evidence = record_startup(home=tmp_path)  # returns while the check is parked

    assert evidence is not None and evidence["state_db_integrity"] == "pending"
    assert [r["tag"] for r in _exit_diag_records(tmp_path)] == ["gateway.previous_unclean_exit"]
    release.set()
    ledger._integrity_thread.join(timeout=5)
    failure = _exit_diag_records(tmp_path)[-1]
    assert failure["tag"] == "gateway.state_db_integrity_failed"
    assert failure["state_db_integrity"] == "*** in database main ***"


def test_background_check_on_a_real_torn_store_flags_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_STARTUP_DB_CHECK", "background")
    _make_state_db(tmp_path, corrupt=True)
    _write_sentinel(tmp_path)

    record_startup(home=tmp_path)
    ledger._integrity_thread.join(timeout=30)

    tags = [r["tag"] for r in _exit_diag_records(tmp_path)]
    assert tags == ["gateway.previous_unclean_exit", "gateway.state_db_integrity_failed"]


def test_background_check_on_a_healthy_store_adds_no_record(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_STARTUP_DB_CHECK", "background")
    _make_state_db(tmp_path, corrupt=False)
    _write_sentinel(tmp_path)

    record_startup(home=tmp_path)
    ledger._integrity_thread.join(timeout=30)

    assert [r["tag"] for r in _exit_diag_records(tmp_path)] == ["gateway.previous_unclean_exit"]


def test_off_skips_the_check_but_still_reports_the_death(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_STARTUP_DB_CHECK", "off")
    _make_state_db(tmp_path, corrupt=True)
    _write_sentinel(tmp_path)
    called = []
    monkeypatch.setattr(ledger, "check_state_db_integrity", lambda **kw: called.append(1) or "ok")

    evidence = record_startup(home=tmp_path)

    assert not called
    assert evidence is not None and evidence["state_db_integrity"] == "skipped"
    assert _exit_diag_records(tmp_path)[0]["state_db_integrity"] == "skipped"


@pytest.mark.parametrize("raw, expected", [
    ("", "background"), ("BLOCKING", "blocking"), (" off ", "off"), ("sometimes", "background"),
])
def test_mode_parsing(monkeypatch, raw: str, expected: str) -> None:
    monkeypatch.setenv("HERMES_STARTUP_DB_CHECK", raw)
    assert ledger.startup_db_check_mode() == expected
