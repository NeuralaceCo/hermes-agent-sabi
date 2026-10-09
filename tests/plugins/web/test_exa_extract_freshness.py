"""Exa extract must return the page as it is now, not Exa's old copy (SW-1105).

The fake client below follows Exa's /contents freshness rules as measured against the live
API on 2026-10-09 (exa-py 2.10.2): with no ``max_age_hours`` Exa serves its stored copy,
however old (the live WTA order-of-play PDF came back with the previous day's sheet);
``livecrawl="preferred"`` is ignored and also serves that copy; ``max_age_hours=N`` fetches
the page when the stored copy is older than N hours, and serves the stored copy when that
fetch runs past ``livecrawl_timeout`` (default 10 s); ``max_age_hours=0`` returns no page
at all when the fetch times out.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

URL = "https://wtafiles.wtatennis.com/pdf/draws/2026/1020/OP.pdf"
STORED = "ORDER OF PLAY - THURSDAY, 8 OCTOBER 2026"
LIVE = "ORDER OF PLAY - FRIDAY, 9 OCTOBER 2026"


class _ExaWithOldCopy:
    def __init__(self, stored_age_hours: float, fetch_ms: int) -> None:
        self.headers: dict = {}
        self.stored_age_hours, self.fetch_ms = stored_age_hours, fetch_ms

    def get_contents(self, urls, text=True, max_age_hours=None, livecrawl_timeout=10_000, **_):
        fetch = max_age_hours is not None and max_age_hours >= 0 and (
            max_age_hours == 0 or self.stored_age_hours > max_age_hours
        )
        timed_out = fetch and self.fetch_ms > livecrawl_timeout
        if timed_out and max_age_hours == 0:
            return SimpleNamespace(results=[])  # status error CRAWL_LIVECRAWL_TIMEOUT, no text
        body = LIVE if fetch and not timed_out else STORED
        return SimpleNamespace(results=[SimpleNamespace(url=u, title="Order of play", text=body) for u in urls])


@pytest.fixture
def exa(monkeypatch: pytest.MonkeyPatch):
    import tools.web_tools

    monkeypatch.setenv("EXA_API_KEY", "test-key")

    def install(stored_age_hours: float, fetch_ms: int):
        monkeypatch.setattr(tools.web_tools, "_exa_client", _ExaWithOldCopy(stored_age_hours, fetch_ms), raising=False)
        from plugins.web.exa.provider import ExaWebSearchProvider
        return ExaWebSearchProvider()

    return install


def test_extract_returns_the_page_as_it_is_now(exa) -> None:
    provider = exa(stored_age_hours=14, fetch_ms=2_000)
    [page] = provider.extract([URL])
    assert page["content"] == LIVE


def test_a_slow_fetch_falls_back_to_the_stored_copy_instead_of_losing_the_page(exa) -> None:
    provider = exa(stored_age_hours=14, fetch_ms=60_000)
    [page] = provider.extract([URL])
    assert page["content"] == STORED
