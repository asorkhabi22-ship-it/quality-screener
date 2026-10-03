"""EdgarClient: rate limiting, caching, CIK padding, retries. No real network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests
from fakes import FakeClock, FakeResponse, FakeSession, NoLimit, json_response

from quality_screener import config
from quality_screener.edgar import (
    COMPANY_FACTS_URL,
    SUBMISSIONS_URL,
    TICKER_MAP_URL,
    EdgarClient,
    EdgarError,
    TokenBucket,
    archive_url,
    backoff_delay,
    pad_cik,
)

AAPL = "0000320193"
FACTS_URL = COMPANY_FACTS_URL.format(cik=AAPL)
FACTS = {"cik": 320193, "entityName": "Apple Inc.", "facts": {}}


def make_client(tmp_path: Path, routes: dict, clock: FakeClock | None = None) -> EdgarClient:
    clock = clock or FakeClock()
    return EdgarClient(
        tmp_path, session=FakeSession(routes), limiter=NoLimit(), sleep=clock.sleep
    )


# --- rate limiter -------------------------------------------------------------------------


def test_token_bucket_spaces_requests_evenly() -> None:
    clock = FakeClock()
    bucket = TokenBucket(rate=8, clock=clock, sleep=clock.sleep)
    stamps = []
    for _ in range(25):
        bucket.acquire()
        stamps.append(clock.now)
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert stamps[0] == 0.0
    assert min(gaps) == pytest.approx(1 / 8)
    # never more than 8 requests inside any 1-second window
    assert all(stamps[i + 8] - stamps[i] >= 1.0 - 1e-9 for i in range(len(stamps) - 8))


def test_token_bucket_does_not_wait_when_idle() -> None:
    clock = FakeClock()
    bucket = TokenBucket(rate=8, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        bucket.acquire()
        clock.now += 1.0  # caller is slower than the limit
    assert clock.sleeps == []


def test_default_limiter_uses_config_rate(tmp_path: Path) -> None:
    client = EdgarClient(tmp_path, session=FakeSession({}))
    assert isinstance(client.limiter, TokenBucket)
    assert client.limiter._rate == config.MAX_REQUESTS_PER_SEC


# --- headers and CIK padding --------------------------------------------------------------


def test_session_sends_sec_headers(tmp_path: Path) -> None:
    client = make_client(tmp_path, {})
    assert client.session.headers["User-Agent"] == config.USER_AGENT
    assert client.session.headers["Accept-Encoding"] == "gzip, deflate"


@pytest.mark.parametrize("raw", [320193, "320193", " 320193 ", "CIK0000320193", "0000320193"])
def test_pad_cik(raw: int | str) -> None:
    assert pad_cik(raw) == AAPL


@pytest.mark.parametrize("raw", ["", "AAPL", "12345678901", 320193.0])
def test_pad_cik_rejects_garbage(raw: object) -> None:
    with pytest.raises(ValueError):
        pad_cik(raw)  # type: ignore[arg-type]


def test_ticker_map_pads_ciks(tmp_path: Path) -> None:
    payload = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"},
    }
    client = make_client(tmp_path, {TICKER_MAP_URL: [json_response(payload)]})
    tickers = client.get_ticker_map()
    assert list(tickers.columns) == ["cik", "ticker", "title"]
    assert tickers["cik"].tolist() == [AAPL, "0001067983"]


def test_company_facts_url_uses_padded_cik(tmp_path: Path) -> None:
    client = make_client(tmp_path, {FACTS_URL: [json_response(FACTS)]})
    client.get_company_facts(320193)
    assert client.session.calls == [
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"
    ]


# --- cache --------------------------------------------------------------------------------


def test_first_call_writes_cache_second_call_skips_network(tmp_path: Path) -> None:
    client = make_client(tmp_path, {FACTS_URL: [json_response(FACTS)]})
    assert client.get_company_facts(AAPL) == FACTS
    assert (tmp_path / "companyfacts" / f"CIK{AAPL}.json").exists()

    assert client.get_company_facts(AAPL) == FACTS
    assert client.network_calls == 1
    assert client.limiter.acquired == 1  # cache hits do not consume rate-limit tokens


def test_pre_existing_cache_file_means_zero_requests(tmp_path: Path) -> None:
    cached = tmp_path / "companyfacts" / f"CIK{AAPL}.json"
    cached.parent.mkdir(parents=True)
    cached.write_text(json.dumps(FACTS))
    client = make_client(tmp_path, {})  # any request would raise AssertionError
    assert client.get_company_facts(AAPL) == FACTS
    assert client.network_calls == 0


def test_refresh_bypasses_cache(tmp_path: Path) -> None:
    updated = {**FACTS, "entityName": "Apple Inc. (refreshed)"}
    client = make_client(tmp_path, {FACTS_URL: [json_response(FACTS), json_response(updated)]})
    client.get_company_facts(AAPL)
    assert client.get_company_facts(AAPL, refresh=True) == updated
    assert client.network_calls == 2


def test_non_json_body_raises_and_is_not_cached(tmp_path: Path) -> None:
    html = FakeResponse(200, b"<html>Request Rate Threshold Exceeded</html>")
    client = make_client(tmp_path, {FACTS_URL: [html]})
    with pytest.raises(EdgarError, match="non-JSON"):
        client.get_company_facts(AAPL)
    assert not (tmp_path / "companyfacts" / f"CIK{AAPL}.json").exists()


# --- retries ------------------------------------------------------------------------------


def test_retries_on_429_with_exponential_backoff(tmp_path: Path) -> None:
    clock = FakeClock()
    routes = {FACTS_URL: [FakeResponse(429), FakeResponse(429), json_response(FACTS)]}
    client = make_client(tmp_path, routes, clock)
    assert client.get_company_facts(AAPL) == FACTS
    assert client.network_calls == 3
    assert clock.sleeps == [1.0, 2.0]


@pytest.mark.parametrize("status", [403, 500, 503])
def test_retries_on_403_and_5xx(tmp_path: Path, status: int) -> None:
    routes = {FACTS_URL: [FakeResponse(status), json_response(FACTS)]}
    client = make_client(tmp_path, routes)
    assert client.get_company_facts(AAPL) == FACTS
    assert client.network_calls == 2


def test_retry_after_header_is_respected(tmp_path: Path) -> None:
    clock = FakeClock()
    routes = {FACTS_URL: [FakeResponse(429, headers={"Retry-After": "7"}), json_response(FACTS)]}
    make_client(tmp_path, routes, clock).get_company_facts(AAPL)
    assert clock.sleeps == [7.0]


def test_connection_error_is_retried(tmp_path: Path) -> None:
    routes = {FACTS_URL: [requests.ConnectionError("reset"), json_response(FACTS)]}
    client = make_client(tmp_path, routes)
    assert client.get_company_facts(AAPL) == FACTS


def test_gives_up_after_five_tries(tmp_path: Path) -> None:
    clock = FakeClock()
    client = make_client(tmp_path, {FACTS_URL: [FakeResponse(429)] * 5}, clock)
    with pytest.raises(EdgarError, match="failed after 5 tries.*HTTP 429"):
        client.get_company_facts(AAPL)
    assert client.network_calls == 5
    assert clock.sleeps == [1.0, 2.0, 4.0, 8.0]


def test_404_is_not_retried(tmp_path: Path) -> None:
    client = make_client(tmp_path, {FACTS_URL: [FakeResponse(404)]})
    with pytest.raises(EdgarError, match="HTTP 404"):
        client.get_company_facts(AAPL)
    assert client.network_calls == 1


def test_backoff_ignores_non_numeric_retry_after() -> None:
    assert backoff_delay(3, 1.0, "Wed, 21 Oct 2026 07:28:00 GMT") == 4.0


# --- submissions --------------------------------------------------------------------------


def test_submissions_merges_paged_filing_files(tmp_path: Path) -> None:
    main = {
        "name": "Apple Inc.",
        "sic": "3571",
        "sicDescription": "Electronic Computers",
        "fiscalYearEnd": "0926",
        "filings": {
            "recent": {
                "accessionNumber": ["a-3", "a-2"],
                "form": ["10-K", "10-K/A"],
                "filingDate": ["2024-11-01", "2024-02-01"],
                "reportDate": ["2024-09-28", ""],
            },
            "files": [{"name": f"CIK{AAPL}-submissions-001.json"}],
        },
    }
    older = {
        "accessionNumber": ["a-1"],
        "form": ["10-K"],
        "filingDate": ["2009-10-27"],
        "reportDate": ["2009-09-26"],
    }
    routes = {
        SUBMISSIONS_URL.format(name=f"CIK{AAPL}.json"): [json_response(main)],
        SUBMISSIONS_URL.format(name=f"CIK{AAPL}-submissions-001.json"): [json_response(older)],
    }
    client = make_client(tmp_path, routes)
    sub = client.get_submissions(320193)

    assert (sub.cik, sub.sic, sub.fiscal_year_end) == (AAPL, 3571, "0926")
    assert sub.filings["accessionNumber"].tolist() == ["a-3", "a-2", "a-1"]
    assert sub.filings["reportDate"].isna().tolist() == [False, True, False]
    assert (tmp_path / "submissions" / f"CIK{AAPL}-submissions-001.json").exists()

    client.get_submissions(320193)
    assert client.network_calls == 2  # both pages served from cache the second time


def test_blank_sic_is_none(tmp_path: Path) -> None:
    main = {"name": "Shell Co", "sic": "", "filings": {"recent": {}}}
    url = SUBMISSIONS_URL.format(name=f"CIK{AAPL}.json")
    sub = make_client(tmp_path, {url: [json_response(main)]}).get_submissions(AAPL)
    assert sub.sic is None
    assert sub.filings.empty


# --- EDGAR Archives -------------------------------------------------------------------------

COF = "0000927628"
ACCN = "0001140361-11-014724"
FOLDER = "https://www.sec.gov/Archives/edgar/data/927628/000114036111014724/"


def test_archive_url_uses_unpadded_cik_and_bare_accession() -> None:
    assert archive_url(COF, ACCN) == FOLDER
    assert archive_url(927628, ACCN, "form10ka.htm") == FOLDER + "form10ka.htm"


def test_filing_index_and_document_are_cached(tmp_path: Path) -> None:
    index = {"directory": {"item": [{"name": "form10ka.htm"}]}}
    routes = {
        FOLDER + "index.json": [json_response(index)],
        FOLDER + "form10ka.htm": [FakeResponse(200, b"<p>Explanatory Note</p>")],
    }
    client = make_client(tmp_path, routes)
    assert client.get_filing_index(COF, ACCN) == index
    assert client.get_filing_document(COF, ACCN, "form10ka.htm") == "<p>Explanatory Note</p>"
    again = make_client(tmp_path, {})  # any request would fail: both served from cache
    assert again.get_filing_index(COF, ACCN) == index
    assert "Explanatory" in again.get_filing_document(COF, ACCN, "form10ka.htm")
    assert again.network_calls == 0
