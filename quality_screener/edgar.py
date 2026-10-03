"""Rate-limited, cached HTTP client for SEC EDGAR JSON endpoints.

Every request to sec.gov:
- sends ``config.USER_AGENT`` (SEC fair-access policy: https://www.sec.gov/os/accessing-edgar-data),
- passes through a token-bucket limiter at ``config.MAX_REQUESTS_PER_SEC`` (SEC hard cap: 10/s),
- is retried with exponential backoff on 403, 429 and 5xx (at most ``MAX_TRIES`` attempts),
- is cached byte-for-byte under ``data/raw/<endpoint>/<key>.json``; a cache hit makes no
  network call unless ``refresh=True``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import requests

from quality_screener import config

log = logging.getLogger(__name__)

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{name}"

MAX_TRIES = 5
BACKOFF_BASE_SECONDS = 1.0
TIMEOUT_SECONDS = 30.0


class EdgarError(RuntimeError):
    """A sec.gov request failed permanently (non-retryable status or retries exhausted)."""


class Limiter(Protocol):
    def acquire(self) -> None: ...


class TokenBucket:
    """Token-bucket rate limiter: refills at ``rate`` tokens/second, holds at most ``capacity``.

    ``capacity=1`` (default) enforces even spacing of ``1 / rate`` seconds. A larger bucket allows
    bursts, and a full bucket plus refill can exceed ``rate`` requests inside one second, which
    would breach SEC's 10/s cap.
    """

    def __init__(
        self,
        rate: float,
        capacity: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate <= 0 or capacity < 1:
            raise ValueError("rate must be > 0 and capacity >= 1")
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._clock = clock
        self._sleep = sleep
        self._last = clock()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until one token is available, then consume it."""
        with self._lock:
            self._refill()
            if self._tokens < 1:
                self._sleep((1 - self._tokens) / self._rate)
                self._refill()
            self._tokens -= 1

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._rate)
        self._last = now


@dataclass(frozen=True)
class Submissions:
    """Parsed ``submissions`` endpoint: company metadata plus every filing on record.

    ``filings`` merges ``filings.recent`` with the older pages SEC lists under ``filings.files``,
    one row per filing (10-K/A amendments included), newest first.
    """

    cik: str
    name: str
    sic: int | None
    sic_description: str
    fiscal_year_end: str  # "MMDD", e.g. "0926"
    filings: pd.DataFrame


def pad_cik(cik: int | str) -> str:
    """Return a CIK as the 10-digit zero-padded string SEC URLs use (320193 -> "0000320193")."""
    digits = str(cik).strip().upper().removeprefix("CIK")
    if not digits.isdigit() or len(digits) > 10:
        raise ValueError(f"not a valid CIK: {cik!r}")
    return digits.zfill(10)


def is_retryable(status: int) -> bool:
    """SEC answers 403/429 when throttling and 5xx when overloaded; both are worth retrying."""
    return status in (403, 429) or 500 <= status < 600


def backoff_delay(attempt: int, base: float, retry_after: str | None = None) -> float:
    """Exponential backoff (base, 2*base, 4*base, ...), never shorter than a numeric Retry-After."""
    delay = base * 2 ** (attempt - 1)
    if retry_after is not None and retry_after.strip().isdigit():
        delay = max(delay, float(retry_after))
    return delay


class EdgarClient:
    """Small client for the SEC EDGAR JSON API. See module docstring for request discipline."""

    def __init__(
        self,
        cache_dir: Path = config.RAW_DIR,
        *,
        session: requests.Session | None = None,
        limiter: Limiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_tries: int = MAX_TRIES,
        backoff_base: float = BACKOFF_BASE_SECONDS,
        timeout: float = TIMEOUT_SECONDS,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.session = session if session is not None else requests.Session()
        self.session.headers.update(
            {"User-Agent": config.USER_AGENT, "Accept-Encoding": "gzip, deflate"}
        )
        self.limiter = limiter if limiter is not None else TokenBucket(config.MAX_REQUESTS_PER_SEC)
        self.max_tries = max_tries
        self.backoff_base = backoff_base
        self.timeout = timeout
        self.network_calls = 0
        self._sleep = sleep

    def get_ticker_map(self, refresh: bool = False) -> pd.DataFrame:
        """All SEC-registered tickers as columns ``cik`` (10-digit str), ``ticker``, ``title``.

        One row per ticker, so a CIK with several listed share classes appears several times.
        """
        data = self._get_json(TICKER_MAP_URL, "company_tickers", "company_tickers", refresh)
        rows = pd.DataFrame.from_records(list(data.values()))
        return pd.DataFrame(
            {
                "cik": rows["cik_str"].map(pad_cik),
                "ticker": rows["ticker"].str.upper(),
                "title": rows["title"],
            }
        )

    def get_company_facts(self, cik: int | str, refresh: bool = False) -> dict[str, Any]:
        """Raw XBRL ``companyfacts`` JSON: every reported fact, every filing that reported it."""
        cik10 = pad_cik(cik)
        url = COMPANY_FACTS_URL.format(cik=cik10)
        return self._get_json(url, "companyfacts", f"CIK{cik10}", refresh)

    def get_submissions(self, cik: int | str, refresh: bool = False) -> Submissions:
        """Company metadata and the full filing list, including older paged-out filing files."""
        cik10 = pad_cik(cik)
        main = self._get_json(
            SUBMISSIONS_URL.format(name=f"CIK{cik10}.json"), "submissions", f"CIK{cik10}", refresh
        )
        pages = [main["filings"]["recent"]]
        for extra in main["filings"].get("files", []):
            name = extra["name"]
            url = SUBMISSIONS_URL.format(name=name)
            pages.append(self._get_json(url, "submissions", Path(name).stem, refresh))
        return Submissions(
            cik=cik10,
            name=main.get("name") or "",
            sic=_parse_sic(main.get("sic")),
            sic_description=main.get("sicDescription") or "",
            fiscal_year_end=main.get("fiscalYearEnd") or "",
            filings=_filings_frame(pages),
        )

    def get_filing_index(self, cik: int | str, accn: str, refresh: bool = False) -> dict[str, Any]:
        """The filing's folder listing on EDGAR Archives (``index.json``): every document in
        the submission, e.g. to see whether it carried XBRL exhibits."""
        url = archive_url(cik, accn, "index.json")
        return self._get_json(url, f"archives/{pad_cik(cik)}", f"{accn}-index", refresh)

    def get_filing_document(
        self, cik: int | str, accn: str, name: str, refresh: bool = False
    ) -> str:
        """One document of a filing (e.g. the 10-K/A's primary HTML), cached as fetched."""
        path = self.cache_dir / "archives" / pad_cik(cik) / accn / name
        if path.exists() and not refresh:
            return path.read_bytes().decode("utf-8", errors="replace")
        body = self._fetch(archive_url(cik, accn, name))
        _atomic_write(path, body)
        return body.decode("utf-8", errors="replace")

    def _get_json(self, url: str, endpoint: str, key: str, refresh: bool) -> Any:
        """Return parsed JSON for ``url``, from ``cache_dir/endpoint/key.json`` when cached."""
        path = self.cache_dir / endpoint / f"{key}.json"
        if path.exists() and not refresh:
            return json.loads(path.read_bytes())
        body = self._fetch(url)
        try:
            data = json.loads(body)
        except ValueError as exc:
            raise EdgarError(f"GET {url} returned a non-JSON body; not cached") from exc
        _atomic_write(path, body)
        return data

    def _fetch(self, url: str) -> bytes:
        """GET ``url`` with rate limiting and retries; return the response body."""
        last_problem = ""
        for attempt in range(1, self.max_tries + 1):
            self.limiter.acquire()
            self.network_calls += 1
            retry_after = None
            try:
                resp = self.session.get(url, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_problem = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code == 200:
                    return resp.content
                if not is_retryable(resp.status_code):
                    raise EdgarError(f"GET {url} -> HTTP {resp.status_code} (not retryable)")
                last_problem = f"HTTP {resp.status_code}"
                retry_after = resp.headers.get("Retry-After")
            if attempt < self.max_tries:
                delay = backoff_delay(attempt, self.backoff_base, retry_after)
                log.warning("GET %s: %s; retry %d in %.1fs", url, last_problem, attempt, delay)
                self._sleep(delay)
        raise EdgarError(
            f"GET {url} failed after {self.max_tries} tries (last: {last_problem}). "
            "Repeated 403/429 from SEC usually means a missing User-Agent or a rate-limit "
            "block, which SEC typically lifts after about 10 minutes."
        )


def archive_url(cik: int | str, accn: str, name: str = "") -> str:
    """EDGAR Archives URL of a filing folder or one of its documents."""
    return ARCHIVES_URL.format(cik=int(pad_cik(cik)), folder=accn.replace("-", ""), name=name)


def _parse_sic(raw: Any) -> int | None:
    text = str(raw).strip() if raw is not None else ""
    return int(text) if text.isdigit() else None


def _filings_frame(pages: list[dict[str, list[Any]]]) -> pd.DataFrame:
    """Stack SEC's column-oriented filing pages into one DataFrame, newest filing first."""
    frames = [pd.DataFrame(page) for page in pages if page]
    if not frames:
        return pd.DataFrame()
    filings = pd.concat(frames, ignore_index=True)
    for col in ("filingDate", "reportDate"):
        if col in filings:
            filings[col] = pd.to_datetime(filings[col], errors="coerce")
    if "filingDate" in filings:
        filings = filings.sort_values("filingDate", ascending=False, ignore_index=True)
    return filings


def _atomic_write(path: Path, body: bytes) -> None:
    """Write via a temp file so an interrupted run never leaves a truncated cache entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(body)
    tmp.replace(path)
