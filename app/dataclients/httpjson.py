"""Tiny stdlib-only HTTP helper shared by the data clients (Yahoo, Finnhub,
SEC EDGAR, Google News). Uses urllib instead of httpx so this project has
zero third-party dependencies beyond numpy/pandas -- important for running
in places where `pip install` isn't possible (locked-down networks)."""
from __future__ import annotations

import gzip
import json
import logging
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib

from app.config import HTTP_TIMEOUT

log = logging.getLogger("stockgraph.http")

# HTTP statuses worth retrying: rate-limited or a transient server-side
# hiccup. Deliberately NOT 401/403/404 -- those mean "blocked" or "wrong
# URL/ticker", and retrying an auth failure or a not-found just burns time
# and rate-limit budget for the same outcome. This distinction is what lets
# a real rate-limit blip recover instead of permanently downgrading a
# ticker to demo data for the rest of the day.
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class HTTPError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _build_url(url: str, params: dict | None) -> str:
    if not params:
        return url
    qs = urllib.parse.urlencode(params)
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{qs}"


def _sleep_backoff(attempt: int, base: float) -> None:
    # Exponential backoff with full jitter (AWS's recommended shape): avoids
    # every failed ticker in a thread-pool batch waking up and retrying in
    # lockstep, which would just recreate the same burst that got
    # rate-limited in the first place.
    time.sleep(random.uniform(0, base * (2 ** attempt)))


def get_text(url: str, params: dict | None = None, headers: dict | None = None,
             timeout: float = HTTP_TIMEOUT, opener=None,
             retries: int = 2, backoff_base: float = 0.75) -> str:
    """`opener` is an optional urllib.request.OpenerDirector (e.g. one built
    with an HTTPCookieProcessor) for endpoints that need cookies carried
    across requests, such as Yahoo Finance's crumb-protected APIs. Falls
    back to a plain one-off request when omitted.

    `retries` applies only to RETRYABLE_STATUSES (429/5xx) and to network-
    level failures (DNS/connection errors) -- a 401/403/404 raises
    immediately since retrying it can't succeed. Pass retries=0 to disable."""
    full_url = _build_url(url, params)
    opener_fn = opener.open if opener is not None else urllib.request.urlopen
    attempt = 0
    while True:
        req = urllib.request.Request(full_url, headers=headers or {})
        try:
            with opener_fn(req, timeout=timeout) as resp:
                raw = resp.read()
                encoding = (resp.headers.get("Content-Encoding") or "").lower()
                if encoding == "gzip":
                    raw = gzip.decompress(raw)
                elif encoding == "deflate":
                    raw = zlib.decompress(raw)
                return raw.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code in RETRYABLE_STATUSES and attempt < retries:
                log.debug("HTTP %s for %s, retrying (attempt %d/%d)", e.code, full_url, attempt + 1, retries)
                _sleep_backoff(attempt, backoff_base)
                attempt += 1
                continue
            raise HTTPError(f"HTTP {e.code} for {full_url}: {e.reason}", status=e.code) from e
        except urllib.error.URLError as e:
            if attempt < retries:
                _sleep_backoff(attempt, backoff_base)
                attempt += 1
                continue
            raise HTTPError(f"failed to reach {full_url}: {e.reason}") from e


def get_json(url: str, params: dict | None = None, headers: dict | None = None,
             timeout: float = HTTP_TIMEOUT, opener=None,
             retries: int = 2, backoff_base: float = 0.75):
    text = get_text(url, params=params, headers=headers, timeout=timeout, opener=opener,
                     retries=retries, backoff_base=backoff_base)
    return json.loads(text)
