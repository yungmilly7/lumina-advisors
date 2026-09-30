"""SEC EDGAR client: recent filings per ticker (10-K, 10-Q, 8-K).

SEC requires a descriptive User-Agent with contact info on every request,
and rate-limits aggressively, so we keep requests minimal and cache the
ticker->CIK map for the whole run.
"""
from __future__ import annotations

import threading

from app.config import SEC_USER_AGENT
from app.dataclients.httpjson import get_json

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"

_headers = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}

_cik_map_cache: dict[str, str] | None = None
# The ingestion pipeline fetches multiple tickers concurrently (see
# app.pipeline), and every one of them calls _load_cik_map() on its first
# filings lookup -- without this lock, several worker threads could all
# see the cache as empty at once and each fire off its own redundant
# fetch of the (fairly large) company_tickers.json file.
_cik_map_lock = threading.Lock()


def _load_cik_map() -> dict[str, str]:
    global _cik_map_cache
    if _cik_map_cache is not None:
        return _cik_map_cache
    with _cik_map_lock:
        if _cik_map_cache is not None:  # another thread filled it while we waited
            return _cik_map_cache
        data = get_json(TICKERS_URL, headers=_headers)
        _cik_map_cache = {
            row["ticker"].upper(): str(row["cik_str"]) for row in data.values()
        }
        return _cik_map_cache


def fetch_recent_filings(ticker: str, limit: int = 10) -> list[dict]:
    cik_map = _load_cik_map()
    cik = cik_map.get(ticker.upper())
    if not cik:
        return []
    data = get_json(SUBMISSIONS_URL.format(cik=cik), headers=_headers)

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accns = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])

    # SEC's "recent" list is every filing (10-Ks/10-Qs/8-Ks, but also the far
    # more frequent Form 4s, DEF 14As, S-8s, SC 13Gs, ...), newest first. An
    # earlier version of this loop only looked at the first `limit` raw
    # entries and then filtered those down to the big three forms -- for any
    # company whose most recent filings happen to be mostly Form 4s (which
    # is most companies, most of the time), that returned zero or almost no
    # rows even though 10-Ks/10-Qs/8-Ks were sitting right there further
    # down the list. Scan the whole list and collect the first `limit`
    # *matching* filings instead.
    rows = []
    for i in range(len(forms)):
        if forms[i] not in ("10-K", "10-Q", "8-K"):
            continue
        accn_nodash = accns[i].replace("-", "")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{accn_nodash}/{docs[i]}"
        )
        rows.append(
            {
                "filed_date": dates[i],
                "form_type": forms[i],
                "title": f"{forms[i]} filed {dates[i]}",
                "url": url,
            }
        )
        if len(rows) >= limit:
            break
    return rows
