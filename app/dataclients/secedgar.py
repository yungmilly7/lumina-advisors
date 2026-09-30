"""SEC EDGAR client: recent filings per ticker (10-K, 10-Q, 8-K), plus
insider transactions (Form 4).

SEC requires a descriptive User-Agent with contact info on every request,
and rate-limits aggressively, so we keep requests minimal and cache the
ticker->CIK map for the whole run.
"""
from __future__ import annotations

import logging
import threading
import xml.etree.ElementTree as ET

from app.config import SEC_USER_AGENT
from app.dataclients.httpjson import get_json, get_text

log = logging.getLogger("stockgraph.secedgar")

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


def _value_text(parent: ET.Element | None, tag: str) -> str | None:
    """Form 4 XML wraps most leaf values in a nested <value> element (this
    lets the schema attach a "footnoteId" sibling to the same field), but a
    few filers emit the value directly on the tag itself. Handle both.
    """
    if parent is None:
        return None
    node = parent.find(tag)
    if node is None:
        return None
    value_node = node.find("value")
    text = value_node.text if value_node is not None else node.text
    return text.strip() if text else None


def _parse_form4_xml(raw_xml: str) -> list[dict]:
    """Parse a Form 4 ownershipDocument XML into open-market buy/sell rows.

    Only transactionCode P (open-market purchase) and S (open-market sale)
    are kept -- these are the only codes that reflect a genuine discretionary
    trading decision by the insider. Everything else (A = grant/award,
    F = tax withholding, M = option exercise, G = gift, ...) is
    compensation/administrative noise by academic convention and would just
    dilute the signal.

    Deliberately defensive: SEC filings are produced by hundreds of
    different filer agents over decades and are not perfectly uniform, so
    any row that doesn't parse cleanly is skipped rather than raising.
    """
    try:
        root = ET.fromstring(raw_xml)
    except ET.ParseError:
        return []

    owner = root.find("reportingOwner")
    owner_name = _value_text(owner, "reportingOwnerId/rptOwnerName")
    relationship = owner.find("reportingOwnerRelationship") if owner is not None else None
    is_officer = (_value_text(relationship, "isOfficer") or "0") in ("1", "true")
    is_director = (_value_text(relationship, "isDirector") or "0") in ("1", "true")
    is_ten_pct = (_value_text(relationship, "isTenPercentOwner") or "0") in ("1", "true")

    rows: list[dict] = []
    table = root.find("nonDerivativeTable")
    if table is None:
        return rows
    for txn in table.findall("nonDerivativeTransaction"):
        coding = txn.find("transactionCoding")
        code = _value_text(coding, "transactionCode")
        if code not in ("P", "S"):
            continue
        amounts = txn.find("transactionAmounts")
        shares_text = _value_text(amounts, "transactionShares")
        price_text = _value_text(amounts, "transactionPricePerShare")
        acquired_disposed = _value_text(amounts, "transactionAcquiredDisposedCode")
        txn_date = _value_text(txn, "transactionDate")
        if not shares_text or not txn_date:
            continue
        try:
            shares = float(shares_text)
            price = float(price_text) if price_text else 0.0
        except ValueError:
            continue
        rows.append(
            {
                "transaction_date": txn_date,
                "owner_name": owner_name or "unknown",
                "is_officer": is_officer,
                "is_director": is_director,
                "is_ten_pct_owner": is_ten_pct,
                "transaction_code": code,
                "acquired_disposed": acquired_disposed or ("A" if code == "P" else "D"),
                "shares": shares,
                "price": price,
                "value_usd": shares * price,
            }
        )
    return rows


def fetch_insider_transactions(
    ticker: str, limit_filings: int = 10, already_seen: frozenset[str] | None = None
) -> tuple[list[dict], list[str]]:
    """Fetch and parse the most recent, not-already-processed Form 4 filings
    for a ticker.

    Each Form 4 costs a *separate* document fetch beyond the one submissions-
    list request every other filing type needs -- multiplied across 448+
    tickers on every scheduled refresh (every few hours, indefinitely; see
    app.engine's background refresh), an unbounded "always re-fetch the last
    N filings" here would mean thousands of redundant SEC requests per run
    for filings whose content never changes once filed. `already_seen` (the
    set of accession numbers already fetched+parsed in a previous run -- see
    app.pipeline._fetch_live_ticker / app.db's insider_txn_filings_seen
    table) makes this genuinely incremental: only filings not already
    fetched are ever fetched again. A ticker with a deep backlog catches up
    `limit_filings` at a time over its first several runs (bounding worst-
    case cost per run even then); once caught up, steady state is just the
    0-2 new Form 4s that typically appear between runs, not `limit_filings`.

    Returns (rows, fetched_accessions): `rows` is the flat list of open-
    market buy/sell rows (see _parse_form4_xml) from filings fetched *this
    call*, and `fetched_accessions` is every accession number actually
    fetched this call (whether or not it produced any P/S rows) -- the
    caller persists these as newly "seen" so they're skipped next time.
    Best-effort: any network or parse failure for an individual filing is
    skipped (not added to fetched_accessions, so it'll be retried next run)
    rather than aborting the whole fetch, since one malformed/unreachable
    filing shouldn't cost us every other one.
    """
    already_seen = already_seen or frozenset()
    cik_map = _load_cik_map()
    cik = cik_map.get(ticker.upper())
    if not cik:
        return [], []
    data = get_json(SUBMISSIONS_URL.format(cik=cik), headers=_headers)

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accns = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])

    rows: list[dict] = []
    fetched_accessions: list[str] = []
    filings_checked = 0
    for i in range(len(forms)):
        if forms[i] != "4":
            continue
        if accns[i] in already_seen:
            continue
        if filings_checked >= limit_filings:
            break
        filings_checked += 1
        accn_nodash = accns[i].replace("-", "")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{accn_nodash}/{docs[i]}"
        )
        try:
            raw_xml = get_text(url, headers=_headers)
        except Exception:
            log.warning("failed to fetch Form 4 document for %s at %s", ticker, url)
            continue
        rows.extend(_parse_form4_xml(raw_xml))
        fetched_accessions.append(accns[i])
    return rows, fetched_accessions


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
