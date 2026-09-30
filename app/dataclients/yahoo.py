"""Minimal Yahoo Finance client built on plain HTTP (no `yfinance` dependency,
so there's one less fast-moving third-party wrapper to break). Uses the same
public chart/quoteSummary endpoints yfinance itself calls under the hood.

Needs outbound internet access to query1.finance.yahoo.com / query2. In
sandboxes without that access these calls will raise, and callers should
fall back to app.dataclients.demo.
"""
from __future__ import annotations

import http.cookiejar
import logging
import threading
import time
import urllib.request

from app.config import HTTP_TIMEOUT
from app.dataclients.httpjson import get_json

log = logging.getLogger("stockgraph.yahoo")

# A real browser UA -- Yahoo's edge (Akamai) blocks the generic
# "LuminaAdvisorsBot" UA that used to be here outright for some endpoints.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
QUOTE_SUMMARY_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
_CRUMB_URL = "https://query1.finance.yahoo.com/v1/test/getcrumb"
_COOKIE_SEED_URL = "https://fc.yahoo.com"

# Yahoo's quoteSummary endpoint (unlike the chart endpoint) has required a
# session cookie + "crumb" token since Yahoo tightened its API -- the same
# fix the `yfinance` library ships. We seed a cookie jar once per process
# and cache the crumb; if the handshake itself fails (e.g. this network
# can't reach fc.yahoo.com), we fall back to an unauthenticated request,
# which is exactly the old behavior.
_cookie_jar = http.cookiejar.CookieJar()
_opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_cookie_jar))
_crumb: str | None = None
_crumb_attempted = False
# The ingestion pipeline fetches many tickers concurrently, and every one
# of them calls _get_crumb() -- without this lock, several worker threads
# could all see _crumb_attempted as False at once and each redo the
# cookie-seed + crumb-fetch handshake (and, worse, all write into the
# shared, non-thread-safe CookieJar/OpenerDirector at the same time).
_crumb_lock = threading.Lock()


def _get_crumb() -> str | None:
    global _crumb, _crumb_attempted
    if _crumb_attempted:
        return _crumb
    with _crumb_lock:
        if _crumb_attempted:  # another thread finished the handshake while we waited
            return _crumb
        return _do_crumb_handshake()


def _do_crumb_handshake() -> str | None:
    global _crumb, _crumb_attempted
    _crumb_attempted = True

    # Step 1: seed a session cookie. fc.yahoo.com often answers this GET with
    # a 404 (it's an edge/accelerator endpoint, not a real page) but still
    # sets the ".yahoo.com" cookie via Set-Cookie on that same response --
    # the cookie is what matters, not the status code, so a non-2xx here is
    # expected and must NOT abort the handshake.
    try:
        seed_req = urllib.request.Request(_COOKIE_SEED_URL, headers=_HEADERS)
        _opener.open(seed_req, timeout=HTTP_TIMEOUT).read()
    except Exception as e:
        log.debug("Yahoo cookie-seed request returned %s: %s (often expected, continuing)",
                  type(e).__name__, e)

    # Step 2: fetch the crumb using whatever cookies we picked up above (or
    # none, if even that failed) -- this is the step that actually matters,
    # so it gets its own try/except rather than sharing one with step 1.
    try:
        crumb_req = urllib.request.Request(_CRUMB_URL, headers=_HEADERS)
        crumb = _opener.open(crumb_req, timeout=HTTP_TIMEOUT).read().decode("utf-8").strip()
        if crumb and "{" not in crumb:  # a JSON error body means no real crumb
            _crumb = crumb
        else:
            log.warning("Yahoo crumb handshake returned no usable crumb (got %r)", crumb[:120])
    except Exception as e:
        log.warning("Yahoo crumb fetch failed (%s: %s) -- falling back to unauthenticated "
                    "quoteSummary requests, which Yahoo may 401", type(e).__name__, e)
        _crumb = None
    return _crumb


def fetch_daily_bars(ticker: str, range_: str = "2y") -> list[dict]:
    """Returns [{date, open, high, low, close, volume}, ...] ascending by date."""
    params = {"range": range_, "interval": "1d", "events": "div,splits"}
    payload = get_json(CHART_URL.format(ticker=ticker), params=params, headers=_HEADERS)

    result = payload.get("chart", {}).get("result")
    if not result:
        raise ValueError(f"no chart data for {ticker}: {payload.get('chart', {}).get('error')}")
    r0 = result[0]
    timestamps = r0.get("timestamp", [])
    quote = r0.get("indicators", {}).get("quote", [{}])[0]
    opens, highs, lows, closes, vols = (
        quote.get("open", []),
        quote.get("high", []),
        quote.get("low", []),
        quote.get("close", []),
        quote.get("volume", []),
    )
    rows = []
    for i, ts in enumerate(timestamps):
        if i >= len(closes) or closes[i] is None:
            continue
        date = time.strftime("%Y-%m-%d", time.gmtime(ts))
        rows.append(
            {
                "date": date,
                "open": opens[i] if i < len(opens) else None,
                "high": highs[i] if i < len(highs) else None,
                "low": lows[i] if i < len(lows) else None,
                "close": closes[i],
                "volume": vols[i] if i < len(vols) else None,
            }
        )
    return rows


def fetch_earnings_calendar(ticker: str) -> list[dict]:
    """Returns known past + next earnings dates with EPS estimate/actual/surprise."""
    modules = "earningsHistory,calendarEvents,earningsTrend"
    params = {"modules": modules}
    crumb = _get_crumb()
    if crumb:
        params["crumb"] = crumb
    payload = get_json(
        QUOTE_SUMMARY_URL.format(ticker=ticker),
        params=params,
        headers=_HEADERS,
        opener=_opener if crumb else None,
    )

    result = payload.get("quoteSummary", {}).get("result")
    if not result:
        return []
    r0 = result[0]
    rows: list[dict] = []

    hist = r0.get("earningsHistory", {}).get("history", [])
    for h in hist:
        rows.append(
            {
                "report_date": _fmt_date(h.get("quarter", {})),
                "eps_estimate": _fmt_num(h.get("epsEstimate")),
                "eps_actual": _fmt_num(h.get("epsActual")),
                "surprise_pct": _fmt_num(h.get("surprisePercent")),
                "fiscal_period": h.get("period", {}).get("fmt") if isinstance(h.get("period"), dict) else None,
                "is_future": 0,
            }
        )

    cal = r0.get("calendarEvents", {}).get("earnings", {})
    for ts in cal.get("earningsDate", []):
        rows.append(
            {
                "report_date": _fmt_date(ts),
                "eps_estimate": _fmt_num(cal.get("earningsAverage")),
                "eps_actual": None,
                "surprise_pct": None,
                "fiscal_period": "upcoming",
                "is_future": 1,
            }
        )
    return rows


def fetch_fundamentals(ticker: str) -> dict:
    """Returns a flat dict of richer per-company info: business summary,
    market cap, valuation ratios, dividend yield, beta, and analyst target
    price/recommendation -- everything the site shows beyond raw price bars.
    Missing fields are simply omitted (Yahoo doesn't populate every module
    for every ticker, e.g. non-dividend payers have no dividendYield).
    """
    modules = "summaryProfile,summaryDetail,defaultKeyStatistics,financialData"
    params = {"modules": modules}
    crumb = _get_crumb()
    if crumb:
        params["crumb"] = crumb
    payload = get_json(
        QUOTE_SUMMARY_URL.format(ticker=ticker),
        params=params,
        headers=_HEADERS,
        opener=_opener if crumb else None,
    )

    result = payload.get("quoteSummary", {}).get("result")
    if not result:
        return {}
    r0 = result[0]
    profile = r0.get("summaryProfile", {}) or {}
    detail = r0.get("summaryDetail", {}) or {}
    stats = r0.get("defaultKeyStatistics", {}) or {}
    fin = r0.get("financialData", {}) or {}

    summary = profile.get("longBusinessSummary")
    if summary and len(summary) > 700:
        summary = summary[:697].rsplit(" ", 1)[0] + "..."

    out = {
        "description": summary,
        "market_cap": _fmt_num(detail.get("marketCap")),
        "pe_ratio": _fmt_num(detail.get("trailingPE")),
        "forward_pe": _fmt_num(stats.get("forwardPE")),
        "peg_ratio": _fmt_num(stats.get("pegRatio")),
        "dividend_yield": _fmt_num(detail.get("dividendYield")),
        "beta": _fmt_num(detail.get("beta")) or _fmt_num(stats.get("beta")),
        "profit_margin": _fmt_num(stats.get("profitMargins")),
        "revenue_growth": _fmt_num(fin.get("revenueGrowth")),
        "analyst_target_mean": _fmt_num(fin.get("targetMeanPrice")),
        "analyst_target_high": _fmt_num(fin.get("targetHighPrice")),
        "analyst_target_low": _fmt_num(fin.get("targetLowPrice")),
        "analyst_recommendation": fin.get("recommendationKey"),
        "num_analyst_opinions": _fmt_num(fin.get("numberOfAnalystOpinions")),
    }
    return out


def _fmt_date(v) -> str | None:
    if isinstance(v, dict) and "raw" in v:
        return time.strftime("%Y-%m-%d", time.gmtime(v["raw"]))
    if isinstance(v, (int, float)):
        return time.strftime("%Y-%m-%d", time.gmtime(v))
    return None


def _fmt_num(v):
    if isinstance(v, dict):
        return v.get("raw")
    return v
