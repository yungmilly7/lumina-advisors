"""Finnhub client for earnings history/calendar and company fundamentals --
a real, licensed API (free tier: 60 calls/min, no card required) used in
place of Yahoo Finance's undocumented quoteSummary endpoint for exactly the
two data types Yahoo has started blocking hardest (see pipeline.py and the
README's data-sources note). Yahoo's own price-chart endpoint keeps working
for the large majority of tickers, so bars are left on yahoo.py -- this
client only covers earnings + fundamentals, which is where Yahoo now fails
almost universally.

Needs STOCKGRAPH_FINNHUB_API_KEY set to a free key from finnhub.io/register
-- that signup is something only Danny can do (it's a real account), so
`is_configured()` is False until he's pasted a key into .env, and every
caller in pipeline.py treats that exactly like "this source isn't
reachable right now": skip straight to the next fallback (Yahoo, then
demo). Nothing breaks if this is never configured.

Field-name caveat: Finnhub's `/stock/metric?metric=all` response is large
and its exact key names are not fully documented in one place. The
extraction below tries a short list of plausible aliases per metric (all
sourced from Finnhub's own published examples/SDKs) and simply leaves a
field as None if none match -- exactly like Yahoo's fetch_fundamentals,
which already omits whatever a given ticker's response doesn't populate.
If fundamentals come back mostly-empty in practice once a real key is in
place, log.debug output from _extract_metric (or a raw metric dump) will
show what Finnhub is actually naming that field, and the alias list below
is the only thing that needs updating.
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta

from app.config import FINNHUB_API_KEY, HTTP_TIMEOUT
from app.dataclients.httpjson import HTTPError, get_json
from app.dataclients.ratelimit import TokenBucket

log = logging.getLogger("stockgraph.finnhub")

BASE_URL = "https://finnhub.io/api/v1"

# Free tier is 60 calls/min; stay comfortably under that (55) so a burst
# from this process plus any manual testing against the same key doesn't
# tip it into 429s. TokenBucket is shared across every worker thread in the
# ingestion pool, so this is a real global cap, not a per-thread one.
_bucket = TokenBucket(rate_per_sec=55 / 60, capacity=10)


def is_configured() -> bool:
    return bool(FINNHUB_API_KEY)


def _get(path: str, **params):
    if not FINNHUB_API_KEY:
        raise HTTPError("Finnhub API key not configured (set STOCKGRAPH_FINNHUB_API_KEY)")
    _bucket.acquire()
    params["token"] = FINNHUB_API_KEY
    return get_json(f"{BASE_URL}{path}", params=params, timeout=HTTP_TIMEOUT)


def fetch_earnings_history(ticker: str) -> list[dict]:
    """Past reported quarters: EPS estimate/actual/surprise. Finnhub returns
    these newest-first; we don't re-sort since callers only care about the
    set, not order (db.upsert_earnings keys on (ticker, report_date))."""
    payload = _get("/stock/earnings", symbol=ticker)
    rows: list[dict] = []
    for e in payload or []:
        period = e.get("period")
        if not period:
            continue
        quarter, year = e.get("quarter"), e.get("year")
        rows.append(
            {
                "report_date": period,
                "eps_estimate": e.get("estimate"),
                "eps_actual": e.get("actual"),
                "surprise_pct": e.get("surprisePercent"),
                "fiscal_period": f"Q{quarter} {year}" if quarter and year else None,
                "is_future": 0,
            }
        )
    return rows


def fetch_earnings_calendar_bulk(tickers: set[str], window_days: int = 120) -> dict[str, dict]:
    """The *whole universe's* upcoming earnings dates in ONE call, keyed by
    ticker -- Finnhub's /calendar/earnings takes a date range rather than a
    symbol, so pulling this once per ingestion run (instead of once per
    ticker, the way Yahoo's per-ticker calendarEvents module worked) saves
    hundreds of calls out of the free-tier budget. Tickers outside `tickers`
    are dropped since the endpoint returns the whole market."""
    today = date.today()
    payload = _get(
        "/calendar/earnings",
        **{"from": today.isoformat(), "to": (today + timedelta(days=window_days)).isoformat()},
    )
    out: dict[str, dict] = {}
    for e in payload.get("earningsCalendar", []) or []:
        sym = e.get("symbol")
        report_date = e.get("date")
        if not sym or sym not in tickers or not report_date:
            continue
        # A symbol can appear more than once in a wide window (estimate
        # revisions); keep the earliest upcoming date.
        if sym in out and out[sym]["report_date"] <= report_date:
            continue
        out[sym] = {
            "report_date": report_date,
            "eps_estimate": e.get("epsEstimate"),
            "eps_actual": None,
            "surprise_pct": None,
            "fiscal_period": "upcoming",
            "is_future": 1,
        }
    return out


def _first(d: dict, *keys):
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def _pct_to_fraction(v):
    """Finnhub reports several ratios (margins, growth, yield) as whole
    percents (e.g. 23.4 meaning 23.4%); the rest of this app stores them as
    fractions (0.234), matching Yahoo's `raw` values. A value already given
    as a small fraction (<1 in magnitude) is left alone defensively, in case
    a particular metric turns out not to follow the whole-percent
    convention."""
    if v is None:
        return None
    return v / 100.0 if abs(v) > 1.5 else v


def _extract_metric(metric: dict) -> dict:
    return {
        "pe_ratio": _first(metric, "peBasicExclExtraTTM", "peExclExtraTTM", "peTTM", "peAnnual"),
        "forward_pe": _first(metric, "peForward", "forwardPE", "peNormalizedAnnual"),
        "peg_ratio": _first(metric, "pegRatioTTM", "pegRatio", "pegRatioAnnual"),
        "dividend_yield": _pct_to_fraction(
            _first(metric, "dividendYieldIndicatedAnnual", "currentDividendYieldTTM", "dividendYield5Y")
        ),
        "beta": _first(metric, "beta"),
        "profit_margin": _pct_to_fraction(_first(metric, "netProfitMarginTTM", "netProfitMarginAnnual")),
        "revenue_growth": _pct_to_fraction(
            _first(metric, "revenueGrowthTTMYoy", "revenueGrowthQuarterlyYoy", "revenueGrowth5Y")
        ),
    }


def _summarize_recommendation(rec: dict) -> tuple[str | None, int | None]:
    """Finnhub's /stock/recommendation returns analyst buy/hold/sell head
    counts per period rather than Yahoo's single `recommendationKey` label,
    so we derive an equivalent label from whichever bucket has the most
    votes -- same five-point scale the UI already renders (see
    RECOMMENDATION_LABELS in app.js)."""
    buckets = {
        "strong_buy": rec.get("strongBuy") or 0,
        "buy": rec.get("buy") or 0,
        "hold": rec.get("hold") or 0,
        "underperform": rec.get("sell") or 0,
        "sell": rec.get("strongSell") or 0,
    }
    total = sum(buckets.values())
    if total == 0:
        return None, None
    top = max(buckets, key=buckets.get)
    return top, total


def fetch_fundamentals(ticker: str) -> dict:
    profile = _get("/stock/profile2", symbol=ticker) or {}
    metric = (_get("/stock/metric", symbol=ticker, metric="all") or {}).get("metric", {}) or {}

    target: dict = {}
    try:
        target = _get("/stock/price-target", symbol=ticker) or {}
    except HTTPError as e:
        log.debug("no price target for %s: %s", ticker, e)

    recommendation_key, num_opinions = None, None
    try:
        recs = _get("/stock/recommendation", symbol=ticker) or []
        if recs:
            recommendation_key, num_opinions = _summarize_recommendation(recs[0])
    except HTTPError as e:
        log.debug("no recommendation trend for %s: %s", ticker, e)

    name = profile.get("name")
    industry = profile.get("finnhubIndustry")
    description = None
    if name:
        description = f"{name} ({industry})." if industry else f"{name}."
        description += " Business description not provided by this data source."

    market_cap = profile.get("marketCapitalization")  # Finnhub reports this in millions
    out = {
        "description": description,
        "market_cap": market_cap * 1_000_000 if market_cap is not None else None,
        "analyst_target_mean": _first(target, "targetMean"),
        "analyst_target_high": _first(target, "targetHigh"),
        "analyst_target_low": _first(target, "targetLow"),
        "analyst_recommendation": recommendation_key,
        "num_analyst_opinions": num_opinions,
    }
    out.update(_extract_metric(metric))
    return out
