"""Orchestrates ingestion for the whole universe into SQLite.

STOCKGRAPH_DATA_MODE controls the strategy:
  live  - always hit the network; raise if any ticker ends up with no bars
  demo  - always use the synthetic generator; no network calls at all
  auto  - try live per-ticker/per-field, backfill exactly what's missing
          with synthetic data (see the per-ticker-gap handling below)

Per data category, "live" now means:
  bars          Yahoo Finance chart endpoint (still reliable for most
                tickers -- see README's data-sources note)
  earnings      Finnhub if STOCKGRAPH_FINNHUB_API_KEY is set, else Yahoo
  fundamentals  Finnhub if configured, else Yahoo
  filings       SEC EDGAR (public, no key, unaffected by any of this)
  news          Google News RSS (public, no key, unaffected by any of this)

Yahoo's undocumented quoteSummary endpoint (earnings + fundamentals) has
started getting blocked by Yahoo's anti-bot layer for a large fraction of
requests as this universe has grown -- Finnhub is a real, licensed API
(free tier, no card) that covers the same two data types without that
risk. It's optional: everything here works exactly as before if
STOCKGRAPH_FINNHUB_API_KEY is unset, just leaning more on Yahoo/demo.

Every run is incremental: a ticker/field combination fetched within the
last STOCKGRAPH_FRESH_HOURS is skipped rather than re-fetched, tracked via
app.db's data_provenance table. This is what makes both a background
scheduled refresh (see app/engine.py) and repeated manual "Refresh data"
clicks fast and rate-limit-friendly after the first run -- without it,
every refresh would re-pull all 340+ companies from scratch every time.
"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from app.config import DATA_MODE
from app.dataclients import demo, finnhub, news, secedgar, yahoo
from app.universe import COMPANIES, TICKERS
from app import db

log = logging.getLogger("stockgraph.pipeline")

# Fetching is almost entirely spent waiting on network I/O, so running
# tickers concurrently -- rather than one full ticker at a time -- cuts a
# full live run from many minutes down to a couple. Kept modest (well under
# SEC EDGAR's published 10 req/sec fair-use guideline; Finnhub's own
# TokenBucket separately caps that source's own rate regardless of how many
# threads are in flight) so this speeds ingestion up without tripping any
# host's rate limiting. Overridable via env for tuning without a code change.
LIVE_FETCH_WORKERS = int(os.environ.get("STOCKGRAPH_LIVE_FETCH_WORKERS", "8"))

# How long a field is considered "fresh" before a run will re-fetch it live.
# 20h (not 24h) so a roughly-daily scheduled refresh (see engine.py) doesn't
# drift later and later each day waiting for a field to turn exactly stale.
FRESH_HOURS = float(os.environ.get("STOCKGRAPH_FRESH_HOURS", "20"))


def _is_fresh(updated_at: str | None, max_age_hours: float = FRESH_HOURS) -> bool:
    if not updated_at:
        return False
    try:
        ts = datetime.fromisoformat(updated_at)
    except ValueError:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_seconds = (datetime.now(timezone.utc) - ts).total_seconds()
    return age_seconds < max_age_hours * 3600


def _needed_fields(ticker: str, provenance: dict[str, dict], force: bool) -> dict[str, bool]:
    """Which of the 5 data categories are worth fetching live for this
    ticker right now. filings/news are cheap and time-sensitive (new
    filings/headlines can land any day) so they're always re-fetched;
    bars/earnings/fundamentals are the rate-limited/quota-constrained ones,
    so those respect the freshness window."""
    row = provenance.get(ticker, {})
    if force:
        return {f: True for f in db.PROVENANCE_FIELDS}
    return {
        "bars": not _is_fresh(row.get("bars_updated_at")),
        "earnings": not _is_fresh(row.get("earnings_updated_at")),
        "fundamentals": not _is_fresh(row.get("fundamentals_updated_at")),
        "filings": True,
        "news": True,
    }


def _fetch_live_ticker(ticker: str, company_name: str, need: dict[str, bool]) -> dict:
    """Fetches whichever live data sources `need` marks True and returns
    whatever succeeded, plus a `sources` map recording which provider
    supplied each field that came back -- pure network I/O, no DB access,
    so it's safe to run from a worker thread. Failures are logged and
    simply omitted, never raised, so one bad ticker can't derail the
    others or the thread pool."""
    result: dict = {
        "ticker": ticker, "bars": None, "earnings": None, "filings": None,
        "news": None, "fundamentals": None, "sources": {},
    }

    if need["bars"]:
        try:
            result["bars"] = yahoo.fetch_daily_bars(ticker)
            if result["bars"]:
                result["sources"]["bars"] = "yahoo"
        except Exception as e:
            log.warning("live price fetch failed for %s: %s", ticker, e)

    if need["earnings"]:
        if finnhub.is_configured():
            try:
                result["earnings"] = finnhub.fetch_earnings_history(ticker)
                if result["earnings"]:
                    result["sources"]["earnings"] = "finnhub"
            except Exception as e:
                log.warning("finnhub earnings fetch failed for %s: %s (falling back to yahoo)", ticker, e)
        if not result["earnings"]:
            try:
                yahoo_earnings = yahoo.fetch_earnings_calendar(ticker)
                if yahoo_earnings:
                    result["earnings"] = yahoo_earnings
                    result["sources"]["earnings"] = "yahoo"
            except Exception as e:
                log.warning("live earnings fetch failed for %s: %s", ticker, e)

    if need["filings"]:
        try:
            result["filings"] = secedgar.fetch_recent_filings(ticker)
            if result["filings"]:
                result["sources"]["filings"] = "secedgar"
        except Exception as e:
            log.warning("live filings fetch failed for %s: %s", ticker, e)

    if need["news"]:
        try:
            result["news"] = news.fetch_headlines(ticker, company_name)
            if result["news"]:
                result["sources"]["news"] = "google_news"
        except Exception as e:
            log.warning("live news fetch failed for %s: %s", ticker, e)

    if need["fundamentals"]:
        if finnhub.is_configured():
            try:
                fund = finnhub.fetch_fundamentals(ticker)
                if fund:
                    result["fundamentals"] = fund
                    result["sources"]["fundamentals"] = "finnhub"
            except Exception as e:
                log.warning("finnhub fundamentals fetch failed for %s: %s (falling back to yahoo)", ticker, e)
        if not result["fundamentals"]:
            try:
                fund = yahoo.fetch_fundamentals(ticker)
                if fund:
                    result["fundamentals"] = fund
                    result["sources"]["fundamentals"] = "yahoo"
            except Exception as e:
                log.warning("live fundamentals fetch failed for %s: %s", ticker, e)

    return result


def _write_ticker_result(result: dict) -> None:
    """Applies one ticker's already-fetched data to the DB. Always called
    from the main thread (see run_ingestion below) so ingestion never has
    more than one thread touching SQLite at a time, however many fetch
    workers are in flight."""
    ticker = result["ticker"]
    if result["bars"]:
        db.upsert_prices(ticker, result["bars"])
    if result["earnings"]:
        db.upsert_earnings(ticker, result["earnings"])
    if result["filings"]:
        db.upsert_filings(ticker, result["filings"])
    if result["news"]:
        db.upsert_news(ticker, result["news"])
    if result["fundamentals"]:
        db.upsert_fundamentals(ticker, result["fundamentals"])
    if result["sources"]:
        db.update_provenance(ticker, result["sources"])


def _ingest_demo_all() -> None:
    log.info("generating synthetic demo dataset for %d companies...", len(COMPANIES))
    dataset = demo.generate_universe_demo_data()
    for ticker, bundle in dataset.items():
        db.upsert_prices(ticker, bundle["prices"])
        db.upsert_earnings(ticker, bundle["earnings"])
        db.upsert_filings(ticker, bundle["filings"])
        db.upsert_news(ticker, bundle["news"])
        db.upsert_fundamentals(ticker, demo.generate_fundamentals(ticker))
        db.update_provenance(ticker, {f: "demo" for f in db.PROVENANCE_FIELDS})


def run_ingestion(force: bool = False) -> dict:
    """`force=True` bypasses the freshness check and re-fetches everything
    live, regardless of how recently it was last updated."""
    db.init_db()
    t0 = time.time()
    mode = DATA_MODE
    run_id = db.start_ingestion_run(mode)
    live_ok, live_fail, skipped_fresh = 0, 0, 0
    error: str | None = None

    try:
        if mode == "demo":
            _ingest_demo_all()
            db.set_meta("data_mode_active", "demo")
        else:
            provenance = db.get_all_provenance()
            needs = {c.ticker: _needed_fields(c.ticker, provenance, force) for c in COMPANIES}
            skipped_fresh = sum(
                1 for t, n in needs.items() if not n["bars"] and not n["earnings"] and not n["fundamentals"]
            )

            # Finnhub's earnings calendar is one call for the WHOLE universe
            # (it takes a date range, not a symbol), so pull it once up
            # front rather than per-ticker -- a large chunk of the free-tier
            # call budget saved versus the old per-ticker calendarEvents
            # module Yahoo used.
            bulk_calendar: dict[str, dict] = {}
            if finnhub.is_configured() and any(n["earnings"] for n in needs.values()):
                try:
                    bulk_calendar = finnhub.fetch_earnings_calendar_bulk(set(TICKERS))
                except Exception as e:
                    log.warning("finnhub bulk earnings calendar fetch failed: %s", e)

            results: dict[str, dict] = {}
            with ThreadPoolExecutor(max_workers=LIVE_FETCH_WORKERS) as pool:
                futures = {
                    pool.submit(_fetch_live_ticker, c.ticker, c.name, needs[c.ticker]): c
                    for c in COMPANIES
                }
                for future in as_completed(futures):
                    c = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:  # pragma: no cover - defensive; fetch fn already catches
                        log.warning("unexpected error fetching %s: %s", c.ticker, e)
                        result = {
                            "ticker": c.ticker, "bars": None, "earnings": None,
                            "filings": None, "news": None, "fundamentals": None, "sources": {},
                        }
                    upcoming = bulk_calendar.get(c.ticker)
                    if upcoming and needs[c.ticker]["earnings"]:
                        result["earnings"] = (result["earnings"] or []) + [upcoming]
                        result["sources"].setdefault("earnings", "finnhub")
                    results[c.ticker] = result
                    _write_ticker_result(result)
                    n = needs[c.ticker]
                    bars_ok = bool(result["bars"]) or not n["bars"]
                    if bars_ok:
                        live_ok += 1
                    else:
                        live_fail += 1

            if mode == "live" and live_fail:
                raise RuntimeError(
                    f"live ingestion failed for {live_fail} tickers and DATA_MODE=live "
                    "(no fallback). Set STOCKGRAPH_DATA_MODE=auto to fall back to demo data."
                )
            if live_ok == 0 and skipped_fresh == 0:
                log.warning("no live data reachable at all; falling back to full demo dataset")
                _ingest_demo_all()
                db.set_meta("data_mode_active", "demo")
            else:
                # Per-ticker/per-field demo fallback. A field only counts as
                # a gap if it was actually attempted-and-failed this run --
                # a field skipped because it's still fresh already has good
                # data sitting in the DB from an earlier run, so it's not a
                # gap at all. build_feature_panel() indexes every universe
                # ticker (see signals.py's sector-momentum grouping), so a
                # ticker with literally no bars at all -- not skipped, not
                # fetched, just missing -- would crash bootstrap outright;
                # that's the one case that always needs backfilling.
                gaps: dict[str, dict] = {}
                for t, r in results.items():
                    n = needs[t]
                    missing_bars = n["bars"] and not r["bars"]
                    missing_earnings = n["earnings"] and not r["earnings"]
                    missing_fundamentals = n["fundamentals"] and not r["fundamentals"]
                    missing_filings = n["filings"] and not r["filings"]
                    missing_news = n["news"] and not r["news"]
                    if missing_bars or missing_earnings or missing_fundamentals or missing_filings or missing_news:
                        gaps[t] = r
                if gaps:
                    log.warning(
                        "backfilling demo data for %d ticker(s) with incomplete live data: %s",
                        len(gaps), ", ".join(sorted(gaps)),
                    )
                    demo_dataset = demo.generate_universe_demo_data()
                    for ticker, result in gaps.items():
                        n = needs[ticker]
                        bundle = demo_dataset.get(ticker)
                        demo_sources = {}
                        if bundle:
                            if n["bars"] and not result["bars"]:
                                db.upsert_prices(ticker, bundle["prices"])
                                demo_sources["bars"] = "demo"
                            if n["earnings"] and not result["earnings"]:
                                db.upsert_earnings(ticker, bundle["earnings"])
                                demo_sources["earnings"] = "demo"
                            if n["filings"] and not result["filings"]:
                                db.upsert_filings(ticker, bundle["filings"])
                                demo_sources["filings"] = "demo"
                            if n["news"] and not result["news"]:
                                db.upsert_news(ticker, bundle["news"])
                                demo_sources["news"] = "demo"
                        if n["fundamentals"] and not result["fundamentals"]:
                            db.upsert_fundamentals(ticker, demo.generate_fundamentals(ticker))
                            demo_sources["fundamentals"] = "demo"
                        db.update_provenance(ticker, demo_sources)
                db.set_meta(
                    "data_mode_active",
                    f"auto (live={live_ok}, demo_fallback={live_fail})" if live_fail else "live",
                )

        db.set_meta("last_ingested_at", db.now_iso())
    except Exception as e:
        error = str(e)
        raise
    finally:
        elapsed = round(time.time() - t0, 1)
        db.finish_ingestion_run(run_id, live_ok, live_fail, skipped_fresh, elapsed, error)

    active_mode = db.get_meta("data_mode_active")
    log.info(
        "ingestion complete in %ss, mode=%s (live_ok=%d, live_fail=%d, skipped_fresh=%d)",
        elapsed, active_mode, live_ok, live_fail, skipped_fresh,
    )
    return {
        "elapsed_sec": elapsed, "mode": active_mode,
        "live_ok": live_ok, "live_fail": live_fail, "skipped_fresh": skipped_fresh,
    }
