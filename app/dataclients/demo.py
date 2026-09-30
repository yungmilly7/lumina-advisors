"""Deterministic synthetic data generator.

This is what runs when live network access isn't available (like in this
sandbox) or when STOCKGRAPH_DATA_MODE=demo is set explicitly. It is NOT
meant to resemble real quotes -- every place this data reaches the UI it's
labeled "synthetic / demo data" -- but it is built to have the same
*structure* real markets have, which is what makes it useful for exercising
the rest of the pipeline honestly:

  - a shared market factor + a per-sector factor + company idiosyncratic
    noise, so companies in the same sector co-move like they do in reality
  - a graph pass-through step: part of each neighbor's prior-day shock
    leaks into a company's return, scaled by edge weight and sign
    (competitor edges push the opposite way, supplier/customer/partner
    edges push the same way) -- this is the same mechanism the live
    "graph spillover" signal is trying to detect, so the demo actually
    exercises that code path meaningfully instead of feeding it noise
  - scheduled quarterly earnings events with a sampled surprise that moves
    the price, plus a handful of random "news" events (product launch,
    deal, recall, downgrade, ...) with matching synthetic headlines whose
    sentiment/tags line up with the jump that day

Everything is seeded from the ticker (+ a fixed universe seed), so re-runs
are reproducible, which matters for the backtest/scorecard.
"""
from __future__ import annotations

import hashlib
import random
from datetime import date, datetime, timedelta, timezone

import numpy as np

from app.config import PRICE_HISTORY_DAYS
from app.universe import COMPANIES, COMPANY_BY_TICKER, TICKERS, adjacency

UNIVERSE_SEED = 20260101

# Rough, illustrative starting price levels so the demo "looks" plausible.
# These are NOT live quotes.
_BASE_PRICE_HINTS = {
    "AAPL": 230, "MSFT": 430, "GOOGL": 170, "AMZN": 190, "META": 570,
    "NVDA": 130, "AMD": 165, "INTC": 22, "TSM": 190, "ASML": 700,
    "TSLA": 250, "NFLX": 700, "DIS": 110, "V": 300, "MA": 500,
    "JPM": 220, "BA": 180, "WMT": 90, "COST": 900, "XOM": 115,
}


def _seed_for(ticker: str) -> int:
    h = hashlib.sha256(f"{UNIVERSE_SEED}:{ticker}".encode()).hexdigest()
    return int(h[:8], 16)


def _base_price(ticker: str) -> float:
    if ticker in _BASE_PRICE_HINTS:
        return float(_BASE_PRICE_HINTS[ticker])
    rng = np.random.default_rng(_seed_for(ticker))
    return float(rng.uniform(25, 350))


def _trading_dates(n_days: int, end: date) -> list[date]:
    dates = []
    d = end
    while len(dates) < n_days:
        if d.weekday() < 5:  # Mon-Fri
            dates.append(d)
        d -= timedelta(days=1)
    return list(reversed(dates))


EVENT_KINDS = [
    # (tag, headline_template, sentiment_sign, magnitude)
    ("product_launch", "{name} unveils new flagship product to strong early demand", 1, 0.02),
    ("deal_partnership", "{name} announces major partnership deal", 1, 0.015),
    ("analyst_action", "{name} upgraded by Wall Street analysts, price target raised", 1, 0.012),
    ("guidance", "{name} raises full-year guidance on strong bookings", 1, 0.018),
    ("ma", "{name} to acquire smaller rival, expanding market share", 1, 0.01),
    ("product_delay", "{name} delays product launch amid supply constraints", -1, 0.015),
    ("regulatory", "{name} faces regulatory investigation over business practices", -1, 0.02),
    ("recall", "{name} issues recall over safety concerns", -1, 0.025),
    ("analyst_action", "{name} downgraded as analysts flag demand concerns", -1, 0.012),
    ("layoffs", "{name} announces layoffs amid cost-cutting push", -1, 0.014),
]


def _generate_market_and_sector_factors(dates: list[date], rng: np.random.Generator):
    n = len(dates)
    market = rng.normal(0.0003, 0.009, n)
    sectors = {c.sector for c in COMPANIES}
    sector_factors = {s: rng.normal(0.0, 0.007, n) for s in sectors}
    return market, sector_factors


def generate_universe_demo_data(as_of: date | None = None) -> dict[str, dict]:
    as_of = as_of or datetime.now(timezone.utc).date()
    dates = _trading_dates(PRICE_HISTORY_DAYS, as_of)
    n = len(dates)

    master_rng = np.random.default_rng(UNIVERSE_SEED)
    market, sector_factors = _generate_market_and_sector_factors(dates, master_rng)

    adj = adjacency()
    idio: dict[str, np.ndarray] = {}
    events: dict[str, list[tuple[int, str, str, int, float]]] = {t: [] for t in TICKERS}

    # Pass 1: idiosyncratic returns + scheduled events, before graph pass-through.
    for c in COMPANIES:
        rng = np.random.default_rng(_seed_for(c.ticker))
        vol = rng.uniform(0.012, 0.03)
        idio_ret = rng.normal(0.0002, vol, n)

        # Quarterly earnings events (~every 63 trading days), offset per ticker.
        offset = int(rng.integers(0, 63))
        q = offset
        while q < n:
            surprise = rng.normal(0.0, 1.0)  # in "std beats" units
            move = float(np.clip(surprise * 0.02, -0.09, 0.09))
            idio_ret[q] += move
            tag = "earnings"
            headline = (
                f"{c.name} {'beats' if move >= 0 else 'misses'} quarterly "
                f"earnings estimates, shares react"
            )
            events[c.ticker].append((q, tag, headline, 1 if move >= 0 else -1, abs(move)))
            q += 63

        # A handful of random news events.
        n_events = int(rng.integers(2, 5))
        for _ in range(n_events):
            day = int(rng.integers(5, n - 1))
            kind_tag, template, sign, mag = EVENT_KINDS[rng.integers(0, len(EVENT_KINDS))]
            jitter = float(rng.uniform(0.5, 1.3))
            move = sign * mag * jitter
            idio_ret[day] += move
            headline = template.format(name=c.name)
            events[c.ticker].append((day, kind_tag, headline, sign, abs(move)))

        idio[c.ticker] = idio_ret

    # Pass 2: graph pass-through -- blend in a slice of neighbors' prior-day shocks.
    total_ret: dict[str, np.ndarray] = {t: idio[t].copy() for t in TICKERS}
    for t in TICKERS:
        for other, edge in adj[t]:
            sign = -1.0 if edge.kind == "competitor" else 1.0
            leak = 0.12 * edge.weight * sign
            shifted = np.zeros(n)
            shifted[1:] = idio[other][:-1]
            total_ret[t] += leak * shifted

    # Pass 3: assemble prices/OHLCV.
    result: dict[str, dict] = {}
    for c in COMPANIES:
        t = c.ticker
        sector_ret = sector_factors[c.sector]
        daily_ret = market + sector_ret + total_ret[t]
        base = _base_price(t)
        closes = base * np.cumprod(1.0 + daily_ret)

        rng = np.random.default_rng(_seed_for(t) ^ 0x9E3779B9)
        rows = []
        prev_close = base
        for i, d in enumerate(dates):
            close = float(closes[i])
            intraday = abs(rng.normal(0.006, 0.003))
            high = close * (1 + intraday)
            low = close * (1 - intraday)
            open_ = prev_close * (1 + rng.normal(0, 0.003))
            high = max(high, open_, close)
            low = min(low, open_, close)
            base_vol = rng.uniform(3e6, 2.5e7)
            vol_mult = 1 + 6 * min(abs(daily_ret[i]), 0.1)
            volume = base_vol * vol_mult
            rows.append(
                {
                    "date": d.isoformat(),
                    "open": round(open_, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(close, 2),
                    "volume": round(volume),
                }
            )
            prev_close = close

        # Earnings table rows.
        earnings_rows = []
        for day_idx, tag, headline, sign, mag in events[t]:
            if tag != "earnings" or day_idx >= n:
                continue
            eps_est = round(rng.uniform(0.5, 5.0), 2)
            surprise_pct = round(sign * mag * 100 / 2, 1)
            eps_actual = round(eps_est * (1 + surprise_pct / 100), 2)
            earnings_rows.append(
                {
                    "report_date": dates[day_idx].isoformat(),
                    "eps_estimate": eps_est,
                    "eps_actual": eps_actual,
                    "surprise_pct": surprise_pct,
                    "fiscal_period": f"Q{(day_idx // 63) % 4 + 1}",
                    "is_future": 0,
                }
            )
        # One upcoming (future) earnings date, ~1-8 weeks out.
        days_out = int(rng.integers(7, 56))
        next_date = as_of + timedelta(days=days_out)
        earnings_rows.append(
            {
                "report_date": next_date.isoformat(),
                "eps_estimate": round(rng.uniform(0.5, 5.0), 2),
                "eps_actual": None,
                "surprise_pct": None,
                "fiscal_period": "upcoming",
                "is_future": 1,
            }
        )

        # Filings: annual 10-K, quarterlies near earnings, occasional 8-K near events.
        filing_rows = []
        for idx, (day_idx, tag, headline, sign, mag) in enumerate(events[t]):
            if day_idx >= n:
                continue
            form = "10-Q" if tag == "earnings" else "8-K"
            filing_rows.append(
                {
                    "filed_date": dates[min(day_idx + 1, n - 1)].isoformat(),
                    "form_type": form,
                    "title": f"{form} filed following: {headline}",
                    "url": "",
                }
            )
        filing_rows.append(
            {
                "filed_date": dates[max(0, n - 200)].isoformat(),
                "form_type": "10-K",
                "title": f"{c.name} annual report (10-K)",
                "url": "",
            }
        )

        # News: one headline per event, mapped to a real sentiment score via
        # the same lexicon scorer live mode uses, so the two paths agree.
        from app.dataclients.news import score_headline

        news_rows = []
        for day_idx, tag, headline, sign, mag in events[t]:
            if day_idx >= n:
                continue
            sentiment, auto_tags = score_headline(headline)
            published_dt = datetime.combine(dates[day_idx], datetime.min.time(), tzinfo=timezone.utc)
            news_rows.append(
                {
                    "published": published_dt.isoformat(),
                    "headline": headline,
                    "source": "Synthetic Wire (demo data)",
                    "url": "",
                    "sentiment": sentiment if auto_tags or sentiment else float(sign) * min(mag * 20, 1.0),
                    "event_tags": list(set(auto_tags + [tag])),
                }
            )

        result[t] = {
            "prices": rows,
            "earnings": earnings_rows,
            "filings": filing_rows,
            "news": news_rows,
        }

    return result


_REC_OPTIONS = ["strong_buy", "buy", "hold", "underperform", "sell"]
_REC_WEIGHTS = [0.15, 0.30, 0.35, 0.15, 0.05]


def generate_fundamentals(ticker: str) -> dict:
    """Synthetic business summary + valuation/analyst stats for demo mode,
    deterministic per ticker so re-runs agree. Clearly not real data -- the
    description says so explicitly, and the UI labels the whole site as
    demo data whenever this path is active."""
    c = COMPANY_BY_TICKER[ticker]
    rng = np.random.default_rng(_seed_for(ticker) ^ 0x51ED270B)
    price = _base_price(ticker)
    shares_out = rng.uniform(3e8, 1.2e10)
    market_cap = price * shares_out
    pe = float(rng.uniform(8, 45))
    forward_pe = pe * float(rng.uniform(0.75, 1.05))
    peg = float(rng.uniform(0.6, 3.0))
    pays_dividend = rng.random() < 0.55
    dividend_yield = float(rng.uniform(0.002, 0.045)) if pays_dividend else None
    beta = float(rng.uniform(0.5, 1.9))
    profit_margin = float(rng.uniform(-0.05, 0.35))
    revenue_growth = float(rng.uniform(-0.08, 0.35))
    target_mean = price * float(rng.uniform(0.92, 1.25))
    target_high = target_mean * float(rng.uniform(1.05, 1.25))
    target_low = target_mean * float(rng.uniform(0.75, 0.95))
    recommendation = str(rng.choice(_REC_OPTIONS, p=_REC_WEIGHTS))
    num_analysts = int(rng.integers(3, 35))
    summary = (
        f"{c.name} operates in the {c.industry} industry within the {c.sector} "
        f"sector. (Synthetic demo description, not the company's real business "
        f"summary -- turn on live data to see the real one.)"
    )
    return {
        "description": summary,
        "market_cap": round(market_cap),
        "pe_ratio": round(pe, 2),
        "forward_pe": round(forward_pe, 2),
        "peg_ratio": round(peg, 2),
        "dividend_yield": round(dividend_yield, 4) if dividend_yield else None,
        "beta": round(beta, 2),
        "profit_margin": round(profit_margin, 4),
        "revenue_growth": round(revenue_growth, 4),
        "analyst_target_mean": round(target_mean, 2),
        "analyst_target_high": round(target_high, 2),
        "analyst_target_low": round(target_low, 2),
        "analyst_recommendation": recommendation,
        "num_analyst_opinions": num_analysts,
    }


_SYNTHETIC_OWNER_NAMES = [
    "A. Whitfield", "B. Ncube", "C. Delacroix", "D. Yamashita", "E. Okafor",
    "F. Sørensen", "G. Petrov", "H. Alvarado", "I. Nakamura", "J. Fitzgerald",
]


def generate_insider_transactions(ticker: str, as_of: date | None = None) -> list[dict]:
    """Synthetic Form 4 open-market buy/sell history for demo mode,
    deterministic per ticker. Real insider activity is bursty and often
    absent for months at a time -- some tickers get zero rows here, same as
    a real company that simply hasn't had an open-market insider trade
    lately, which is deliberate rather than a bug."""
    as_of = as_of or datetime.now(timezone.utc).date()
    rng = np.random.default_rng(_seed_for(ticker) ^ 0x1D5B7E31)
    base_price = _base_price(ticker)

    # Most companies have a handful of insiders who trade occasionally, not
    # constantly -- weight toward few-or-zero transactions in the trailing
    # window rather than a uniform spread.
    n_txns = int(rng.choice([0, 1, 2, 3, 4, 5, 6, 8], p=[0.15, 0.2, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05]))
    if n_txns == 0:
        return []

    owners = rng.choice(_SYNTHETIC_OWNER_NAMES, size=min(len(_SYNTHETIC_OWNER_NAMES), rng.integers(1, 4)), replace=False)
    rows = []
    for _ in range(n_txns):
        owner_name = str(rng.choice(owners))
        days_ago = int(rng.integers(1, 180))
        txn_date = as_of - timedelta(days=days_ago)
        # Insiders sell (rebalancing, tax, diversification) far more often
        # than they buy on the open market, historically -- an open-market
        # buy is the rarer, more information-laden signal.
        code = "S" if rng.random() < 0.78 else "P"
        is_officer = bool(rng.random() < 0.6)
        is_director = bool(not is_officer and rng.random() < 0.5)
        is_ten_pct = bool(rng.random() < 0.1)
        shares = float(rng.integers(500, 50_000))
        price = float(base_price * (1 + rng.normal(0, 0.08)))
        price = max(price, 0.5)
        rows.append(
            {
                "transaction_date": txn_date.isoformat(),
                "owner_name": owner_name,
                "is_officer": is_officer,
                "is_director": is_director,
                "is_ten_pct_owner": is_ten_pct,
                "transaction_code": code,
                "acquired_disposed": "A" if code == "P" else "D",
                "shares": shares,
                "price": round(price, 2),
                "value_usd": round(shares * price, 2),
            }
        )
    rows.sort(key=lambda r: r["transaction_date"], reverse=True)
    return rows


def generate_macro_series(as_of: date | None = None) -> list[dict]:
    """Synthetic VIX + Treasury yield curve history for demo mode, in the
    same {series, date, value} shape app.db.upsert_macro_series expects.

    Each series is a mean-reverting (Ornstein-Uhlenbeck-style) random walk
    around a plausible real-world level -- not calibrated to any specific
    historical period, just structurally similar (VIX clusters low with
    occasional spikes; yields drift slowly) so it exercises the same
    feature-engineering code path as live data honestly."""
    as_of = as_of or datetime.now(timezone.utc).date()
    dates = _trading_dates(PRICE_HISTORY_DAYS, as_of)
    n = len(dates)
    rng = np.random.default_rng(UNIVERSE_SEED ^ 0x4D4143524F)  # "MACRO" bytes, arbitrary

    def mean_reverting(level: float, vol: float, reversion: float, floor: float) -> np.ndarray:
        vals = np.empty(n)
        vals[0] = level
        shocks = rng.normal(0, vol, n)
        for i in range(1, n):
            vals[i] = vals[i - 1] + reversion * (level - vals[i - 1]) + shocks[i]
            vals[i] = max(vals[i], floor)
        return vals

    vix = mean_reverting(level=17.0, vol=1.3, reversion=0.08, floor=9.0)
    # A handful of synthetic "vol spike" events, same spirit as the
    # idiosyncratic news-event jumps in generate_universe_demo_data.
    n_spikes = int(rng.integers(2, 5))
    for _ in range(n_spikes):
        start = int(rng.integers(10, max(11, n - 15)))
        spike_mag = float(rng.uniform(8, 25))
        decay = np.exp(-np.arange(min(15, n - start)) / 4.0)
        vix[start:start + len(decay)] += spike_mag * decay

    yield_3m = mean_reverting(level=5.0, vol=0.03, reversion=0.03, floor=0.0)
    yield_2y = mean_reverting(level=4.2, vol=0.04, reversion=0.03, floor=0.0)
    yield_10y = mean_reverting(level=4.3, vol=0.03, reversion=0.02, floor=0.0)

    rows: list[dict] = []
    for i, d in enumerate(dates):
        iso = d.isoformat()
        rows.append({"series": "vix_close", "date": iso, "value": round(float(vix[i]), 2)})
        rows.append({"series": "yield_3m", "date": iso, "value": round(float(yield_3m[i]), 3)})
        rows.append({"series": "yield_2y", "date": iso, "value": round(float(yield_2y[i]), 3)})
        rows.append({"series": "yield_10y", "date": iso, "value": round(float(yield_10y[i]), 3)})
    return rows
