"""Recommendation engine: turns today's forecasts into a short, ranked
"what to invest in, when, why, and how much" list.

This is NOT a new signal. Every number here traces back to a forecast row
app.forecast already computed and a ticker's own backtested track record
app.scoring already logged -- this module only filters, ranks, and sizes
what's already on the site into something actionable instead of a 448-row
table. It never gives advice in its own voice; it exposes the ranking
math (conviction score, how sizing was computed) so the "why" is always
inspectable, matching this project's explainable-by-design ethos (see
app/trading.py's docstring for the same philosophy applied to the opt-in
paper-trading bot). Nothing here places an order -- see app.trading for
that, which is a separate, explicitly opt-in feature.

Ranking ("what" and, via horizon, "when"):
  A ticker's forecast confidence (app.forecast.generate_forecast's
  prob_up-spread-times-holdout-accuracy figure) is blended with that
  SAME ticker's own historical backtested hit rate (db.scorecard_by_ticker)
  -- a confident call today on a ticker this model has historically been
  coin-flip-or-worse on gets discounted, not taken at face value. Small
  samples are Bayesian-shrunk toward 50% (TRACK_RECORD_SHRINKAGE_K pseudo-
  observations) so one ticker's lucky/unlucky handful of outcomes can't
  swing its score wildly. See _conviction().

  Deliberately NOT part of the ranking: expected_move_pct's magnitude.
  Mixing "how sure" (confidence/track record) with "how big" (expected
  move) into one score would blur what a rank actually means; expected
  move and target price are still shown per pick so the person can weigh
  reward size themselves.

  Deliberately NO hard confidence/move floor that could return zero picks
  (unlike app.trading's TRADING_MIN_CONFIDENCE, which guards real paper
  orders). This model's confidence values run far lower in "demo" mode
  (synthetic data, holdout accuracy near a coin flip -- see
  tests/smoke_test.py and README) than they would against real live data,
  so a floor tuned for one would silently empty out the other. Instead
  every pick is labeled with signal_strength (see _signal_strength) so a
  weak field of candidates reads as weak, not as "no picks".

Sizing ("how much"):
  A configured hypothetical portfolio (see portfolio_size/set_portfolio_size
  -- a planning number stored in the meta table, not a real account) is
  split across the top picks proportional to conviction score, capped per
  position at RECO_MAX_POSITION_PCT so no single ticker dominates
  regardless of score, with the capped remainder redistributed across the
  rest (iterative water-filling, see _size_positions). RECO_CASH_RESERVE_PCT
  of the portfolio is held back as cash, never allocated.

  "Short" candidates (model expects the price down) are ranked and shown
  alongside the buys but never sized in dollars -- shorting needs a
  margin/short-selling account a plain brokerage account may not have,
  and this module has no way to know if the person has one.

Why ("rationale"):
  Reuses the same plain-English rationale string generate_forecast already
  builds for every ticker (not just the ones a person happens to click
  into) -- identical text to what the forecast detail page shows for that
  ticker, so there's exactly one place that sentence is written.
"""
from __future__ import annotations

from app import db
from app.config import (
    RECO_CASH_RESERVE_PCT,
    RECO_DEFAULT_HORIZON_DAYS,
    RECO_DEFAULT_PORTFOLIO_USD,
    RECO_MAX_PICKS,
    RECO_MAX_POSITION_PCT,
    RECO_MIN_CONFIDENCE,
    RECO_MIN_MOVE_PCT,
)
from app.engine import engine
from app.universe import COMPANY_BY_TICKER

PORTFOLIO_META_KEY = "reco_portfolio_usd"

# Pseudo-observations at 50% a ticker's own backtested hit rate gets
# shrunk toward -- with this project's universe-wide sample sizes (~175
# held-out outcomes per ticker, see db.scorecard_by_ticker), this pulls a
# ticker with a handful of lucky/unlucky calls most of the way back to
# even while still letting a large, consistent track record show through.
TRACK_RECORD_SHRINKAGE_K = 30

# Thresholds for the plain-English signal_strength label shown next to
# every pick's raw confidence number, calibrated against this project's
# OWN observed confidence range (demo mode currently tops out well under
# 0.2 -- see the module docstring) rather than an arbitrary 0-1 scale
# that would label everything "weak" without context.
_STRENGTH_BANDS = [(0.35, "strong"), (0.15, "moderate"), (0.0, "weak")]


def _signal_strength(confidence: float) -> str:
    for floor, label in _STRENGTH_BANDS:
        if confidence >= floor:
            return label
    return "weak"


def _shrunk_hit_rate(hit_rate: float | None, n: int) -> float:
    if hit_rate is None or not n:
        return 0.5
    k = TRACK_RECORD_SHRINKAGE_K
    return (hit_rate * n + 0.5 * k) / (n + k)


def _conviction(confidence: float, shrunk_hit_rate: float) -> float:
    """confidence, discounted (never boosted above face value) by how
    trustworthy this ticker's own track record has actually been.
    edge=0 (track record at or below a coin flip) halves the raw
    confidence rather than zeroing it out entirely -- a single bad-luck
    stretch on low sample size shouldn't fully disqualify a ticker,
    it should just stop getting extra credit."""
    edge = max(0.0, (shrunk_hit_rate - 0.5) * 2.0)  # 0..1
    return confidence * (0.5 + 0.5 * edge)


def portfolio_size() -> float:
    raw = db.get_meta(PORTFOLIO_META_KEY)
    if raw is None:
        return RECO_DEFAULT_PORTFOLIO_USD
    try:
        return float(raw)
    except ValueError:
        return RECO_DEFAULT_PORTFOLIO_USD


def set_portfolio_size(usd: float) -> None:
    if usd <= 0 or usd != usd:  # NaN guard
        raise ValueError("portfolio size must be a positive number")
    db.set_meta(PORTFOLIO_META_KEY, str(float(usd)))


def _size_positions(candidates: list[dict], investable_usd: float) -> None:
    """Confidence-weighted allocation across `candidates` (mutates each
    dict in place with allocation_usd/allocation_pct/suggested_shares),
    capped per position at RECO_MAX_POSITION_PCT of `investable_usd`.

    Plain proportional splitting can put more than the cap on a dominant
    top pick, so this "water-fills" instead: whichever picks would exceed
    the cap get pinned there and removed from the pool, then the
    remaining budget is re-split across what's left, repeating until
    nothing overflows. With <=1 candidate left the cap stops applying
    (there's nowhere else for the money to go)."""
    for c in candidates:
        c["allocation_usd"] = 0.0
    remaining = list(candidates)
    pool = investable_usd
    cap = RECO_MAX_POSITION_PCT * investable_usd
    while remaining:
        total_conviction = sum(c["conviction_score"] for c in remaining) or 1.0
        overflowed = []
        for c in remaining:
            share = pool * (c["conviction_score"] / total_conviction)
            if share > cap and len(remaining) > 1:
                overflowed.append(c)
        if not overflowed:
            for c in remaining:
                c["allocation_usd"] += pool * (c["conviction_score"] / total_conviction)
            break
        for c in overflowed:
            c["allocation_usd"] = cap
            pool -= cap
            remaining.remove(c)

    for c in candidates:
        usd = c["allocation_usd"]
        c["allocation_usd"] = round(usd, 2)
        c["allocation_pct_of_portfolio"] = round(usd / investable_usd, 4) if investable_usd else 0.0
        c["suggested_shares"] = round(usd / c["base_price"], 3) if c.get("base_price") else None


def top_picks(
    horizon: int | None = None,
    portfolio_usd: float | None = None,
    max_picks: int | None = None,
) -> dict:
    """Ranked, sized picks for `horizon` trading days out ("when" = "over
    the next `horizon` trading days", same window the rest of the site
    already presents that forecast in).

    Returns {"generated_at", "horizon_days", "portfolio_usd",
    "investable_usd", "cash_reserve_usd", "buys": [...], "shorts": [...],
    "method_note": "..."}. `buys` are sized against the portfolio;
    `shorts` are ranked and shown with the same rationale/confidence
    fields but never sized in dollars (see module docstring).
    """
    horizon = horizon or RECO_DEFAULT_HORIZON_DAYS
    if horizon not in engine.horizons():
        raise ValueError(f"no model for horizon={horizon}; available={engine.horizons()}")
    portfolio_usd = portfolio_usd if portfolio_usd is not None else portfolio_size()
    max_picks = max_picks or RECO_MAX_PICKS
    cash_reserve_usd = portfolio_usd * RECO_CASH_RESERVE_PCT
    investable_usd = portfolio_usd - cash_reserve_usd

    forecasts = engine.list_forecasts(horizon)
    track_records = {r["ticker"]: dict(r) for r in db.scorecard_by_ticker()}

    candidates = []
    for f in forecasts:
        confidence = f.get("confidence") or 0.0
        if confidence < RECO_MIN_CONFIDENCE or abs(f["expected_move_pct"]) < RECO_MIN_MOVE_PCT:
            continue
        tr = track_records.get(f["ticker"])
        shrunk = _shrunk_hit_rate(tr["hit_rate"] if tr else None, tr["n"] if tr else 0)
        c = COMPANY_BY_TICKER.get(f["ticker"])
        candidates.append({
            "ticker": f["ticker"],
            "name": c.name if c else f["ticker"],
            "sector": c.sector if c else None,
            "action": "buy" if f["direction"] == "up" else "short",
            "horizon_days": horizon,
            "expected_move_pct": f["expected_move_pct"],
            "confidence": confidence,
            "signal_strength": _signal_strength(confidence),
            "base_price": f["base_price"],
            "target_price": f["target_price"],
            "days_to_earnings": f.get("days_to_earnings"),
            "rationale": f["rationale"],
            "track_record": (
                {"n": tr["n"], "hit_rate": round(tr["hit_rate"], 4), "mae": round(tr["mae"], 4)}
                if tr else None
            ),
            "conviction_score": round(_conviction(confidence, shrunk), 4),
        })

    candidates.sort(key=lambda c: c["conviction_score"], reverse=True)
    buys = [c for c in candidates if c["action"] == "buy"][:max_picks]
    shorts = [c for c in candidates if c["action"] == "short"][:max_picks]

    _size_positions(buys, investable_usd)
    for c in shorts:
        c["allocation_usd"] = None
        c["allocation_pct_of_portfolio"] = None
        c["suggested_shares"] = None

    return {
        "generated_at": db.now_iso(),
        "horizon_days": horizon,
        "portfolio_usd": round(portfolio_usd, 2),
        "cash_reserve_usd": round(cash_reserve_usd, 2),
        "investable_usd": round(investable_usd, 2),
        "buys": buys,
        "shorts": shorts,
        "method_note": (
            "Ranked by model confidence, discounted by each ticker's own backtested "
            "track record (small samples shrunk toward a coin flip). Sized proportional "
            f"to that ranking, capped at {RECO_MAX_POSITION_PCT:.0%} of the investable "
            f"amount per position, with {RECO_CASH_RESERVE_PCT:.0%} of the portfolio held "
            "back as cash. This is model output, not personalized financial advice -- see "
            "each pick's track record and the site-wide scorecard for how reliable this "
            "model has actually been."
        ),
    }
