"""Paper-trading layer: turns today's forecasts into simulated orders on
Alpaca's paper-trading account (see app/broker.py). This never touches
real money -- Alpaca's paper environment is a fully simulated account, and
nothing in this codebase points at Alpaca's live-trading host.

Off by default. Two independent switches both have to be true before a
single broker call happens: app.broker.is_configured() (Danny has pasted
his own free paper-trading keys into .env -- a real account signup, so
it's something only he can do) AND STOCKGRAPH_TRADING_ENABLED is explicitly
set. See app/config.py and .env.example. Deliberately NOT wired into
render.yaml / the public Render deployment -- this is meant to stay a
local, opt-in experiment on Danny's own machine against his own paper
account, not something a visitor to the public demo can trigger.

Deliberately simple and conservative for a first version:
  - only the shortest trained horizon is traded (fewest days for the
    forecast to be wrong before the position is closed again)
  - only forecasts at or above a confidence floor are acted on at all
  - a fixed number of positions are held at once, each sized as a fixed
    dollar amount (not a % of account equity, so sizing can't spiral)
  - a kill switch checks the paper account's own reported day-over-day P&L
    *before* opening anything, and skips the rest of the pass if it's
    already past the configured loss threshold -- important because the
    underlying model's holdout accuracy is currently close to a coin flip
    (~50%, see the scorecard), so this should fail safe on a bad day
    rather than compound it
  - every decision -- opened, closed, or skipped, and why -- is logged to
    the paper_trades table so the site can show what the bot did and why,
    matching this project's "explainable" ethos instead of being a black
    box that silently moves (paper) money around
"""
from __future__ import annotations

import logging

from app import broker, db
from app.config import (
    TRADING_ENABLED,
    TRADING_HORIZON_DAYS,
    TRADING_MAX_DAILY_LOSS_PCT,
    TRADING_MAX_POSITIONS,
    TRADING_MIN_CONFIDENCE,
    TRADING_POSITION_USD,
)

log = logging.getLogger("stockgraph.trading")


def enabled() -> bool:
    return TRADING_ENABLED and broker.is_configured()


def _note(as_of: str, reason: str, **overrides) -> dict:
    row = {
        "as_of": as_of, "ticker": None, "action": "skipped", "side": None,
        "notional_usd": None, "forecast_confidence": None, "forecast_direction": None,
        "reason": reason, "alpaca_order_id": None, "status": "ok",
    }
    row.update(overrides)
    return row


def run_trading_pass(engine) -> dict:
    """One daily-rebalance pass: check the kill switch, close every
    position this bot currently holds (each forecast is a fresh "as of
    right now" call, so yesterday's position is stale info, not a thesis
    to hold through today), then open fresh ones from today's top-
    confidence forecasts. Never raises -- a failed pass is logged and
    returned as a result dict, the same way a failed scheduled refresh is,
    so it can't take down bootstrap or the background refresh loop."""
    as_of = db.now_iso()
    if not enabled():
        return {"ran": False, "reason": "trading not enabled/configured"}

    try:
        account = broker.get_account()
    except broker.BrokerError as e:
        log.warning("trading pass aborted: could not reach broker: %s", e)
        return {"ran": False, "reason": f"broker unreachable: {e}"}

    rows = []
    equity = float(account.get("equity", 0) or 0)
    last_equity = float(account.get("last_equity") or equity)
    daily_pnl_pct = (equity - last_equity) / last_equity if last_equity else 0.0
    if daily_pnl_pct <= -TRADING_MAX_DAILY_LOSS_PCT:
        rows.append(_note(
            as_of,
            f"kill switch: daily P&L {daily_pnl_pct:.2%} breached the "
            f"-{TRADING_MAX_DAILY_LOSS_PCT:.0%} limit -- skipping this pass entirely",
        ))
        db.insert_paper_trades_batch(rows)
        log.warning("trading pass halted by kill switch: daily P&L %.2f%%", daily_pnl_pct * 100)
        return {"ran": True, "halted_by_kill_switch": True, "daily_pnl_pct": daily_pnl_pct}

    try:
        positions = broker.list_positions()
    except broker.BrokerError as e:
        log.warning("trading pass: could not list positions: %s", e)
        positions = []

    for p in positions:
        symbol = p.get("symbol")
        try:
            order = broker.close_position(symbol)
            rows.append(_note(
                as_of, "daily rebalance: closing prior position",
                ticker=symbol, action="closed", side="sell",
                alpaca_order_id=order.get("id"),
            ))
        except broker.BrokerError as e:
            rows.append(_note(
                as_of, f"close failed: {e}",
                ticker=symbol, action="closed", side="sell", status="error",
            ))

    forecasts = engine.list_forecasts(TRADING_HORIZON_DAYS) if TRADING_HORIZON_DAYS in engine.horizons() else []
    candidates = sorted(
        (f for f in forecasts if (f.get("confidence") or 0) >= TRADING_MIN_CONFIDENCE),
        key=lambda f: f["confidence"],
        reverse=True,
    )[:TRADING_MAX_POSITIONS]

    if not candidates:
        rows.append(_note(
            as_of, f"no forecast met the {TRADING_MIN_CONFIDENCE:.0%} confidence floor today",
        ))

    for f in candidates:
        side = "buy" if f["direction"] == "up" else "sell"
        try:
            order = broker.submit_order(f["ticker"], side, TRADING_POSITION_USD)
            rows.append(_note(
                as_of, f"confidence {f['confidence']:.0%} >= floor {TRADING_MIN_CONFIDENCE:.0%}",
                ticker=f["ticker"], action="opened", side=side,
                notional_usd=TRADING_POSITION_USD, forecast_confidence=f["confidence"],
                forecast_direction=f["direction"], alpaca_order_id=order.get("id"),
            ))
        except broker.BrokerError as e:
            rows.append(_note(
                as_of, f"order failed: {e}",
                ticker=f["ticker"], action="opened", side=side,
                notional_usd=TRADING_POSITION_USD, forecast_confidence=f["confidence"],
                forecast_direction=f["direction"], status="error",
            ))

    db.insert_paper_trades_batch(rows)
    log.info("trading pass complete: %d closed, %d opened", len(positions), len(candidates))
    return {"ran": True, "halted_by_kill_switch": False, "closed": len(positions), "opened": len(candidates)}
