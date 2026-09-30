"""Thin stdlib-only client for Alpaca's paper-trading REST API.

Alpaca (alpaca.markets) is a real brokerage with a free paper-trading
environment: the exact same API shape as live trading, but every order
fills against a simulated account funded with fake money -- nothing here
can move real dollars as long as ALPACA_BASE_URL points at
paper-api.alpaca.markets, which is this project's hard-coded default (see
app/config.py). Talks to Alpaca the same way app/chat.py and
app/forecast.py talk to api.anthropic.com: a direct HTTPS call via
urllib, no SDK dependency, consistent with this project's zero-third-
party-dependency stance.

`is_configured()` is False until Danny pastes his own free paper-trading
keys into .env (a real account signup, so it's something only he can do --
exactly like ANTHROPIC_API_KEY/STOCKGRAPH_FINNHUB_API_KEY elsewhere in this
project). app/trading.py treats "not configured" as "this feature is off",
never as an error.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from app.config import ALPACA_API_KEY, ALPACA_BASE_URL, ALPACA_SECRET_KEY

log = logging.getLogger("stockgraph.broker")


class BrokerError(Exception):
    """Raised for anything from a missing key pair to a rejected order --
    callers (app.trading) catch this per-call and log it as a skipped/
    failed trade rather than letting it crash a bootstrap or refresh."""


def is_configured() -> bool:
    return bool(ALPACA_API_KEY and ALPACA_SECRET_KEY)


def _request(method: str, path: str, body: dict | None = None, timeout: float = 15):
    if not is_configured():
        raise BrokerError("Alpaca is not configured (set ALPACA_API_KEY / ALPACA_SECRET_KEY)")
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        f"{ALPACA_BASE_URL}{path}",
        data=data,
        method=method,
        headers={
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise BrokerError(f"Alpaca {method} {path} failed: HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise BrokerError(f"Alpaca {method} {path} unreachable: {e.reason}") from e


def get_account() -> dict:
    """Includes `equity` and `last_equity` (prior trading day's closing
    equity) -- app.trading's kill switch compares these directly instead
    of reconstructing P&L from the order/position log itself."""
    return _request("GET", "/v2/account")


def list_positions() -> list[dict]:
    return _request("GET", "/v2/positions")


def close_position(symbol: str) -> dict:
    return _request("DELETE", f"/v2/positions/{symbol}")


def submit_order(symbol: str, side: str, notional_usd: float) -> dict:
    """Market order sized in dollars (Alpaca's `notional` field) rather
    than share count -- the simplest way to respect a fixed per-trade
    dollar budget across tickers priced anywhere from $5 to $2,000.
    Fractional/notional orders require time_in_force="day" on Alpaca."""
    return _request(
        "POST",
        "/v2/orders",
        body={
            "symbol": symbol,
            "notional": round(notional_usd, 2),
            "side": side,
            "type": "market",
            "time_in_force": "day",
        },
    )
