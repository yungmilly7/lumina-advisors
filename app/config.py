"""Central configuration for Lumina Advisors.

Everything is driven by environment variables so the same code runs the
same way in this sandbox (DATA_MODE=demo, no network) and on a real
server with internet access (DATA_MODE=live).
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def _load_dotenv(path: Path) -> None:
    """Tiny stdlib-only ".env" loader (no python-dotenv dependency).

    Real environment variables always win -- this only fills in ones that
    aren't already set, so `set FOO=bar` before launching still overrides
    whatever's in .env.
    """
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(BASE_DIR / ".env")

DB_PATH = os.environ.get("STOCKGRAPH_DB", str(DATA_DIR / "stockgraph.db"))

# "live"  -> hit Finnhub / Yahoo Finance / SEC EDGAR / Google News over the
#            network; raise if any ticker comes back with nothing at all
# "demo"  -> generate deterministic synthetic data, no network required
# "auto"  -> try live per-ticker/per-field; whatever a given ticker's live
#            fetch doesn't cover (a blocked endpoint, a rate limit, one bad
#            symbol) is backfilled with synthetic data for exactly that
#            gap, not the whole ticker and not the whole universe. Every
#            row this produces is honest about which parts are real: see
#            forecasts_list()'s per-field "source" tagging in api.py.
DATA_MODE = os.environ.get("STOCKGRAPH_DATA_MODE", "auto").lower()

# Optional: if set, the forecast agent asks Claude for a plain-English
# rationale on top of the quant model. Talks to api.anthropic.com directly
# over HTTPS (no SDK dependency) so it works anywhere that host is reachable.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# Haiku by default: this gets called on every forecast-detail page view (the
# narrative) and every chat message, so a fast/cheap model keeps latency and
# API cost sane if this ever gets real traffic. Override via .env if you'd
# rather trade cost for a stronger model.
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

# SEC requires a descriptive User-Agent with contact info on every request.
SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "Lumina Advisors research tool (contact: set SEC_USER_AGENT env var)"
)

# Optional: a free Finnhub API key (finnhub.io/register -- takes a minute,
# no card needed) makes earnings/fundamentals fetch through a real,
# licensed API instead of Yahoo's undocumented quoteSummary endpoint, which
# Yahoo's anti-bot layer has been blocking increasingly aggressively as this
# universe has grown (see app/dataclients/finnhub.py). Without a key, the
# pipeline just keeps using Yahoo/demo exactly as before -- this is
# additive, not a replacement that requires action.
FINNHUB_API_KEY = os.environ.get("STOCKGRAPH_FINNHUB_API_KEY", "")

HTTP_TIMEOUT = float(os.environ.get("STOCKGRAPH_HTTP_TIMEOUT", "10"))

# How many days of daily bars to keep per ticker.
PRICE_HISTORY_DAYS = int(os.environ.get("STOCKGRAPH_PRICE_DAYS", "400"))

HOST = os.environ.get("STOCKGRAPH_HOST", "0.0.0.0")
# Most hosts (Render, Railway, Heroku-style buildpacks, ...) assign a port at
# deploy time via the bare $PORT env var rather than letting you pick one --
# STOCKGRAPH_PORT still wins if it's explicitly set (e.g. running locally),
# but PORT is used as the fallback so this runs unmodified on those hosts.
PORT = int(os.environ.get("STOCKGRAPH_PORT") or os.environ.get("PORT") or "8000")

# Optional: paper-trading integration with Alpaca (alpaca.markets), a real
# brokerage whose API has a first-class paper-trading mode -- identical
# request/response shape to live trading, but every order fills against a
# simulated account funded with fake money. Get free paper keys at
# alpaca.markets -> sign up -> Paper Trading tab -> "Generate New Keys";
# no funding or approval needed since nothing here is real money. This is
# a real account signup, so it's something only Danny can do -- exactly
# like ANTHROPIC_API_KEY/STOCKGRAPH_FINNHUB_API_KEY above, this is unset
# (and the feature entirely off) until he pastes his own keys in.
#
# Two independent switches have to both be true before app.trading ever
# calls the broker: a key pair configured (app.broker.is_configured()) AND
# TRADING_ENABLED explicitly set -- so this can sit configured-but-off, or
# never run at all just by never setting STOCKGRAPH_TRADING_ENABLED, with
# zero risk of turning itself on. ALPACA_BASE_URL defaults to the paper
# host; pointing it at api.alpaca.markets (live trading, real money) is a
# deliberate override this project has never been run against and doesn't
# guard against.
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
TRADING_ENABLED = os.environ.get("STOCKGRAPH_TRADING_ENABLED", "false").strip().lower() in ("1", "true", "yes")

# Which trained horizon to trade (shortest by default -- fewer days for the
# forecast to be wrong before the position is closed out again).
TRADING_HORIZON_DAYS = int(os.environ.get("STOCKGRAPH_TRADING_HORIZON_DAYS", "1"))
# Only act on forecasts at or above this model confidence.
TRADING_MIN_CONFIDENCE = float(os.environ.get("STOCKGRAPH_TRADING_MIN_CONFIDENCE", "0.6"))
# At most this many open positions at once.
TRADING_MAX_POSITIONS = int(os.environ.get("STOCKGRAPH_TRADING_MAX_POSITIONS", "5"))
# Fixed dollar size per position (not a % of equity, so sizing can't spiral
# with account balance).
TRADING_POSITION_USD = float(os.environ.get("STOCKGRAPH_TRADING_POSITION_USD", "500"))
# Kill switch: if the paper account's own reported day-over-day P&L is
# already worse than -this fraction, skip opening any new positions for
# the rest of that pass.
TRADING_MAX_DAILY_LOSS_PCT = float(os.environ.get("STOCKGRAPH_TRADING_MAX_DAILY_LOSS_PCT", "0.03"))

# Recommendations ("what to invest in, when, why, how much" -- app.recommend):
# a read-only ranking + sizing layer on top of the same forecasts everything
# else on the site already shows. Unlike TRADING_MIN_CONFIDENCE above, there
# is deliberately no confidence/move floor that could return zero picks --
# this always surfaces the best *available* signals today and labels how
# strong they actually are (see recommend.py's signal_strength), rather than
# silently going empty. That matters because this model's confidence values
# run much lower in "demo" mode (synthetic data, holdout accuracy near a coin
# flip) than they might against real live data -- a fixed absolute floor
# tuned for one would misbehave in the other.
RECO_MAX_PICKS = int(os.environ.get("STOCKGRAPH_RECO_MAX_PICKS", "10"))
# Default horizon a fresh /api/recommendations call uses if none is given --
# matches the site's other default (HORIZONS[1] in forecast.py).
RECO_DEFAULT_HORIZON_DAYS = int(os.environ.get("STOCKGRAPH_RECO_DEFAULT_HORIZON", "5"))
# No single position may be sized above this fraction of the investable
# pool, however high its conviction score -- basic diversification even
# when one ticker dominates the ranking.
RECO_MAX_POSITION_PCT = float(os.environ.get("STOCKGRAPH_RECO_MAX_POSITION_PCT", "0.25"))
# Fraction of the configured portfolio size held back as cash rather than
# allocated across picks -- a small built-in buffer, not a recommendation
# to go 100% invested.
RECO_CASH_RESERVE_PCT = float(os.environ.get("STOCKGRAPH_RECO_CASH_RESERVE_PCT", "0.10"))
# Optional hard floors, off (0.0) by default -- see the note above on why a
# floor tuned for live-mode confidence would zero out demo mode. Set these
# via env if running against real live data and you want them enforced.
RECO_MIN_CONFIDENCE = float(os.environ.get("STOCKGRAPH_RECO_MIN_CONFIDENCE", "0.0"))
RECO_MIN_MOVE_PCT = float(os.environ.get("STOCKGRAPH_RECO_MIN_MOVE_PCT", "0.0"))
# Default hypothetical portfolio size ($) sizing is computed against, until
# the user sets their own via the Recommendations page (stored in the meta
# table, see app.recommend.portfolio_size/set_portfolio_size). Purely a
# paper/planning number -- nothing here executes a trade.
RECO_DEFAULT_PORTFOLIO_USD = float(os.environ.get("STOCKGRAPH_RECO_DEFAULT_PORTFOLIO_USD", "10000"))
