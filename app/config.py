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
