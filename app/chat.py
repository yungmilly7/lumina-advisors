"""A grounded Q&A layer on top of the forecast data -- lets a person ask
plain-English questions ("why is NVDA down?", "which company has the
most confident call today?", "what's Apple's biggest connected-company
risk?") instead of only reading tables and charts.

Talks to Claude the same way app/forecast.py's narrative feature does: a
direct HTTPS call to api.anthropic.com via urllib (no SDK dependency,
consistent with this project's zero-third-party-dependency stance -- see
README's "Why zero dependencies"). Every answer is grounded in a text
snapshot of this platform's own data (the selected company's forecast,
drivers, news, neighbors, earnings -- or a whole-market snapshot when no
company is selected) built fresh per request, so the model is reasoning
over real numbers from this site rather than its own background
knowledge of the real world.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from app import db, scoring
from app.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL
from app.engine import engine
from app.universe import COMPANY_BY_TICKER, neighbors

log = logging.getLogger("stockgraph.chat")

SYSTEM_PROMPT = """You are the research assistant embedded in Lumina Advisors, a stock-forecasting \
website that a finance student built. You answer questions using ONLY the DATA SNAPSHOT provided \
in each message -- it is real data from this platform (as of right now), not from your own \
training. Do not bring in outside knowledge about real companies, real current events, or real \
stock prices; if the snapshot doesn't contain what's needed to answer, say so plainly rather than \
guessing or using what you otherwise know.

Style: concise (2-5 sentences unless the question genuinely needs a short list), plain English, no \
hedging filler. Never give direct buy/sell/hold investment advice or tell the person what to do \
with their money -- explain what the model and data show instead. If the question drifts toward \
"should I buy/sell", redirect to explaining the data and note forecasts here are probabilistic \
estimates from a transparent statistical model, not guarantees. This is not investment advice."""

MAX_HISTORY_MESSAGES = 12  # ~6 user/assistant turns of prior context


def _fmt_pct(x, digits=2):
    return f"{x * 100:.{digits}f}%" if x is not None else "n/a"


def _company_snapshot(ticker: str, horizon: int) -> str:
    c = COMPANY_BY_TICKER[ticker]
    f = engine.get_forecast(ticker, horizon, with_llm=False)
    drivers = f.get("drivers")
    if isinstance(drivers, str):
        drivers = json.loads(drivers)
    driver_lines = "\n".join(
        f"  - {d['label']}: contribution={d['contribution']:+.3f} (value={d['value']:.3f})"
        for d in (drivers or [])[:8]
    )

    nb_lines = []
    for edge in neighbors(ticker)[:10]:
        other = edge.dst if edge.src == ticker else edge.src
        oc = COMPANY_BY_TICKER[other]
        nb_lines.append(f"  - {edge.kind} <-> {other} ({oc.name}): {edge.note} (weight={edge.weight})")

    news_rows = db.get_news(ticker, limit=6)
    news_lines = [f"  - ({n['published'][:10]}) {n['headline']} [sentiment={n['sentiment']:.2f}]" for n in news_rows]

    earn_rows = db.get_earnings(ticker)
    next_earn = next((e for e in earn_rows if e["is_future"]), None)
    last_earn = next((e for e in reversed(earn_rows) if not e["is_future"]), None)

    prices = db.get_prices(ticker, limit_days=30)
    price_line = ""
    if prices:
        closes = [p["close"] for p in prices]
        price_line = (
            f"Last 30 sessions: close range ${min(closes):.2f}-${max(closes):.2f}, "
            f"most recent close ${closes[-1]:.2f} on {prices[-1]['date']}."
        )

    return f"""COMPANY: {c.name} ({ticker}) -- sector: {c.sector}, industry: {c.industry}
FORECAST ({horizon}-trading-day horizon, as of {f['as_of']}):
  direction={f['direction']}, probability_up={_fmt_pct(f['prob_up'], 1)}, expected_move={_fmt_pct(f['expected_move_pct'])}, confidence={_fmt_pct(f['confidence'], 1)}
  base_price=${f['base_price']}, target_price=${f['target_price']}, ~80% range=[{_fmt_pct(f['low_pct'])}, {_fmt_pct(f['high_pct'])}]
  plain-English rationale: {f['rationale']}
TOP MODEL DRIVERS (standardized contribution to the direction call; positive=bullish):
{driver_lines or "  (none)"}
{price_line}
NEXT EARNINGS: {next_earn['report_date'] if next_earn else 'not scheduled in data'}
LAST EARNINGS SURPRISE: {f"{last_earn['surprise_pct']:+.1f}%" if last_earn and last_earn['surprise_pct'] is not None else 'n/a'}
CONNECTED COMPANIES (relationship graph, {len(neighbors(ticker))} total, showing up to 10):
{chr(10).join(nb_lines) or "  (none)"}
RECENT NEWS (up to 6, most recent first):
{chr(10).join(news_lines) or "  (none)"}"""


def _macro_regime_line() -> str:
    """One line of shared macro context (VIX, yield curve) for the
    whole-market chat snapshot -- best-effort, since a fresh DB before the
    first ingestion run has no macro_series rows yet."""
    vals = db.latest_macro_values()
    if not vals.get("vix_close"):
        return ""
    bits = [f"VIX {vals['vix_close']:.1f}"]
    if vals.get("yield_10y") is not None and vals.get("yield_2y") is not None:
        spread = vals["yield_10y"] - vals["yield_2y"]
        bits.append(f"10y-2y Treasury spread {spread:+.2f}pp{' (inverted)' if spread < 0 else ''}")
    return "  macro regime: " + ", ".join(bits)


def _market_snapshot(horizon: int) -> str:
    forecasts = engine.list_forecasts(horizon)
    if not forecasts:
        return "No forecasts are available yet."
    n = len(forecasts)
    up = sum(1 for f in forecasts if f["direction"] == "up")
    avg_conf = sum(f["confidence"] for f in forecasts) / n
    avg_move = sum(abs(f["expected_move_pct"]) for f in forecasts) / n
    sorted_by_move = sorted(forecasts, key=lambda f: f["expected_move_pct"], reverse=True)
    gainers = sorted_by_move[:8]
    decliners = list(reversed(sorted_by_move[-8:]))
    most_confident = sorted(forecasts, key=lambda f: f["confidence"], reverse=True)[:8]

    def line(f):
        c = COMPANY_BY_TICKER[f["ticker"]]
        return f"  - {f['ticker']} ({c.name}, {c.sector}): {f['direction']} {_fmt_pct(f['expected_move_pct'])}, confidence={_fmt_pct(f['confidence'], 1)}"

    sc = scoring.scorecard()
    overall = sc.get("overall", {})

    return f"""WHOLE-MARKET SNAPSHOT ({horizon}-trading-day horizon, {n} companies tracked):
  breadth: {up} called UP / {n - up} called DOWN ({_fmt_pct(up / n, 0)} up)
  average confidence: {_fmt_pct(avg_conf, 1)}, average |expected move|: {_fmt_pct(avg_move)}
{_macro_regime_line()}
TOP GAINERS (by model expected move):
{chr(10).join(line(f) for f in gainers)}
TOP DECLINERS (by model expected move):
{chr(10).join(line(f) for f in decliners)}
MOST CONFIDENT CALLS:
{chr(10).join(line(f) for f in most_confident)}
MODEL ACCURACY SCORECARD (honest, walk-forward backtested, out-of-sample):
  overall: n={overall.get('n', 0)}, direction hit rate={_fmt_pct(overall.get('hit_rate'), 1) if overall.get('hit_rate') is not None else 'n/a'}, mean abs error={_fmt_pct(overall.get('mae'), 2) if overall.get('mae') is not None else 'n/a'}"""


def _preferences_block(preferences: dict | None) -> str:
    if not preferences:
        return ""
    bits = []
    if preferences.get("goals"):
        bits.append(f"goals={', '.join(preferences['goals'])}")
    if preferences.get("risk_tolerance"):
        bits.append(f"risk_tolerance={preferences['risk_tolerance']}")
    if preferences.get("horizon"):
        bits.append(f"preferred_horizon={preferences['horizon']}")
    if preferences.get("sectors"):
        bits.append(f"sectors_of_interest={', '.join(preferences['sectors'])}")
    if preferences.get("experience"):
        bits.append(f"experience={preferences['experience']}")
    if not bits:
        return ""
    return (
        "\n\nTHIS USER'S SAVED INVESTOR PROFILE (from their own onboarding survey -- use it to "
        "tailor tone/emphasis, e.g. lean into sectors they said they care about or match their "
        "stated experience level, but never as a reason to give direct buy/sell advice):\n  "
        + "; ".join(bits)
    )


def build_snapshot(ticker: str | None, horizon: int, preferences: dict | None = None) -> str:
    if ticker and ticker in COMPANY_BY_TICKER:
        base = _company_snapshot(ticker, horizon)
    else:
        base = _market_snapshot(horizon)
    return base + _preferences_block(preferences)


def is_configured() -> bool:
    return bool(ANTHROPIC_API_KEY)


def answer(message: str, ticker: str | None, horizon: int, history: list[dict],
           preferences: dict | None = None) -> dict:
    if not is_configured():
        return {
            "configured": False,
            "reply": "AI chat isn't turned on for this deployment yet -- set ANTHROPIC_API_KEY in "
            ".env to enable it. Everything else on the site works without it.",
        }

    try:
        snapshot = build_snapshot(ticker, horizon, preferences)
    except Exception as e:  # pragma: no cover - defensive
        log.exception("failed to build chat snapshot")
        return {"configured": True, "reply": f"Sorry, I couldn't load the data for that ({e}). Try again?"}

    messages = []
    for h in (history or [])[-MAX_HISTORY_MESSAGES:]:
        role = h.get("role")
        content = h.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content[:4000]})

    user_turn = f"DATA SNAPSHOT:\n{snapshot}\n\nQUESTION: {message.strip()[:2000]}"
    messages.append({"role": "user", "content": user_turn})

    body = json.dumps(
        {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 500,
            "system": SYSTEM_PROMPT,
            "messages": messages,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        method="POST",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        blocks = data.get("content", [])
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
        return {"configured": True, "reply": text or "I didn't get a usable response -- try rephrasing?"}
    except urllib.error.HTTPError as e:
        log.warning("chat call failed: HTTP %s: %s", e.code, e.read()[:500])
        return {"configured": True, "reply": "The AI service returned an error. Try again in a moment."}
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        log.warning("chat call failed: %s", e)
        return {"configured": True, "reply": "Couldn't reach the AI service just now. Try again in a moment."}
