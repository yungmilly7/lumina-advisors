"""HTTP API. Built on app.httpserver -- a minimal stdlib http.server-based
router -- instead of Starlette/FastAPI, so this project has zero
third-party web dependencies. Response shapes are simple enough that
hand-written JSON is clearer than a schema layer here.
"""
from __future__ import annotations

import json
import logging

from app import auth, chat, db
from app.config import BASE_DIR
from app.engine import engine
from app.forecast import HORIZONS
from app.httpserver import JSONResponse, Router
from app.universe import COMPANIES, COMPANY_BY_TICKER, EDGES, neighbors

log = logging.getLogger("stockgraph.api")


def _company_json(c) -> dict:
    return {"ticker": c.ticker, "name": c.name, "sector": c.sector, "industry": c.industry}


def _current_user(request) -> dict | None:
    token = request.cookies.get(auth.SESSION_COOKIE)
    return auth.user_from_token(token)


def _require_user(request):
    """Returns (user, None) or (None, error_response)."""
    user = _current_user(request)
    if not user:
        return None, JSONResponse({"error": "not signed in"}, status_code=401)
    return user, None


def status(request):
    return JSONResponse(engine.status())


def health(request):
    """Pipeline observability: is data live or synthetic, per field, for
    the whole universe -- not just the single global data_mode_active
    string /api/status gives you. Backs the Data Health panel so "is the
    live pipeline actually working" is answerable by looking at the site,
    not by reading server logs. Also doubles as the plain liveness check
    (a 200 with `ready` in the body) that a process supervisor or uptime
    monitor would poll."""
    from app.dataclients import finnhub

    runs = [dict(r) for r in db.recent_ingestion_runs(limit=10)]
    return JSONResponse({
        "ready": engine.ready,
        "trained_at": engine.trained_at,
        "last_ingested_at": db.get_meta("last_ingested_at"),
        "data_mode_active": db.get_meta("data_mode_active"),
        "finnhub_configured": finnhub.is_configured(),
        "refresh_interval_hours": engine.refresh_interval_hours,
        "last_refresh_error": engine.last_refresh_error,
        "source_summary": db.provenance_summary(),
        "recent_runs": runs,
        "universe_size": len(COMPANIES),
    })


def companies(request):
    return JSONResponse([_company_json(c) for c in COMPANIES])


def graph(request):
    nodes = [{**_company_json(c), "id": c.ticker} for c in COMPANIES]
    edges = [
        {"source": e.src, "target": e.dst, "kind": e.kind, "weight": e.weight, "note": e.note}
        for e in EDGES
    ]
    return JSONResponse({"nodes": nodes, "edges": edges})


def _data_sources_for(prov: dict) -> dict[str, dict | None]:
    """Turns one data_provenance row (or {} for a never-ingested ticker)
    into {field: {"source": ..., "updated_at": ...} | None} -- the shape
    the UI's per-field live/synthetic badges read directly."""
    out = {}
    for field in db.PROVENANCE_FIELDS:
        src = prov.get(f"{field}_source")
        out[field] = {"source": src, "updated_at": prov.get(f"{field}_updated_at")} if src else None
    return out


def forecasts_list(request):
    horizon = int(request.query_params.get("horizon", HORIZONS[1]))
    if horizon not in engine.horizons():
        return JSONResponse({"error": f"no model for horizon={horizon}", "available": engine.horizons()}, status_code=400)
    user = _current_user(request)
    watched = set(db.get_watchlist(user["id"])) if user else set()
    prefs = db.get_preferences(user["id"]) if user else None
    interest_sectors = set(prefs["sectors"]) if prefs and prefs.get("sectors") else set()
    fundamentals = db.all_fundamentals()
    provenance = db.get_all_provenance()
    ticker_scorecards = {r["ticker"]: dict(r) for r in db.scorecard_by_ticker()}
    out = []
    for f in engine.list_forecasts(horizon):
        row = dict(f)
        row.pop("_model_holdout_accuracy", None)
        row.pop("_model_holdout_mae", None)
        row["drivers"] = json.loads(row["drivers"]) if isinstance(row["drivers"], str) else row["drivers"]
        c = COMPANY_BY_TICKER[row["ticker"]]
        row["name"] = c.name
        row["sector"] = c.sector
        row["in_watchlist"] = row["ticker"] in watched
        row["matches_interest"] = c.sector in interest_sectors
        fd = fundamentals.get(row["ticker"])
        row["market_cap"] = fd.get("market_cap") if fd else None
        prov = provenance.get(row["ticker"], {})
        live_fields = sum(1 for f2 in db.PROVENANCE_FIELDS if prov.get(f"{f2}_source") not in (None, "demo"))
        row["is_live"] = live_fields > 0
        row["live_field_count"] = live_fields
        row["track_record"] = ticker_scorecards.get(row["ticker"])
        out.append(row)
    if interest_sectors:
        out.sort(key=lambda r: (not r["matches_interest"], -r["expected_move_pct"]))
    else:
        out.sort(key=lambda r: r["expected_move_pct"], reverse=True)
    return JSONResponse(out)


def forecast_detail(request):
    ticker = request.path_params["ticker"].upper()
    if ticker not in COMPANY_BY_TICKER:
        return JSONResponse({"error": "unknown ticker"}, status_code=404)
    horizon = int(request.query_params.get("horizon", HORIZONS[1]))
    with_llm = request.query_params.get("narrative", "1") != "0"
    if horizon not in engine.horizons():
        return JSONResponse({"error": f"no model for horizon={horizon}", "available": engine.horizons()}, status_code=400)

    try:
        f = engine.get_forecast(ticker, horizon, with_llm=with_llm)
    except Exception as e:
        log.exception("forecast_detail failed")
        return JSONResponse({"error": str(e)}, status_code=500)

    row = dict(f)
    row.pop("_model_holdout_accuracy", None)
    row.pop("_model_holdout_mae", None)
    row["drivers"] = json.loads(row["drivers"]) if isinstance(row["drivers"], str) else row["drivers"]
    c = COMPANY_BY_TICKER[ticker]
    row["name"] = c.name
    row["sector"] = c.sector
    row["industry"] = c.industry
    user = _current_user(request)
    row["in_watchlist"] = bool(user and ticker in db.get_watchlist(user["id"]))

    prices_rows = db.get_prices(ticker, limit_days=180)
    row["price_history"] = [
        {"date": p["date"], "close": p["close"], "volume": p["volume"]} for p in prices_rows
    ]

    # "more data": 52-week range + volume stats, computed from up to a year of history
    yr_rows = db.get_prices(ticker, limit_days=252)
    if yr_rows:
        closes = [p["close"] for p in yr_rows]
        vols = [p["volume"] for p in yr_rows if p["volume"] is not None]
        last_close = closes[-1]
        wk52_high = max(closes)
        wk52_low = min(closes)
        row["wk52_high"] = round(wk52_high, 2)
        row["wk52_low"] = round(wk52_low, 2)
        row["pct_off_52wk_high"] = round((last_close - wk52_high) / wk52_high, 4) if wk52_high else None
        row["pct_off_52wk_low"] = round((last_close - wk52_low) / wk52_low, 4) if wk52_low else None
        if vols:
            avg_vol_30 = sum(v for v in vols[-30:]) / len(vols[-30:])
            avg_vol_90 = sum(v for v in vols[-90:]) / len(vols[-90:]) if len(vols) >= 1 else avg_vol_30
            row["avg_volume_30d"] = round(avg_vol_30)
            row["avg_volume_90d"] = round(avg_vol_90)
            row["latest_volume"] = vols[-1]
            row["volume_ratio"] = round(vols[-1] / avg_vol_30, 2) if avg_vol_30 else None

    row["fundamentals"] = db.get_fundamentals(ticker)
    row["data_sources"] = _data_sources_for(db.get_provenance(ticker))
    row["track_record"] = db.get_ticker_scorecard(ticker) or None

    earn = db.get_earnings(ticker)
    row["earnings"] = [dict(e) for e in earn]

    filings = db.get_filings(ticker, limit=10)
    row["filings"] = [dict(fl) for fl in filings]

    insider_rows = db.get_insider_transactions(ticker, limit=15)
    row["insider_transactions"] = [dict(i) for i in insider_rows]

    news_rows = db.get_news(ticker, limit=12)
    row["news"] = [
        {**dict(n), "event_tags": json.loads(n["event_tags"] or "[]")} for n in news_rows
    ]

    nb = []
    for edge in neighbors(ticker):
        other = edge.dst if edge.src == ticker else edge.src
        oc = COMPANY_BY_TICKER[other]
        nb.append(
            {
                "ticker": other,
                "name": oc.name,
                "kind": edge.kind,
                "weight": edge.weight,
                "note": edge.note,
                "direction": "src" if edge.src == ticker else "dst",
            }
        )
    row["neighbors"] = nb

    return JSONResponse(row)


def prices(request):
    ticker = request.path_params["ticker"].upper()
    if ticker not in COMPANY_BY_TICKER:
        return JSONResponse({"error": "unknown ticker"}, status_code=404)
    days = int(request.query_params.get("days", 180))
    rows = db.get_prices(ticker, limit_days=days)
    return JSONResponse([dict(r) for r in rows])


def scorecard_view(request):
    from app import scoring

    return JSONResponse(scoring.scorecard())


def refresh(request):
    try:
        engine.bootstrap()
    except Exception as e:
        log.exception("refresh failed")
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse(engine.status())


def trading_status(request):
    """Paper-trading visibility: whether it's turned on, the last pass's
    result, and a log of every decision (opened/closed/skipped, and why)
    -- matches this project's "explainable" ethos instead of a bot that
    silently moves (paper) money with no record of its reasoning."""
    from app import trading

    return JSONResponse({
        "enabled": trading.enabled(),
        "last_result": engine.last_trading_result,
        "recent_trades": [dict(r) for r in db.recent_paper_trades(limit=50)],
    })


# ---------- accounts ----------


def signup(request):
    body = request.json()
    try:
        user = auth.signup(
            body.get("email", ""),
            body.get("password", ""),
            body.get("display_name", ""),
            client_ip=request.client_addr,
        )
    except auth.AuthError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    token = auth.create_session(user["id"])
    resp = JSONResponse({"user": auth.public_user(user)})
    resp.set_cookie(auth.SESSION_COOKIE, token, max_age=auth.SESSION_TTL_SECONDS)
    return resp


def login(request):
    body = request.json()
    try:
        user = auth.login(body.get("email", ""), body.get("password", ""), client_ip=request.client_addr)
    except auth.AuthError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    token = auth.create_session(user["id"])
    resp = JSONResponse({"user": auth.public_user(user)})
    resp.set_cookie(auth.SESSION_COOKIE, token, max_age=auth.SESSION_TTL_SECONDS)
    return resp


def logout(request):
    auth.logout(request.cookies.get(auth.SESSION_COOKIE))
    resp = JSONResponse({"ok": True})
    resp.clear_cookie(auth.SESSION_COOKIE)
    return resp


def me(request):
    user = _current_user(request)
    if not user:
        return JSONResponse({"user": None})
    return JSONResponse({
        "user": auth.public_user(user),
        "watchlist": db.get_watchlist(user["id"]),
        "preferences": db.get_preferences(user["id"]),
    })


# ---------- investor preferences (onboarding survey) ----------

VALID_GOALS = {"growth", "income", "value", "speculation", "learning"}
VALID_RISK = {"conservative", "moderate", "aggressive"}
VALID_HORIZON = {"short", "medium", "long"}
VALID_EXPERIENCE = {"new", "some", "experienced"}


def preferences_get(request):
    user, err = _require_user(request)
    if err:
        return err
    return JSONResponse({"preferences": db.get_preferences(user["id"])})


def preferences_save(request):
    user, err = _require_user(request)
    if err:
        return err
    body = request.json()
    goals = [g for g in (body.get("goals") or []) if g in VALID_GOALS]
    risk = body.get("risk_tolerance")
    if risk not in VALID_RISK:
        risk = None
    horizon = body.get("horizon")
    if horizon not in VALID_HORIZON:
        horizon = None
    sectors = [s for s in (body.get("sectors") or []) if isinstance(s, str)][:12]
    experience = body.get("experience")
    if experience not in VALID_EXPERIENCE:
        experience = None
    db.save_preferences(
        user["id"],
        {
            "goals": goals,
            "risk_tolerance": risk,
            "horizon": horizon,
            "sectors": sectors,
            "experience": experience,
        },
    )
    return JSONResponse({"preferences": db.get_preferences(user["id"])})


def sectors_list(request):
    return JSONResponse(sorted({c.sector for c in COMPANIES}))


# ---------- chat ----------


def chat_ask(request):
    body = request.json()
    message = (body.get("message") or "").strip()
    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)
    ticker = body.get("ticker")
    if ticker:
        ticker = ticker.upper()
        if ticker not in COMPANY_BY_TICKER:
            return JSONResponse({"error": "unknown ticker"}, status_code=404)
    horizon = int(body.get("horizon") or HORIZONS[1])
    if horizon not in engine.horizons():
        horizon = HORIZONS[1]
    history = body.get("history") or []
    user = _current_user(request)
    preferences = db.get_preferences(user["id"]) if user else None
    result = chat.answer(message, ticker, horizon, history, preferences=preferences)
    return JSONResponse(result)


def chat_status(request):
    return JSONResponse({"configured": chat.is_configured()})


# ---------- watchlist ----------


def watchlist_list(request):
    user, err = _require_user(request)
    if err:
        return err
    return JSONResponse({"watchlist": db.get_watchlist(user["id"])})


def watchlist_add(request):
    user, err = _require_user(request)
    if err:
        return err
    ticker = request.path_params["ticker"].upper()
    if ticker not in COMPANY_BY_TICKER:
        return JSONResponse({"error": "unknown ticker"}, status_code=404)
    db.add_watchlist(user["id"], ticker)
    return JSONResponse({"watchlist": db.get_watchlist(user["id"])})


def watchlist_remove(request):
    user, err = _require_user(request)
    if err:
        return err
    ticker = request.path_params["ticker"].upper()
    db.remove_watchlist(user["id"], ticker)
    return JSONResponse({"watchlist": db.get_watchlist(user["id"])})


def build_router() -> Router:
    router = Router()
    router.add("/api/health", health)
    router.add("/api/status", status)
    router.add("/api/companies", companies)
    router.add("/api/graph", graph)
    router.add("/api/forecasts", forecasts_list)
    router.add("/api/forecast/{ticker}", forecast_detail)
    router.add("/api/prices/{ticker}", prices)
    router.add("/api/scorecard", scorecard_view)
    router.add("/api/refresh", refresh, methods=["POST"])
    router.add("/api/trading/status", trading_status)
    router.add("/api/auth/signup", signup, methods=["POST"])
    router.add("/api/auth/login", login, methods=["POST"])
    router.add("/api/auth/logout", logout, methods=["POST"])
    router.add("/api/auth/me", me)
    router.add("/api/preferences", preferences_get)
    router.add("/api/preferences", preferences_save, methods=["POST"])
    router.add("/api/sectors", sectors_list)
    router.add("/api/watchlist", watchlist_list)
    router.add("/api/watchlist/{ticker}", watchlist_add, methods=["POST"])
    router.add("/api/watchlist/{ticker}", watchlist_remove, methods=["DELETE"])
    router.add("/api/chat", chat_ask, methods=["POST"])
    router.add("/api/chat/status", chat_status)
    router.serve_static(BASE_DIR / "static")
    return router


router = build_router()
