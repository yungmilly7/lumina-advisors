"""SQLite storage. One file, no server to run, easy to ship."""
from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from datetime import datetime, timezone

from app.config import DB_PATH

_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,          -- YYYY-MM-DD
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON prices(ticker, date);

CREATE TABLE IF NOT EXISTS earnings (
    ticker TEXT NOT NULL,
    report_date TEXT NOT NULL,   -- YYYY-MM-DD, most recent/next known date
    eps_estimate REAL,
    eps_actual REAL,
    surprise_pct REAL,
    fiscal_period TEXT,
    is_future INTEGER DEFAULT 0,
    PRIMARY KEY (ticker, report_date)
);

CREATE TABLE IF NOT EXISTS filings (
    ticker TEXT NOT NULL,
    filed_date TEXT NOT NULL,
    form_type TEXT,
    title TEXT,
    url TEXT,
    PRIMARY KEY (ticker, filed_date, form_type)
);

CREATE TABLE IF NOT EXISTS news (
    ticker TEXT NOT NULL,
    published TEXT NOT NULL,     -- ISO timestamp
    headline TEXT NOT NULL,
    source TEXT,
    url TEXT,
    sentiment REAL,              -- -1..1
    event_tags TEXT,             -- JSON list, e.g. ["product_launch","deal"]
    PRIMARY KEY (ticker, published, headline)
);
CREATE INDEX IF NOT EXISTS idx_news_ticker_pub ON news(ticker, published);

CREATE TABLE IF NOT EXISTS forecasts (
    ticker TEXT NOT NULL,
    as_of TEXT NOT NULL,          -- ISO timestamp forecast was generated
    horizon_days INTEGER NOT NULL,
    direction TEXT NOT NULL,      -- up | down
    prob_up REAL NOT NULL,
    expected_move_pct REAL NOT NULL,
    low_pct REAL,
    high_pct REAL,
    confidence REAL,
    rationale TEXT,
    drivers TEXT,                 -- JSON list of {factor, contribution, detail}
    llm_narrative TEXT,
    base_price REAL,
    target_price REAL,
    PRIMARY KEY (ticker, as_of, horizon_days)
);
CREATE INDEX IF NOT EXISTS idx_forecasts_ticker ON forecasts(ticker, as_of);

CREATE TABLE IF NOT EXISTS forecast_outcomes (
    ticker TEXT NOT NULL,
    as_of TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    predicted_direction TEXT,
    predicted_move_pct REAL,
    actual_move_pct REAL,
    direction_correct INTEGER,
    abs_error_pct REAL,
    evaluated_at TEXT,
    PRIMARY KEY (ticker, as_of, horizon_days)
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    display_name TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

CREATE TABLE IF NOT EXISTS watchlist (
    user_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    added_at TEXT NOT NULL,
    PRIMARY KEY (user_id, ticker),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS preferences (
    user_id INTEGER PRIMARY KEY,
    goals TEXT,              -- JSON list: growth, income, value, speculation, learning
    risk_tolerance TEXT,     -- conservative | moderate | aggressive
    horizon TEXT,            -- short | medium | long
    sectors TEXT,            -- JSON list of sector names the user cares about
    experience TEXT,         -- new | some | experienced
    updated_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Per-ticker, per-data-category provenance: which source last supplied
-- this field ("finnhub" | "yahoo" | "secedgar" | "google_news" | "demo")
-- and when. Two jobs: lets the UI show an honest live-vs-synthetic badge
-- per section instead of one all-or-nothing banner (see api.py's
-- forecasts_list/forecast_detail), and lets the ingestion pipeline decide
-- what's still "fresh" and skip re-fetching it (see pipeline.py's
-- incremental refresh) instead of hammering rate-limited APIs on every run.
CREATE TABLE IF NOT EXISTS data_provenance (
    ticker TEXT PRIMARY KEY,
    bars_source TEXT, bars_updated_at TEXT,
    earnings_source TEXT, earnings_updated_at TEXT,
    fundamentals_source TEXT, fundamentals_updated_at TEXT,
    news_source TEXT, news_updated_at TEXT,
    filings_source TEXT, filings_updated_at TEXT
);

-- Short history of ingestion runs (bootstrap or scheduled refresh), for the
-- /api/health endpoint and the Data Health panel -- otherwise the only way
-- to know the pipeline is healthy is to read the scrolling console log.
CREATE TABLE IF NOT EXISTS ingestion_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    mode TEXT,
    live_ok INTEGER,
    live_fail INTEGER,
    skipped_fresh INTEGER,
    elapsed_sec REAL,
    error TEXT
);

CREATE TABLE IF NOT EXISTS fundamentals (
    ticker TEXT PRIMARY KEY,
    description TEXT,
    market_cap REAL,
    pe_ratio REAL,
    forward_pe REAL,
    peg_ratio REAL,
    dividend_yield REAL,
    beta REAL,
    profit_margin REAL,
    revenue_growth REAL,
    analyst_target_mean REAL,
    analyst_target_high REAL,
    analyst_target_low REAL,
    analyst_recommendation TEXT,
    num_analyst_opinions INTEGER,
    updated_at TEXT NOT NULL
);

-- Every decision app.trading's paper-trading pass makes, whether it acted
-- or not (a skipped pass -- kill switch tripped, nothing met the
-- confidence floor -- is logged too, not just executed trades), so the
-- site can show a plain-English "what did the bot do and why" history
-- instead of that only being visible in server logs.
CREATE TABLE IF NOT EXISTS paper_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    as_of TEXT NOT NULL,            -- ISO timestamp this pass ran
    ticker TEXT,                    -- NULL for a whole-pass note (e.g. kill switch)
    action TEXT NOT NULL,           -- opened | closed | skipped
    side TEXT,                      -- buy | sell, when applicable
    notional_usd REAL,
    forecast_confidence REAL,
    forecast_direction TEXT,
    reason TEXT NOT NULL,
    alpaca_order_id TEXT,
    status TEXT NOT NULL            -- ok | error
);
CREATE INDEX IF NOT EXISTS idx_paper_trades_as_of ON paper_trades(as_of);

-- Open-market insider buy/sell transactions (SEC Form 4, transaction codes
-- P and S only -- see app.dataclients.secedgar._parse_form4_xml). Treated
-- like filings/news: append-only historical data fetched best-effort, with
-- no per-field freshness tracking in data_provenance, since a Form 4 filed
-- last month doesn't go "stale" the way a live price quote does.
CREATE TABLE IF NOT EXISTS insider_transactions (
    ticker TEXT NOT NULL,
    transaction_date TEXT NOT NULL,
    owner_name TEXT NOT NULL,
    is_officer INTEGER DEFAULT 0,
    is_director INTEGER DEFAULT 0,
    is_ten_pct_owner INTEGER DEFAULT 0,
    transaction_code TEXT NOT NULL,   -- P (buy) | S (sell)
    acquired_disposed TEXT,           -- A | D
    shares REAL,
    price REAL,
    value_usd REAL,
    PRIMARY KEY (ticker, transaction_date, owner_name, transaction_code, shares)
);
CREATE INDEX IF NOT EXISTS idx_insider_txn_ticker_date ON insider_transactions(ticker, transaction_date);
"""


def get_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            # WAL is nicer for concurrent access, but its shared-memory/mmap
            # requirements fail with "disk I/O error" on some mounted/synced
            # folders (network drives, OneDrive/iCloud-backed dirs, FUSE
            # bridges) -- fall back to the plain rollback journal, which
            # works everywhere and is plenty for this single-process app.
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            conn.execute("PRAGMA journal_mode=DELETE")
        _local.conn = conn
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()


@contextlib.contextmanager
def tx():
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def set_meta(key: str, value: str) -> None:
    with tx() as conn:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )


def get_meta(key: str, default: str | None = None) -> str | None:
    row = get_conn().execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def upsert_prices(ticker: str, rows: list[dict]) -> None:
    """rows: [{date, open, high, low, close, volume}, ...]"""
    if not rows:
        return
    with tx() as conn:
        conn.executemany(
            "INSERT INTO prices(ticker, date, open, high, low, close, volume) "
            "VALUES (:ticker, :date, :open, :high, :low, :close, :volume) "
            "ON CONFLICT(ticker, date) DO UPDATE SET "
            "open=excluded.open, high=excluded.high, low=excluded.low, "
            "close=excluded.close, volume=excluded.volume",
            [{**r, "ticker": ticker} for r in rows],
        )


def get_prices(ticker: str, limit_days: int | None = None) -> list[sqlite3.Row]:
    q = "SELECT * FROM prices WHERE ticker=? ORDER BY date ASC"
    rows = get_conn().execute(q, (ticker,)).fetchall()
    if limit_days:
        rows = rows[-limit_days:]
    return rows


def upsert_earnings(ticker: str, rows: list[dict]) -> None:
    if not rows:
        return
    with tx() as conn:
        conn.executemany(
            "INSERT INTO earnings(ticker, report_date, eps_estimate, eps_actual, "
            "surprise_pct, fiscal_period, is_future) VALUES "
            "(:ticker, :report_date, :eps_estimate, :eps_actual, :surprise_pct, "
            ":fiscal_period, :is_future) "
            "ON CONFLICT(ticker, report_date) DO UPDATE SET "
            "eps_estimate=excluded.eps_estimate, eps_actual=excluded.eps_actual, "
            "surprise_pct=excluded.surprise_pct, fiscal_period=excluded.fiscal_period, "
            "is_future=excluded.is_future",
            [{**r, "ticker": ticker} for r in rows],
        )


def get_earnings(ticker: str) -> list[sqlite3.Row]:
    return get_conn().execute(
        "SELECT * FROM earnings WHERE ticker=? ORDER BY report_date ASC", (ticker,)
    ).fetchall()


def upsert_filings(ticker: str, rows: list[dict]) -> None:
    if not rows:
        return
    with tx() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO filings(ticker, filed_date, form_type, title, url) "
            "VALUES (:ticker, :filed_date, :form_type, :title, :url)",
            [{**r, "ticker": ticker} for r in rows],
        )


def get_filings(ticker: str, limit: int = 20) -> list[sqlite3.Row]:
    return get_conn().execute(
        "SELECT * FROM filings WHERE ticker=? ORDER BY filed_date DESC LIMIT ?",
        (ticker, limit),
    ).fetchall()


def upsert_insider_transactions(ticker: str, rows: list[dict]) -> None:
    if not rows:
        return
    with tx() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO insider_transactions(ticker, transaction_date, "
            "owner_name, is_officer, is_director, is_ten_pct_owner, transaction_code, "
            "acquired_disposed, shares, price, value_usd) VALUES "
            "(:ticker, :transaction_date, :owner_name, :is_officer, :is_director, "
            ":is_ten_pct_owner, :transaction_code, :acquired_disposed, :shares, "
            ":price, :value_usd)",
            [
                {
                    **r,
                    "ticker": ticker,
                    "is_officer": int(bool(r.get("is_officer"))),
                    "is_director": int(bool(r.get("is_director"))),
                    "is_ten_pct_owner": int(bool(r.get("is_ten_pct_owner"))),
                }
                for r in rows
            ],
        )


def get_insider_transactions(ticker: str, limit: int = 50) -> list[sqlite3.Row]:
    return get_conn().execute(
        "SELECT * FROM insider_transactions WHERE ticker=? "
        "ORDER BY transaction_date DESC LIMIT ?",
        (ticker, limit),
    ).fetchall()


def upsert_news(ticker: str, rows: list[dict]) -> None:
    if not rows:
        return
    with tx() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO news(ticker, published, headline, source, url, "
            "sentiment, event_tags) VALUES "
            "(:ticker, :published, :headline, :source, :url, :sentiment, :event_tags)",
            [
                {
                    **r,
                    "ticker": ticker,
                    "event_tags": json.dumps(r.get("event_tags", [])),
                }
                for r in rows
            ],
        )


def get_news(ticker: str, limit: int = 30) -> list[sqlite3.Row]:
    return get_conn().execute(
        "SELECT * FROM news WHERE ticker=? ORDER BY published DESC LIMIT ?",
        (ticker, limit),
    ).fetchall()


def insert_forecast(row: dict) -> None:
    with tx() as conn:
        conn.execute(
            "INSERT INTO forecasts(ticker, as_of, horizon_days, direction, prob_up, "
            "expected_move_pct, low_pct, high_pct, confidence, rationale, drivers, "
            "llm_narrative, base_price, target_price) VALUES "
            "(:ticker, :as_of, :horizon_days, :direction, :prob_up, :expected_move_pct, "
            ":low_pct, :high_pct, :confidence, :rationale, :drivers, :llm_narrative, "
            ":base_price, :target_price) "
            "ON CONFLICT(ticker, as_of, horizon_days) DO UPDATE SET "
            "direction=excluded.direction, prob_up=excluded.prob_up, "
            "expected_move_pct=excluded.expected_move_pct, low_pct=excluded.low_pct, "
            "high_pct=excluded.high_pct, confidence=excluded.confidence, "
            "rationale=excluded.rationale, drivers=excluded.drivers, "
            "llm_narrative=excluded.llm_narrative, base_price=excluded.base_price, "
            "target_price=excluded.target_price",
            row,
        )


def insert_forecasts_batch(rows: list[dict]) -> None:
    """Same upsert as insert_forecast, batched into a single transaction.

    A full bootstrap regenerates every (ticker, horizon) forecast at once
    (448 companies x however many horizons) -- doing that as one INSERT-
    plus-commit per row meant paying a separate disk fsync per row for
    what is, in total, a couple of megabytes of data. One commit for the
    whole batch is the same data, written far faster."""
    if not rows:
        return
    with tx() as conn:
        conn.executemany(
            "INSERT INTO forecasts(ticker, as_of, horizon_days, direction, prob_up, "
            "expected_move_pct, low_pct, high_pct, confidence, rationale, drivers, "
            "llm_narrative, base_price, target_price) VALUES "
            "(:ticker, :as_of, :horizon_days, :direction, :prob_up, :expected_move_pct, "
            ":low_pct, :high_pct, :confidence, :rationale, :drivers, :llm_narrative, "
            ":base_price, :target_price) "
            "ON CONFLICT(ticker, as_of, horizon_days) DO UPDATE SET "
            "direction=excluded.direction, prob_up=excluded.prob_up, "
            "expected_move_pct=excluded.expected_move_pct, low_pct=excluded.low_pct, "
            "high_pct=excluded.high_pct, confidence=excluded.confidence, "
            "rationale=excluded.rationale, drivers=excluded.drivers, "
            "llm_narrative=excluded.llm_narrative, base_price=excluded.base_price, "
            "target_price=excluded.target_price",
            rows,
        )


def latest_forecast(ticker: str) -> sqlite3.Row | None:
    return get_conn().execute(
        "SELECT * FROM forecasts WHERE ticker=? ORDER BY as_of DESC LIMIT 1", (ticker,)
    ).fetchone()


def all_latest_forecasts() -> list[sqlite3.Row]:
    return get_conn().execute(
        """
        SELECT f.* FROM forecasts f
        INNER JOIN (
            SELECT ticker, MAX(as_of) AS max_as_of FROM forecasts GROUP BY ticker
        ) latest ON f.ticker = latest.ticker AND f.as_of = latest.max_as_of
        ORDER BY f.ticker ASC
        """
    ).fetchall()


def insert_outcome(row: dict) -> None:
    with tx() as conn:
        conn.execute(
            "INSERT INTO forecast_outcomes(ticker, as_of, horizon_days, "
            "predicted_direction, predicted_move_pct, actual_move_pct, "
            "direction_correct, abs_error_pct, evaluated_at) VALUES "
            "(:ticker, :as_of, :horizon_days, :predicted_direction, :predicted_move_pct, "
            ":actual_move_pct, :direction_correct, :abs_error_pct, :evaluated_at) "
            "ON CONFLICT(ticker, as_of, horizon_days) DO UPDATE SET "
            "actual_move_pct=excluded.actual_move_pct, "
            "direction_correct=excluded.direction_correct, "
            "abs_error_pct=excluded.abs_error_pct, evaluated_at=excluded.evaluated_at",
            row,
        )


def insert_outcomes_batch(rows: list[dict]) -> None:
    """Same upsert as insert_outcome, batched into a single transaction.

    run_backtest scores every held-out (ticker, date) row per horizon --
    tens of thousands of rows across the held-out slice x 3 horizons. One
    INSERT-plus-commit per row was, empirically, the dominant cost of a
    full bootstrap (each commit is a separate disk fsync; on a slow or
    throttled disk that's minutes of pure I/O wait). Batching the whole
    horizon's rows into one transaction writes the identical data with a
    single commit instead."""
    if not rows:
        return
    with tx() as conn:
        conn.executemany(
            "INSERT INTO forecast_outcomes(ticker, as_of, horizon_days, "
            "predicted_direction, predicted_move_pct, actual_move_pct, "
            "direction_correct, abs_error_pct, evaluated_at) VALUES "
            "(:ticker, :as_of, :horizon_days, :predicted_direction, :predicted_move_pct, "
            ":actual_move_pct, :direction_correct, :abs_error_pct, :evaluated_at) "
            "ON CONFLICT(ticker, as_of, horizon_days) DO UPDATE SET "
            "actual_move_pct=excluded.actual_move_pct, "
            "direction_correct=excluded.direction_correct, "
            "abs_error_pct=excluded.abs_error_pct, evaluated_at=excluded.evaluated_at",
            rows,
        )


def pending_outcomes(as_of_before: str) -> list[sqlite3.Row]:
    """Forecasts old enough that their horizon has elapsed but haven't been scored."""
    return get_conn().execute(
        """
        SELECT f.* FROM forecasts f
        LEFT JOIN forecast_outcomes o
          ON f.ticker=o.ticker AND f.as_of=o.as_of AND f.horizon_days=o.horizon_days
        WHERE o.ticker IS NULL AND f.as_of < ?
        """,
        (as_of_before,),
    ).fetchall()


def scorecard_summary() -> dict:
    row = get_conn().execute(
        """
        SELECT
            COUNT(*) AS n,
            AVG(direction_correct) AS hit_rate,
            AVG(abs_error_pct) AS mae,
            AVG(predicted_move_pct - actual_move_pct) AS bias
        FROM forecast_outcomes
        """
    ).fetchone()
    return dict(row) if row else {}


def scorecard_by_horizon() -> list[sqlite3.Row]:
    return get_conn().execute(
        """
        SELECT horizon_days, COUNT(*) AS n, AVG(direction_correct) AS hit_rate,
               AVG(abs_error_pct) AS mae
        FROM forecast_outcomes GROUP BY horizon_days ORDER BY horizon_days ASC
        """
    ).fetchall()


def scorecard_by_ticker() -> list[sqlite3.Row]:
    return get_conn().execute(
        """
        SELECT ticker, COUNT(*) AS n, AVG(direction_correct) AS hit_rate,
               AVG(abs_error_pct) AS mae
        FROM forecast_outcomes GROUP BY ticker ORDER BY n DESC
        """
    ).fetchall()


def get_ticker_scorecard(ticker: str) -> dict:
    """Single-ticker version of scorecard_by_ticker() -- this company's own
    walk-forward track record (n held-out calls, direction hit rate, mean
    absolute error), for showing next to its live forecast in the detail
    panel and the Compare tab rather than only in the aggregate scorecard.
    Returns {} for a ticker with no recorded outcomes yet (e.g. too little
    price history for a walk-forward split, or newly added this run)."""
    row = get_conn().execute(
        """
        SELECT ticker, COUNT(*) AS n, AVG(direction_correct) AS hit_rate,
               AVG(abs_error_pct) AS mae
        FROM forecast_outcomes WHERE ticker=? GROUP BY ticker
        """,
        (ticker,),
    ).fetchone()
    return dict(row) if row else {}


# ---------- accounts ----------


def create_user(email: str, password_hash: str, salt: str, display_name: str) -> int:
    with tx() as conn:
        cur = conn.execute(
            "INSERT INTO users(email, password_hash, salt, display_name, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (email, password_hash, salt, display_name, now_iso()),
        )
        return cur.lastrowid


def get_user(user_id: int) -> sqlite3.Row | None:
    return get_conn().execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()


def get_user_by_email(email: str) -> sqlite3.Row | None:
    return get_conn().execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()


def create_session(token: str, user_id: int, ttl_seconds: int) -> None:
    with tx() as conn:
        conn.execute(
            "INSERT INTO sessions(token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, now_iso(), _iso_plus_seconds(ttl_seconds)),
        )


def get_session_user(token: str) -> sqlite3.Row | None:
    row = get_conn().execute(
        """
        SELECT u.* FROM sessions s
        INNER JOIN users u ON u.id = s.user_id
        WHERE s.token=? AND s.expires_at > ?
        """,
        (token, now_iso()),
    ).fetchone()
    return row


def delete_session(token: str) -> None:
    with tx() as conn:
        conn.execute("DELETE FROM sessions WHERE token=?", (token,))


def _iso_plus_seconds(seconds: int) -> str:
    from datetime import timedelta

    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat(timespec="seconds")


# ---------- watchlist ----------


def add_watchlist(user_id: int, ticker: str) -> None:
    with tx() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO watchlist(user_id, ticker, added_at) VALUES (?, ?, ?)",
            (user_id, ticker, now_iso()),
        )


def remove_watchlist(user_id: int, ticker: str) -> None:
    with tx() as conn:
        conn.execute("DELETE FROM watchlist WHERE user_id=? AND ticker=?", (user_id, ticker))


def get_watchlist(user_id: int) -> list[str]:
    rows = get_conn().execute(
        "SELECT ticker FROM watchlist WHERE user_id=? ORDER BY added_at DESC", (user_id,)
    ).fetchall()
    return [r["ticker"] for r in rows]


# ---------- investor preferences (onboarding survey) ----------


def save_preferences(user_id: int, data: dict) -> None:
    with tx() as conn:
        conn.execute(
            "INSERT INTO preferences(user_id, goals, risk_tolerance, horizon, sectors, "
            "experience, updated_at) VALUES (:user_id, :goals, :risk_tolerance, :horizon, "
            ":sectors, :experience, :updated_at) "
            "ON CONFLICT(user_id) DO UPDATE SET "
            "goals=excluded.goals, risk_tolerance=excluded.risk_tolerance, "
            "horizon=excluded.horizon, sectors=excluded.sectors, "
            "experience=excluded.experience, updated_at=excluded.updated_at",
            {
                "user_id": user_id,
                "goals": json.dumps(data.get("goals") or []),
                "risk_tolerance": data.get("risk_tolerance"),
                "horizon": data.get("horizon"),
                "sectors": json.dumps(data.get("sectors") or []),
                "experience": data.get("experience"),
                "updated_at": now_iso(),
            },
        )


def get_preferences(user_id: int) -> dict | None:
    row = get_conn().execute(
        "SELECT * FROM preferences WHERE user_id=?", (user_id,)
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["goals"] = json.loads(d["goals"] or "[]")
    d["sectors"] = json.loads(d["sectors"] or "[]")
    return d


# ---------- fundamentals (richer per-stock data) ----------


def upsert_fundamentals(ticker: str, data: dict) -> None:
    with tx() as conn:
        conn.execute(
            "INSERT INTO fundamentals(ticker, description, market_cap, pe_ratio, "
            "forward_pe, peg_ratio, dividend_yield, beta, profit_margin, revenue_growth, "
            "analyst_target_mean, analyst_target_high, analyst_target_low, "
            "analyst_recommendation, num_analyst_opinions, updated_at) VALUES "
            "(:ticker, :description, :market_cap, :pe_ratio, :forward_pe, :peg_ratio, "
            ":dividend_yield, :beta, :profit_margin, :revenue_growth, :analyst_target_mean, "
            ":analyst_target_high, :analyst_target_low, :analyst_recommendation, "
            ":num_analyst_opinions, :updated_at) "
            "ON CONFLICT(ticker) DO UPDATE SET "
            "description=excluded.description, market_cap=excluded.market_cap, "
            "pe_ratio=excluded.pe_ratio, forward_pe=excluded.forward_pe, "
            "peg_ratio=excluded.peg_ratio, dividend_yield=excluded.dividend_yield, "
            "beta=excluded.beta, profit_margin=excluded.profit_margin, "
            "revenue_growth=excluded.revenue_growth, "
            "analyst_target_mean=excluded.analyst_target_mean, "
            "analyst_target_high=excluded.analyst_target_high, "
            "analyst_target_low=excluded.analyst_target_low, "
            "analyst_recommendation=excluded.analyst_recommendation, "
            "num_analyst_opinions=excluded.num_analyst_opinions, "
            "updated_at=excluded.updated_at",
            {
                "ticker": ticker,
                "description": data.get("description"),
                "market_cap": data.get("market_cap"),
                "pe_ratio": data.get("pe_ratio"),
                "forward_pe": data.get("forward_pe"),
                "peg_ratio": data.get("peg_ratio"),
                "dividend_yield": data.get("dividend_yield"),
                "beta": data.get("beta"),
                "profit_margin": data.get("profit_margin"),
                "revenue_growth": data.get("revenue_growth"),
                "analyst_target_mean": data.get("analyst_target_mean"),
                "analyst_target_high": data.get("analyst_target_high"),
                "analyst_target_low": data.get("analyst_target_low"),
                "analyst_recommendation": data.get("analyst_recommendation"),
                "num_analyst_opinions": data.get("num_analyst_opinions"),
                "updated_at": now_iso(),
            },
        )


def get_fundamentals(ticker: str) -> dict | None:
    row = get_conn().execute(
        "SELECT * FROM fundamentals WHERE ticker=?", (ticker,)
    ).fetchone()
    return dict(row) if row else None


def all_fundamentals() -> dict[str, dict]:
    rows = get_conn().execute("SELECT * FROM fundamentals").fetchall()
    return {r["ticker"]: dict(r) for r in rows}


# ---------- data provenance (live vs. synthetic, per field) ----------

PROVENANCE_FIELDS = ("bars", "earnings", "fundamentals", "news", "filings")


def get_all_provenance() -> dict[str, dict]:
    """One query for the whole universe -- callers that need to decide
    per-ticker staleness for hundreds of companies should call this once,
    not per-ticker (that's the N+1 this exists to avoid)."""
    rows = get_conn().execute("SELECT * FROM data_provenance").fetchall()
    return {r["ticker"]: dict(r) for r in rows}


def get_provenance(ticker: str) -> dict:
    """Single-ticker lookup for the per-company detail view (see
    api.py's forecast_detail) -- returns {} rather than raising for a
    ticker that hasn't been ingested yet (e.g. a brand-new company added
    to the universe before the next ingestion run)."""
    row = get_conn().execute(
        "SELECT * FROM data_provenance WHERE ticker=?", (ticker,)
    ).fetchone()
    return dict(row) if row else {}


def provenance_summary() -> dict[str, dict[str, int]]:
    """Per-field counts of how many tickers are currently backed by each
    source ({"bars": {"yahoo": 340, "demo": 3}, ...}) -- powers the Data
    Health panel's source-breakdown table without shipping the whole
    per-ticker table to the browser. A ticker with no data_provenance row
    at all (never ingested) isn't counted in any bucket."""
    rows = get_conn().execute("SELECT * FROM data_provenance").fetchall()
    summary: dict[str, dict[str, int]] = {f: {} for f in PROVENANCE_FIELDS}
    for r in rows:
        for f in PROVENANCE_FIELDS:
            src = r[f"{f}_source"]
            if src:
                summary[f][src] = summary[f].get(src, 0) + 1
    return summary


def update_provenance(ticker: str, sources: dict[str, str]) -> None:
    """`sources`: {field: source_name} for whichever of PROVENANCE_FIELDS
    were actually (re)fetched this run -- fields not present are left
    untouched, so a run that only refreshed news doesn't clobber an
    earlier bars_source/bars_updated_at."""
    if not sources:
        return
    ts = now_iso()
    set_clauses, params = [], {"ticker": ticker}
    for field, source in sources.items():
        if field not in PROVENANCE_FIELDS:
            continue
        set_clauses.append(f"{field}_source=:{field}_source")
        set_clauses.append(f"{field}_updated_at=:{field}_updated_at")
        params[f"{field}_source"] = source
        params[f"{field}_updated_at"] = ts
    if not set_clauses:
        return
    insert_cols = ["ticker"] + [f"{f}_source" for f in sources if f in PROVENANCE_FIELDS] + \
        [f"{f}_updated_at" for f in sources if f in PROVENANCE_FIELDS]
    insert_vals = [":ticker"] + [f":{f}_source" for f in sources if f in PROVENANCE_FIELDS] + \
        [f":{f}_updated_at" for f in sources if f in PROVENANCE_FIELDS]
    with tx() as conn:
        conn.execute(
            f"INSERT INTO data_provenance({', '.join(insert_cols)}) VALUES ({', '.join(insert_vals)}) "
            f"ON CONFLICT(ticker) DO UPDATE SET {', '.join(set_clauses)}",
            params,
        )


# ---------- ingestion run history ----------


def start_ingestion_run(mode: str) -> int:
    with tx() as conn:
        cur = conn.execute(
            "INSERT INTO ingestion_runs(started_at, mode) VALUES (?, ?)", (now_iso(), mode)
        )
        return cur.lastrowid


def finish_ingestion_run(run_id: int, live_ok: int, live_fail: int, skipped_fresh: int,
                          elapsed_sec: float, error: str | None = None) -> None:
    with tx() as conn:
        conn.execute(
            "UPDATE ingestion_runs SET finished_at=?, live_ok=?, live_fail=?, "
            "skipped_fresh=?, elapsed_sec=?, error=? WHERE id=?",
            (now_iso(), live_ok, live_fail, skipped_fresh, elapsed_sec, error, run_id),
        )


# ---------- paper trading log (app.trading) ----------


def insert_paper_trades_batch(rows: list[dict]) -> None:
    """One pass logs several rows at once (a close per prior position, a
    skip/open per candidate) -- batched into a single transaction for the
    same reason insert_forecasts_batch/insert_outcomes_batch are (see
    those for the full story on why per-row commits were a real cost)."""
    if not rows:
        return
    with tx() as conn:
        conn.executemany(
            "INSERT INTO paper_trades(as_of, ticker, action, side, notional_usd, "
            "forecast_confidence, forecast_direction, reason, alpaca_order_id, status) "
            "VALUES (:as_of, :ticker, :action, :side, :notional_usd, :forecast_confidence, "
            ":forecast_direction, :reason, :alpaca_order_id, :status)",
            rows,
        )


def recent_paper_trades(limit: int = 50) -> list[sqlite3.Row]:
    return get_conn().execute(
        "SELECT * FROM paper_trades ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()


def recent_ingestion_runs(limit: int = 20) -> list[sqlite3.Row]:
    return get_conn().execute(
        "SELECT * FROM ingestion_runs ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
