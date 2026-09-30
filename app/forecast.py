"""The forecasting "agent".

Trains a small, transparent machine-learning model per horizon on the
whole universe's history (pooled across all ~100 tickers, which is what
makes a ~400-day history workable for scikit-learn at all):

  - LogisticRegression -> P(price is higher in `h` trading days)
  - Ridge regression    -> expected % move over the same horizon

Both run on standardized versions of the feature panel from app.signals.
Logistic regression coefficients are linear and additive after scaling,
so at inference time we decompose each prediction into per-feature
contributions -- that decomposition *is* the rationale shown in the UI,
not a separate explanation bolted on after the fact.

An optional second pass asks Claude (if ANTHROPIC_API_KEY is set) to turn
the structured driver list into a paragraph of plain English. That call
is a plain HTTPS POST to api.anthropic.com so it has no SDK dependency;
if it's not configured or the request fails, the numeric/structured
forecast still stands on its own.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app import db
from app.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, DATA_MODE
from app.ml import LogisticRegression, Pipeline, RidgeRegression, StandardScaler
from app.signals import FeaturePanel, build_feature_panel, split_boundary_dates
from app.universe import COMPANY_BY_TICKER, adjacency

log = logging.getLogger("stockgraph.forecast")

HORIZONS = [1, 5, 20]

# Demo mode's data is synthetic and rebuilt from scratch on every cold boot
# (the public host has no persistent disk -- see DEPLOY.md), so there's no
# real backtest track record riding on training to full convergence the
# way there is for auto/live mode. Fewer gradient-descent steps reaches
# essentially the same decision boundary on this data in a fraction of the
# time, all 448 companies and all 3 horizons still included. auto/live
# mode (Danny's own laptop) is untouched -- this only shortens the
# iteration count, nothing else about what gets trained.
LOGISTIC_ITERATIONS = 150 if DATA_MODE == "demo" else 500

FACTOR_LABELS = {
    "mom_5": "5-day price momentum",
    "mom_10": "10-day price momentum",
    "mom_20": "20-day price momentum",
    "mom_60": "60-day price momentum",
    "vol_20": "recent volatility",
    "rsi_14": "RSI (overbought/oversold)",
    "ma_gap_20": "distance from 20-day average",
    "ma_gap_50": "distance from 50-day average",
    "vol_zscore": "unusual trading volume",
    "sentiment_7d": "news sentiment (7d, decayed)",
    "news_event_count_7d": "news event frequency (7d)",
    "days_to_earnings": "days until next earnings",
    "days_since_earnings": "days since last earnings",
    "last_earnings_surprise_pct": "last earnings surprise",
    "earnings_imminent": "earnings report imminent (<=5d)",
    "post_earnings_drift_window": "post-earnings drift window",
    "sector_mom_5": "sector peer momentum",
    "market_mom_5": "broad market momentum",
    "graph_mom_spillover": "connected-company momentum spillover",
    "graph_sent_spillover": "connected-company news sentiment spillover",
    "macd_hist": "MACD histogram (trend momentum)",
    "bollinger_pct_b": "position within Bollinger Bands",
    "bollinger_bandwidth": "Bollinger Band width (volatility squeeze/expansion)",
    "atr_pct": "average true range (volatility, % of price)",
    "days_since_filing": "days since last SEC filing",
    "filing_count_30d": "SEC filing activity (30d)",
    "insider_net_buy_ratio_90d": "insider open-market buy/sell balance (90d)",
    "insider_buy_count_90d": "insider open-market buys (90d)",
}


@dataclass
class HorizonModel:
    direction: Pipeline
    magnitude: Pipeline
    resid_std: float
    holdout_accuracy: float
    holdout_mae: float
    n_train: int


@dataclass
class ForecastModels:
    feature_names: list[str]
    by_horizon: dict[int, HorizonModel] = field(default_factory=dict)


def train_models(panel: FeaturePanel, horizons: list[int] = HORIZONS) -> ForecastModels:
    feature_cols = list(panel.features.keys())
    long_df = panel.as_long_frame(horizons).sort_values("date")
    models = ForecastModels(feature_names=feature_cols)

    for h in horizons:
        df = long_df.dropna(subset=[f"fwd_ret_{h}"]).copy()
        if len(df) < 300:
            log.warning("not enough rows to train horizon=%sd (%d rows)", h, len(df))
            continue

        X = df[feature_cols].to_numpy(dtype=float)
        y_dir = df[f"fwd_dir_{h}"].to_numpy(dtype=float)
        y_ret = df[f"fwd_ret_{h}"].to_numpy(dtype=float)

        # Chronological split with a purge gap of `h` trading days between
        # train and test: a training row's label already looks `h` days
        # into the future, so without the gap, rows just before the
        # boundary would have labels computed from prices inside the
        # nominally held-out test window. See signals.split_boundary_dates.
        train_end_date, test_start_date = split_boundary_dates(df["date"], h)
        train_mask = (df["date"] <= train_end_date).to_numpy()
        test_mask = (df["date"] >= test_start_date).to_numpy()

        X_train, X_test = X[train_mask], X[test_mask]
        ydir_train, ydir_test = y_dir[train_mask], y_dir[test_mask]
        yret_train, yret_test = y_ret[train_mask], y_ret[test_mask]

        dir_pipe = Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(alpha=2.0, lr=0.5, iterations=LOGISTIC_ITERATIONS))]
        )
        dir_pipe.fit(X_train, ydir_train)

        mag_pipe = Pipeline([("scaler", StandardScaler()), ("reg", RidgeRegression(alpha=8.0))])
        mag_pipe.fit(X_train, yret_train)

        if len(X_test) > 20:
            acc = float((dir_pipe.predict(X_test) == ydir_test).mean())
            resid = yret_test - mag_pipe.predict(X_test)
            mae = float(np.mean(np.abs(resid)))
            resid_std = float(np.std(resid)) if len(resid) > 5 else float(np.std(yret_train))
        else:
            acc, mae = float("nan"), float("nan")
            resid_std = float(np.std(yret_train))

        models.by_horizon[h] = HorizonModel(
            direction=dir_pipe,
            magnitude=mag_pipe,
            resid_std=resid_std,
            holdout_accuracy=acc,
            holdout_mae=mae,
            n_train=len(X_train),
        )
        log.info(
            "trained horizon=%sd n_train=%d holdout_acc=%.3f holdout_mae=%.4f",
            h, len(X_train), acc, mae,
        )

    return models


def _logit_contributions(dir_pipe: Pipeline, feature_names: list[str], raw_row: np.ndarray) -> dict[str, float]:
    scaler: StandardScaler = dir_pipe.named_steps["scaler"]
    clf: LogisticRegression = dir_pipe.named_steps["clf"]
    scaled = (raw_row - scaler.mean_) / scaler.scale_
    contributions = scaled * clf.coef_[0]
    return dict(zip(feature_names, contributions))


def _top_graph_driver(ticker: str, panel: FeaturePanel) -> dict | None:
    adj = adjacency()
    neighbors = adj.get(ticker, [])
    if not neighbors:
        return None
    latest_date = panel.dates[-1]
    mom5 = panel.features["mom_5"]
    total_w = sum(e.weight for _, e in neighbors) or 1.0
    best = None
    for other, edge in neighbors:
        sign = -1.0 if edge.kind == "competitor" else 1.0
        neighbor_mom = float(mom5.loc[latest_date, other]) if other in mom5.columns else 0.0
        contribution = sign * edge.weight * neighbor_mom / total_w
        if best is None or abs(contribution) > abs(best["contribution"]):
            best = {
                "ticker": other,
                "name": COMPANY_BY_TICKER[other].name,
                "kind": edge.kind,
                "note": edge.note,
                "contribution": contribution,
                "neighbor_mom_5": neighbor_mom,
            }
    return best


def _build_rationale(ticker: str, direction: str, drivers: list[dict], top_graph: dict | None) -> str:
    name = COMPANY_BY_TICKER[ticker].name
    parts = [f"Model leans {direction.upper()} on {name}."]
    top3 = drivers[:3]
    dir_sign = 1.0 if direction == "up" else -1.0
    for d in top3:
        effective = d["contribution"] * dir_sign
        sign_word = "supporting" if effective > 0 else "weighing against"
        parts.append(f"{d['label']} is {sign_word} that call (value {d['value']:.3f}).")
    if top_graph and abs(top_graph["contribution"]) > 0.001:
        rel = "boosting" if top_graph["contribution"] * dir_sign > 0 else "dragging on"
        parts.append(
            f"Its {top_graph['kind']} relationship with {top_graph['name']} "
            f"({top_graph['note']}) is currently {rel} the outlook."
        )
    return " ".join(parts)


def _call_claude_narrative(ticker: str, direction: str, prob_up: float, expected_move_pct: float,
                            horizon_days: int, drivers: list[dict], top_graph: dict | None) -> str | None:
    if not ANTHROPIC_API_KEY:
        return None
    import json as _json
    import urllib.request as _urlreq
    import urllib.error as _urlerr

    name = COMPANY_BY_TICKER[ticker].name
    driver_lines = "\n".join(
        f"- {d['label']}: value={d['value']:.3f}, contribution={d['contribution']:+.3f}"
        for d in drivers[:6]
    )
    graph_line = ""
    if top_graph:
        graph_line = (
            f"\nStrongest graph relationship effect: {top_graph['kind']} link to "
            f"{top_graph['name']} ({top_graph['note']}), contribution={top_graph['contribution']:+.3f}"
        )
    prompt = (
        f"You are a markets analyst writing a 2-3 sentence note for a retail investor "
        f"dashboard. Ticker: {ticker} ({name}). Model output: direction={direction}, "
        f"probability_up={prob_up:.2f}, expected_move_pct={expected_move_pct*100:.2f}%, "
        f"horizon={horizon_days} trading days.\nTop model drivers (standardized logistic "
        f"regression contributions, positive=bullish):\n{driver_lines}{graph_line}\n\n"
        f"Write a short, concrete, non-hedgy explanation a person could actually use. "
        f"Do not give investment advice or tell them to buy/sell. Do not repeat numbers "
        f"that were already given verbatim; interpret them."
    )
    body = _json.dumps(
        {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    req = _urlreq.Request(
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
        with _urlreq.urlopen(req, timeout=20) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        blocks = data.get("content", [])
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        return text.strip() or None
    except (_urlerr.URLError, _urlerr.HTTPError, TimeoutError, ValueError) as e:
        log.warning("claude narrative call failed for %s: %s", ticker, e)
        return None


def generate_forecast(ticker: str, panel: FeaturePanel, models: ForecastModels, horizon: int,
                       use_llm: bool = True) -> dict:
    hm = models.by_horizon.get(horizon)
    if hm is None:
        raise ValueError(f"no trained model for horizon={horizon}")

    latest_date = panel.dates[-1]
    raw_row = np.array(
        [float(panel.features[f].loc[latest_date, ticker]) for f in models.feature_names]
    )
    prob_up = float(hm.direction.predict_proba(raw_row.reshape(1, -1))[0, 1])
    direction = "up" if prob_up >= 0.5 else "down"

    # The direction classifier and the magnitude regressor are two
    # independently-fit models and can disagree on sign (e.g. classifier
    # says 51% up, regressor's raw point-estimate is slightly negative).
    # The classifier is what "direction" means in this product, so it's
    # authoritative for sign; the regressor only sizes the magnitude.
    # Showing a "DOWN" call next to a "+0.7%" move would just be confusing.
    raw_magnitude = float(hm.magnitude.predict(raw_row.reshape(1, -1))[0])
    sign = 1.0 if direction == "up" else -1.0
    expected_move = sign * abs(raw_magnitude)

    contributions = _logit_contributions(hm.direction, models.feature_names, raw_row)
    drivers = sorted(
        (
            {
                "factor": f,
                "label": FACTOR_LABELS.get(f, f),
                "value": float(raw_row[i]),
                "contribution": float(contributions[f]),
            }
            for i, f in enumerate(models.feature_names)
        ),
        key=lambda d: abs(d["contribution"]),
        reverse=True,
    )

    top_graph = _top_graph_driver(ticker, panel)
    rationale = _build_rationale(ticker, direction, drivers, top_graph)

    z = 1.28  # ~80% band
    low = expected_move - z * hm.resid_std
    high = expected_move + z * hm.resid_std

    base_price = float(panel.close.loc[latest_date, ticker])
    target_price = base_price * (1 + expected_move)

    confidence = float(min(1.0, abs(prob_up - 0.5) * 2 * (0.5 + 0.5 * (hm.holdout_accuracy or 0.5))))

    # 999 is signals.py's "no known earnings date nearby" sentinel -- surface
    # that as null rather than a nonsense "999 days away".
    days_to_earnings_raw = float(panel.features["days_to_earnings"].loc[latest_date, ticker])
    days_to_earnings = int(days_to_earnings_raw) if days_to_earnings_raw < 999 else None

    llm_narrative = None
    if use_llm:
        llm_narrative = _call_claude_narrative(
            ticker, direction, prob_up, expected_move, horizon, drivers, top_graph
        )

    return {
        "ticker": ticker,
        "as_of": db.now_iso(),
        "horizon_days": horizon,
        "direction": direction,
        "prob_up": round(prob_up, 4),
        "expected_move_pct": round(expected_move, 5),
        "low_pct": round(low, 5),
        "high_pct": round(high, 5),
        "confidence": round(confidence, 4),
        "rationale": rationale,
        "drivers": json.dumps(drivers[:8]),
        "llm_narrative": llm_narrative,
        "base_price": round(base_price, 2),
        "target_price": round(target_price, 2),
        "days_to_earnings": days_to_earnings,
        "_model_holdout_accuracy": hm.holdout_accuracy,
        "_model_holdout_mae": hm.holdout_mae,
    }


def generate_all_forecasts(panel: FeaturePanel, models: ForecastModels, horizon: int,
                            use_llm: bool = False) -> list[dict]:
    out = []
    for t in panel.close.columns:
        if horizon not in models.by_horizon:
            continue
        try:
            out.append(generate_forecast(t, panel, models, horizon, use_llm=use_llm))
        except Exception as e:
            log.warning("forecast failed for %s: %s", t, e)
    return out
