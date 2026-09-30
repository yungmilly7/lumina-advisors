"""Walk-forward scorecard: how good is the model actually?

Reuses the exact same chronological train/test split `forecast.train_models`
used (roughly the first 85% of history to fit, the last 15% held out, with
a purge gap of `horizon` trading days at the boundary -- see
signals.split_boundary_dates) so there's no leakage -- the held-out slice
was never seen during fitting, and no training label reaches into it. For
every (ticker, date) row in that held-out slice we already know the real
outcome (it's history), so we can score the frozen model against it
immediately instead of waiting for real time to pass, and store the
results the same way we would for live forecasts that mature later.
"""
from __future__ import annotations

import logging

import numpy as np

from app import db
from app.forecast import ForecastModels
from app.signals import FeaturePanel, split_boundary_dates

log = logging.getLogger("stockgraph.scoring")


def run_backtest(panel: FeaturePanel, models: ForecastModels) -> dict:
    feature_cols = models.feature_names
    long_df = panel.as_long_frame(list(models.by_horizon.keys())).sort_values("date")
    summary = {}
    evaluated_at = db.now_iso()

    for h, hm in models.by_horizon.items():
        df = long_df.dropna(subset=[f"fwd_ret_{h}"]).copy()
        _, test_start_date = split_boundary_dates(df["date"], h)
        test_df = df[df["date"] >= test_start_date]
        if test_df.empty:
            continue

        X_test = test_df[feature_cols].to_numpy(dtype=float)
        pred_prob = hm.direction.predict_proba(X_test)[:, 1]
        pred_move = hm.magnitude.predict(X_test)
        actual_move = test_df[f"fwd_ret_{h}"].to_numpy(dtype=float)
        actual_dir = test_df[f"fwd_dir_{h}"].to_numpy(dtype=float)

        # Vectorized rather than a per-row iterrows() loop: with tens of
        # thousands of held-out rows per horizon, building each row's dict
        # via pandas' row-by-row iteration was itself a meaningful chunk of
        # bootstrap time, on top of the per-row insert cost that
        # insert_outcomes_batch below now avoids.
        pred_dir_up = pred_prob >= 0.5
        predicted_direction = np.where(pred_dir_up, "up", "down")
        direction_correct = (pred_dir_up == actual_dir.astype(bool)).astype(int)
        abs_error = np.abs(pred_move - actual_move)
        tickers = test_df["ticker"].to_numpy()
        as_of_dates = test_df["date"].dt.strftime("%Y-%m-%d").to_numpy()

        rows = [
            {
                "ticker": tickers[i],
                "as_of": as_of_dates[i],
                "horizon_days": int(h),
                "predicted_direction": predicted_direction[i],
                "predicted_move_pct": float(pred_move[i]),
                "actual_move_pct": float(actual_move[i]),
                "direction_correct": int(direction_correct[i]),
                "abs_error_pct": float(abs_error[i]),
                "evaluated_at": evaluated_at,
            }
            for i in range(len(test_df))
        ]
        db.insert_outcomes_batch(rows)

        hit_rate = float(direction_correct.mean())
        mae = float(abs_error.mean())
        summary[h] = {"n": len(rows), "hit_rate": hit_rate, "mae": mae}
        log.info("backtest horizon=%sd n=%d hit_rate=%.3f mae=%.4f", h, len(rows), hit_rate, mae)

    return summary


def scorecard() -> dict:
    return {
        "overall": db.scorecard_summary(),
        "by_horizon": [dict(r) for r in db.scorecard_by_horizon()],
        "by_ticker": [dict(r) for r in db.scorecard_by_ticker()],
    }
