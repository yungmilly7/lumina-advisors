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

        rows = []
        for i, (_, r) in enumerate(test_df.iterrows()):
            predicted_direction = "up" if pred_prob[i] >= 0.5 else "down"
            direction_correct = int((pred_prob[i] >= 0.5) == bool(actual_dir[i]))
            rows.append(
                {
                    "ticker": r["ticker"],
                    "as_of": str(r["date"].date()) if hasattr(r["date"], "date") else str(r["date"]),
                    "horizon_days": int(h),
                    "predicted_direction": predicted_direction,
                    "predicted_move_pct": float(pred_move[i]),
                    "actual_move_pct": float(actual_move[i]),
                    "direction_correct": direction_correct,
                    "abs_error_pct": float(abs(pred_move[i] - actual_move[i])),
                    "evaluated_at": evaluated_at,
                }
            )
        for row in rows:
            db.insert_outcome(row)

        hit_rate = float(np.mean([r["direction_correct"] for r in rows]))
        mae = float(np.mean([r["abs_error_pct"] for r in rows]))
        summary[h] = {"n": len(rows), "hit_rate": hit_rate, "mae": mae}
        log.info("backtest horizon=%sd n=%d hit_rate=%.3f mae=%.4f", h, len(rows), hit_rate, mae)

    return summary


def scorecard() -> dict:
    return {
        "overall": db.scorecard_summary(),
        "by_horizon": [dict(r) for r in db.scorecard_by_horizon()],
        "by_ticker": [dict(r) for r in db.scorecard_by_ticker()],
    }
