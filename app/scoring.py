"""Scorecard: how good is the model actually?

Two distinct validations live here, and it matters which one a given number
comes from:

- `run_backtest` scores the actual production model (the one forecasts are
  served from) against the single chronological train/test split
  `forecast.train_models` used to fit it (roughly the first 85% of history,
  the last 15% held out, with a purge gap of `horizon` trading days at the
  boundary -- see signals.split_boundary_dates), so there's no leakage: the
  held-out slice was never seen during fitting, and no training label
  reaches into it. For every (ticker, date) row in that held-out slice we
  already know the real outcome (it's history), so we can score the frozen
  model against it immediately instead of waiting for real time to pass,
  and store the results the same way we would for live forecasts that
  mature later.

- `run_walk_forward_backtest` answers a different question: was that one
  split's number representative, or did the model just get a lucky (or
  unlucky) draw of holdout dates? It re-fits several *disposable* models,
  each on its own expanding window, and walks forward through a sequence of
  test windows the way `run_backtest`'s single split never does. See
  signals.walk_forward_splits for the fold construction. Its results never
  touch the production model -- they exist purely to validate the method.
"""
from __future__ import annotations

import gc
import logging

import numpy as np

from app import db
from app.config import DATA_MODE
from app.forecast import ForecastModels, fit_direction_and_magnitude, MIN_TRAIN_ROWS
from app.signals import FeaturePanel, split_boundary_dates, walk_forward_splits

log = logging.getLogger("stockgraph.scoring")

MIN_FOLD_TEST_ROWS = 20
# Fewer folds in demo mode -- same reasoning as forecast.LOGISTIC_ITERATIONS:
# demo data is synthetic and rebuilt from scratch every cold boot, and this
# background pass fits n_folds extra disposable models *per horizon* on top
# of everything else a bootstrap already does. Render's free-tier deploy
# (always demo mode -- see render.yaml) runs on a 512MB cap; trimming this
# is a real, deliberate safety margin, not just a speed tweak. auto/live
# mode (Danny's own laptop) keeps the full 5 folds.
DEFAULT_WALK_FORWARD_FOLDS = 3 if DATA_MODE == "demo" else 5
DEFAULT_WALK_FORWARD_MIN_TRAIN_FRAC = 0.5


def run_backtest(panel: FeaturePanel, models: ForecastModels) -> dict:
    feature_cols = models.feature_names
    long_df = panel.as_long_frame(list(models.by_horizon.keys())).sort_values("date")
    summary = {}
    evaluated_at = db.now_iso()

    for h, hm in models.by_horizon.items():
        cols = feature_cols + ["ticker", "date", f"fwd_ret_{h}", f"fwd_dir_{h}"]
        df = long_df.loc[long_df[f"fwd_ret_{h}"].notna(), cols].copy()
        _, test_start_date = split_boundary_dates(df["date"], h)
        test_df = df[df["date"] >= test_start_date]
        if test_df.empty:
            del df, test_df
            continue

        # float32: same reasoning as forecast.train_models -- this is the
        # single largest recurring allocation in this loop, x3 horizons.
        X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
        pred_prob = hm.direction.predict_proba(X_test)[:, 1]
        pred_move = hm.magnitude.predict(X_test)
        actual_move = test_df[f"fwd_ret_{h}"].to_numpy(dtype=np.float32)
        actual_dir = test_df[f"fwd_dir_{h}"].to_numpy(dtype=np.float32)

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

        # See forecast.train_models's matching comment: pandas DataFrames'
        # internal BlockManager commonly creates reference cycles that
        # outlive simple refcounting, and on a memory-capped host (Render's
        # free tier) that lag across horizons was enough to tip a bootstrap
        # into an OOM kill.
        del df, test_df, X_test, pred_prob, pred_move, actual_move, actual_dir, rows
        gc.collect()

    del long_df
    gc.collect()
    return summary


def run_walk_forward_backtest(
    panel: FeaturePanel,
    horizons: list[int],
    n_folds: int = DEFAULT_WALK_FORWARD_FOLDS,
    min_train_frac: float = DEFAULT_WALK_FORWARD_MIN_TRAIN_FRAC,
) -> dict:
    """Genuine walk-forward validation: for each horizon, repeatedly fits a
    fresh model on an expanding training window and scores it on the very
    next slice of dates it never saw, walking forward through several such
    folds -- see the module docstring for how this differs from
    `run_backtest` above, and signals.walk_forward_splits for how the folds
    themselves are carved out.

    Every fold's models are fit here and discarded; they're never stored,
    never used for live forecasts, and don't touch `ForecastModels` at all.
    This exists purely to validate that the production model's single-split
    holdout number is representative rather than a fluke, so it's naturally
    much more expensive than `run_backtest` (n_folds separate model fits
    per horizon instead of one) -- fine for a background bootstrap, not
    something to run per-request.
    """
    feature_cols = list(panel.features.keys())
    long_df = panel.as_long_frame(horizons).sort_values("date")
    evaluated_at = db.now_iso()
    fold_rows = []

    for h in horizons:
        cols = feature_cols + ["date", f"fwd_ret_{h}", f"fwd_dir_{h}"]
        df = long_df.loc[long_df[f"fwd_ret_{h}"].notna(), cols].copy()
        if len(df) < MIN_TRAIN_ROWS:
            del df
            continue

        dates = df["date"]
        # float32: this is the most memory-hungry loop in the whole
        # bootstrap -- up to n_folds extra full-size copies of X per
        # horizon (each fold's expanding train_mask/test_mask slice is a
        # fresh numpy copy, not a view), on top of everything
        # forecast.train_models/run_backtest above already allocate in the
        # same process. See their matching comments for why this matters
        # on Render's 512MB free tier.
        X = df[feature_cols].to_numpy(dtype=np.float32)
        y_dir = df[f"fwd_dir_{h}"].to_numpy(dtype=np.float32)
        y_ret = df[f"fwd_ret_{h}"].to_numpy(dtype=np.float32)

        for fold_idx, train_end_date, test_start_date, test_end_date in walk_forward_splits(
            dates, h, n_folds=n_folds, min_train_frac=min_train_frac
        ):
            train_mask = (dates <= train_end_date).to_numpy()
            test_mask = ((dates >= test_start_date) & (dates <= test_end_date)).to_numpy()
            n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
            if n_train < MIN_TRAIN_ROWS or n_test < MIN_FOLD_TEST_ROWS:
                continue

            dir_pipe, mag_pipe = fit_direction_and_magnitude(
                X[train_mask], y_dir[train_mask], y_ret[train_mask]
            )

            X_test = X[test_mask]
            direction_correct = dir_pipe.predict(X_test) == y_dir[test_mask]
            hit_rate = float(direction_correct.mean())
            mae = float(np.mean(np.abs(mag_pipe.predict(X_test) - y_ret[test_mask])))

            fold_rows.append(
                {
                    "horizon_days": int(h),
                    "fold_index": fold_idx,
                    "train_end": train_end_date.strftime("%Y-%m-%d"),
                    "test_start": test_start_date.strftime("%Y-%m-%d"),
                    "test_end": test_end_date.strftime("%Y-%m-%d"),
                    "n": n_test,
                    "hit_rate": hit_rate,
                    "mae": mae,
                    "evaluated_at": evaluated_at,
                }
            )
            log.info(
                "walk-forward horizon=%sd fold=%d n_train=%d n_test=%d hit_rate=%.3f mae=%.4f "
                "(train_end=%s test=%s..%s)",
                h, fold_idx, n_train, n_test, hit_rate, mae,
                train_end_date.date(), test_start_date.date(), test_end_date.date(),
            )

            # Each fold fits and discards a full model pair on its own
            # expanding-window slice -- by far the most memory this
            # function touches, repeated up to n_folds times per horizon.
            # See forecast.train_models's matching comment on why an
            # explicit collection (not just the del) matters on a
            # memory-capped host.
            del dir_pipe, mag_pipe, X_test, direction_correct, train_mask, test_mask
            gc.collect()

        del df, dates, X, y_dir, y_ret
        gc.collect()

    db.replace_walk_forward_folds(fold_rows)
    del long_df
    gc.collect()
    return {"n_folds_scored": len(fold_rows)}


def scorecard() -> dict:
    return {
        "overall": db.scorecard_summary(),
        "by_horizon": [dict(r) for r in db.scorecard_by_horizon()],
        "by_ticker": [dict(r) for r in db.scorecard_by_ticker()],
        "walk_forward": {
            "overall": db.walk_forward_summary(),
            "by_horizon": [dict(r) for r in db.walk_forward_by_horizon()],
            "folds": [dict(r) for r in db.walk_forward_folds()],
        },
    }
