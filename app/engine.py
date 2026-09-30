"""In-process singleton tying ingestion -> features -> models -> forecasts
-> backtest together, so the API layer just asks this object for things
instead of re-running the pipeline on every request.

Bulk forecast generation never calls the optional Claude narrative step
(that would be ~100 companies x 3 horizons of API calls on every refresh);
the narrative is generated lazily, only when a single ticker's detail view
is requested.
"""
from __future__ import annotations

import logging
import os
import threading

from app import db, pipeline, scoring, trading
from app.forecast import ForecastModels, generate_all_forecasts, generate_forecast, train_models
from app.signals import FeaturePanel, build_feature_panel

log = logging.getLogger("stockgraph.engine")

# Ingestion used to only ever run once, at process startup -- "live" data
# then quietly went stale until someone noticed and manually restarted the
# server. 0 disables the background refresh entirely (e.g. for tests or a
# demo-mode deployment where nothing ever changes anyway).
REFRESH_INTERVAL_HOURS = float(os.environ.get("STOCKGRAPH_REFRESH_INTERVAL_HOURS", "4"))


class Engine:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.panel: FeaturePanel | None = None
        self.models: ForecastModels | None = None
        self.forecast_cache: dict[int, list[dict]] = {}
        self.forecast_by_ticker: dict[tuple[str, int], dict] = {}
        self.trained_at: str | None = None
        self.backtest_summary: dict = {}
        self.ingestion_info: dict = {}
        self.ready = False
        self._refresh_thread: threading.Thread | None = None
        self._stop_refresh = threading.Event()
        self.refresh_interval_hours: float = 0.0
        self.last_refresh_error: str | None = None
        self.last_trading_result: dict = {}

    def bootstrap(self) -> None:
        with self.lock:
            log.info("bootstrap: running ingestion...")
            self.ingestion_info = pipeline.run_ingestion()
            log.info("bootstrap: building feature panel...")
            self.panel = build_feature_panel()
            log.info("bootstrap: training models...")
            self.models = train_models(self.panel)
            self.trained_at = db.now_iso()
            log.info("bootstrap: backtesting...")
            self.backtest_summary = scoring.run_backtest(self.panel, self.models)
            log.info("bootstrap: generating forecasts...")
            self._regenerate_forecasts()
            self.ready = True
            log.info("bootstrap complete.")
            # Off unless Danny has both set STOCKGRAPH_TRADING_ENABLED and
            # pasted his own Alpaca paper-trading keys in -- see
            # app/trading.py. A failure here is logged and recorded, never
            # allowed to undo the bootstrap that already succeeded above
            # (the site should keep serving forecasts even if the paper-
            # trading pass itself breaks).
            if trading.enabled():
                log.info("bootstrap: running paper-trading pass...")
                try:
                    self.last_trading_result = trading.run_trading_pass(self)
                except Exception as e:
                    log.exception("paper-trading pass failed; forecasts/site are unaffected")
                    self.last_trading_result = {"ran": False, "reason": f"unexpected error: {e}"}

    def start_background_refresh(self, interval_hours: float = REFRESH_INTERVAL_HOURS) -> None:
        """Periodically re-runs the full bootstrap (ingest -> features ->
        train -> backtest -> forecast) in a daemon thread, so data and
        models actually stay current without a person restarting the
        process. Safe to call only after an initial bootstrap() has
        succeeded. Idempotent -- calling twice doesn't start two threads.

        A failed refresh is caught, logged, and recorded on
        last_refresh_error; it does NOT crash the thread or affect what the
        site is currently serving, since a transient data-source outage
        (or a bug in a new data client) shouldn't take down a site that was
        working fine a moment ago with its last-known-good data/models.
        Incremental ingestion (see pipeline.py's freshness check) is what
        keeps each of these re-runs cheap after the first one.
        """
        if interval_hours <= 0 or self._refresh_thread is not None:
            return
        self.refresh_interval_hours = interval_hours

        def _loop() -> None:
            while not self._stop_refresh.wait(interval_hours * 3600):
                log.info("scheduled refresh: starting...")
                try:
                    self.bootstrap()
                    self.last_refresh_error = None
                    log.info("scheduled refresh: complete.")
                except Exception as e:
                    self.last_refresh_error = str(e)
                    log.exception("scheduled refresh failed; continuing to serve previous data/models")

        self._refresh_thread = threading.Thread(target=_loop, name="stockgraph-refresh", daemon=True)
        self._refresh_thread.start()
        log.info("background refresh scheduled every %.1fh", interval_hours)

    def stop_background_refresh(self) -> None:
        self._stop_refresh.set()

    def _regenerate_forecasts(self) -> None:
        for h in self.models.by_horizon:
            forecasts = generate_all_forecasts(self.panel, self.models, h, use_llm=False)
            self.forecast_cache[h] = forecasts
            rows = []
            for f in forecasts:
                self.forecast_by_ticker[(f["ticker"], h)] = f
                rows.append({k: v for k, v in f.items() if not k.startswith("_")})
            # Batched into one commit per horizon instead of one commit per
            # company -- 448 individual disk fsyncs was a meaningful chunk
            # of a full bootstrap. See db.insert_forecasts_batch.
            db.insert_forecasts_batch(rows)

    def list_forecasts(self, horizon: int) -> list[dict]:
        with self.lock:
            return list(self.forecast_cache.get(horizon, []))

    def get_forecast(self, ticker: str, horizon: int, with_llm: bool = False) -> dict:
        with self.lock:
            if self.models is None or horizon not in self.models.by_horizon:
                raise ValueError(f"no model for horizon={horizon}")
            cached = self.forecast_by_ticker.get((ticker, horizon))
            if cached and not with_llm:
                return cached
            if cached and with_llm and cached.get("llm_narrative"):
                return cached
            f = generate_forecast(ticker, self.panel, self.models, horizon, use_llm=with_llm)
            db.insert_forecast({k: v for k, v in f.items() if not k.startswith("_")})
            self.forecast_by_ticker[(ticker, horizon)] = f
            return f

    def horizons(self) -> list[int]:
        with self.lock:
            return sorted(self.models.by_horizon.keys()) if self.models else []

    def status(self) -> dict:
        with self.lock:
            return {
                "ready": self.ready,
                "trained_at": self.trained_at,
                "ingestion": self.ingestion_info,
                "data_mode_active": db.get_meta("data_mode_active"),
                "last_ingested_at": db.get_meta("last_ingested_at"),
                "horizons": self.horizons(),
                "model_metrics": {
                    str(h): {
                        "holdout_accuracy": hm.holdout_accuracy,
                        "holdout_mae": hm.holdout_mae,
                        "n_train": hm.n_train,
                    }
                    for h, hm in (self.models.by_horizon.items() if self.models else [])
                },
                "backtest_summary": {str(k): v for k, v in self.backtest_summary.items()},
                "refresh_interval_hours": self.refresh_interval_hours,
                "last_refresh_error": self.last_refresh_error,
                "trading_enabled": trading.enabled(),
                "last_trading_result": self.last_trading_result,
            }


engine = Engine()
