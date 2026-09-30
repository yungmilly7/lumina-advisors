"""Entrypoint: `python run.py` starts the API + serves the website.

First boot runs the full pipeline (ingest -> features -> train -> backtest
-> forecast) synchronously before accepting requests, so there's a short
pause (a few seconds in demo mode; longer in live mode depending on network
latency to the live data sources) before the site responds. After that,
engine.start_background_refresh() re-runs the same pipeline on a schedule
(STOCKGRAPH_REFRESH_INTERVAL_HOURS, default 4h) in a daemon thread, so data
and forecasts keep themselves current without a manual restart -- ingestion
is incremental (see pipeline.py), so these re-runs are fast, not a repeat
of the full first-run cost.

Built on the standard library's http.server (see app/httpserver.py) rather
than uvicorn/Starlette, so running this needs nothing beyond numpy and
pandas -- no `pip install` required in places that can't reach PyPI.
"""
from __future__ import annotations

import logging

from app.config import HOST, PORT
from app.engine import engine
from app.httpserver import run

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    engine.bootstrap()
    engine.start_background_refresh()

    from app.api import router

    run(router, HOST, PORT)
