"""Macro/market-regime data: the VIX (equity fear gauge) and the Treasury
par yield curve. Unlike everything else in dataclients/, these aren't
per-company -- they're one shared signal broadcast to every ticker (see
signals.py's _macro_features), fetched once per ingestion run rather than
once per ticker (see pipeline.py's _ingest_macro).

Both sources are free and keyless:
  - VIX: the same Yahoo Finance chart endpoint already used for equity
    prices (app.dataclients.yahoo), just pointed at the ^VIX index.
  - Treasury par yield curve: Treasury.gov's own published CSV export -- no
    API key, no meaningful rate limit, no registration.
"""
from __future__ import annotations

import csv
import io
import logging
import urllib.parse
from datetime import date, datetime

from app.dataclients.httpjson import get_text
from app.dataclients.yahoo import fetch_daily_bars

log = logging.getLogger("stockgraph.macro")

TREASURY_CSV_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all"
)

# Tenors behind the two classic curve-inversion spreads: 10y-2y is the most
# commonly cited recession indicator; 10y-3mo is the New York Fed's own
# preferred one. Keyed by the CSV's own column header so parsing is by name,
# not position -- Treasury has occasionally added columns (e.g. "1.5 Month")
# without warning, which would silently misalign a position-based parse.
_TENOR_COLUMNS = {"3 Mo": "yield_3m", "2 Yr": "yield_2y", "10 Yr": "yield_10y"}


def fetch_vix_history() -> list[dict]:
    """Returns [{date, value}, ...] ascending by date. The VIX index has no
    meaningful open/high/low/volume for this purpose, just a daily close
    level, so only that's kept. `^VIX` is percent-encoded (%5EVIX) before
    being spliced into the chart URL's path -- Yahoo's endpoint takes the
    raw ticker directly in the path rather than as a query parameter, and
    unlike the query-string params get_json builds, that path segment isn't
    otherwise escaped for us."""
    encoded = urllib.parse.quote("^VIX", safe="")
    rows = fetch_daily_bars(encoded, range_="2y")
    return [
        {"series": "vix_close", "date": r["date"], "value": r["close"]}
        for r in rows
        if r["close"] is not None
    ]


def _parse_treasury_date(raw: str) -> str | None:
    # Treasury's CSV uses MM/DD/YYYY.
    try:
        return datetime.strptime(raw.strip(), "%m/%d/%Y").date().isoformat()
    except ValueError:
        return None


def _fetch_treasury_year(year: int) -> list[dict]:
    params = {
        "type": "daily_treasury_yield_curve",
        "field_tdr_date_value": year,
        "page": "",
        "_format": "csv",
    }
    text = get_text(TREASURY_CSV_URL.format(year=year), params=params)
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict] = []
    for raw_row in reader:
        iso_date = _parse_treasury_date(raw_row.get("Date", ""))
        if not iso_date:
            continue
        for col, series in _TENOR_COLUMNS.items():
            raw_val = raw_row.get(col)
            if raw_val in (None, "", "N/A"):
                continue
            try:
                value = float(raw_val)
            except ValueError:
                continue
            rows.append({"series": series, "date": iso_date, "value": value})
    return rows


def fetch_treasury_yield_curve(years_back: int = 2) -> list[dict]:
    """Returns [{series, date, value}, ...] for yield_3m/yield_2y/yield_10y,
    covering the current year plus `years_back - 1` prior years (Treasury's
    CSV export is one calendar year per request) so the history comfortably
    covers PRICE_HISTORY_DAYS worth of trading days even right after a
    January 1st rollover."""
    this_year = date.today().year
    rows: list[dict] = []
    for year in range(this_year - years_back + 1, this_year + 1):
        rows.extend(_fetch_treasury_year(year))
    return rows
