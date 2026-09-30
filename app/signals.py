"""Turns raw prices/earnings/news into a per-ticker, per-day feature panel.

Everything here is vectorized across the whole universe at once (pandas
DataFrames shaped dates x tickers) so the "graph spillover" features --
the whole point of this project -- can be computed as a single matrix
multiply against the relationship graph's adjacency matrix, rather than
looping ticker by ticker.

Feature groups per (ticker, date):
  price/technical : momentum over a few windows, realized vol, RSI,
                     distance from moving averages, volume z-score
  earnings        : days since/until known earnings date, last surprise
  news             : decayed trailing sentiment, recent event-tag counts
  graph            : weighted spillover of neighbors' momentum and
                      sentiment, signed by relationship type (competitor
                      pressure works the opposite way from supplier/
                      customer/partner) and scaled by edge weight
  sector/market    : cross-sectional sector and universe momentum, as a
                      cheap proxy for the systematic factors that make
                      stocks in the same neighborhood move together
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app import db
from app.universe import COMPANY_BY_TICKER, EDGES, TICKERS

MOMENTUM_WINDOWS = (5, 10, 20, 60)
VOL_WINDOW = 20
RSI_WINDOW = 14
FEATURE_COLUMNS: list[str] = []  # populated at import time below


def _build_weight_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two |tickers| x |tickers| matrices: raw signed weights, and the
    same weights normalized per-column so a spillover feature comes out
    as a weighted *average* (bounded, comparable across companies with
    different numbers of neighbors) rather than a weighted sum."""
    idx = TICKERS
    w = pd.DataFrame(0.0, index=idx, columns=idx)
    for e in EDGES:
        sign = -1.0 if e.kind == "competitor" else 1.0
        w.loc[e.src, e.dst] += sign * e.weight
        w.loc[e.dst, e.src] += sign * e.weight
    denom = w.abs().sum(axis=0).replace(0, np.nan)
    w_norm = w.div(denom, axis=1).fillna(0.0)
    return w, w_norm


_RAW_W, _NORM_W = _build_weight_matrix()


def _load_price_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (close, volume, high, low, dates_index) as dates x tickers
    DataFrames, forward/backward filled lightly so missing single days
    don't break rolling windows across a universe of tickers with slightly
    different trading histories."""
    frames_close = {}
    frames_vol = {}
    frames_high = {}
    frames_low = {}
    for t in TICKERS:
        rows = db.get_prices(t)
        if not rows:
            continue
        frames_close[t] = pd.Series({r["date"]: r["close"] for r in rows}, name=t)
        frames_vol[t] = pd.Series({r["date"]: r["volume"] for r in rows}, name=t)
        frames_high[t] = pd.Series({r["date"]: r["high"] for r in rows}, name=t)
        frames_low[t] = pd.Series({r["date"]: r["low"] for r in rows}, name=t)

    close = pd.DataFrame(frames_close).sort_index()
    volume = pd.DataFrame(frames_vol).sort_index()
    high = pd.DataFrame(frames_high).sort_index()
    low = pd.DataFrame(frames_low).sort_index()
    close.index = pd.to_datetime(close.index)
    volume.index = pd.to_datetime(volume.index)
    high.index = pd.to_datetime(high.index)
    low.index = pd.to_datetime(low.index)
    close = close.ffill(limit=3)
    volume = volume.ffill(limit=3)
    high = high.ffill(limit=3)
    low = low.ffill(limit=3)
    return close, volume, high, low, close.index


def _macd_histogram(close: pd.DataFrame) -> pd.DataFrame:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return hist / close  # scale by price so it's comparable across tickers


def _bollinger_features(close: pd.DataFrame, window: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    lower = sma - 2 * std
    upper = sma + 2 * std
    band_width = ((upper - lower) / sma.replace(0, np.nan)).fillna(0.0)
    percent_b = ((close - lower) / (upper - lower).replace(0, np.nan)).fillna(0.5)
    return percent_b, band_width


def _atr_pct(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Average True Range as a fraction of price -- a volatility measure
    that (unlike a plain stdev of returns) accounts for intraday gaps."""
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = true_range.rolling(window).mean()
    return (atr / close.replace(0, np.nan)).fillna(0.0)


def _filing_recency_features(dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Days since the most recent SEC filing (any of 10-K/10-Q/8-K), and a
    trailing 30-day filing count -- 8-Ks in particular are filed for material
    corporate events (M&A, executive changes, major agreements), so a burst
    of filings is itself a signal, independent of price/news. Built from
    genuinely dated historical events (unlike a fundamentals snapshot), so
    unlike analyst targets this carries no look-ahead risk in the backtest.

    Vectorized with searchsorted rather than a per-date Python loop (400
    dates x 232 tickers x up to 50 filings each was measurably slow)."""
    days_since = pd.DataFrame(999.0, index=dates, columns=TICKERS)
    trailing_count = pd.DataFrame(0.0, index=dates, columns=TICKERS)
    dates_arr = dates.values  # datetime64[ns], ascending
    window_start = dates_arr - np.timedelta64(30, "D")
    for t in TICKERS:
        rows = db.get_filings(t, limit=50)
        if not rows:
            continue
        filed_list = sorted(pd.Timestamp(r["filed_date"]) for r in rows if r["filed_date"])
        if not filed_list:
            continue
        filed = np.array(filed_list, dtype="datetime64[ns]")

        end_idx = np.searchsorted(filed, dates_arr, side="right")
        has_past = end_idx > 0
        last_filed = np.where(has_past, filed[np.clip(end_idx - 1, 0, None)], np.datetime64("NaT"))
        diff_days = (dates_arr - last_filed) / np.timedelta64(1, "D")
        days_since[t] = np.where(has_past, diff_days, 999.0)

        start_idx = np.searchsorted(filed, window_start, side="right")
        trailing_count[t] = (end_idx - start_idx).astype(float)
    return days_since, trailing_count


def _rsi(close: pd.DataFrame, window: int = RSI_WINDOW) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _news_sentiment_panel(dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trailing exponentially-decayed sentiment and trailing event-tag
    counts, reindexed onto the price calendar."""
    sent_frames = {}
    event_frames = {}
    for t in TICKERS:
        rows = db.get_news(t, limit=200)
        if not rows:
            sent_frames[t] = pd.Series(dtype=float)
            event_frames[t] = pd.Series(dtype=float)
            continue
        idx = pd.to_datetime([r["published"][:10] for r in rows])
        sent = pd.Series([r["sentiment"] or 0.0 for r in rows], index=idx)
        sent = sent.groupby(level=0).mean().sort_index()
        n_events = pd.Series(
            [1 if json.loads(r["event_tags"] or "[]") else 0 for r in rows], index=idx
        )
        n_events = n_events.groupby(level=0).sum().sort_index()
        sent_frames[t] = sent
        event_frames[t] = n_events

    sent_df = pd.DataFrame(sent_frames).reindex(dates).fillna(0.0)
    event_df = pd.DataFrame(event_frames).reindex(dates).fillna(0.0)
    # Exponential decay (halflife ~4 trading days) trailing sum -> smooth signal.
    decayed_sent = sent_df.ewm(halflife=4, min_periods=1).mean()
    trailing_events = event_df.rolling(7, min_periods=1).sum()
    return decayed_sent, trailing_events


def _earnings_features(dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    days_to_next = pd.DataFrame(np.nan, index=dates, columns=TICKERS)
    days_since_last = pd.DataFrame(np.nan, index=dates, columns=TICKERS)
    last_surprise = pd.DataFrame(0.0, index=dates, columns=TICKERS)

    for t in TICKERS:
        rows = db.get_earnings(t)
        if not rows:
            continue
        report_dates = pd.to_datetime([r["report_date"] for r in rows if r["report_date"]])
        surprises = {
            pd.Timestamp(r["report_date"]): (r["surprise_pct"] or 0.0)
            for r in rows
            if r["report_date"] and not r["is_future"]
        }
        if len(report_dates) == 0:
            continue
        rd_sorted = sorted(report_dates)
        for d in dates:
            future = [rd for rd in rd_sorted if rd >= d]
            past = [rd for rd in rd_sorted if rd < d]
            if future:
                days_to_next.loc[d, t] = (future[0] - d).days
            if past:
                last = past[-1]
                days_since_last.loc[d, t] = (d - last).days
                last_surprise.loc[d, t] = surprises.get(last, 0.0)

    return days_to_next.fillna(999), days_since_last.fillna(999), last_surprise


@dataclass
class FeaturePanel:
    close: pd.DataFrame
    features: dict[str, pd.DataFrame]  # feature_name -> (dates x tickers)
    dates: pd.DatetimeIndex

    def latest_row(self, ticker: str) -> dict:
        d = self.dates[-1]
        return {name: float(df.loc[d, ticker]) if pd.notna(df.loc[d, ticker]) else 0.0
                for name, df in self.features.items()}

    def as_long_frame(self, horizons: list[int]) -> pd.DataFrame:
        """Stack into one (ticker, date) row per observation, with forward
        return / direction labels for each horizon where available. Used
        to train the forecasting models."""
        pieces = []
        for t in TICKERS:
            df = pd.DataFrame({name: self.features[name][t] for name in self.features})
            df["ticker"] = t
            df["date"] = self.dates
            close_t = self.close[t]
            for h in horizons:
                fwd = close_t.shift(-h) / close_t - 1.0
                df[f"fwd_ret_{h}"] = fwd.values
                df[f"fwd_dir_{h}"] = (fwd.values > 0).astype(float)
            pieces.append(df)
        long_df = pd.concat(pieces, ignore_index=True)
        return long_df


def split_boundary_dates(dates, horizon: int, train_frac: float = 0.85):
    """Returns (train_end_date, test_start_date) for a chronological
    train/test split with a purge gap of `horizon` trading days between
    them.

    Without the gap, a training row dated just before the boundary still
    has its label -- the return `horizon` days later -- computed from
    prices that fall inside the nominally "held out" test window. That
    silently leaks test-period price action into training and can inflate
    the reported backtest accuracy versus what the model would actually
    achieve forecasting genuinely unseen future data. `forecast.train_models`
    and `scoring.run_backtest` both call this so the split they evaluate
    against is identical and leak-free.
    """
    dates = pd.DatetimeIndex(sorted(pd.Index(dates).unique()))
    n = len(dates)
    test_start_idx = min(max(int(n * train_frac), 1), n - 1)
    train_end_idx = max(test_start_idx - horizon, 0)
    return dates[train_end_idx], dates[test_start_idx]


def build_feature_panel() -> FeaturePanel:
    close, volume, high, low, dates = _load_price_panel()
    if close.empty:
        raise RuntimeError("no price data loaded; run the ingestion pipeline first")

    ret1 = close.pct_change()
    features: dict[str, pd.DataFrame] = {}

    for w in MOMENTUM_WINDOWS:
        features[f"mom_{w}"] = close.pct_change(w)

    features["vol_20"] = ret1.rolling(VOL_WINDOW).std() * np.sqrt(252)
    features["rsi_14"] = _rsi(close)
    features["ma_gap_20"] = close / close.rolling(20).mean() - 1.0
    features["ma_gap_50"] = close / close.rolling(50).mean() - 1.0

    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std().replace(0, np.nan)
    features["vol_zscore"] = ((volume - vol_mean) / vol_std).fillna(0.0)

    # More technical signal: MACD histogram, Bollinger %B/bandwidth, ATR.
    features["macd_hist"] = _macd_histogram(close)
    percent_b, band_width = _bollinger_features(close)
    features["bollinger_pct_b"] = percent_b
    features["bollinger_bandwidth"] = band_width
    features["atr_pct"] = _atr_pct(close, high, low)

    sent, events = _news_sentiment_panel(dates)
    features["sentiment_7d"] = sent
    features["news_event_count_7d"] = events

    days_to_next, days_since_last, last_surprise = _earnings_features(dates)
    features["days_to_earnings"] = days_to_next
    features["days_since_earnings"] = days_since_last
    features["last_earnings_surprise_pct"] = last_surprise
    features["earnings_imminent"] = (days_to_next <= 5).astype(float)
    features["post_earnings_drift_window"] = (days_since_last <= 3).astype(float)

    filing_days_since, filing_trailing_count = _filing_recency_features(dates)
    features["days_since_filing"] = filing_days_since
    features["filing_count_30d"] = filing_trailing_count

    # Sector momentum: mean mom_5 across the sector, excluding self.
    mom5 = features["mom_5"].fillna(0.0)
    sector_of = {t: COMPANY_BY_TICKER[t].sector for t in TICKERS}
    sector_groups: dict[str, list[str]] = {}
    for t, s in sector_of.items():
        sector_groups.setdefault(s, []).append(t)
    sector_mom = pd.DataFrame(index=mom5.index, columns=TICKERS, dtype=float)
    for s, members in sector_groups.items():
        grp = mom5[members]
        grp_sum = grp.sum(axis=1)
        n = len(members)
        for t in members:
            sector_mom[t] = (grp_sum - grp[t]) / max(n - 1, 1)
    features["sector_mom_5"] = sector_mom
    market_series = mom5.mean(axis=1)
    features["market_mom_5"] = pd.DataFrame({t: market_series for t in TICKERS})

    # Graph spillover: weighted (normalized) neighbor momentum & sentiment.
    features["graph_mom_spillover"] = mom5 @ _NORM_W
    sent_filled = sent.fillna(0.0)
    features["graph_sent_spillover"] = sent_filled @ _NORM_W

    for name, df in features.items():
        features[name] = df.reindex(columns=TICKERS).fillna(0.0)

    global FEATURE_COLUMNS
    FEATURE_COLUMNS = list(features.keys())

    return FeaturePanel(close=close, features=features, dates=dates)
