# Lumina Advisors

A stock research website built around one idea: companies don't move in
isolation. It ingests prices, earnings, SEC filings, and news for ~232
interconnected public companies, encodes their real-world relationships
(supplier, customer, competitor, partner) as a graph, and trains a model
that uses that graph — plus price action, sentiment, and earnings timing —
to forecast direction and expected move size over 1/5/20-day horizons.

Accounts are built in: sign up to save a personal watchlist (star any
company) that's kept in your account and follows you across visits/devices.
Signing in is optional -- everything else works signed out.

Every forecast comes with a plain-English rationale, a driver breakdown
(which factors pushed the call which way, with numbers, and a hover
explainer in plain English for each one), 52-week high/low and volume
context, an "earnings soon" badge (📅) on any ticker reporting within the
next few trading days, and an honest, walk-forward-tested accuracy
scorecard, because a forecasting tool that hides its own error rate isn't
trustworthy. A chat
widget (bottom-right, on every page) lets you ask plain-English questions
about any of it -- "why is NVDA called up?", "which sector looks weakest
right now?" -- answered only from this site's own live data, never from
outside/trained knowledge about real markets.

**This is not investment advice**, and the scorecard tab will tell you
exactly how good (or not) the model actually is — read that before trusting
any single prediction. Predicting daily stock moves is famously hard; this
tool is a structured, explainable estimate, not an edge.

## Quickstart

Only needs **numpy and pandas** -- no other third-party dependencies (see
"Why zero dependencies" below). Both are pre-installed in most Python
setups (Anaconda, most data-science images); if not:

```bash
pip install -r requirements.txt   # just numpy + pandas
cp .env.example .env   # optional: set STOCKGRAPH_DATA_MODE, ANTHROPIC_API_KEY, etc.
python run.py
```

Then open http://localhost:8000. First boot runs the whole pipeline
(ingest → build features → train models → backtest → generate forecasts)
before the site responds — a few seconds in demo mode, longer in live mode
depending on network latency.

## How it's built

```
app/
  universe.py     the ~232-company universe + the real supplier/customer/
                   competitor/partner relationship graph between them
  dataclients/
    yahoo.py       live daily prices + earnings calendar (Yahoo Finance's
                    public chart/quoteSummary endpoints, no API key)
    secedgar.py     live SEC filings (10-K/10-Q/8-K) via EDGAR's JSON API,
                    plus open-market insider buy/sell transactions parsed
                    out of Form 4 filings (see "Insider transactions" below)
    macro.py        live VIX + Treasury yield curve -- one shared market-
                    regime signal for the whole universe, not per-company
                    (see "Macro/market regime signal" below)
    news.py         Google News RSS + a transparent lexicon-based
                    sentiment/event-tag scorer (no ML black box, no API key)
    demo.py         deterministic synthetic data generator -- same
                    statistical structure as real markets (market factor +
                    sector factor + idiosyncratic noise + graph pass-through
                    + scheduled earnings/news events), used when live data
                    isn't reachable
  pipeline.py      orchestrates ingestion, with automatic live -> demo
                   fallback per ticker
  db.py            SQLite storage (one file, nothing to run/host)
  signals.py       turns raw data into a per-company/day feature panel:
                   momentum, volatility, RSI, volume anomalies, earnings
                   timing, decayed news sentiment, sector/market momentum,
                   insider buy/sell balance, VIX/yield-curve macro regime,
                   and the graph spillover features (a matrix multiply of
                   neighbor momentum/sentiment against the relationship
                   graph's signed, weighted adjacency matrix)
  ml.py            small pure-numpy StandardScaler / RidgeRegression /
                   LogisticRegression / Pipeline -- see "Why zero
                   dependencies" below
  auth.py          accounts: PBKDF2-HMAC-SHA256 password hashing (stdlib
                   hashlib, no bcrypt dependency) + opaque session tokens
                   stored server-side, handed to the browser as an
                   HttpOnly cookie
  chat.py          the Q&A chatbot: builds a plain-text snapshot of this
                   platform's own data (a company's forecast/drivers/news/
                   neighbors/earnings, or a whole-market summary) and asks
                   Claude to answer strictly from it -- grounded, not
                   trained-knowledge, so it can't "hallucinate" real-market
                   facts
  forecast.py      the "agent": a LogisticRegression (direction) + Ridge
                   (magnitude) pair trained per horizon on the whole
                   universe's pooled history, decomposed into per-feature
                   contributions at inference time for the rationale, with
                   an optional Claude API call for a plain-English narrative
  scoring.py       scores the frozen model against a held-out slice of
                   history it never trained on, plus a separate genuine
                   walk-forward validation (several re-trained folds) that
                   checks whether that single split's number is real
  engine.py        in-process singleton wiring it all together for the API
  httpserver.py    a minimal router + server on the standard library's
                   http.server -- see "Why zero dependencies" below
  api.py           the HTTP API, built on httpserver.py
static/
  index.html, app.css, app.js, graph.js
                   the website -- vanilla JS, a hand-rolled canvas
                   force-directed graph (no CDN dependency to fail if a
                   network is locked down), SVG sparklines, sortable
                   tables, hand-rolled canvas bar charts (sector
                   performance, confidence distribution, hit-rate by
                   horizon), and a floating chat widget
tests/
  smoke_test.py    unittest suite (stdlib only) covering the universe,
                   pipeline, models, and every API route
```

## Why zero dependencies

This was built and tested across two different locked-down, managed-device
networks that could not reach PyPI, npm, or GitHub at all (only
`api.anthropic.com` was reachable in one of them). So rather than assume
`pip install` will work wherever this ends up running, everything beyond
numpy/pandas was replaced with a small hand-rolled equivalent:

- `app/ml.py` — logistic regression (gradient descent) + ridge regression
  (closed-form) + standardization, in ~100 lines of numpy, instead of
  scikit-learn.
- `app/httpserver.py` — a small router on top of the standard library's
  `http.server`, instead of Starlette/FastAPI + uvicorn.
- `app/dataclients/httpjson.py` — `urllib.request`-based HTTP calls,
  instead of `httpx`.
- `static/graph.js` — a from-scratch canvas force-directed graph layout
  and renderer, instead of a CDN graph library that may not load on a
  restricted network.

If you're running this somewhere with normal internet access and would
rather use the real libraries (better solver convergence, HTTP/2, etc.),
swapping `app/ml.py`'s classes for scikit-learn's or `app/httpserver.py`
for Starlette is a small, contained change -- the rest of the codebase
only depends on the small interface those modules expose (`.fit`,
`.predict`, `.predict_proba`, `Router.add`, `JSONResponse`).

## Data modes

Set `STOCKGRAPH_DATA_MODE` in `.env`:

- `auto` (default, recommended) — tries live data per ticker, falls back to
  synthetic data for any ticker it can't reach. If literally nothing is
  reachable, the whole universe falls back to demo data so the site still
  works.
- `live` — always hits Yahoo Finance / SEC EDGAR / Google News; raises if
  any ticker fails, so you know immediately if something's broken.
- `demo` — never touches the network; generates synthetic data. This is
  what ran during development in this sandbox, which has locked-down
  network egress, and it's what the shipped site is currently running.

The badge in the top-right of the site always tells you which mode is
actually active. Demo data is clearly not real market data and is labeled
as such everywhere it appears — it exists so you can see the whole product
working before pointing it at real data.

## Turning on live data

The sandbox this was built in could not reach `query1.finance.yahoo.com`,
`data.sec.gov`, or `news.google.com` (its network is locked to package
registries only), so the original live-mode code was untested against real
traffic. It's since been run against the real internet and three issues
came up, all fixed and confirmed live (232/232 companies ingested in pure
`live` mode, no demo fallback needed):

- Yahoo now requires a "crumb" token plus a session cookie on
  `quoteSummary` calls (the same handshake the `yfinance` library does
  internally) -- `app/dataclients/yahoo.py` seeds a cookie from
  `fc.yahoo.com`, fetches a crumb from the `getcrumb` endpoint, and sends
  both automatically. No configuration needed. (The cookie-seed request
  itself routinely 404s -- that's expected, Yahoo still sets the cookie on
  that response -- so the seed and crumb steps are allowed to fail/succeed
  independently rather than one failure aborting the whole handshake.)
- SEC EDGAR 403s any `User-Agent` that isn't a real, descriptive one (a
  name plus a contact email) -- set `SEC_USER_AGENT` in `.env` accordingly.
- SEC's per-company filing list is newest-first and dominated by routine
  forms (Form 4 insider trades, proxies, ...); the 10-Ks/10-Qs/8-Ks this
  site actually shows are a small fraction of it.
  `app/dataclients/secedgar.py` now scans the whole list for matches
  instead of just filtering the first page, which fixed roughly 90
  companies that were showing zero filings despite having them.

`.env` is loaded automatically on startup (a small stdlib-only parser in
`app/config.py` -- no extra dependency), and real environment variables
always take priority over it. Set `STOCKGRAPH_DATA_MODE=auto` (or `live`),
fill in `SEC_USER_AGENT` in `.env` with real contact info, and restart --
edits to `.env` need a restart, not just the in-app "Refresh data" button,
since that only re-runs the pipeline without reloading code. Hit
`POST /api/refresh` (or the "Refresh data" button) any time after that to
re-pull data and retrain.

## Insider transactions

Every company's forecast page shows recent open-market insider buys/sells,
and the model gets two features built from the same data:
`insider_net_buy_ratio_90d` (a dollar-weighted -1..+1 balance of buying vs.
selling in the trailing 90 days) and `insider_buy_count_90d`.

This comes from SEC EDGAR Form 4 filings -- free, no API key, same source
as the 10-K/10-Q/8-K filings already shown. Only transaction codes `P`
(open-market purchase) and `S` (open-market sale) are counted; codes like
`A` (grants/awards), `F` (tax withholding), `M` (option exercises), and `G`
(gifts) are compensation/administrative noise by academic convention, not a
genuine discretionary trading decision, so they're filtered out before this
ever reaches the model. Like filings and news, the per-ticker check for
*new* filings runs on every ingestion cycle rather than being tracked for
"freshness" -- an empty result usually just means that company's insiders
haven't made an open-market trade recently, which is a real, common state,
not a fetch failure.

Unlike filings, though, each Form 4 costs a *separate* document fetch on top
of the one list request (parsing the actual transaction requires the
filing's own XML document, not just its listing entry) -- multiplied across
448+ companies on every scheduled refresh, indefinitely, an unbounded
"always re-fetch the last N filings" would mean thousands of redundant SEC
requests per run for filings whose content never changes once filed. So
this one genuinely is incremental: `insider_txn_filings_seen` tracks which
accession numbers have already been fetched+parsed per ticker, and only
new ones are ever fetched again -- a ticker with a deep backlog catches up
in bounded batches over its first several runs, and steady state afterward
is just the 0-2 new Form 4s that typically appear between runs.

## Macro/market regime signal

Every forecast also sees four shared, market-wide features: `vix_level` and
`vix_change_5d` (the VIX volatility/fear-gauge index and its 5-day change),
and `yield_curve_10y_2y` / `yield_curve_10y_3m` (the two classic Treasury
yield-curve-inversion spreads -- 10y-2y is the most commonly cited, 10y-3mo
is the New York Fed's own preferred recession indicator). Unlike every other
feature, these are identical across every company on a given date -- it's
shared regime context, not a per-company signal, the same way `market_mom_5`
already broadcasts the cross-sectional average momentum to every ticker.

Both sources are free and keyless: VIX comes from the same Yahoo Finance
chart endpoint already used for equity prices (just pointed at the `^VIX`
index), and the yield curve comes from Treasury.gov's own published CSV
export. Fetched once per ingestion run (two small HTTP calls total), not
once per company -- there's no per-ticker rate-limit pressure to manage
here, so unlike bars/earnings/fundamentals this has no freshness window and
just always re-fetches fresh. A fetch failure for either source falls back
to a synthetic series for exactly that source (never both, and never
clobbering the other source's real data) rather than failing the run.

## SEC XBRL financials

Every forecast page also shows a "Financial trend" strip -- the last six
reported quarters of revenue and net income, straight from each company's
own SEC filings -- and the model gets three features built from the same
history: `xbrl_revenue_yoy_growth` (year-over-year revenue growth),
`xbrl_revenue_trend_8q` (an 8-quarter revenue trend direction), and
`xbrl_net_income_yoy_ratio` (a -1..+1 dollar-weighted year-over-year change
in net income, using the same symmetric ratio as the insider buy/sell
balance above so a swing across zero doesn't blow up).

This comes from the SEC's XBRL `companyconcept` API
(`data.sec.gov/api/xbrl/companyconcept/...`) -- free, no API key, structured
data straight out of each 10-Q/10-K rather than screen-scraped text. It's a
narrower, per-concept request than the full `companyfacts` blob, asked for
just the `Revenues`/`SalesRevenueNet` and `NetIncomeLoss` tags (trying a
short list of common tag aliases per concept, since not every filer uses
the same GAAP tag name), and only entries actually tagged `10-Q`/`10-K` with
a ~one-quarter reporting duration are kept -- year-to-date and full-year
cumulative entries are filtered out so the trend is quarter-over-quarter,
not a mix of durations.

The one thing this feature had to get right is *when* each quarter's numbers
actually became knowable. XBRL data carries two dates -- the quarter's own
`period_end` and the much-later `filed_date` the company actually disclosed
it -- and a naive version of this feature would use `period_end`, which
leaks a quarter's real results into the model weeks before any real investor
could have seen them. Every value here is gated strictly on `filed_date`,
so the model only ever sees a company's Q2 numbers once Q2's 10-Q was
actually public. Unlike insider transactions and macro data, XBRL results
carry real freshness semantics (a company's financials only change once a
quarter), so it's tracked in the same `data_provenance`/staleness system as
filings and fundamentals, just with its own much longer refresh window
(7 days, vs. the general 20 hours) -- there's no reason to re-check a
company's XBRL data every few hours between earnings.

A ticker with clean, complete SEC tagging shows six quarters immediately;
one with unusual or foreign-filer tagging (not everyone tags `us-gaap`
consistently, and some foreign private issuers don't file 10-Q/10-K at all)
may show fewer, or none yet -- same "legitimate empty state, not a fetch
failure" reasoning as insider transactions above.

## Adding the AI narrative layer and chat

Set `ANTHROPIC_API_KEY` in `.env` and two things turn on: the forecast
detail view will ask Claude for a short plain-English read on top of the
numeric forecast, and the chat widget (bottom-right) will answer
questions. Both are direct HTTPS calls to `api.anthropic.com`, no SDK
needed. Leave the key unset and the site works exactly the same, just
without those two things -- the chat widget stays reachable and explains
that it isn't turned on yet, rather than disappearing or erroring.

The chat widget is scoped automatically: if you've got a company open in
the relationship graph, it answers about that company (its forecast,
drivers, news, neighbors, earnings); otherwise it answers about the whole
tracked universe (breadth, top movers, overall accuracy). It only ever
answers from a fresh snapshot of this platform's own data built at request
time -- the system prompt explicitly tells it not to reach for outside
knowledge about real companies or markets, so answers stay consistent with
whatever's actually on screen (including in demo mode, where the numbers
are synthetic).

## Extending the company universe

Add companies to `COMPANIES` and relationships to `EDGES` in
`app/universe.py` -- that's the entire surface area. Everything downstream
(ingestion, features, graph spillover, training, the UI graph) picks up
new tickers automatically. Each edge needs a `kind` (supplier / customer /
competitor / partner), a `weight` (0-1, roughly "how much this matters to
both companies' stock"), and a one-line `note`. The universe currently
spans ~232 companies across tech, semis, autos, retail, financials, energy,
materials, staples, utilities/REITs, industrials/defense, healthcare/biotech,
insurance, homebuilders, rail/trucking, metals & mining, restaurants,
crypto-adjacent, agriculture, regional banks, media, apparel/luxury,
packaging, and water utilities.

## Accounts and watchlists

Sign up (email + password) from the top-right of the site to save a
personal watchlist -- star any company from the forecasts table or its
detail panel to add it. Watchlists are tied to your account and persist
in the same SQLite database as everything else (`users`, `sessions`, and
`watchlist` tables in `app/db.py`). Passwords are never stored in plain
text (PBKDF2-HMAC-SHA256, 100k iterations, random per-user salt); sessions
are random opaque tokens in an HttpOnly cookie, not a client-readable JWT.
Signing in is entirely optional -- forecasts, the graph, and the scorecard
all work the same either way.

## Paper trading (optional, off by default)

Lumina can turn its own top-confidence forecasts into simulated orders on
a free [Alpaca](https://alpaca.markets) paper-trading account -- the real
brokerage API, but every order fills against fake money in a simulated
account. Nothing here can place a real trade: `ALPACA_BASE_URL` defaults
to Alpaca's paper host, and this project has never been pointed at the
live one.

**It's off until you turn it on, on purpose, in two places:**

1. Sign up free at [alpaca.markets](https://alpaca.markets) -> Paper
   Trading tab -> "Generate New Keys" (no funding or approval needed --
   it's simulated). Put the two keys in `.env` as `ALPACA_API_KEY` /
   `ALPACA_SECRET_KEY`.
2. Set `STOCKGRAPH_TRADING_ENABLED=true` in `.env`.

With both set, every bootstrap/scheduled refresh runs one more step after
generating forecasts: it closes whatever positions the bot is currently
holding (each forecast is a fresh "as of right now" call, so yesterday's
position isn't a thesis worth holding through today), then opens new
positions from today's highest-confidence calls on the shortest trained
horizon, up to a fixed number of positions, each a fixed dollar size. A
kill switch checks the paper account's own day-over-day P&L before opening
anything and skips the whole pass if it's already past a configurable loss
threshold. Every decision -- opened, closed, or skipped, and exactly why
-- is logged to the `paper_trades` table and readable at
`/api/trading/status`. All the knobs (which horizon, confidence floor,
position count/size, kill-switch threshold) are environment variables --
see `.env.example`.

**Read this before turning it on:** the "Honest limitations" section right
below says the model's holdout accuracy is roughly coin-flip. That's fine
for a forecast you read and think about; it's a real reason not to expect
this to make (paper) money as shipped. This exists so you can *watch* that
play out safely with fake money, not because the strategy is validated.

This is deliberately **not** wired into `render.yaml` / the public Render
deployment -- it's meant to stay a local, opt-in experiment against your
own paper account on your own machine, not something every visitor to the
public demo shares or can affect.

## Walk-forward validation

The scorecard tab's headline numbers (and `scoring.run_backtest`) come from
scoring the actual production model -- the one forecasts are served from --
against a single chronological train/test split: roughly the first 85% of
history to fit, the last 15% held out, with a purge gap of `horizon`
trading days at the boundary so no training label's forward-return window
reaches into the holdout (see `signals.split_boundary_dates`). That's a
real, leakage-free out-of-sample score, but it's still one number from one
split -- if that particular holdout window happened to be an unusually easy
or hard stretch of the market, the number would be misleading either way
without any way to tell.

`scoring.run_walk_forward_backtest` (`signals.walk_forward_splits`) exists
to answer exactly that question. It carves the most recent half of history
into several sequential test windows and walks forward through them: fold 0
trains on everything up through a purge gap before the first window and
tests on it; fold 1 trains on everything up through fold 0's test window
(an *expanding* window -- it never forgets earlier folds' history) and
tests on the next one; and so on, by default 5 folds per horizon, each
scored by a fresh model fit just for that fold and then discarded --
walk-forward validation is checking the *method*, not producing a model
anyone actually uses. If hit rates are consistent fold to fold, the single
split's headline number is real and repeatable; if they swing wildly, that
inconsistency is the whole point of running this at all. Both the
per-fold results and the aggregate are on the scorecard tab and in
`/api/scorecard`'s `walk_forward` key.

This is also the one place in the bootstrap pipeline that deliberately runs
*after* the site is already serving traffic, rather than inline like
everything else. Fitting 5 extra models per horizon roughly doubles a
bootstrap's total time in local testing -- fine for periodic validation,
not something the site's cold-boot response time (see the Render sleep/wake
behavior in `DEPLOY.md`) should ever wait on. `Engine.bootstrap()` kicks it
off on a background thread once `ready=True`, so the very first request
(and every 4-hourly scheduled refresh) is never blocked on it; the
walk-forward numbers just fill in a few/several seconds later.

## What to invest in (ranked, sized picks)

The "What to Invest In" tab (and `/api/recommendations`) turns the day's
448-row forecast table into a short, ranked, sized shortlist -- what to
look at, when, why, and (for buys) how much of a hypothetical portfolio to
put into it. It's implemented in `app/recommend.py` and is **not** a new
signal: every field on a pick traces back to a forecast row `app.forecast`
already produced and a ticker's own backtested track record `app.scoring`
already logged. This module only filters, ranks, and sizes what's already
on the site.

**Ranking ("what" and "when").** Each ticker's forecast confidence is
blended with that *same* ticker's own historical hit rate
(`db.scorecard_by_ticker`) into a conviction score -- a confident call
today on a ticker this model has historically been coin-flip-or-worse on
gets discounted, not taken at face value. Small sample sizes are
Bayesian-shrunk toward 50% first, so a ticker's lucky or unlucky handful of
outcomes can't swing its score wildly. "When" is just the forecast
horizon already in use elsewhere on the site (5/10/20 trading days, etc.).
Expected-move size is deliberately *not* part of the ranking (mixing "how
sure" with "how big" would blur what the rank means) -- it's shown per
pick instead, alongside the target price, so you can weigh reward size
yourself. There's also deliberately no hard confidence/move floor that
could silently return zero picks; every pick is labeled `signal_strength`
(weak/moderate/strong), calibrated against this model's own observed
confidence range, so a weak field of candidates reads as weak rather than
as "no picks".

**Sizing ("how much").** A configured hypothetical portfolio size (set via
the tab's input, or `POST /api/recommendations/portfolio` -- a planning
number stored in the `meta` table, not a real account) is split across the
top buy picks proportional to conviction score, capped per position at
`RECO_MAX_POSITION_PCT` (25% by default) of the investable amount so no
single ticker can dominate regardless of score, with any capped overflow
iteratively redistributed across the remaining picks. `RECO_CASH_RESERVE_PCT`
(10% by default) of the portfolio is always held back as cash, never
allocated. "Short" candidates (the model expects the price to fall) are
ranked and shown with the same rationale/confidence/track-record detail as
buys, but are **never** sized in dollars -- shorting needs a margin/
short-selling account this module has no way to confirm you have.

**Recommend-only, by design.** This module never places an order --
separate from, and with no connection to, the opt-in paper-trading bot
above. It only surfaces ranked ideas and suggested sizing for a human to
read and decide on.

**Three ways to see it,** all reading from the same engine:
1. The **"What to Invest In" tab** on the site.
2. **Ask the chat assistant** ("what should I buy?", "what looks good this
   week?") -- the assistant is told it may cite the picks as the model's
   current output (with the same rationale/confidence/track-record detail
   shown on the tab), never phrased as its own personal advice, consistent
   with the rest of the site's "not a licensed financial advisor" framing.
3. `GET /api/recommendations` directly (optional `horizon`/`portfolio_usd`
   query params override the saved default for that one call without
   changing it).

All the knobs (`STOCKGRAPH_RECO_MAX_PICKS`, `STOCKGRAPH_RECO_DEFAULT_HORIZON`,
`STOCKGRAPH_RECO_MAX_POSITION_PCT`, `STOCKGRAPH_RECO_CASH_RESERVE_PCT`,
`STOCKGRAPH_RECO_MIN_CONFIDENCE`, `STOCKGRAPH_RECO_MIN_MOVE_PCT`,
`STOCKGRAPH_RECO_DEFAULT_PORTFOLIO_USD`) are environment variables -- see
`app/config.py` and `.env.example`.

## Honest limitations

- The forecasting model is intentionally simple and transparent (linear
  models on ~20 hand-built features) rather than a deep model, specifically
  so every prediction can be decomposed into "here's exactly why" -- the
  scorecard tab shows this gets roughly coin-flip-to-slightly-better
  accuracy on directional calls in backtesting, which is in line with how
  hard short-horizon stock prediction actually is. Don't expect more than
  that from it as shipped; the value is the structure (graph relationships,
  explainability, tracked accuracy), not a secret edge.
- The relationship graph is hand-curated from well-known, real supplier/
  customer/competitor/partner relationships -- it's a reasonable starting
  set for ~232 companies, not an exhaustive or auto-updating one.
- News sentiment is a transparent keyword lexicon, not a trained NLP model
  -- easy to audit, easy to extend, not state-of-the-art.
- Demo/synthetic data is clearly labeled but is, by construction, more
  learnable than real markets (the graph pass-through and event jumps are
  baked into how it's generated) -- don't read the demo-mode backtest
  numbers as a claim about real-market performance.
- The insider-transaction signal is a real, published academic effect on
  average across large samples, not a reliable predictor for any single
  company or trade -- most companies have only a handful of open-market
  Form 4 transactions in a given quarter, which is too small a sample to
  read much into on its own. It's one input among ~20+, not a standalone
  "insiders are buying" call.
