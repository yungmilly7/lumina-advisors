/* Lumina Advisors — frontend. Vanilla JS, one file, talks to the /api/* routes. */

// Escapes text before it's interpolated into an innerHTML template string.
// Needed anywhere the value isn't one of our own hardcoded strings (company
// names, driver labels, etc.) -- specifically: user-supplied account fields
// (display_name, email -- the signup validation only checks shape, not
// content, so these can contain arbitrary characters), news headlines
// (pulled from Google News RSS, a third-party feed we don't control), and
// the AI narrative (Claude's free-text output, which can end up echoing
// characters from that same untrusted news text). Without this, any of
// those becomes a stored/reflected XSS vector.
function escapeHtml(str) {
  if (str == null) return "";
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

const state = {
  horizon: 5,
  companies: [],
  companyByTicker: {},
  graph: { nodes: [], edges: [] },
  forecasts: [],
  forecastByTicker: {},
  scorecard: null,
  selectedTicker: null,
  network: null,
  sort: { key: "expected_move_pct", dir: -1 },
  user: null,
  watchlist: new Set(),
  authMode: "login",
  sectorFilter: "All",
  chat: { history: [], sending: false },
  preferences: null,
  sectors: [],
  health: null,
  compare: [null, null],
};

const DRIVER_GLOSSARY = {
  "5-day price momentum": "How much the price has moved over the last 5 trading days — short-term trend.",
  "10-day price momentum": "How much the price has moved over the last 10 trading days.",
  "20-day price momentum": "How much the price has moved over the last 20 trading days — medium-term trend.",
  "60-day price momentum": "How much the price has moved over the last 60 trading days — longer-term trend.",
  "recent volatility": "How much the price has been swinging lately. Higher volatility means wider, less certain outcomes.",
  "RSI (overbought/oversold)": "Relative Strength Index — a 0-100 momentum gauge. Above ~70 is considered 'overbought', below ~30 'oversold'.",
  "distance from 20-day average": "How far the current price sits above or below its own 20-day moving average.",
  "distance from 50-day average": "How far the current price sits above or below its own 50-day moving average.",
  "unusual trading volume": "Whether today's trading volume is abnormally high or low versus its recent norm — a spike often signals news.",
  "news sentiment (7d, decayed)": "Average tone of recent headlines (positive vs negative), weighted more heavily toward the most recent days.",
  "news event frequency (7d)": "How many notable news events (tagged product launches, deals, etc.) hit this company in the last 7 days.",
  "days until next earnings": "Trading days remaining until the company's next scheduled earnings report.",
  "days since last earnings": "Trading days elapsed since the company's most recent earnings report.",
  "last earnings surprise": "How far the last actual EPS came in versus what analysts expected, in percent.",
  "earnings report imminent (<=5d)": "Flag for whether an earnings report is due within the next 5 trading days — a known volatility trigger.",
  "post-earnings drift window": "Stocks often keep drifting in the direction of an earnings surprise for a while afterward; this flags that window.",
  "sector peer momentum": "How this company's sector as a whole has been trending — a rising tide effect.",
  "broad market momentum": "How the overall market has been trending recently.",
  "connected-company momentum spillover": "Momentum flowing in from this company's suppliers, customers, competitors, and partners in the relationship graph.",
  "connected-company news sentiment spillover": "News sentiment flowing in from this company's connected companies in the relationship graph.",
};

const SECTOR_COLORS = {
  "Technology": "#2757c9",
  "Consumer Discretionary": "#c9852f",
  "Communication Services": "#6b3fc9",
  "Materials": "#3f8f6b",
  "Consumer Staples": "#0f9b8e",
  "Financials": "#b5790a",
  "Energy": "#4a4a4a",
  "Industrials": "#8a6d3b",
  "Healthcare": "#c0362c",
  "Utilities": "#2f8fa3",
  "Real Estate": "#a34f8f",
};

function sectorColor(sector) {
  return SECTOR_COLORS[sector] || "#7a8699";
}

// ---------- company autocomplete (used by the topbar jump-search and the
// Compare tab's picker slots) ----------

function attachAutocomplete(inputEl, resultsEl, opts = {}) {
  const excludeTickers = opts.excludeTickers || (() => new Set());
  const onPick = opts.onPick || (() => {});

  function render(query) {
    const q = query.trim().toLowerCase();
    if (!q) {
      resultsEl.classList.add("hidden");
      resultsEl.innerHTML = "";
      return;
    }
    const excluded = excludeTickers();
    const matches = state.companies
      .filter(
        (c) =>
          !excluded.has(c.ticker) &&
          (c.ticker.toLowerCase().includes(q) || c.name.toLowerCase().includes(q))
      )
      .sort((a, b) => {
        const aStarts = a.ticker.toLowerCase().startsWith(q) ? 0 : 1;
        const bStarts = b.ticker.toLowerCase().startsWith(q) ? 0 : 1;
        if (aStarts !== bStarts) return aStarts - bStarts;
        return a.ticker.localeCompare(b.ticker);
      })
      .slice(0, 8);
    if (!matches.length) {
      resultsEl.innerHTML = `<div class="autocomplete-empty">No matching companies</div>`;
      resultsEl.classList.remove("hidden");
      return;
    }
    resultsEl.innerHTML = matches
      .map(
        (c) => `
      <div class="autocomplete-item" data-ticker="${c.ticker}">
        <span class="ac-ticker">${c.ticker}</span>
        <span class="ac-name">${escapeHtml(c.name)}</span>
        <span class="ac-sector">${escapeHtml(c.sector)}</span>
      </div>`
      )
      .join("");
    resultsEl.classList.remove("hidden");
  }

  function pick(ticker) {
    resultsEl.classList.add("hidden");
    resultsEl.innerHTML = "";
    inputEl.value = "";
    onPick(ticker);
  }

  inputEl.addEventListener("input", () => render(inputEl.value));
  inputEl.addEventListener("focus", () => {
    if (inputEl.value.trim()) render(inputEl.value);
  });
  // mousedown (not click) fires before the input's blur hides the list,
  // otherwise blur would remove the results before the click registers.
  resultsEl.addEventListener("mousedown", (e) => {
    const item = e.target.closest(".autocomplete-item");
    if (!item) return;
    e.preventDefault();
    pick(item.dataset.ticker);
  });
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      resultsEl.classList.add("hidden");
      inputEl.blur();
    } else if (e.key === "Enter") {
      const first = resultsEl.querySelector(".autocomplete-item");
      if (first) {
        e.preventDefault();
        pick(first.dataset.ticker);
      }
    }
  });
  document.addEventListener("click", (e) => {
    if (e.target !== inputEl && !resultsEl.contains(e.target)) {
      resultsEl.classList.add("hidden");
    }
  });
}

async function fetchJSON(url, opts) {
  const res = await fetch(url, { credentials: "same-origin", ...opts });
  if (!res.ok) {
    let msg;
    try {
      msg = (await res.json()).error;
    } catch {
      msg = await res.text();
    }
    const err = new Error(msg || `${url} -> ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

async function postJSON(url, body) {
  return fetchJSON(url, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
}

async function deleteJSON(url) {
  return fetchJSON(url, { method: "DELETE" });
}

// ---------- toasts ----------

function toast(message, type = "info") {
  const stack = document.getElementById("toastStack");
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.textContent = message;
  stack.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

function fmtPct(x, digits = 2) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(digits)}%`;
}
function fmtSignedPct(x, digits = 2) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const s = x >= 0 ? "+" : "";
  return `${s}${(x * 100).toFixed(digits)}%`;
}
function fmtPrice(x) {
  if (x === null || x === undefined) return "—";
  return `$${Number(x).toFixed(2)}`;
}
function fmtDate(iso) {
  if (!iso) return "—";
  return iso.slice(0, 10);
}
function fmtMarketCap(v) {
  if (v === null || v === undefined) return "—";
  const n = Number(v);
  if (n >= 1e12) return `$${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `$${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `$${(n / 1e6).toFixed(0)}M`;
  return `$${n.toFixed(0)}`;
}
function fmtRatio(v, digits = 1) {
  if (v === null || v === undefined) return "—";
  return Number(v).toFixed(digits);
}
const RECOMMENDATION_LABELS = {
  strong_buy: "Strong buy", buy: "Buy", hold: "Hold",
  underperform: "Underperform", sell: "Sell",
  strongBuy: "Strong buy", outperform: "Outperform",
};
function fmtRecommendation(v) {
  if (!v) return "—";
  return RECOMMENDATION_LABELS[v] || v.replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase());
}

// ---------- boot ----------

async function init() {
  applyStoredTheme();
  wireStaticUI();
  wireAuthUI();
  wireChatUI();
  wireSurveyUI();
  wireHelpUI();
  wireGlobalSearch();
  await loadSectors();
  await loadCurrentUser();
  await loadAll();
}

// ---------- help modal ----------

function wireHelpUI() {
  const backdrop = document.getElementById("helpModalBackdrop");
  document.getElementById("helpBtn").addEventListener("click", () => backdrop.classList.remove("hidden"));
  document.getElementById("helpModalClose").addEventListener("click", () => backdrop.classList.add("hidden"));
  backdrop.addEventListener("click", (e) => {
    if (e.target === backdrop) backdrop.classList.add("hidden");
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") backdrop.classList.add("hidden");
  });
}

// ---------- global jump-to-company search ----------

function wireGlobalSearch() {
  const input = document.getElementById("globalSearch");
  const results = document.getElementById("globalSearchResults");
  attachAutocomplete(input, results, {
    onPick: (ticker) => {
      switchTab("graph");
      selectTicker(ticker);
    },
  });
}

async function loadSectors() {
  try {
    state.sectors = await fetchJSON("/api/sectors");
  } catch (e) {
    console.error(e);
    state.sectors = [];
  }
}

// ---------- theme ----------

function applyStoredTheme() {
  let stored = null;
  try {
    stored = localStorage.getItem("sg_theme");
  } catch {
    // Private browsing / blocked storage: fall back to OS preference each
    // load, which the CSS already handles via prefers-color-scheme.
  }
  if (stored === "dark" || stored === "light") {
    document.documentElement.setAttribute("data-theme", stored);
  }
  updateThemeButton();
}

function updateThemeButton() {
  const btn = document.getElementById("themeToggleBtn");
  if (!btn) return;
  const current = document.documentElement.getAttribute("data-theme");
  btn.textContent = current === "dark" ? "☀" : current === "light" ? "☾" : "◐";
  btn.title =
    current === "dark"
      ? "Dark theme — click for light"
      : current === "light"
      ? "Light theme — click for system default"
      : "Following system theme — click for dark";
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme");
  // Cycle: system -> dark -> light -> system
  const next = current === "dark" ? "light" : current === "light" ? null : "dark";
  if (next) {
    document.documentElement.setAttribute("data-theme", next);
  } else {
    document.documentElement.removeAttribute("data-theme");
  }
  try {
    if (next) localStorage.setItem("sg_theme", next);
    else localStorage.removeItem("sg_theme");
  } catch {
    // Storage unavailable -- theme just won't persist across reloads.
  }
  updateThemeButton();
}

function wireStaticUI() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });
  document.getElementById("horizonSelect").addEventListener("change", async (e) => {
    state.horizon = Number(e.target.value);
    await loadForecasts();
    renderForecastTable();
    renderWatchlistTab();
    renderOverview();
    renderSectorPerfChart();
    renderConfidenceHistogram();
    renderCompareContent();
    buildNetwork();
    if (state.selectedTicker) selectTicker(state.selectedTicker);
  });
  document.getElementById("refreshBtn").addEventListener("click", onRefresh);
  document.getElementById("themeToggleBtn").addEventListener("click", toggleTheme);
  document.getElementById("forecastSearch").addEventListener("input", () => renderForecastTable());
  document.getElementById("exportCsvBtn").addEventListener("click", exportForecastsCSV);
  document.querySelectorAll("#forecastTable thead th[data-sort]").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.sort;
      if (state.sort.key === key) state.sort.dir *= -1;
      else state.sort = { key, dir: 1 };
      renderForecastTable();
    });
  });

  // Star buttons live inside dynamically-rendered rows/panels, so delegate
  // from a stable ancestor rather than re-binding after every render. This
  // must run in the CAPTURE phase: table rows have their own click handler
  // (open detail panel) attached directly on the <tr>, which is closer to
  // the target than document.body and so fires first during the normal
  // bubble phase -- by the time a bubble-phase listener here called
  // stopPropagation, the row's "open detail" handler would already have
  // run. Capturing at body intercepts the click on the way down, before it
  // ever reaches the row.
  document.body.addEventListener(
    "click",
    (e) => {
      const star = e.target.closest(".star-btn");
      if (!star) return;
      e.stopPropagation();
      toggleWatchlist(star.dataset.ticker);
    },
    true
  );
}

function switchTab(tab) {
  document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${tab}`));
  if (tab === "graph" && state.network) setTimeout(() => state.network._resize(), 50);
  if (tab === "watchlist") renderWatchlistTab();
  if (tab === "forecasts") setTimeout(renderSectorPerfChart, 30);
  if (tab === "compare") setTimeout(renderCompareContent, 30);
  if (tab === "scorecard")
    setTimeout(() => {
      renderConfidenceHistogram();
      renderHitRateChart();
    }, 30);
}

let _chartResizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(_chartResizeTimer);
  _chartResizeTimer = setTimeout(() => {
    const active = document.querySelector(".tab-panel.active");
    if (!active) return;
    if (active.id === "tab-forecasts") renderSectorPerfChart();
    if (active.id === "tab-scorecard") {
      renderConfidenceHistogram();
      renderHitRateChart();
    }
  }, 150);
});

let _loadingTimer = null;
let _loadingStartedAt = 0;

function setLoading(on, text) {
  const el = document.getElementById("loadingBanner");
  el.classList.toggle("hidden", !on);
  if (text) document.getElementById("loadingText").textContent = text;
  document.getElementById("refreshBtn").disabled = on;

  const elapsedEl = document.getElementById("loadingElapsed");
  clearInterval(_loadingTimer);
  _loadingTimer = null;
  if (on) {
    // A full live-mode pipeline run can take well over ten minutes end to
    // end (ingest -> features -> train -> backtest across 200+ tickers).
    // Without visible progress, that looks identical to a frozen page --
    // a ticking "elapsed" counter is the cheapest possible reassurance
    // that work is still happening, and after a couple of minutes we add
    // a note that this is normal for a full live-data run.
    _loadingStartedAt = Date.now();
    if (elapsedEl) elapsedEl.textContent = "";
    _loadingTimer = setInterval(() => {
      const secs = Math.floor((Date.now() - _loadingStartedAt) / 1000);
      const mins = Math.floor(secs / 60);
      const rem = secs % 60;
      const clock = mins > 0 ? `${mins}m ${rem}s` : `${rem}s`;
      let msg = `${clock} elapsed`;
      if (secs >= 90) msg += " — a full live-data run can take 10-20 minutes, this is normal";
      if (elapsedEl) elapsedEl.textContent = msg;
    }, 1000);
  } else if (elapsedEl) {
    elapsedEl.textContent = "";
  }
}

function showLoadError(message, retryFn) {
  const banner = document.getElementById("loadErrorBanner");
  const textEl = document.getElementById("loadErrorText");
  const retryBtn = document.getElementById("loadErrorRetryBtn");
  if (!banner || !textEl || !retryBtn) return;
  textEl.textContent = message;
  banner.classList.remove("hidden");
  retryBtn.onclick = async () => {
    banner.classList.add("hidden");
    await retryFn();
  };
}

function hideLoadError() {
  const banner = document.getElementById("loadErrorBanner");
  if (banner) banner.classList.add("hidden");
}

async function loadAll() {
  setLoading(true, "Loading companies, relationship graph, and forecasts…");
  hideLoadError();
  try {
    const [status, companies, graph, health] = await Promise.all([
      fetchJSON("/api/status"),
      fetchJSON("/api/companies"),
      fetchJSON("/api/graph"),
      fetchJSON("/api/health"),
    ]);
    state.companies = companies;
    state.companyByTicker = Object.fromEntries(companies.map((c) => [c.ticker, c]));
    state.graph = graph;
    state.health = health;
    renderModeBadge(status);
    renderDataHealth();

    renderSectorFilterChips();
    await loadForecasts();
    renderForecastTable();
    renderWatchlistTab();
    renderOverview();
    renderSectorPerfChart();
    renderComparePicker();
    renderCompareContent();
    buildNetwork();

    state.scorecard = await fetchJSON("/api/scorecard");
    renderScorecard();
    renderConfidenceHistogram();
    renderHitRateChart();
  } catch (e) {
    console.error(e);
    document.getElementById("modeBadge").textContent = "load error";
    const friendly =
      e.status === undefined
        ? "Couldn't reach the server — it may still be starting up, or your connection just dropped."
        : `The server reported an error (${e.message}).`;
    showLoadError(friendly, loadAll);
  } finally {
    setLoading(false);
  }
}

async function loadForecasts() {
  const list = await fetchJSON(`/api/forecasts?horizon=${state.horizon}`);
  state.forecasts = list;
  state.forecastByTicker = Object.fromEntries(list.map((f) => [f.ticker, f]));
}

function renderModeBadge(status) {
  const badge = document.getElementById("modeBadge");
  const mode = (status.data_mode_active || "unknown").toLowerCase();
  // "live" and "auto (live=X, demo_fallback=Y)" (partial-failure backfill --
  // see pipeline.py) both mean live data is actually driving the site, even
  // though the latter string contains the substring "demo" as part of
  // "demo_fallback". A naive mode.includes("demo") check used to misread
  // that as fully-synthetic and show "SYNTHETIC DEMO DATA" even at 99%+
  // live coverage -- isLive is the correct signal, so use it everywhere
  // below instead of re-deriving (and getting wrong) an includes("demo") check.
  const isLive = mode.startsWith("live") || mode.startsWith("auto");
  badge.textContent = isLive ? `LIVE DATA (${status.data_mode_active})` : "SYNTHETIC DEMO DATA";
  badge.className = "badge " + (isLive ? "badge-live" : "badge-demo");

  const ing = status.ingestion || {};
  let coverageLine = "";
  if (typeof ing.live_ok === "number" && typeof ing.live_fail === "number" && (ing.live_ok + ing.live_fail) > 0) {
    const total = ing.live_ok + ing.live_fail;
    coverageLine = ing.live_fail > 0
      ? ` · Live coverage: ${ing.live_ok}/${total} companies (${ing.live_fail} unreachable this run, left out rather than faked)`
      : ` · Live coverage: ${ing.live_ok}/${total} companies`;
  }
  badge.title = `Model trained: ${status.trained_at || "n/a"} · Last ingested: ${status.last_ingested_at || "n/a"}${coverageLine}`;

  // Keep the footer disclaimer's data-source sentence in sync with the
  // actual active mode -- this used to be a hardcoded "synthetic demo
  // data" claim that stayed wrong forever once live mode was turned on.
  const note = document.getElementById("disclaimerDataNote");
  if (note) {
    note.textContent = isLive
      ? ` and, right now, on live market data pulled from Yahoo Finance, SEC EDGAR, and Google News${coverageLine}`
      : " and, right now, on synthetic demo data (no live source is configured for this instance)";
  }
}

const HEALTH_FIELD_LABELS = {
  bars: "Price bars",
  earnings: "Earnings",
  fundamentals: "Fundamentals",
  filings: "SEC filings",
  news: "News headlines",
};
const HEALTH_FIELD_ORDER = ["bars", "earnings", "fundamentals", "filings", "news"];
const SOURCE_LABELS = {
  yahoo: "Yahoo Finance",
  finnhub: "Finnhub",
  secedgar: "SEC EDGAR",
  google_news: "Google News",
  demo: "Synthetic demo",
};
const SOURCE_COLORS = {
  yahoo: "var(--up)",
  finnhub: "var(--up)",
  secedgar: "var(--up)",
  google_news: "var(--up)",
  demo: "#d9a300",
};

// Renders the Data Health tab (per-field live/synthetic coverage + recent
// ingestion run history). Reads from state.health, set by loadAll() from
// /api/health -- this is the one place the whole universe's provenance
// gets summarized, rather than one row per company.
function renderDataHealth() {
  const h = state.health;
  const summaryEl = document.getElementById("healthSummary");
  const sourceBody = document.querySelector("#healthSourceTable tbody");
  const runsBody = document.querySelector("#healthRunsTable tbody");
  if (!h || !summaryEl || !sourceBody || !runsBody) return;

  const refreshLine = h.refresh_interval_hours
    ? `Refreshes automatically every ${h.refresh_interval_hours}h`
    : "Automatic background refresh is off for this instance";

  summaryEl.innerHTML = `
    <div class="metric-box"><div class="label">Pipeline</div><div class="value">${h.ready ? "Ready" : "Starting…"}</div></div>
    <div class="metric-box"><div class="label">Active mode</div><div class="value">${escapeHtml(h.data_mode_active || "unknown")}</div></div>
    <div class="metric-box"><div class="label">Universe size</div><div class="value">${h.universe_size} companies</div></div>
    <div class="metric-box"><div class="label">Finnhub (earnings/fundamentals)</div><div class="value">${h.finnhub_configured ? "Configured" : "Not configured"}</div></div>
    <div class="metric-box"><div class="label">Background refresh</div><div class="value" style="font-size:13px;">${refreshLine}</div></div>
    ${h.last_refresh_error ? `<div class="metric-box"><div class="label">Last refresh error</div><div class="value run-error" title="${escapeHtml(h.last_refresh_error)}">${escapeHtml(h.last_refresh_error).slice(0, 60)}</div></div>` : ""}
  `;

  sourceBody.innerHTML = HEALTH_FIELD_ORDER.map((field) => {
    const counts = (h.source_summary && h.source_summary[field]) || {};
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    const liveCount = Object.entries(counts).reduce((a, [src, n]) => a + (src === "demo" ? 0 : n), 0);
    const bar = total
      ? Object.entries(counts)
          .map(([src, n]) => `<span style="width:${(n / total) * 100}%; background:${SOURCE_COLORS[src] || "var(--accent)"};" title="${SOURCE_LABELS[src] || src}: ${n}"></span>`)
          .join("")
      : "";
    const chips = total
      ? Object.entries(counts)
          .sort((a, b) => b[1] - a[1])
          .map(([src, n]) => `<span class="src-badge ${src === "demo" ? "src-badge-demo" : "src-badge-live"}" style="margin-left:0;margin-right:6px;">${SOURCE_LABELS[src] || src} · ${n}</span>`)
          .join("")
      : `<span class="hint">No data ingested yet</span>`;
    return `
      <tr>
        <td>${HEALTH_FIELD_LABELS[field] || field}</td>
        <td style="min-width:140px;">
          <div class="health-source-bar">${bar}</div>
          <div class="hint" style="margin-top:4px;">${total ? `${liveCount}/${total} live` : "—"}</div>
        </td>
        <td>${chips}</td>
      </tr>`;
  }).join("");

  const runs = h.recent_runs || [];
  runsBody.innerHTML = runs.length
    ? runs
        .map((r) => `
      <tr title="${r.error ? escapeHtml(r.error) : ""}">
        <td>${fmtDate(r.started_at)}</td>
        <td>${escapeHtml(r.mode || "—")}</td>
        <td>${r.live_ok ?? "—"}</td>
        <td>${r.live_fail ? `<span class="run-error">${r.live_fail}</span>` : "0"}</td>
        <td>${r.skipped_fresh ?? "—"}</td>
        <td>${r.elapsed_sec != null ? `${r.elapsed_sec}s` : "running…"}</td>
      </tr>`)
        .join("")
    : `<tr><td colspan="6" class="hint">No ingestion runs recorded yet.</td></tr>`;
}

async function onRefresh() {
  setLoading(true, "Re-running the full pipeline (ingest → features → train → backtest)…");
  hideLoadError();
  try {
    await fetchJSON("/api/refresh", { method: "POST" });
    await loadAll();
    toast("Data refreshed.", "success");
  } catch (e) {
    console.error(e);
    toast("Refresh failed: " + e.message, "error");
    showLoadError(`Refresh failed (${e.message}). Your existing data is untouched.`, onRefresh);
  } finally {
    setLoading(false);
  }
}

// ---------- auth ----------

async function loadCurrentUser() {
  try {
    const data = await fetchJSON("/api/auth/me");
    state.user = data.user;
    state.watchlist = new Set(data.watchlist || []);
    state.preferences = data.preferences || null;
  } catch (e) {
    console.error(e);
    state.user = null;
    state.watchlist = new Set();
    state.preferences = null;
  }
  renderAuthArea();
}

function renderAuthArea() {
  const el = document.getElementById("authArea");
  if (!state.user) {
    el.innerHTML = `<button class="btn" id="signInBtn">Sign in</button>`;
    document.getElementById("signInBtn").addEventListener("click", () => openAuthModal("login"));
    return;
  }
  const initial = (state.user.display_name || state.user.email || "?").trim()[0].toUpperCase();
  el.innerHTML = `
    <div class="user-chip" id="userChip">
      <span class="user-avatar">${escapeHtml(initial)}</span>
      <span class="user-chip-name">${escapeHtml(state.user.display_name || state.user.email)}</span>
      <div class="user-menu hidden" id="userMenu">
        <div class="user-menu-email">${escapeHtml(state.user.email)}</div>
        <div class="account-profile-summary">
          <span class="apl">Member since</span><span>${fmtDate(state.user.created_at)}</span>
          <span class="apl">Watchlist</span><span>${state.watchlist.size} ${state.watchlist.size === 1 ? "company" : "companies"}</span>
        </div>
        <button class="user-menu-item" id="surveyMenuBtn">${state.preferences ? "Edit investor profile" : "Set up investor profile"}</button>
        <button class="user-menu-item" id="logoutBtn">Log out</button>
      </div>
    </div>`;
  const chip = document.getElementById("userChip");
  const menu = document.getElementById("userMenu");
  chip.addEventListener("click", (e) => {
    e.stopPropagation();
    menu.classList.toggle("hidden");
  });
  document.addEventListener("click", () => menu.classList.add("hidden"), { once: true });
  document.getElementById("logoutBtn").addEventListener("click", onLogout);
  document.getElementById("surveyMenuBtn").addEventListener("click", () => openSurveyModal());
}

function openAuthModal(mode) {
  state.authMode = mode;
  document.getElementById("authModalBackdrop").classList.remove("hidden");
  document.getElementById("authError").classList.add("hidden");
  document.getElementById("authForm").reset();
  setAuthModalMode(mode);
  document.getElementById("authEmail").focus();
}

function closeAuthModal() {
  document.getElementById("authModalBackdrop").classList.add("hidden");
}

function setAuthModalMode(mode) {
  state.authMode = mode;
  document.querySelectorAll(".modal-tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.authtab === mode));
  document.getElementById("signupNameRow").classList.toggle("hidden", mode !== "signup");
  document.getElementById("authPassword").autocomplete = mode === "signup" ? "new-password" : "current-password";
  document.getElementById("authSubmitBtn").textContent = mode === "signup" ? "Create account" : "Sign in";
  document.getElementById("authError").classList.add("hidden");
}

function wireAuthUI() {
  document.getElementById("authModalClose").addEventListener("click", closeAuthModal);
  document.getElementById("authModalBackdrop").addEventListener("click", (e) => {
    if (e.target.id === "authModalBackdrop") closeAuthModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !document.getElementById("authModalBackdrop").classList.contains("hidden")) {
      closeAuthModal();
    }
  });
  document.querySelectorAll(".modal-tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => setAuthModalMode(btn.dataset.authtab));
  });
  document.getElementById("watchlistSignInBtn").addEventListener("click", () => openAuthModal("login"));

  document.getElementById("authForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const email = document.getElementById("authEmail").value;
    const password = document.getElementById("authPassword").value;
    const displayName = document.getElementById("authDisplayName").value;
    const errEl = document.getElementById("authError");
    const btn = document.getElementById("authSubmitBtn");
    btn.disabled = true;
    errEl.classList.add("hidden");
    try {
      const path = state.authMode === "signup" ? "/api/auth/signup" : "/api/auth/login";
      const payload = state.authMode === "signup" ? { email, password, display_name: displayName } : { email, password };
      const data = await postJSON(path, payload);
      const wasSignup = state.authMode === "signup";
      state.user = data.user;
      state.watchlist = new Set();
      const [wl, meData] = await Promise.all([fetchJSON("/api/watchlist"), fetchJSON("/api/auth/me")]);
      state.watchlist = new Set(wl.watchlist || []);
      state.preferences = meData.preferences || null;
      renderAuthArea();
      closeAuthModal();
      toast(wasSignup ? `Welcome, ${state.user.display_name}!` : "Signed in.", "success");
      // A returning user's forecasts were fetched pre-login (as a guest,
      // no preferences) -- re-fetch now so matches_interest/sort order
      // reflect the profile that just loaded, same reasoning as the
      // survey-save handler above.
      await loadForecasts();
      renderForecastTable();
      renderWatchlistTab();
      if (state.selectedTicker) selectTicker(state.selectedTicker);
      if (wasSignup && !state.preferences) {
        openSurveyModal();
      }
    } catch (e2) {
      errEl.textContent = e2.message;
      errEl.classList.remove("hidden");
    } finally {
      btn.disabled = false;
    }
  });
}

// ---------- investor profile survey ----------

function populateSurveySectors() {
  const el = document.getElementById("surveySectorsList");
  el.innerHTML = state.sectors
    .map((s) => `<label><input type="checkbox" name="sectors" value="${escapeHtml(s)}" /> ${escapeHtml(s)}</label>`)
    .join("");
}

function fillSurveyForm(prefs) {
  const form = document.getElementById("surveyForm");
  form.querySelectorAll('input[type="checkbox"], input[type="radio"]').forEach((el) => (el.checked = false));
  if (!prefs) return;
  (prefs.goals || []).forEach((g) => {
    const el = form.querySelector(`input[name="goals"][value="${g}"]`);
    if (el) el.checked = true;
  });
  (prefs.sectors || []).forEach((s) => {
    const el = form.querySelector(`input[name="sectors"][value="${CSS.escape(s)}"]`);
    if (el) el.checked = true;
  });
  if (prefs.risk_tolerance) {
    const el = form.querySelector(`input[name="risk_tolerance"][value="${prefs.risk_tolerance}"]`);
    if (el) el.checked = true;
  }
  if (prefs.horizon) {
    const el = form.querySelector(`input[name="horizon"][value="${prefs.horizon}"]`);
    if (el) el.checked = true;
  }
  if (prefs.experience) {
    const el = form.querySelector(`input[name="experience"][value="${prefs.experience}"]`);
    if (el) el.checked = true;
  }
}

function openSurveyModal() {
  if (state.sectors.length && !document.getElementById("surveySectorsList").children.length) {
    populateSurveySectors();
  }
  fillSurveyForm(state.preferences);
  document.getElementById("surveyError").classList.add("hidden");
  document.getElementById("surveyModalBackdrop").classList.remove("hidden");
}

function closeSurveyModal() {
  document.getElementById("surveyModalBackdrop").classList.add("hidden");
}

function wireSurveyUI() {
  document.getElementById("surveyModalClose").addEventListener("click", closeSurveyModal);
  document.getElementById("surveySkipBtn").addEventListener("click", closeSurveyModal);
  document.getElementById("surveyModalBackdrop").addEventListener("click", (e) => {
    if (e.target.id === "surveyModalBackdrop") closeSurveyModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !document.getElementById("surveyModalBackdrop").classList.contains("hidden")) {
      closeSurveyModal();
    }
  });
  document.getElementById("surveyForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const goals = Array.from(form.querySelectorAll('input[name="goals"]:checked')).map((el) => el.value);
    const sectors = Array.from(form.querySelectorAll('input[name="sectors"]:checked')).map((el) => el.value);
    const riskEl = form.querySelector('input[name="risk_tolerance"]:checked');
    const horizonEl = form.querySelector('input[name="horizon"]:checked');
    const experienceEl = form.querySelector('input[name="experience"]:checked');
    const btn = document.getElementById("surveySaveBtn");
    const errEl = document.getElementById("surveyError");
    btn.disabled = true;
    errEl.classList.add("hidden");
    try {
      const data = await postJSON("/api/preferences", {
        goals,
        sectors,
        risk_tolerance: riskEl ? riskEl.value : null,
        horizon: horizonEl ? horizonEl.value : null,
        experience: experienceEl ? experienceEl.value : null,
      });
      state.preferences = data.preferences;
      closeSurveyModal();
      renderAuthArea();
      // Re-fetch forecasts: matches_interest and the "sorted toward your
      // interests" ordering are computed server-side from the preferences
      // that were in effect at fetch time, so without this, the table below
      // keeps showing the pre-survey snapshot (no badges, old order) even
      // though the toast promises the list already reflects the new profile.
      await loadForecasts();
      renderForecastTable();
      renderWatchlistTab();
      toast("Investor profile saved — forecasts are now sorted toward what you care about.", "success");
    } catch (e2) {
      errEl.textContent = e2.message || "Couldn't save your profile. Try again.";
      errEl.classList.remove("hidden");
    } finally {
      btn.disabled = false;
    }
  });
}

async function onLogout() {
  try {
    await postJSON("/api/auth/logout");
  } catch (e) {
    console.error(e);
  }
  state.user = null;
  state.watchlist = new Set();
  state.preferences = null;
  renderAuthArea();
  // Same reasoning as login/survey-save: state.forecasts still carries
  // whatever matches_interest/ordering was computed while signed in, so
  // re-fetch as a guest or the "FOR YOU" badges would wrongly survive logout.
  await loadForecasts();
  renderForecastTable();
  renderWatchlistTab();
  if (state.selectedTicker) selectTicker(state.selectedTicker);
  toast("Signed out.");
}

// ---------- watchlist ----------

function starButton(ticker, extraClass = "") {
  const starred = state.watchlist.has(ticker);
  const label = `${starred ? "Remove" : "Add"} ${ticker} ${starred ? "from" : "to"} watchlist`;
  return `<button type="button" class="star-btn ${starred ? "starred" : ""} ${extraClass}" data-ticker="${ticker}" title="${label}" aria-label="${label}" aria-pressed="${starred}">${starred ? "★" : "☆"}</button>`;
}

async function toggleWatchlist(ticker) {
  if (!state.user) {
    openAuthModal("login");
    return;
  }
  const wasStarred = state.watchlist.has(ticker);
  try {
    if (wasStarred) {
      await deleteJSON(`/api/watchlist/${ticker}`);
      state.watchlist.delete(ticker);
    } else {
      await postJSON(`/api/watchlist/${ticker}`);
      state.watchlist.add(ticker);
    }
  } catch (e) {
    toast("Couldn't update watchlist: " + e.message, "error");
    return;
  }
  renderForecastTable();
  renderWatchlistTab();
  if (state.selectedTicker === ticker) selectTicker(ticker);
}

function updateWatchlistCount() {
  const el = document.getElementById("watchlistCount");
  const n = state.watchlist.size;
  el.textContent = String(n);
  el.classList.toggle("hidden", n === 0);
}

function renderWatchlistTab() {
  updateWatchlistCount();
  const signedOut = document.getElementById("watchlistSignedOut");
  const empty = document.getElementById("watchlistEmpty");
  const wrap = document.getElementById("watchlistTableWrap");
  const diversification = document.getElementById("watchlistDiversification");

  if (!state.user) {
    signedOut.classList.remove("hidden");
    empty.classList.add("hidden");
    wrap.classList.add("hidden");
    diversification.classList.add("hidden");
    return;
  }
  signedOut.classList.add("hidden");

  const rows = state.forecasts.filter((f) => state.watchlist.has(f.ticker));
  if (rows.length === 0) {
    empty.classList.remove("hidden");
    wrap.classList.add("hidden");
    diversification.classList.add("hidden");
    return;
  }
  empty.classList.add("hidden");
  wrap.classList.remove("hidden");
  renderDiversificationCheck(rows);

  const tbody = document.getElementById("watchlistTableBody");
  tbody.innerHTML = rows
    .slice()
    .sort((a, b) => b.expected_move_pct - a.expected_move_pct)
    .map(
      (f) => `
    <tr data-ticker="${f.ticker}" class="${f.matches_interest ? "interest-match-row" : ""}">
      <td class="star-cell">${starButton(f.ticker)}</td>
      <td>${liveDot(f)}<strong>${f.ticker}</strong>${earningsBadge(f)}</td>
      <td>${f.name}</td>
      <td>${f.sector}${f.matches_interest ? '<span class="interest-badge" title="Matches your investor profile">FOR YOU</span>' : ""}</td>
      <td class="${f.direction === "up" ? "dir-up" : "dir-down"}">${f.direction === "up" ? "▲ UP" : "▼ DOWN"}</td>
      <td>${fmtPct(f.prob_up, 1)}</td>
      <td class="${f.expected_move_pct >= 0 ? "dir-up" : "dir-down"}">${fmtSignedPct(f.expected_move_pct)}</td>
      <td>${fmtPct(f.confidence, 1)}</td>
      <td>${fmtPrice(f.base_price)}</td>
      <td>${fmtPrice(f.target_price)}</td>
    </tr>`
    )
    .join("");
  tbody.querySelectorAll("tr").forEach((tr) => {
    tr.addEventListener("click", () => {
      selectTicker(tr.dataset.ticker);
      switchTab("graph");
    });
  });
}

// Reads the same relationship graph that feeds the model's spillover
// features (state.graph.edges) to flag when a watchlist is really one bet
// spread across several tickers -- either piled into one sector, or
// directly linked (supplier/customer/competitor/partner) to itself, which
// means those companies' momentum/sentiment can move together rather than
// diversify away risk.
function renderDiversificationCheck(rows) {
  const box = document.getElementById("watchlistDiversification");
  if (!box) return;
  if (rows.length < 2) {
    box.classList.add("hidden");
    return;
  }
  box.classList.remove("hidden");

  const sectorCounts = {};
  rows.forEach((f) => {
    sectorCounts[f.sector] = (sectorCounts[f.sector] || 0) + 1;
  });
  const total = rows.length;
  const sectorEntries = Object.entries(sectorCounts).sort((a, b) => b[1] - a[1]);

  document.getElementById("diversificationSectorBar").innerHTML = sectorEntries
    .map(
      ([sector, count]) =>
        `<span style="width:${(count / total) * 100}%;background:${sectorColor(sector)};" title="${escapeHtml(
          sector
        )}: ${count} of ${total}"></span>`
    )
    .join("");

  document.getElementById("diversificationSectorLegend").innerHTML = sectorEntries
    .map(
      ([sector, count]) =>
        `<span class="legend-item"><span class="legend-swatch-dot" style="background:${sectorColor(
          sector
        )};"></span>${escapeHtml(sector)} (${Math.round((count / total) * 100)}%)</span>`
    )
    .join("");

  const [topSector, topCount] = sectorEntries[0];
  const topShare = topCount / total;
  const sectorMsg =
    topShare >= 0.5
      ? `<div class="diversification-warning">⚠ ${Math.round(topShare * 100)}% of your watchlist is in ${escapeHtml(
          topSector
        )} — concentrated exposure to one part of the market.</div>`
      : `<div class="diversification-ok">✓ No single sector dominates your watchlist.</div>`;

  const tickers = new Set(rows.map((f) => f.ticker));
  const links = (state.graph.edges || []).filter((e) => tickers.has(e.source) && tickers.has(e.target));

  let linksMsg;
  if (links.length) {
    linksMsg =
      `<div class="diversification-links-title">Direct relationships among your picks (${links.length})</div>` +
      links
        .map(
          (e) => `
        <div class="diversification-link-item">
          <strong>${e.source}</strong> ${e.kind === "competitor" ? "vs." : "→"} <strong>${e.target}</strong>
          <span class="dl-kind">${escapeHtml(e.kind)}</span>
          <span class="dl-note">${escapeHtml(e.note || "")}</span>
        </div>`
        )
        .join("") +
      `<div class="diversification-warning">These companies are directly linked in the model's relationship graph — a shock to one can spill over into the other(s) through momentum/sentiment, so they may not move as independently as separate tickers usually would.</div>`;
  } else {
    linksMsg = `<div class="diversification-ok">✓ No direct supplier/customer/competitor/partner links among your starred companies.</div>`;
  }

  document.getElementById("diversificationLinks").innerHTML = sectorMsg + linksMsg;
}

// ---------- compare tab ----------

function renderComparePicker() {
  const row = document.getElementById("comparePickerRow");
  if (!row) return;
  row.innerHTML = "";
  state.compare.forEach((ticker, i) => {
    const slot = document.createElement("div");
    slot.className = "compare-slot";
    if (ticker) {
      const c = state.companyByTicker[ticker];
      slot.innerHTML = `
        <div class="compare-slot-filled">
          <span class="cs-ticker">${ticker}</span>
          <span class="cs-name">${escapeHtml(c ? c.name : "")}</span>
          <button class="compare-slot-remove" aria-label="Remove ${ticker}">×</button>
        </div>`;
      slot.querySelector(".compare-slot-remove").addEventListener("click", () => {
        state.compare.splice(i, 1);
        if (state.compare.length < 2) state.compare.push(null);
        renderComparePicker();
        renderCompareContent();
      });
    } else {
      slot.innerHTML = `
        <input type="text" class="search-box" placeholder="Add a company…" autocomplete="off" />
        <div class="autocomplete-results hidden"></div>`;
      const input = slot.querySelector("input");
      const results = slot.querySelector(".autocomplete-results");
      attachAutocomplete(input, results, {
        excludeTickers: () => new Set(state.compare.filter(Boolean)),
        onPick: (picked) => {
          state.compare[i] = picked;
          renderComparePicker();
          renderCompareContent();
        },
      });
    }
    row.appendChild(slot);
  });
  if (state.compare.length < 4) {
    const addBtn = document.createElement("button");
    addBtn.type = "button";
    addBtn.className = "compare-add-slot-btn";
    addBtn.textContent = "+ Add company";
    addBtn.addEventListener("click", () => {
      state.compare.push(null);
      renderComparePicker();
    });
    row.appendChild(addBtn);
  }
}

function renderCompareContent() {
  const empty = document.getElementById("compareEmpty");
  const content = document.getElementById("compareContent");
  if (!empty || !content) return;
  const tickers = state.compare.filter(Boolean);
  const rows = tickers.map((t) => state.forecastByTicker[t]).filter(Boolean);

  if (rows.length < 2) {
    empty.classList.remove("hidden");
    content.classList.add("hidden");
    return;
  }
  empty.classList.add("hidden");
  content.classList.remove("hidden");

  const chartItems = rows.map((r) => ({
    label: r.ticker,
    value: r.expected_move_pct,
    color: r.expected_move_pct >= 0 ? "#17825a" : "#c0362c",
  }));
  drawDivergingBarChart(document.getElementById("compareMoveCanvas"), chartItems, {
    labelWidth: 70,
    valueWidth: 66,
    formatValue: (v) => fmtSignedPct(v),
  });

  const metricRows = [
    ["Sector", (r) => escapeHtml(r.sector)],
    [
      "Call",
      (r) =>
        `<span class="${r.direction === "up" ? "ct-up" : "ct-down"}">${r.direction === "up" ? "▲ Up" : "▼ Down"}</span>`,
    ],
    ["P(up)", (r) => fmtPct(r.prob_up)],
    [
      "Expected move",
      (r) =>
        `<span class="${r.expected_move_pct >= 0 ? "ct-up" : "ct-down"}">${fmtSignedPct(r.expected_move_pct)}</span>`,
    ],
    ["Confidence", (r) => fmtPct(r.confidence, 1)],
    ["Price", (r) => fmtPrice(r.base_price)],
    ["Target", (r) => fmtPrice(r.target_price)],
    ["Market cap", (r) => fmtMarketCap(r.market_cap)],
    [
      "Data",
      (r) =>
        r.is_live
          ? `<span class="src-badge src-badge-live">Live</span>`
          : `<span class="src-badge src-badge-demo">Demo</span>`,
    ],
    [
      "Top driver",
      (r) => {
        const d = r.drivers && r.drivers[0];
        if (!d) return "—";
        const explainer = DRIVER_GLOSSARY[d.label];
        const title = (explainer ? `${d.label} — ${explainer}` : d.label).replace(/"/g, "&quot;");
        return `<span title="${title}">${escapeHtml(d.label)} (${d.contribution >= 0 ? "+" : ""}${d.contribution.toFixed(
          3
        )})</span>`;
      },
    ],
    [
      "Track record",
      (r) => {
        const tr = r.track_record;
        if (!tr || !tr.n) return `<span class="hint">No history yet</span>`;
        const cls = tr.hit_rate >= 0.5 ? "ct-up" : "ct-down";
        return `<span class="${cls}">${fmtPct(tr.hit_rate, 0)}</span> <span class="hint">(${tr.n} calls)</span>`;
      },
    ],
  ];

  let html = '<thead><tr><th class="ct-metric-label"></th>';
  html += rows
    .map(
      (r) =>
        `<th><span class="ct-ticker-head">${r.ticker}</span><span class="ct-name-sub">${escapeHtml(
          r.name
        )}</span></th>`
    )
    .join("");
  html += "</tr></thead><tbody>";
  metricRows.forEach(([label, fn]) => {
    html += `<tr><td class="ct-metric-label">${label}</td>`;
    html += rows.map((r) => `<td>${fn(r)}</td>`).join("");
    html += "</tr>";
  });
  html += "</tbody>";
  document.getElementById("compareTable").innerHTML = html;
}

// ---------- graph ----------

const EDGE_STYLE = {
  supplier: { color: "#2757c9", dashed: null, arrow: true },
  customer: { color: "#6b3fc9", dashed: null, arrow: true },
  competitor: { color: "#c0362c", dashed: [4, 3], arrow: false },
  partner: { color: "#17825a", dashed: [7, 3], arrow: false },
};

function buildNetwork() {
  const nodes = state.graph.nodes.map((n) => {
    const f = state.forecastByTicker[n.ticker];
    const conf = f ? f.confidence : 0.1;
    const radius = 7 + Math.min(15, conf * 34);
    const dirColor = f ? (f.direction === "up" ? "#17825a" : "#c0362c") : "#8a97a8";
    const title = `<strong>${n.name}</strong><br>${n.sector}` + (f ? `<br>${f.direction.toUpperCase()} ${fmtSignedPct(f.expected_move_pct)} (conf ${fmtPct(f.confidence)})` : "");
    return {
      id: n.ticker,
      label: n.ticker,
      title,
      radius,
      fill: sectorColor(n.sector),
      stroke: dirColor,
      strokeWidth: f ? 2.6 : 1.2,
    };
  });

  const edges = state.graph.edges.map((e) => {
    const st = EDGE_STYLE[e.kind] || { color: "#999", dashed: null, arrow: false };
    return {
      source: e.source,
      target: e.target,
      color: st.color,
      dashed: st.dashed,
      arrow: st.arrow,
      width: 0.6 + e.weight * 2.2,
      weight: e.weight,
      title: `${e.kind}: ${e.note}`,
    };
  });

  const container = document.getElementById("network");
  const canvas = document.getElementById("networkCanvas");
  const tooltip = document.getElementById("networkTooltip");

  if (!state.network) {
    state.network = new ForceGraph(container, canvas, tooltip);
    state.network.onNodeClick = (id) => selectTicker(id);
  }
  state.network.setData(nodes, edges);
  state.network.layout(220);
  state.network.draw();

  renderEdgeLegend();
  renderSectorLegend();
}

function renderEdgeLegend() {
  const el = document.getElementById("edgeLegend");
  const items = [
    ["supplier", "#2757c9", "solid"],
    ["customer", "#6b3fc9", "solid"],
    ["competitor", "#c0362c", "dashed"],
    ["partner", "#17825a", "dashed"],
  ];
  el.innerHTML = items
    .map(
      ([kind, color, style]) =>
        `<span class="legend-item"><span class="legend-swatch" style="background:${color};border-bottom:${style === "dashed" ? "2px dashed " + color : "none"}"></span>${kind}</span>`
    )
    .join("");
}

function renderSectorLegend() {
  const el = document.getElementById("sectorLegend");
  const sectors = Array.from(new Set(state.companies.map((c) => c.sector))).sort();
  el.innerHTML = sectors
    .map(
      (s) =>
        `<span class="legend-item"><span class="legend-swatch-dot" style="background:${sectorColor(s)}"></span>${s}</span>`
    )
    .join("");
}

// ---------- forecast table ----------

function renderForecastTable() {
  const tbody = document.getElementById("forecastTableBody");
  const q = document.getElementById("forecastSearch").value.trim().toLowerCase();
  let rows = state.forecasts.filter((f) => {
    if (state.sectorFilter !== "All" && f.sector !== state.sectorFilter) return false;
    if (!q) return true;
    return (
      f.ticker.toLowerCase().includes(q) ||
      f.name.toLowerCase().includes(q) ||
      f.sector.toLowerCase().includes(q)
    );
  });
  const { key, dir } = state.sort;
  rows = rows.slice().sort((a, b) => {
    const av = a[key], bv = b[key];
    if (typeof av === "string") return dir * av.localeCompare(bv);
    return dir * ((av ?? 0) - (bv ?? 0));
  });

  tbody.innerHTML = rows
    .map(
      (f) => `
    <tr data-ticker="${f.ticker}" class="${f.matches_interest ? "interest-match-row" : ""}">
      <td class="star-cell">${starButton(f.ticker)}</td>
      <td>${liveDot(f)}<strong>${f.ticker}</strong>${earningsBadge(f)}</td>
      <td>${f.name}</td>
      <td>${f.sector}${f.matches_interest ? '<span class="interest-badge" title="Matches your investor profile">FOR YOU</span>' : ""}</td>
      <td class="${f.direction === "up" ? "dir-up" : "dir-down"}">${f.direction === "up" ? "▲ UP" : "▼ DOWN"}</td>
      <td>${fmtPct(f.prob_up, 1)}</td>
      <td class="${f.expected_move_pct >= 0 ? "dir-up" : "dir-down"}">${fmtSignedPct(f.expected_move_pct)}</td>
      <td>${fmtPct(f.confidence, 1)}</td>
      <td>${fmtPrice(f.base_price)}</td>
      <td>${fmtPrice(f.target_price)}</td>
    </tr>`
    )
    .join("");

  tbody.querySelectorAll("tr").forEach((tr) => {
    tr.addEventListener("click", () => {
      selectTicker(tr.dataset.ticker);
      switchTab("graph");
    });
  });
}

// Small inline pill next to a ticker when its next earnings report is
// within 5 days -- a known volatility trigger, and the main place Danny's
// "day-specific events" idea is visible at a glance across the whole list
// (the detail panel already shows the full earnings history/date).
function earningsBadge(f) {
  if (f.days_to_earnings == null || f.days_to_earnings > 5) return "";
  const d = f.days_to_earnings;
  const label = d === 0 ? "today" : `${d}d`;
  return `<span class="earnings-badge" title="Earnings report in ${d} trading day${d === 1 ? "" : "s"}">📅 ${label}</span>`;
}

// Per-company live-vs-synthetic indicator for the forecasts/watchlist
// tables -- a quiet dot rather than a loud badge on every row, since most
// rows will be one or the other consistently and the Data Health tab has
// the full per-field breakdown for anyone who wants it.
function liveDot(f) {
  const live = !!f.is_live;
  const n = f.live_field_count ?? 0;
  const title = live
    ? `${n}/5 data fields for ${f.ticker} are from a live source (see Data Health tab)`
    : `All data fields for ${f.ticker} are synthetic demo data (see Data Health tab)`;
  return `<span class="src-dot ${live ? "src-dot-live" : "src-dot-demo"}" title="${title}"></span>`;
}

function renderSectorFilterChips() {
  const el = document.getElementById("sectorFilterRow");
  const sectors = Array.from(new Set(state.companies.map((c) => c.sector))).sort();
  const all = ["All", ...sectors];
  el.innerHTML = all
    .map((s) => `<button type="button" class="filter-chip ${s === state.sectorFilter ? "active" : ""}" data-sector="${s}">${s}</button>`)
    .join("");
  el.querySelectorAll(".filter-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      state.sectorFilter = chip.dataset.sector;
      renderSectorFilterChips();
      renderForecastTable();
    });
  });
}

function exportForecastsCSV() {
  const q = document.getElementById("forecastSearch").value.trim().toLowerCase();
  const rows = state.forecasts.filter((f) => {
    if (state.sectorFilter !== "All" && f.sector !== state.sectorFilter) return false;
    if (!q) return true;
    return (
      f.ticker.toLowerCase().includes(q) ||
      f.name.toLowerCase().includes(q) ||
      f.sector.toLowerCase().includes(q)
    );
  });
  const header = ["Ticker", "Company", "Sector", "Call", "P(up)", "Expected move %", "Confidence", "Price", "Target"];
  const csvRows = rows.map((f) => [
    f.ticker,
    `"${f.name.replace(/"/g, '""')}"`,
    f.sector,
    f.direction,
    (f.prob_up * 100).toFixed(1),
    (f.expected_move_pct * 100).toFixed(2),
    (f.confidence * 100).toFixed(1),
    f.base_price != null ? f.base_price.toFixed(2) : "",
    f.target_price != null ? f.target_price.toFixed(2) : "",
  ]);
  const csv = [header, ...csvRows].map((r) => r.join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `stockgraph-forecasts-${state.horizon}d-${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  toast(`Exported ${rows.length} rows.`, "success");
}

// ---------- market overview ----------

function renderOverview() {
  const rows = state.forecasts;
  const statsEl = document.getElementById("overviewStats");
  const gainersEl = document.getElementById("topGainers");
  const declinersEl = document.getElementById("topDecliners");
  if (!rows.length) {
    statsEl.innerHTML = "";
    gainersEl.innerHTML = "";
    declinersEl.innerHTML = "";
    return;
  }

  const upCount = rows.filter((f) => f.direction === "up").length;
  const downCount = rows.length - upCount;
  const avgConfidence = rows.reduce((s, f) => s + f.confidence, 0) / rows.length;
  const avgMove = rows.reduce((s, f) => s + Math.abs(f.expected_move_pct), 0) / rows.length;
  const upPct = (upCount / rows.length) * 100;

  statsEl.innerHTML = `
    <div class="stat-tile">
      <div class="stat-label">Market breadth</div>
      <div class="stat-value">${upCount} <span class="dir-up">▲</span> / ${downCount} <span class="dir-down">▼</span></div>
      <div class="stat-sub">${upPct.toFixed(0)}% of ${rows.length} companies called UP</div>
    </div>
    <div class="stat-tile">
      <div class="stat-label">Avg. confidence</div>
      <div class="stat-value">${fmtPct(avgConfidence, 1)}</div>
      <div class="stat-sub">${state.horizon}-day horizon</div>
    </div>
    <div class="stat-tile">
      <div class="stat-label">Avg. |expected move|</div>
      <div class="stat-value">${fmtPct(avgMove, 2)}</div>
      <div class="stat-sub">across all ${rows.length} companies</div>
    </div>
    <div class="stat-tile">
      <div class="stat-label">Companies tracked</div>
      <div class="stat-value">${state.companies.length}</div>
      <div class="stat-sub">${state.graph.edges.length} relationships</div>
    </div>`;

  const sorted = rows.slice().sort((a, b) => b.expected_move_pct - a.expected_move_pct);
  const gainers = sorted.slice(0, 5);
  const decliners = sorted.slice(-5).reverse();

  const moverRow = (f) => `
    <div class="mover-row" data-ticker="${f.ticker}">
      <span class="mover-ticker">${f.ticker}</span>
      <span class="mover-name">${f.name}</span>
      <span class="mover-move ${f.expected_move_pct >= 0 ? "dir-up" : "dir-down"}">${fmtSignedPct(f.expected_move_pct)}</span>
    </div>`;

  gainersEl.innerHTML = gainers.map(moverRow).join("");
  declinersEl.innerHTML = decliners.map(moverRow).join("");

  [gainersEl, declinersEl].forEach((el) => {
    el.querySelectorAll(".mover-row").forEach((row) => {
      row.addEventListener("click", () => {
        selectTicker(row.dataset.ticker);
        switchTab("graph");
      });
    });
  });
}

// ---------- charts (plain canvas, no library) ----------

function _chartWidth(canvas, fallback = 480) {
  // clientWidth includes the parent's own left/right padding, but the
  // canvas renders inside that padding (a normal block child), not over
  // it -- sizing the canvas to the raw clientWidth makes it a few px
  // wider than the space actually available and pushes the page into
  // horizontal overflow on narrow (mobile) viewports. Subtract the
  // parent's padding to get the true content width.
  const parent = canvas.parentElement;
  if (!parent) return fallback;
  const cs = getComputedStyle(parent);
  const hPadding = (parseFloat(cs.paddingLeft) || 0) + (parseFloat(cs.paddingRight) || 0);
  const width = parent.clientWidth - hPadding;
  return width > 0 ? width : fallback;
}

function drawDivergingBarChart(canvas, items, opts = {}) {
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = _chartWidth(canvas);
  const rowH = opts.rowHeight || 24;
  const cssHeight = Math.max(40, items.length * rowH + 16);
  canvas.style.width = cssWidth + "px";
  canvas.style.height = cssHeight + "px";
  canvas.width = Math.max(1, cssWidth * dpr);
  canvas.height = Math.max(1, cssHeight * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);
  if (!items.length) return;

  const labelWidth = opts.labelWidth || 150;
  const valueWidth = opts.valueWidth || 60;
  const chartWidth = Math.max(40, cssWidth - labelWidth - valueWidth);
  const midX = labelWidth + chartWidth / 2;
  const halfWidth = chartWidth / 2;
  const maxAbs = Math.max(1e-6, ...items.map((it) => Math.abs(it.value)));

  ctx.font = "12px system-ui, -apple-system, sans-serif";
  ctx.textBaseline = "middle";

  items.forEach((it, i) => {
    const y = 8 + i * rowH + rowH / 2;
    ctx.fillStyle = opts.labelColor || "#5b6b7c";
    ctx.textAlign = "right";
    let label = it.label;
    ctx.fillText(label, labelWidth - 10, y, labelWidth - 14);

    const barLen = (Math.abs(it.value) / maxAbs) * (halfWidth - 48);
    const barColor = it.color || (it.value >= 0 ? "#17825a" : "#c0362c");
    ctx.fillStyle = barColor;
    const barH = rowH - 10;
    if (it.value >= 0) {
      ctx.fillRect(midX, y - barH / 2, barLen, barH);
    } else {
      ctx.fillRect(midX - barLen, y - barH / 2, barLen, barH);
    }

    ctx.fillStyle = opts.labelColor || "#5b6b7c";
    const valueText = opts.formatValue ? opts.formatValue(it.value) : String(it.value);
    if (it.value >= 0) {
      ctx.textAlign = "left";
      ctx.fillText(valueText, midX + barLen + 6, y);
    } else {
      ctx.textAlign = "right";
      ctx.fillText(valueText, midX - barLen - 6, y);
    }
  });

  ctx.strokeStyle = opts.axisColor || "#c7cdd6";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(midX + 0.5, 4);
  ctx.lineTo(midX + 0.5, cssHeight - 4);
  ctx.stroke();
}

function drawVerticalBarChart(canvas, items, opts = {}) {
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = _chartWidth(canvas);
  const cssHeight = opts.height || 170;
  canvas.style.width = cssWidth + "px";
  canvas.style.height = cssHeight + "px";
  canvas.width = Math.max(1, cssWidth * dpr);
  canvas.height = Math.max(1, cssHeight * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);
  if (!items.length) return;

  const padBottom = 30, padTop = 22, padSide = 8;
  const plotH = cssHeight - padBottom - padTop;
  const maxVal = Math.max(1, ...items.map((it) => it.value));
  const n = items.length;
  const gap = 8;
  const barW = Math.max(4, (cssWidth - padSide * 2 - gap * (n - 1)) / n);

  ctx.font = "11px system-ui, -apple-system, sans-serif";
  ctx.textAlign = "center";

  items.forEach((it, i) => {
    const x = padSide + i * (barW + gap);
    const h = maxVal > 0 ? (it.value / maxVal) * plotH : 0;
    const y = padTop + (plotH - h);
    ctx.fillStyle = it.color || opts.barColor || "#2757c9";
    ctx.fillRect(x, y, barW, Math.max(h, it.value > 0 ? 2 : 0));

    const valueLabel = String(it.value);
    if (y - 6 >= padTop) {
      // room above the bar
      ctx.fillStyle = opts.labelColor || "#5b6b7c";
      ctx.textBaseline = "alphabetic";
      ctx.fillText(valueLabel, x + barW / 2, y - 6);
    } else {
      // bar reaches (near) the top -- draw the value inside it, in white
      ctx.fillStyle = "#ffffff";
      ctx.textBaseline = "top";
      ctx.fillText(valueLabel, x + barW / 2, y + 5);
    }

    ctx.fillStyle = opts.labelColor || "#5b6b7c";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(it.label, x + barW / 2, cssHeight - padBottom + 14);
  });
}

function renderSectorPerfChart() {
  const canvas = document.getElementById("sectorPerfCanvas");
  if (!canvas) return;
  const rows = state.forecasts;
  if (!rows.length) return;
  const bySector = {};
  rows.forEach((f) => {
    (bySector[f.sector] = bySector[f.sector] || []).push(f.expected_move_pct);
  });
  const items = Object.entries(bySector)
    .map(([sector, moves]) => ({
      label: sector,
      value: moves.reduce((s, v) => s + v, 0) / moves.length,
      color: sectorColor(sector),
    }))
    .sort((a, b) => b.value - a.value);
  drawDivergingBarChart(canvas, items, { formatValue: (v) => fmtSignedPct(v, 2) });
}

function renderConfidenceHistogram() {
  const canvas = document.getElementById("confidenceHistCanvas");
  if (!canvas) return;
  const rows = state.forecasts;
  if (!rows.length) return;
  const labels = ["0-5%", "5-10%", "10-15%", "15-20%", "20-25%", "25%+"];
  const buckets = labels.map(() => 0);
  rows.forEach((f) => {
    const pct = (f.confidence || 0) * 100;
    let idx = Math.floor(pct / 5);
    if (idx < 0) idx = 0;
    if (idx > 5) idx = 5;
    buckets[idx]++;
  });
  const items = labels.map((label, i) => ({ label, value: buckets[i] }));
  drawVerticalBarChart(canvas, items, { barColor: "#2757c9" });
}

function renderHitRateChart() {
  const canvas = document.getElementById("hitRateCanvas");
  if (!canvas || !state.scorecard) return;
  const rows = state.scorecard.by_horizon || [];
  if (!rows.length) return;
  const items = rows.map((r) => ({
    label: `${r.horizon_days}d`,
    value: r.hit_rate != null ? Math.round(r.hit_rate * 1000) / 10 : 0,
    color: r.hit_rate != null && r.hit_rate >= 0.5 ? "#17825a" : "#c0362c",
  }));
  drawVerticalBarChart(canvas, items, { height: 150 });
}

// ---------- detail panel ----------

async function selectTicker(ticker) {
  state.selectedTicker = ticker;
  updateChatScopeLabel();
  const panel = document.getElementById("detailPanel");
  panel.innerHTML = `<div class="detail-empty">Loading ${ticker}…</div>`;

  if (state.network) {
    state.network.focusNode(ticker);
  }

  try {
    const d = await fetchJSON(`/api/forecast/${ticker}?horizon=${state.horizon}&narrative=1`);
    renderDetail(d);
  } catch (e) {
    panel.innerHTML = `<div class="detail-empty">Failed to load ${ticker}: ${e.message}</div>`;
  }
}

function sparklineSVG(history) {
  if (!history || history.length < 2) return "";
  const w = 320, h = 70, pad = 4;
  const closes = history.map((p) => p.close);
  const min = Math.min(...closes), max = Math.max(...closes);
  const range = max - min || 1;
  const step = (w - pad * 2) / (closes.length - 1);
  const pts = closes.map((c, i) => {
    const x = pad + i * step;
    const y = pad + (1 - (c - min) / range) * (h - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const last = closes[closes.length - 1];
  const first = closes[0];
  const color = last >= first ? "#17825a" : "#c0362c";
  const areaPts = `${pad},${h - pad} ` + pts.join(" ") + ` ${w - pad},${h - pad}`;
  return `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" preserveAspectRatio="none">
    <polyline points="${areaPts}" fill="${color}22" stroke="none"></polyline>
    <polyline points="${pts.join(" ")}" fill="none" stroke="${color}" stroke-width="1.6"></polyline>
  </svg>`;
}

function driverBar(d) {
  const magnitude = Math.min(1, Math.abs(d.contribution) / 0.5);
  const cls = d.contribution >= 0 ? "pos" : "neg";
  const widthPct = (magnitude * 50).toFixed(1);
  const style = cls === "pos" ? `width:${widthPct}%;` : `width:${widthPct}%;`;
  const explainer = DRIVER_GLOSSARY[d.label];
  const tooltip = explainer ? `${d.label} — ${explainer}` : d.label;
  return `
    <div class="driver-row">
      <div class="driver-label" title="${tooltip.replace(/"/g, "&quot;")}">${d.label}${explainer ? '<span class="info-dot">ⓘ</span>' : ""}</div>
      <div class="driver-bar-track"><div class="driver-bar-fill ${cls}" style="${style}"></div></div>
      <div class="driver-val">${d.contribution >= 0 ? "+" : ""}${d.contribution.toFixed(3)}</div>
    </div>`;
}

function sentimentPill(s) {
  if (s > 0.15) return `<span class="sent-pill sent-pos">+${s.toFixed(2)}</span>`;
  if (s < -0.15) return `<span class="sent-pill sent-neg">${s.toFixed(2)}</span>`;
  return `<span class="sent-pill sent-neu">${s.toFixed(2)}</span>`;
}

function fmtVolume(v) {
  if (v === null || v === undefined) return "—";
  if (v >= 1e9) return (v / 1e9).toFixed(2) + "B";
  if (v >= 1e6) return (v / 1e6).toFixed(2) + "M";
  if (v >= 1e3) return (v / 1e3).toFixed(1) + "K";
  return String(Math.round(v));
}

function detailDataGridHTML(d) {
  const items = [];
  if (d.wk52_high != null) items.push(["52-week high", fmtPrice(d.wk52_high)]);
  if (d.wk52_low != null) items.push(["52-week low", fmtPrice(d.wk52_low)]);
  if (d.pct_off_52wk_high != null) items.push(["Off 52wk high", fmtSignedPct(d.pct_off_52wk_high, 1)]);
  if (d.pct_off_52wk_low != null) items.push(["Off 52wk low", fmtSignedPct(d.pct_off_52wk_low, 1)]);
  if (d.latest_volume != null) items.push(["Latest volume", fmtVolume(d.latest_volume)]);
  if (d.avg_volume_30d != null) items.push(["Avg. volume (30d)", fmtVolume(d.avg_volume_30d)]);
  if (d.volume_ratio != null) items.push(["Volume vs. 30d avg", `${d.volume_ratio.toFixed(2)}×`]);
  if (!items.length) return "";
  return `<div class="detail-data-grid">${items
    .map(([label, value]) => `<div class="dd-item"><span class="dd-label">${label}</span><span class="dd-value">${value}</span></div>`)
    .join("")}</div>`;
}

function fundamentalsHTML(fd) {
  if (!fd) return "";
  const items = [];
  if (fd.market_cap != null) items.push(["Market cap", fmtMarketCap(fd.market_cap)]);
  if (fd.pe_ratio != null) items.push(["P/E", fmtRatio(fd.pe_ratio)]);
  if (fd.forward_pe != null) items.push(["Forward P/E", fmtRatio(fd.forward_pe)]);
  if (fd.peg_ratio != null) items.push(["PEG", fmtRatio(fd.peg_ratio, 2)]);
  if (fd.dividend_yield != null) items.push(["Dividend yield", fmtPct(fd.dividend_yield, 2)]);
  if (fd.beta != null) items.push(["Beta", fmtRatio(fd.beta, 2)]);
  if (fd.profit_margin != null) items.push(["Profit margin", fmtPct(fd.profit_margin, 1)]);
  if (fd.revenue_growth != null) items.push(["Revenue growth (YoY)", fmtSignedPct(fd.revenue_growth, 1)]);
  if (fd.analyst_target_mean != null) items.push(["Analyst target (avg)", fmtPrice(fd.analyst_target_mean)]);
  if (fd.analyst_target_low != null && fd.analyst_target_high != null) {
    items.push(["Analyst range", `${fmtPrice(fd.analyst_target_low)} – ${fmtPrice(fd.analyst_target_high)}`]);
  }
  if (fd.analyst_recommendation) {
    const label = fd.num_analyst_opinions
      ? `${fmtRecommendation(fd.analyst_recommendation)} (${fd.num_analyst_opinions} analysts)`
      : fmtRecommendation(fd.analyst_recommendation);
    items.push(["Analyst consensus", label]);
  }
  const gridHTML = items.length
    ? `<div class="detail-data-grid">${items
        .map(([label, value]) => `<div class="dd-item"><span class="dd-label">${label}</span><span class="dd-value">${value}</span></div>`)
        .join("")}</div>`
    : "";
  const descHTML = fd.description ? `<p class="fundamentals-desc">${escapeHtml(fd.description)}</p>` : "";
  if (!gridHTML && !descHTML) return "";
  return `<div class="fundamentals-box">${descHTML}${gridHTML}</div>`;
}

// Small "live · yahoo" / "demo" tag next to a detail-panel section title,
// reading straight from forecast_detail's per-field data_sources -- the
// per-company, per-field replacement for one global "SYNTHETIC DEMO DATA"
// banner (see renderModeBadge above, which still covers the whole-site
// summary case).
// This company's own walk-forward track record (see db.get_ticker_scorecard /
// forecast_detail's "track_record" field) -- grounds the current call in how
// the model has actually done on this specific ticker before, rather than
// only the site-wide scorecard aggregate.
function trackRecordHTML(tr) {
  if (!tr || !tr.n) {
    return `<p class="hint track-record-hint">No track record yet for this company — not enough backtest history to score it.</p>`;
  }
  const cls = tr.hit_rate >= 0.5 ? "ct-up" : "ct-down";
  return `<p class="hint track-record-hint">Track record: <span class="${cls}">${fmtPct(tr.hit_rate, 0)} direction accuracy</span> over ${tr.n} past model calls on this company (avg error ${fmtPct(tr.mae, 2)}) — from the walk-forward backtest, not a guarantee.</p>`;
}

function sourceBadge(d, field) {
  const entry = d.data_sources && d.data_sources[field];
  if (!entry || !entry.source) return "";
  const isDemo = entry.source === "demo";
  const label = SOURCE_LABELS[entry.source] || entry.source;
  const title = entry.updated_at ? `Last updated ${fmtDate(entry.updated_at)}` : "";
  return `<span class="src-badge ${isDemo ? "src-badge-demo" : "src-badge-live"}" title="${title}">${isDemo ? "Demo" : `Live · ${label}`}</span>`;
}

function renderDetail(d) {
  const panel = document.getElementById("detailPanel");
  const dirClass = d.direction === "up" ? "call-up" : "call-down";
  const dirArrow = d.direction === "up" ? "▲" : "▼";

  const drivers = d.drivers.slice(0, 6).map(driverBar).join("");

  const neighborsHTML = d.neighbors
    .sort((a, b) => b.weight - a.weight)
    .map(
      (n) =>
        `<span class="chip chip-${n.kind}" title="${n.kind}: ${n.note}" data-ticker="${n.ticker}">${n.ticker} · ${n.kind}</span>`
    )
    .join("");

  const newsHTML =
    d.news
      .slice(0, 8)
      .map(
        (n) => `
      <div class="news-item">
        <div class="news-headline">${escapeHtml(n.headline)}</div>
        <div class="news-meta">
          ${sentimentPill(n.sentiment || 0)}
          <span>${fmtDate(n.published)}</span>
          <span>${escapeHtml(n.source || "")}</span>
          ${(n.event_tags || []).map((t) => `<span>#${escapeHtml(t)}</span>`).join(" ")}
        </div>
      </div>`
      )
      .join("") || `<div class="hint">No recent headlines.</div>`;

  const nextEarnings = d.earnings.find((e) => e.is_future);
  const lastEarnings = [...d.earnings].reverse().find((e) => !e.is_future);
  const earningsHTML = `
    ${nextEarnings ? `<div class="earn-item"><span>Next report</span><strong>${fmtDate(nextEarnings.report_date)}</strong></div>` : ""}
    ${lastEarnings ? `<div class="earn-item"><span>Last surprise</span><strong>${lastEarnings.surprise_pct >= 0 ? "+" : ""}${(lastEarnings.surprise_pct ?? 0).toFixed(1)}%</strong></div>` : ""}
  `;

  const filingsHTML =
    d.filings
      .slice(0, 6)
      .map((f) => `<div class="filing-item"><span>${f.form_type}</span><span>${fmtDate(f.filed_date)}</span></div>`)
      .join("") || `<div class="hint">No recent filings.</div>`;

  const insiderHTML =
    (d.insider_transactions || [])
      .slice(0, 8)
      .map((i) => {
        const isBuy = i.transaction_code === "P";
        return `<div class="filing-item"><span class="${isBuy ? "dir-up" : "dir-down"}">${isBuy ? "Buy" : "Sell"} · ${i.owner_name}</span><span>${fmtDate(i.transaction_date)} · $${Math.round(i.value_usd).toLocaleString()}</span></div>`;
      })
      .join("") || `<div class="hint">No open-market insider trades in recent filings.</div>`;

  panel.innerHTML = `
    <div class="detail-header">
      <h3>${d.name} <span style="color:var(--text-muted); font-weight:500;">(${d.ticker})</span></h3>
      ${starButton(d.ticker, "detail-star")}
    </div>
    <p class="detail-sub">${d.sector} · ${d.industry}</p>

    <div class="sparkline-wrap">${sparklineSVG(d.price_history)}</div>

    <div class="call-row">
      <span class="call-badge ${dirClass}">${dirArrow} ${d.direction.toUpperCase()}</span>
      <span class="call-meta">P(up) ${fmtPct(d.prob_up, 1)} · confidence ${fmtPct(d.confidence, 1)} · ${d.horizon_days}-day horizon</span>
    </div>

    <div class="metric-grid">
      <div class="metric-box"><div class="label">Price</div><div class="value">${fmtPrice(d.base_price)}</div></div>
      <div class="metric-box"><div class="label">Target</div><div class="value">${fmtPrice(d.target_price)}</div></div>
      <div class="metric-box"><div class="label">Expected move</div><div class="value ${d.expected_move_pct >= 0 ? "dir-up" : "dir-down"}">${fmtSignedPct(d.expected_move_pct)}</div></div>
    </div>
    <p class="hint" style="margin-top:-8px;margin-bottom:6px;">~80% range: ${fmtSignedPct(d.low_pct)} to ${fmtSignedPct(d.high_pct)}</p>
    ${trackRecordHTML(d.track_record)}

    ${detailDataGridHTML(d)}

    <div class="rationale-box">${d.rationale}</div>
    ${d.llm_narrative ? `<div class="narrative-box"><div class="narrative-label">Claude's read</div>${escapeHtml(d.llm_narrative)}</div>` : ""}

    <div class="section-title">Company snapshot${sourceBadge(d, "fundamentals")}</div>
    ${fundamentalsHTML(d.fundamentals) || `<p class="hint">No additional company data available yet.</p>`}

    <div class="section-title">Top model drivers</div>
    ${drivers}

    <div class="section-title">Connected companies (${d.neighbors.length})</div>
    <div class="chip-row">${neighborsHTML}</div>

    <div class="section-title">Earnings${sourceBadge(d, "earnings")}</div>
    ${earningsHTML}

    <div class="section-title">Recent filings${sourceBadge(d, "filings")}</div>
    ${filingsHTML}

    <div class="section-title">Insider transactions (open-market)</div>
    ${insiderHTML}

    <div class="section-title">Recent headlines${sourceBadge(d, "news")}</div>
    ${newsHTML}
  `;

  panel.querySelectorAll(".chip[data-ticker]").forEach((chip) => {
    chip.addEventListener("click", () => selectTicker(chip.dataset.ticker));
  });
}

// ---------- scorecard ----------

function renderScorecard() {
  const sc = state.scorecard;
  if (!sc) return;
  const overall = sc.overall || {};
  const summaryEl = document.getElementById("scorecardSummary");
  summaryEl.innerHTML = `
    <div class="score-card"><div class="label">Predictions scored</div><div class="value">${overall.n ?? 0}</div></div>
    <div class="score-card"><div class="label">Direction hit rate</div><div class="value">${overall.hit_rate != null ? fmtPct(overall.hit_rate, 1) : "—"}</div></div>
    <div class="score-card"><div class="label">Mean abs. error</div><div class="value">${overall.mae != null ? fmtPct(overall.mae, 2) : "—"}</div></div>
  `;

  const hBody = document.querySelector("#scByHorizon tbody");
  hBody.innerHTML = (sc.by_horizon || [])
    .map(
      (r) =>
        `<tr><td>${r.horizon_days}d</td><td>${r.n}</td><td>${fmtPct(r.hit_rate, 1)}</td><td>${fmtPct(r.mae, 2)}</td></tr>`
    )
    .join("");

  const tBody = document.querySelector("#scByTicker tbody");
  tBody.innerHTML = (sc.by_ticker || [])
    .slice(0, 20)
    .map(
      (r) =>
        `<tr data-ticker="${r.ticker}"><td><strong>${r.ticker}</strong></td><td>${r.n}</td><td>${fmtPct(r.hit_rate, 1)}</td><td>${fmtPct(r.mae, 2)}</td></tr>`
    )
    .join("");
  tBody.querySelectorAll("tr").forEach((tr) => {
    tr.addEventListener("click", () => {
      switchTab("graph");
      selectTicker(tr.dataset.ticker);
    });
  });
}

// ---------- chat widget ----------

function updateChatScopeLabel() {
  const el = document.getElementById("chatScopeLabel");
  if (!el) return;
  if (state.selectedTicker && state.companyByTicker[state.selectedTicker]) {
    el.textContent = `Scoped to ${state.selectedTicker} · ${state.companyByTicker[state.selectedTicker].name}`;
  } else {
    el.textContent = "Whole market";
  }
}

function wireChatUI() {
  const fab = document.getElementById("chatFab");
  const panel = document.getElementById("chatPanel");
  const closeBtn = document.getElementById("chatCloseBtn");
  const form = document.getElementById("chatForm");
  const input = document.getElementById("chatInput");
  if (!fab || !panel || !form) return;

  const closeChat = () => {
    panel.classList.add("hidden");
    fab.focus();
  };

  fab.addEventListener("click", () => {
    panel.classList.remove("hidden");
    updateChatScopeLabel();
    input.focus();
  });
  closeBtn.addEventListener("click", closeChat);

  // Esc closes the panel like any other overlay on the site (the auth
  // modal already works this way) -- without this, keyboard users have no
  // way to dismiss it short of clicking the small × button.
  panel.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeChat();
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text || state.chat.sending) return;
    input.value = "";
    await sendChatMessage(text);
  });
}

function appendChatMessage(role, text) {
  const list = document.getElementById("chatMessages");
  const el = document.createElement("div");
  el.className = `chat-msg chat-msg-${role}`;
  el.textContent = text;
  list.appendChild(el);
  list.scrollTop = list.scrollHeight;
  return el;
}

async function sendChatMessage(message) {
  appendChatMessage("user", message);
  state.chat.sending = true;
  const sendBtn = document.getElementById("chatSendBtn");
  const input = document.getElementById("chatInput");
  sendBtn.disabled = true;
  // Disabling the input too (not just the send button) stops someone from
  // queuing up a second question, with its own answer landing out of
  // order, while the first is still in flight.
  if (input) input.disabled = true;
  const pending = appendChatMessage("pending", "Thinking…");

  try {
    const body = {
      message,
      horizon: state.horizon,
      history: state.chat.history,
    };
    if (state.selectedTicker && state.companyByTicker[state.selectedTicker]) {
      body.ticker = state.selectedTicker;
    }
    const res = await postJSON("/api/chat", body);
    pending.remove();
    if (res.configured === false) {
      appendChatMessage("info", res.reply);
    } else {
      appendChatMessage("assistant", res.reply);
      state.chat.history.push({ role: "user", content: message });
      state.chat.history.push({ role: "assistant", content: res.reply });
      if (state.chat.history.length > 12) {
        state.chat.history = state.chat.history.slice(-12);
      }
    }
  } catch (e) {
    pending.remove();
    const errEl = appendChatMessage("error", "Something went wrong reaching the AI assistant: " + e.message + "  ");
    const retryLink = document.createElement("button");
    retryLink.type = "button";
    retryLink.className = "chat-retry-link";
    retryLink.textContent = "Retry";
    retryLink.addEventListener("click", () => {
      errEl.remove();
      sendChatMessage(message);
    });
    errEl.appendChild(retryLink);
  } finally {
    state.chat.sending = false;
    sendBtn.disabled = false;
    if (input) {
      input.disabled = false;
      input.focus();
    }
  }
}

init();
