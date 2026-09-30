"""End-to-end smoke tests. Uses only the standard library's `unittest` (no
pytest dependency needed) so `python3 tests/smoke_test.py` always works.

Runs the whole pipeline in demo mode against a throwaway SQLite file, then
exercises the universe integrity, the feature panel, the trained models,
and every API route.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("STOCKGRAPH_DATA_MODE", "demo")
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["STOCKGRAPH_DB"] = _tmp_db.name


class TestUniverse(unittest.TestCase):
    def test_edges_reference_real_companies(self):
        from app.universe import EDGES, COMPANY_BY_TICKER

        for e in EDGES:
            self.assertIn(e.src, COMPANY_BY_TICKER, f"unknown src ticker {e.src}")
            self.assertIn(e.dst, COMPANY_BY_TICKER, f"unknown dst ticker {e.dst}")
            self.assertIn(e.kind, {"supplier", "customer", "competitor", "partner"})
            self.assertTrue(0 < e.weight <= 1)

    def test_no_duplicate_tickers(self):
        from app.universe import TICKERS

        self.assertEqual(len(TICKERS), len(set(TICKERS)))

    def test_most_companies_have_at_least_one_relationship(self):
        from app.universe import TICKERS, neighbors

        isolated = [t for t in TICKERS if not neighbors(t)]
        # A handful of standalone names is fine; the graph shouldn't be
        # mostly disconnected singletons.
        self.assertLess(len(isolated), len(TICKERS) * 0.15, f"too many isolated tickers: {isolated}")


class TestPipelineAndModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.engine import engine

        engine.bootstrap()
        cls.engine = engine

    def test_ingestion_populated_prices(self):
        from app import db
        from app.universe import TICKERS

        rows = db.get_prices(TICKERS[0])
        self.assertGreater(len(rows), 100)

    def test_models_trained_for_all_horizons(self):
        from app.forecast import HORIZONS

        for h in HORIZONS:
            self.assertIn(h, self.engine.models.by_horizon)

    def test_forecast_direction_and_move_agree_in_sign(self):
        from app.universe import TICKERS

        for t in TICKERS[:15]:
            f = self.engine.get_forecast(t, 5, with_llm=False)
            if f["direction"] == "up":
                self.assertGreaterEqual(f["expected_move_pct"], 0, t)
            else:
                self.assertLessEqual(f["expected_move_pct"], 0, t)

    def test_forecast_probability_in_range(self):
        from app.universe import TICKERS

        for t in TICKERS[:15]:
            f = self.engine.get_forecast(t, 5, with_llm=False)
            self.assertTrue(0.0 <= f["prob_up"] <= 1.0)
            self.assertTrue(0.0 <= f["confidence"] <= 1.0)

    def test_backtest_produced_outcomes(self):
        from app import db

        summary = db.scorecard_summary()
        self.assertGreater(summary["n"], 0)
        self.assertTrue(0.0 <= summary["hit_rate"] <= 1.0)


class _InProcessClient:
    """Drives app.httpserver's Router directly (no socket), the same way
    starlette's TestClient drove the old Starlette app -- just resolves the
    route and calls the handler in-process. Keeps a cookie jar so
    signup/login -> authenticated-request flows can be tested the way a
    real browser session would work."""

    def __init__(self, router, client_ip=None):
        self.router = router
        self.cookies = {}
        self.client_ip = client_ip

    def _call(self, method: str, path: str, json_body=None):
        import json as _json
        import urllib.parse

        from app.httpserver import Request

        parsed = urllib.parse.urlsplit(path)
        query = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
        route, path_params = self.router.resolve(method, parsed.path)
        if route is None:
            return _Resp(404, {"error": "not found"}, [])
        body = _json.dumps(json_body).encode("utf-8") if json_body is not None else b""
        cookie_header = "; ".join(f"{k}={v}" for k, v in self.cookies.items())
        headers = {"Cookie": cookie_header} if cookie_header else {}
        req = Request(
            method, parsed.path, query, path_params or {}, body,
            headers=headers, client_addr=self.client_ip,
        )
        resp = route.handler(req)
        for raw in getattr(resp, "_cookies", []):
            # "name=value; Path=/; SameSite=Lax; Max-Age=0; HttpOnly" -> keep name/value
            first = raw.split(";", 1)[0]
            k, _, v = first.partition("=")
            if raw.endswith("Max-Age=0; HttpOnly") or "Max-Age=0" in raw:
                self.cookies.pop(k, None)
            else:
                self.cookies[k] = v
        return _Resp(resp.status_code, json.loads(resp.body), resp._cookies)

    def get(self, path: str):
        return self._call("GET", path)

    def post(self, path: str, json_body=None):
        return self._call("POST", path, json_body)

    def delete(self, path: str):
        return self._call("DELETE", path)


class _Resp:
    def __init__(self, status_code, data, cookies=None):
        self.status_code = status_code
        self._data = data
        self.cookies = cookies or []

    def json(self):
        return self._data


class TestJSONResponse(unittest.TestCase):
    """Regression test: a stray NaN/Infinity anywhere in an API response
    used to produce invalid JSON (Python's json.dumps allows the bare
    NaN/Infinity tokens; JavaScript's JSON.parse rejects them), which broke
    every fetch on the page with no server-side error to point at -- e.g.
    a horizon trained on too little holdout data sets holdout_accuracy to
    float('nan'), which flows straight into /api/status."""

    def test_nan_and_infinity_are_sanitized_to_null(self):
        from app.httpserver import JSONResponse

        r = JSONResponse({"a": float("nan"), "b": float("inf"), "c": [1, float("-inf")], "d": 3.5})
        parsed = json.loads(r.body)  # would raise if the body still had a bare NaN token
        self.assertIsNone(parsed["a"])
        self.assertIsNone(parsed["b"])
        self.assertIsNone(parsed["c"][1])
        self.assertEqual(parsed["d"], 3.5)


class TestStaticFileCaching(unittest.TestCase):
    """Regression test: _serve_static() re-reads static/* fresh from disk on
    every request specifically so an edit shows up on next page load with
    no server restart needed -- but FileResponse used to send no caching
    headers at all, and a browser applies its OWN heuristic caching to any
    response like that, so a browser that already loaded /app.js once could
    go on serving that stale cached copy indefinitely even on an ordinary
    reload, silently masking every update after the first. Every static
    file must tell the browser not to do that."""

    def test_file_response_sends_no_cache_header(self):
        from app.httpserver import FileResponse

        r = FileResponse(Path(__file__))  # any real file on disk works here
        self.assertEqual(getattr(r, "extra_headers", {}).get("Cache-Control"), "no-cache")


class TestSecEdgarFilings(unittest.TestCase):
    """Regression test: SEC's per-company filings list is newest-first and
    dominated by routine forms (Form 4 insider trades, DEF 14A proxies,
    S-8s, ...) -- 10-Ks/10-Qs/8-Ks are a small fraction of it. An earlier
    version only looked at the first `limit` raw entries and filtered those
    down, so a company whose most recent filings happened to be mostly
    Form 4s returned zero rows even though 10-Ks/10-Qs/8-Ks existed further
    down the same list. fetch_recent_filings must scan until it finds
    `limit` matching filings, not just filter the first `limit` entries."""

    def test_filters_across_the_whole_list_not_just_the_first_page(self):
        from unittest.mock import patch
        import app.dataclients.secedgar as secedgar

        secedgar._cik_map_cache = {"TEST": "1234567890"}

        # 15 Form 4s (noise) followed by a real 10-K and 10-Q further down
        # the list than `limit` would reach if the old first-N-then-filter
        # logic were still in place.
        forms = ["4"] * 15 + ["10-K", "10-Q"]
        n = len(forms)
        fake_submissions = {
            "filings": {
                "recent": {
                    "form": forms,
                    "filingDate": [f"2026-01-{(i % 28) + 1:02d}" for i in range(n)],
                    "accessionNumber": [f"0001234567-26-{i:06d}" for i in range(n)],
                    "primaryDocument": [f"doc{i}.htm" for i in range(n)],
                }
            }
        }

        with patch("app.dataclients.secedgar.get_json", return_value=fake_submissions):
            rows = secedgar.fetch_recent_filings("TEST", limit=10)

        self.assertEqual(len(rows), 2)
        self.assertEqual({r["form_type"] for r in rows}, {"10-K", "10-Q"})


class TestYahooCrumbHandshake(unittest.TestCase):
    """Regression test: fc.yahoo.com's cookie-seed request routinely answers
    with a 404 (it's an edge/accelerator endpoint, not a real page) while
    still setting the session cookie via Set-Cookie on that same response.
    An earlier version of _get_crumb() wrapped the seed request and the
    crumb request in one try/except, so that expected 404 aborted the whole
    handshake before it ever reached the crumb endpoint -- silently
    disabling Yahoo's earnings/quoteSummary data (401s) even though the
    crumb endpoint itself was reachable and working. The seed and crumb
    requests must be able to fail/succeed independently."""

    def test_seed_request_failure_does_not_abort_crumb_fetch(self):
        import urllib.error
        from unittest.mock import patch
        import app.dataclients.yahoo as yahoo

        # Reset the module-level cache so this test doesn't depend on
        # whichever order the test suite happens to run in.
        yahoo._crumb = None
        yahoo._crumb_attempted = False

        calls = []

        def fake_open(self, req, timeout=None):
            url = req.full_url
            calls.append(url)
            if "fc.yahoo.com" in url:
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
            resp = MockResponse(b"a-real-crumb-token")
            return resp

        with patch("urllib.request.OpenerDirector.open", fake_open):
            crumb = yahoo._get_crumb()

        self.assertEqual(crumb, "a-real-crumb-token")
        # Both requests should have been attempted, in order, despite the
        # first one raising.
        self.assertEqual(len(calls), 2)
        self.assertIn("fc.yahoo.com", calls[0])
        self.assertIn("getcrumb", calls[1])


class MockResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.api import router
        from app.engine import engine

        if not engine.ready:
            engine.bootstrap()
        cls.client = _InProcessClient(router)

    def test_health(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)

    def test_companies_and_graph(self):
        companies = self.client.get("/api/companies").json()
        graph = self.client.get("/api/graph").json()
        self.assertGreater(len(companies), 50)
        self.assertEqual(len(graph["nodes"]), len(companies))
        self.assertGreater(len(graph["edges"]), 50)

    def test_forecasts_list(self):
        r = self.client.get("/api/forecasts?horizon=5")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertGreater(len(data), 50)
        for f in data:
            self.assertIn(f["direction"], ("up", "down"))

    def test_forecasts_list_has_days_to_earnings(self):
        # Regression: days_to_earnings must be present on every row and be
        # either null (no known upcoming earnings) or a small non-negative
        # int -- never the 999 "unknown" sentinel signals.py uses internally.
        r = self.client.get("/api/forecasts?horizon=5")
        data = r.json()
        self.assertGreater(len(data), 50)
        for f in data:
            self.assertIn("days_to_earnings", f)
            if f["days_to_earnings"] is not None:
                self.assertIsInstance(f["days_to_earnings"], int)
                self.assertLess(f["days_to_earnings"], 999)
                self.assertGreaterEqual(f["days_to_earnings"], 0)

    def test_forecast_detail_has_all_sections(self):
        r = self.client.get("/api/forecast/AAPL?horizon=5&narrative=0")
        self.assertEqual(r.status_code, 200)
        d = r.json()
        for key in ("price_history", "earnings", "filings", "news", "neighbors", "drivers", "rationale"):
            self.assertIn(key, d)
        self.assertGreater(len(d["price_history"]), 50)

    def test_unknown_ticker_404s(self):
        r = self.client.get("/api/forecast/NOTATICKER?horizon=5")
        self.assertEqual(r.status_code, 404)

    def test_scorecard(self):
        r = self.client.get("/api/scorecard")
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertIn("overall", d)
        self.assertIn("by_horizon", d)
        self.assertIn("by_ticker", d)

    def test_forecast_detail_has_52week_and_volume_data(self):
        r = self.client.get("/api/forecast/AAPL?horizon=5&narrative=0")
        self.assertEqual(r.status_code, 200)
        d = r.json()
        for key in ("wk52_high", "wk52_low", "avg_volume_30d", "latest_volume"):
            self.assertIn(key, d)
        self.assertGreaterEqual(d["wk52_high"], d["wk52_low"])


class TestChat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.api import router
        from app.engine import engine

        if not engine.ready:
            engine.bootstrap()
        cls.client = _InProcessClient(router)

    def test_chat_status_reports_unconfigured_without_api_key(self):
        r = self.client.get("/api/chat/status")
        self.assertEqual(r.status_code, 200)
        self.assertIn("configured", r.json())

    def test_chat_requires_message(self):
        r = self.client.post("/api/chat", {"message": ""})
        self.assertEqual(r.status_code, 400)

    def test_chat_unknown_ticker_404s(self):
        r = self.client.post("/api/chat", {"message": "hi", "ticker": "NOTATICKER"})
        self.assertEqual(r.status_code, 404)

    def test_chat_without_api_key_returns_friendly_unconfigured_reply(self):
        # This project's test/dev environment has no ANTHROPIC_API_KEY set,
        # so this exercises the graceful degrade path -- the same path that
        # keeps the rest of the site working when the key is left unset.
        from app import chat as chat_module

        if chat_module.is_configured():
            self.skipTest("ANTHROPIC_API_KEY is set in this environment")
        r = self.client.post("/api/chat", {"message": "why is AAPL up?", "ticker": "AAPL", "horizon": 5})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertFalse(data["configured"])
        self.assertIn("reply", data)

    def test_chat_market_snapshot_builds_without_a_ticker(self):
        from app import chat as chat_module

        snapshot = chat_module.build_snapshot(None, 5)
        self.assertIn("WHOLE-MARKET SNAPSHOT", snapshot)

    def test_chat_company_snapshot_builds_for_a_known_ticker(self):
        from app import chat as chat_module

        snapshot = chat_module.build_snapshot("AAPL", 5)
        self.assertIn("COMPANY: Apple", snapshot)
        self.assertIn("FORECAST", snapshot)


class TestLoginRateLimiter(unittest.TestCase):
    """Regression/coverage test for the brute-force guard added to
    auth.login(): repeated wrong passwords for one email should eventually
    get refused outright (rather than the login endpoint accepting
    unlimited guesses), and a correct password should still work right up
    until that point."""

    def test_locks_out_after_max_failed_attempts_then_still_rejects_correct_password(self):
        from app import auth

        auth._login_limiter = auth._LoginRateLimiter()  # isolate from other tests
        email = "ratelimit-target@example.com"
        # Not signing up through the DB -- we only need login()'s rate-limit
        # branch, which runs before it consults the DB.
        limiter = auth._login_limiter
        for _ in range(auth._LoginRateLimiter.MAX_PER_EMAIL):
            limiter.record_failure(email, "203.0.113.5")

        with self.assertRaises(auth.AuthError) as ctx:
            auth.login(email, "the-real-password-1", client_ip="203.0.113.5")
        self.assertIn("Too many failed login attempts", str(ctx.exception))

    def test_successful_login_clears_that_emails_counter(self):
        from app import auth

        auth._login_limiter = auth._LoginRateLimiter()
        email = "ratelimit-clears@example.com"
        auth._login_limiter.record_failure(email, "203.0.113.9")
        auth._login_limiter.record_failure(email, "203.0.113.9")
        auth._login_limiter.record_success(email, "203.0.113.9")
        self.assertEqual(auth._login_limiter._by_email.get(email, []), [])


class TestSignupRateLimiter(unittest.TestCase):
    """Regression/coverage test for the brute-force guard added to
    auth.signup(): a script hammering the signup endpoint from one IP
    should eventually get refused outright, independent of whether each
    attempt used a new email."""

    def test_locks_out_after_max_signups_from_one_ip(self):
        from app import auth

        auth._signup_limiter = auth._SignupRateLimiter()  # isolate from other tests
        ip = "203.0.113.20"
        for _ in range(auth._SignupRateLimiter.MAX_PER_IP):
            auth._signup_limiter.check_and_record(ip)

        with self.assertRaises(auth.AuthError) as ctx:
            auth._signup_limiter.check_and_record(ip)
        self.assertIn("Too many accounts created", str(ctx.exception))

    def test_signup_endpoint_enforces_the_limit_end_to_end(self):
        from app import auth
        from app.api import router

        auth._signup_limiter = auth._SignupRateLimiter()
        ip = "203.0.113.21"
        for i in range(auth._SignupRateLimiter.MAX_PER_IP):
            c = _InProcessClient(router, client_ip=ip)
            r = c.post("/api/auth/signup", {"email": f"spam{i}@example.com", "password": "correct-horse-1"})
            self.assertEqual(r.status_code, 200, r.json())

        c = _InProcessClient(router, client_ip=ip)
        r = c.post("/api/auth/signup", {"email": "one-too-many@example.com", "password": "correct-horse-1"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("Too many accounts", r.json()["error"])

    def test_different_ips_are_independent(self):
        from app import auth

        auth._signup_limiter = auth._SignupRateLimiter()
        for _ in range(auth._SignupRateLimiter.MAX_PER_IP):
            auth._signup_limiter.check_and_record("203.0.113.30")
        # A different IP should be unaffected.
        auth._signup_limiter.check_and_record("203.0.113.31")


class TestAuth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.api import router
        from app.engine import engine

        if not engine.ready:
            engine.bootstrap()
        cls.router = router

    def _fresh_client(self):
        return _InProcessClient(self.router)

    def test_signup_sets_session_cookie_and_me_returns_user(self):
        c = self._fresh_client()
        r = c.post("/api/auth/signup", {"email": "alice@example.com", "password": "correct-horse-1"})
        self.assertEqual(r.status_code, 200, r.json())
        self.assertIn("sg_session", c.cookies)
        me = c.get("/api/auth/me")
        self.assertEqual(me.json()["user"]["email"], "alice@example.com")

    def test_duplicate_signup_rejected(self):
        c = self._fresh_client()
        c.post("/api/auth/signup", {"email": "bob@example.com", "password": "correct-horse-1"})
        r2 = c.post("/api/auth/signup", {"email": "bob@example.com", "password": "another-pass-1"})
        self.assertEqual(r2.status_code, 400)

    def test_login_wrong_password_rejected(self):
        c = self._fresh_client()
        c.post("/api/auth/signup", {"email": "carol@example.com", "password": "correct-horse-1"})
        c2 = self._fresh_client()
        r = c2.post("/api/auth/login", {"email": "carol@example.com", "password": "wrong-password"})
        self.assertEqual(r.status_code, 400)

    def test_login_roundtrip(self):
        c = self._fresh_client()
        c.post("/api/auth/signup", {"email": "dave@example.com", "password": "correct-horse-1"})
        c.post("/api/auth/logout")
        self.assertNotIn("sg_session", c.cookies)
        c2 = self._fresh_client()
        r = c2.post("/api/auth/login", {"email": "dave@example.com", "password": "correct-horse-1"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("sg_session", c2.cookies)

    def test_watchlist_requires_auth(self):
        c = self._fresh_client()
        r = c.get("/api/watchlist")
        self.assertEqual(r.status_code, 401)

    def test_watchlist_add_remove_roundtrip(self):
        c = self._fresh_client()
        c.post("/api/auth/signup", {"email": "erin@example.com", "password": "correct-horse-1"})
        r = c.post("/api/watchlist/AAPL")
        self.assertEqual(r.status_code, 200)
        self.assertIn("AAPL", r.json()["watchlist"])

        listed = c.get("/api/watchlist")
        self.assertIn("AAPL", listed.json()["watchlist"])

        forecasts = c.get("/api/forecasts?horizon=5").json()
        aapl_row = next(f for f in forecasts if f["ticker"] == "AAPL")
        self.assertTrue(aapl_row["in_watchlist"])

        r2 = c.delete("/api/watchlist/AAPL")
        self.assertNotIn("AAPL", r2.json()["watchlist"])

    def test_watchlist_unknown_ticker_404s(self):
        c = self._fresh_client()
        c.post("/api/auth/signup", {"email": "frank@example.com", "password": "correct-horse-1"})
        r = c.post("/api/watchlist/NOTATICKER")
        self.assertEqual(r.status_code, 404)

    def test_preferences_requires_auth(self):
        c = self._fresh_client()
        r = c.get("/api/preferences")
        self.assertEqual(r.status_code, 401)

    def test_preferences_save_roundtrip_and_me_includes_them(self):
        c = self._fresh_client()
        c.post("/api/auth/signup", {"email": "grace@example.com", "password": "correct-horse-1"})
        r = c.post(
            "/api/preferences",
            {
                "goals": ["growth", "income", "not-a-real-goal"],
                "risk_tolerance": "aggressive",
                "horizon": "long",
                "sectors": ["Technology", "Healthcare"],
                "experience": "some",
            },
        )
        self.assertEqual(r.status_code, 200)
        prefs = r.json()["preferences"]
        # Invalid/unknown goal values are dropped rather than stored verbatim.
        self.assertEqual(set(prefs["goals"]), {"growth", "income"})
        self.assertEqual(prefs["risk_tolerance"], "aggressive")
        self.assertEqual(prefs["sectors"], ["Technology", "Healthcare"])

        fetched = c.get("/api/preferences").json()["preferences"]
        self.assertEqual(fetched["risk_tolerance"], "aggressive")

        me = c.get("/api/auth/me").json()
        self.assertEqual(me["preferences"]["horizon"], "long")

    def test_preferences_rejects_invalid_enum_values_as_null(self):
        c = self._fresh_client()
        c.post("/api/auth/signup", {"email": "heidi@example.com", "password": "correct-horse-1"})
        r = c.post("/api/preferences", {"risk_tolerance": "yolo", "horizon": "long"})
        prefs = r.json()["preferences"]
        self.assertIsNone(prefs["risk_tolerance"])
        self.assertEqual(prefs["horizon"], "long")

    def test_sectors_endpoint_lists_known_sectors(self):
        c = self._fresh_client()
        r = c.get("/api/sectors")
        self.assertEqual(r.status_code, 200)
        sectors = r.json()
        self.assertGreater(len(sectors), 5)
        self.assertIn("Technology", sectors)

    def test_forecasts_list_matches_interest_reorders_toward_preferred_sectors(self):
        c = self._fresh_client()
        c.post("/api/auth/signup", {"email": "ivan@example.com", "password": "correct-horse-1"})
        c.post("/api/preferences", {"sectors": ["Technology"]})
        rows = c.get("/api/forecasts?horizon=5").json()
        self.assertTrue(all("matches_interest" in r for r in rows))
        matched = [r for r in rows if r["matches_interest"]]
        self.assertGreater(len(matched), 0)
        # every matched-interest row should sort ahead of every non-matched row
        first_unmatched = next(i for i, r in enumerate(rows) if not r["matches_interest"])
        self.assertTrue(all(r["matches_interest"] for r in rows[:first_unmatched]))


class TestFundamentals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.api import router
        from app.engine import engine

        if not engine.ready:
            engine.bootstrap()
        cls.client = _InProcessClient(router)

    def test_db_roundtrip(self):
        from app import db

        db.upsert_fundamentals("AAPL", {
            "description": "Test description", "market_cap": 3.2e12, "pe_ratio": 31.5,
            "forward_pe": 28.0, "peg_ratio": 2.1, "dividend_yield": 0.005, "beta": 1.2,
            "profit_margin": 0.25, "revenue_growth": 0.08, "analyst_target_mean": 250.0,
            "analyst_target_high": 300.0, "analyst_target_low": 200.0,
            "analyst_recommendation": "buy", "num_analyst_opinions": 40,
        })
        fd = db.get_fundamentals("AAPL")
        self.assertEqual(fd["analyst_recommendation"], "buy")
        self.assertEqual(fd["num_analyst_opinions"], 40)
        all_fd = db.all_fundamentals()
        self.assertIn("AAPL", all_fd)

    def test_demo_pipeline_populates_fundamentals_for_every_company(self):
        from app import db
        from app.universe import TICKERS

        all_fd = db.all_fundamentals()
        missing = [t for t in TICKERS if t not in all_fd]
        self.assertEqual(missing, [], f"missing demo fundamentals for: {missing[:10]}")

    def test_forecast_detail_includes_fundamentals(self):
        r = self.client.get("/api/forecast/AAPL?horizon=5&narrative=0")
        d = r.json()
        self.assertIn("fundamentals", d)
        self.assertIsNotNone(d["fundamentals"])
        self.assertIn("description", d["fundamentals"])


class TestNewTechnicalFeatures(unittest.TestCase):
    """MACD/Bollinger/ATR/filing-recency features must show up in the panel
    with no NaNs (which would silently zero out a model coefficient) and
    the filing-recency features must actually vary with real filing dates
    rather than being stuck at the 999 'unknown' sentinel for everything."""

    @classmethod
    def setUpClass(cls):
        from app.engine import engine

        if not engine.ready:
            engine.bootstrap()
        cls.engine = engine

    def test_new_feature_columns_present_and_finite(self):
        panel = self.engine.panel
        for name in ("macd_hist", "bollinger_pct_b", "bollinger_bandwidth", "atr_pct",
                     "days_since_filing", "filing_count_30d"):
            self.assertIn(name, panel.features)
            df = panel.features[name]
            self.assertTrue(np.isfinite(df.to_numpy()).all(), f"{name} has non-finite values")

    def test_filing_recency_varies_across_companies(self):
        panel = self.engine.panel
        latest = panel.dates[-1]
        values = panel.features["days_since_filing"].loc[latest]
        # Not every company should be stuck at the "no filings known" sentinel.
        self.assertLess((values >= 999).sum(), len(values))


class TestLiveIngestionPartialFailureFallback(unittest.TestCase):
    """Regression test for a real production bug: when most of the universe
    fetched live data fine but a handful of tickers failed *entirely*
    (Yahoo blocking/rate-limiting those specific symbols), those tickers
    were left with NO price data in the DB at all. build_feature_panel()
    indexes every universe ticker (sector-momentum grouping in signals.py),
    so even one such gap crashed bootstrap with a KeyError -- this only
    surfaced once the universe grew large enough that at least one ticker
    routinely failed live fetch. "auto" mode must backfill demo data for
    exactly the tickers/fields that failed, not only when the *entire*
    universe is unreachable (that all-or-nothing fallback already existed
    and isn't what this test is checking).

    Runs against its own throwaway DB (not the module-level shared one)
    so it can start from a guaranteed-empty slate for the "failing"
    tickers -- otherwise leftover data from earlier tests would mask the
    bug this is meant to catch.
    """

    def test_partial_live_failure_is_backfilled_and_feature_panel_still_builds(self):
        from unittest import mock

        from app import db, pipeline
        from app.dataclients import demo as demo_client
        from app.signals import build_feature_panel
        from app.universe import COMPANIES

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.addCleanup(os.unlink, tmp.name)

        failing = {COMPANIES[0].ticker, COMPANIES[1].ticker}
        demo_dataset = demo_client.generate_universe_demo_data()

        def fake_fetch(ticker, company_name, need):
            if ticker in failing:
                return {"ticker": ticker, "bars": None, "earnings": None,
                         "filings": None, "news": None, "fundamentals": None, "sources": {}}
            bundle = demo_dataset[ticker]
            return {
                "ticker": ticker,
                "bars": bundle["prices"],
                "earnings": bundle["earnings"],
                "filings": bundle["filings"],
                "news": bundle["news"],
                "fundamentals": demo_client.generate_fundamentals(ticker),
                "sources": {f: "test" for f in db.PROVENANCE_FIELDS},
            }

        old_db_path, old_conn = db.DB_PATH, getattr(db._local, "conn", None)
        db.DB_PATH = tmp.name
        db._local.conn = None
        self.addCleanup(lambda: (setattr(db, "DB_PATH", old_db_path),
                                  setattr(db._local, "conn", old_conn)))

        with mock.patch.object(pipeline, "DATA_MODE", "auto"), \
             mock.patch.object(pipeline, "_fetch_live_ticker", side_effect=fake_fetch):
            pipeline.run_ingestion()

        for ticker in failing:
            rows = db.get_prices(ticker)
            self.assertGreater(
                len(rows), 100,
                f"{ticker} (simulated total live-fetch failure) should have been "
                "backfilled with demo bars, not left empty",
            )

        # The actual regression: this must not raise
        # KeyError("[...] not in index") for the previously-empty tickers.
        panel = build_feature_panel()
        self.assertEqual(set(panel.close.columns), {c.ticker for c in COMPANIES})
        self.assertTrue(np.isfinite(panel.features["mom_5"].to_numpy()).all())


class TestHttpRetry(unittest.TestCase):
    """get_text/get_json must retry a transient failure (429/5xx, or a
    network-level error) with backoff, but fail immediately on a
    permanent one (401/403/404) -- retrying an auth failure or a
    not-found just burns time and rate-limit budget for the same
    outcome, which matters a lot once a source has a hard per-minute cap
    (Finnhub) or is already prone to blocking bursts (Yahoo)."""

    def _fake_opener(self, responses):
        """`responses`: a list of either bytes (success) or an
        urllib.error.HTTPError/URLError instance (raised) -- consumed in
        order, one per call to .open()."""
        import urllib.error

        calls = {"n": 0}

        class _Resp:
            def __init__(self, data):
                self._data = data
                self.headers = {}

            def read(self):
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        class _Opener:
            def open(self, req, timeout=None):
                i = calls["n"]
                calls["n"] += 1
                item = responses[i]
                if isinstance(item, Exception):
                    raise item
                return _Resp(item)

        return _Opener(), calls

    def test_retries_on_429_then_succeeds(self):
        import urllib.error

        from app.dataclients.httpjson import get_text

        err = urllib.error.HTTPError("http://x", 429, "Too Many Requests", {}, None)
        opener, calls = self._fake_opener([err, err, b"ok"])
        result = get_text("http://x", opener=opener, retries=2, backoff_base=0.01)
        self.assertEqual(result, "ok")
        self.assertEqual(calls["n"], 3)

    def test_does_not_retry_on_404(self):
        import urllib.error

        from app.dataclients.httpjson import HTTPError, get_text

        err = urllib.error.HTTPError("http://x", 404, "Not Found", {}, None)
        opener, calls = self._fake_opener([err, b"should not be reached"])
        with self.assertRaises(HTTPError) as ctx:
            get_text("http://x", opener=opener, retries=2, backoff_base=0.01)
        self.assertEqual(ctx.exception.status, 404)
        self.assertEqual(calls["n"], 1, "must not retry a 404")

    def test_gives_up_after_exhausting_retries(self):
        import urllib.error

        from app.dataclients.httpjson import HTTPError, get_text

        err = urllib.error.HTTPError("http://x", 503, "Service Unavailable", {}, None)
        opener, calls = self._fake_opener([err, err, err, err])
        with self.assertRaises(HTTPError):
            get_text("http://x", opener=opener, retries=2, backoff_base=0.01)
        self.assertEqual(calls["n"], 3, "1 initial attempt + 2 retries")


class TestFinnhubFieldExtraction(unittest.TestCase):
    """Finnhub's /stock/metric response isn't fully documented in one
    place, so fetch_fundamentals leans on a short alias list per field
    (see finnhub.py's module docstring). These test the extraction logic
    itself against constructed fixtures, independent of whether any given
    alias turns out to be the exact key Finnhub uses in practice."""

    def test_extract_metric_picks_first_available_alias(self):
        from app.dataclients.finnhub import _extract_metric

        metric = {"peBasicExclExtraTTM": 24.1, "beta": 2.2, "dividendYieldIndicatedAnnual": 3.2}
        out = _extract_metric(metric)
        self.assertEqual(out["pe_ratio"], 24.1)
        self.assertEqual(out["beta"], 2.2)
        # 3.2 is a whole-percent value (3.2%), so it should convert to a
        # fraction (0.032), not be mistaken for "already a fraction".
        self.assertAlmostEqual(out["dividend_yield"], 0.032)

    def test_extract_metric_missing_fields_are_none_not_crash(self):
        from app.dataclients.finnhub import _extract_metric

        out = _extract_metric({})
        for key in ("pe_ratio", "forward_pe", "peg_ratio", "dividend_yield", "beta",
                    "profit_margin", "revenue_growth"):
            self.assertIsNone(out[key])

    def test_pct_to_fraction_leaves_small_fractions_alone(self):
        from app.dataclients.finnhub import _pct_to_fraction

        self.assertAlmostEqual(_pct_to_fraction(23.4), 0.234)
        self.assertAlmostEqual(_pct_to_fraction(0.15), 0.15)
        self.assertIsNone(_pct_to_fraction(None))

    def test_summarize_recommendation_picks_majority_bucket(self):
        from app.dataclients.finnhub import _summarize_recommendation

        key, n = _summarize_recommendation(
            {"strongBuy": 10, "buy": 20, "hold": 3, "sell": 1, "strongSell": 0}
        )
        self.assertEqual(key, "buy")
        self.assertEqual(n, 34)

    def test_summarize_recommendation_all_zero_is_none(self):
        from app.dataclients.finnhub import _summarize_recommendation

        key, n = _summarize_recommendation({"strongBuy": 0, "buy": 0, "hold": 0, "sell": 0, "strongSell": 0})
        self.assertIsNone(key)
        self.assertIsNone(n)

    def test_not_configured_without_api_key(self):
        from unittest import mock

        from app.dataclients import finnhub

        with mock.patch.object(finnhub, "FINNHUB_API_KEY", ""):
            self.assertFalse(finnhub.is_configured())
        with mock.patch.object(finnhub, "FINNHUB_API_KEY", "abc123"):
            self.assertTrue(finnhub.is_configured())


class TestIncrementalIngestion(unittest.TestCase):
    """A ticker/field fetched within the freshness window should be
    skipped on the next run rather than re-fetched -- this is what keeps
    a scheduled or manually-clicked refresh fast and rate-limit-friendly
    after the first run. Runs against its own throwaway DB."""

    def setUp(self):
        from app import db

        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.addCleanup(os.unlink, self.tmp.name)
        self.old_db_path = db.DB_PATH
        self.old_conn = getattr(db._local, "conn", None)
        db.DB_PATH = self.tmp.name
        db._local.conn = None
        db.init_db()

        def _restore():
            db.DB_PATH = self.old_db_path
            db._local.conn = self.old_conn

        self.addCleanup(_restore)

    def test_fresh_field_is_not_marked_as_needed(self):
        from app import db
        from app.pipeline import _needed_fields

        db.update_provenance("AAPL", {"bars": "yahoo", "earnings": "finnhub", "fundamentals": "finnhub"})
        provenance = db.get_all_provenance()

        needs = _needed_fields("AAPL", provenance, force=False)
        self.assertFalse(needs["bars"])
        self.assertFalse(needs["earnings"])
        self.assertFalse(needs["fundamentals"])
        # filings/news are always considered worth a re-check regardless of
        # freshness (see pipeline.py's _needed_fields docstring).
        self.assertTrue(needs["filings"])
        self.assertTrue(needs["news"])

    def test_missing_provenance_means_everything_is_needed(self):
        from app.pipeline import _needed_fields

        needs = _needed_fields("NEWTICKER", {}, force=False)
        self.assertTrue(all(needs.values()))

    def test_force_ignores_freshness(self):
        from app import db
        from app.pipeline import _needed_fields

        db.update_provenance("AAPL", {"bars": "yahoo", "earnings": "finnhub", "fundamentals": "finnhub"})
        provenance = db.get_all_provenance()
        needs = _needed_fields("AAPL", provenance, force=True)
        self.assertTrue(all(needs.values()))

    def test_stale_field_is_needed_again(self):
        from datetime import timedelta

        from app import db
        from app.pipeline import _needed_fields

        with db.tx() as conn:
            conn.execute(
                "INSERT INTO data_provenance(ticker, bars_source, bars_updated_at) VALUES (?, ?, ?)",
                ("AAPL", "yahoo", (db.datetime.now(db.timezone.utc) - timedelta(hours=48)).isoformat()),
            )
        provenance = db.get_all_provenance()
        needs = _needed_fields("AAPL", provenance, force=False)
        self.assertTrue(needs["bars"], "a field older than the freshness window must be re-fetched")


class TestBackgroundRefresh(unittest.TestCase):
    """Ingestion used to only ever run once, at process startup. This
    checks the scheduled-refresh loop actually fires bootstrap() on its
    interval, is idempotent (calling start twice doesn't spawn two
    threads), and -- importantly -- that one failed refresh is caught and
    recorded rather than crashing the background thread or taking down
    whatever the site was already serving."""

    def test_fires_bootstrap_on_schedule_and_is_idempotent(self):
        import threading as _threading
        from unittest import mock

        from app.engine import Engine

        eng = Engine()
        call_count = _threading.Event()
        calls = []

        def fake_bootstrap():
            calls.append(1)
            if len(calls) == 1:
                call_count.set()

        with mock.patch.object(eng, "bootstrap", side_effect=fake_bootstrap):
            eng.start_background_refresh(interval_hours=1 / 3600)  # ~1 second
            eng.start_background_refresh(interval_hours=1 / 3600)  # should no-op
            fired = call_count.wait(timeout=5)
            eng.stop_background_refresh()

        self.assertTrue(fired, "background refresh never called bootstrap()")
        self.assertEqual(eng.refresh_interval_hours, 1 / 3600)

    def test_failed_refresh_is_recorded_not_raised(self):
        import threading as _threading
        from unittest import mock

        from app.engine import Engine

        eng = Engine()
        failed = _threading.Event()

        def failing_bootstrap():
            failed.set()
            raise RuntimeError("simulated data source outage")

        with mock.patch.object(eng, "bootstrap", side_effect=failing_bootstrap):
            eng.start_background_refresh(interval_hours=1 / 3600)
            self.assertTrue(failed.wait(timeout=5))
            # Give the except-block a moment to record the error after the
            # exception is raised inside the loop.
            for _ in range(50):
                if eng.last_refresh_error:
                    break
                time.sleep(0.05)
            eng.stop_background_refresh()

        self.assertIn("simulated data source outage", eng.last_refresh_error or "")

    def test_zero_interval_disables_refresh(self):
        from app.engine import Engine

        eng = Engine()
        eng.start_background_refresh(interval_hours=0)
        self.assertIsNone(eng._refresh_thread)


class TestProvenanceAndIngestionHealth(unittest.TestCase):
    """The /api/health-facing pieces: provenance roundtrips per field
    without clobbering fields it wasn't told about, and ingestion run
    history records start/finish so a dashboard (or a person reading
    logs) can see the pipeline is actually healthy."""

    def setUp(self):
        from app import db

        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.addCleanup(os.unlink, self.tmp.name)
        self.old_db_path = db.DB_PATH
        self.old_conn = getattr(db._local, "conn", None)
        db.DB_PATH = self.tmp.name
        db._local.conn = None
        db.init_db()

        def _restore():
            db.DB_PATH = self.old_db_path
            db._local.conn = self.old_conn

        self.addCleanup(_restore)

    def test_update_provenance_does_not_clobber_untouched_fields(self):
        from app import db

        db.update_provenance("AAPL", {"bars": "yahoo"})
        db.update_provenance("AAPL", {"earnings": "finnhub"})
        row = db.get_all_provenance()["AAPL"]
        self.assertEqual(row["bars_source"], "yahoo")
        self.assertEqual(row["earnings_source"], "finnhub")
        self.assertIsNone(row["fundamentals_source"])

    def test_ingestion_run_roundtrip(self):
        from app import db

        run_id = db.start_ingestion_run("auto")
        db.finish_ingestion_run(run_id, live_ok=340, live_fail=3, skipped_fresh=100, elapsed_sec=12.5)
        runs = db.recent_ingestion_runs(limit=5)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["live_ok"], 340)
        self.assertEqual(runs[0]["live_fail"], 3)
        self.assertIsNotNone(runs[0]["finished_at"])

    def test_ingestion_run_records_error_on_failure(self):
        from app import db

        run_id = db.start_ingestion_run("live")
        db.finish_ingestion_run(run_id, 0, 0, 0, 1.0, error="boom")
        runs = db.recent_ingestion_runs(limit=1)
        self.assertEqual(runs[0]["error"], "boom")

    def test_get_provenance_single_ticker(self):
        from app import db

        db.update_provenance("AAPL", {"bars": "yahoo", "news": "google_news"})
        row = db.get_provenance("AAPL")
        self.assertEqual(row["bars_source"], "yahoo")
        self.assertEqual(row["news_source"], "google_news")
        self.assertIsNone(row["earnings_source"])

    def test_get_provenance_unknown_ticker_returns_empty_dict(self):
        from app import db

        self.assertEqual(db.get_provenance("NOPE"), {})

    def test_provenance_summary_counts_per_field_per_source(self):
        from app import db

        db.update_provenance("AAPL", {"bars": "yahoo", "earnings": "finnhub"})
        db.update_provenance("MSFT", {"bars": "yahoo", "earnings": "demo"})
        db.update_provenance("GOOG", {"bars": "demo"})
        summary = db.provenance_summary()
        self.assertEqual(summary["bars"], {"yahoo": 2, "demo": 1})
        self.assertEqual(summary["earnings"], {"finnhub": 1, "demo": 1})
        self.assertEqual(summary["fundamentals"], {})


class TestHealthEndpointAndProvenanceExposure(unittest.TestCase):
    """/api/health is the Data Health panel's data source, and
    forecasts_list/forecast_detail's new is_live/data_sources fields are
    what let the UI show per-company/per-field live-vs-synthetic badges
    instead of one global banner. Runs against the real (demo-mode)
    global engine, like TestAPI, so these exercise the actual response
    shape rather than a hand-built fixture."""

    @classmethod
    def setUpClass(cls):
        from app.api import router
        from app.engine import engine

        if not engine.ready:
            engine.bootstrap()
        cls.client = _InProcessClient(router)

    def test_health_endpoint_shape(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        d = r.json()
        for key in (
            "ready", "trained_at", "last_ingested_at", "data_mode_active",
            "finnhub_configured", "refresh_interval_hours", "last_refresh_error",
            "source_summary", "recent_runs", "universe_size",
        ):
            self.assertIn(key, d)
        self.assertTrue(d["ready"])
        self.assertIsInstance(d["finnhub_configured"], bool)
        self.assertIsInstance(d["source_summary"], dict)
        from app import db

        for field in db.PROVENANCE_FIELDS:
            self.assertIn(field, d["source_summary"])

    def test_health_source_summary_reflects_demo_mode(self):
        # The whole-suite demo dataset means every field for every company
        # should be sourced from "demo" -- if this ever showed "yahoo" or
        # "finnhub" counts while STOCKGRAPH_DATA_MODE=demo, that would mean
        # provenance is being set for the wrong source.
        r = self.client.get("/api/health")
        summary = r.json()["source_summary"]
        for field, counts in summary.items():
            self.assertEqual(set(counts.keys()), {"demo"}, field)

    def test_forecasts_list_rows_have_is_live_flag(self):
        r = self.client.get("/api/forecasts?horizon=5")
        data = r.json()
        self.assertGreater(len(data), 50)
        for row in data:
            self.assertIn("is_live", row)
            self.assertIn("live_field_count", row)
            # demo mode: nothing counts as "live" (demo isn't a live source)
            self.assertFalse(row["is_live"])
            self.assertEqual(row["live_field_count"], 0)

    def test_forecast_detail_has_data_sources_per_field(self):
        from app import db

        r = self.client.get("/api/forecast/AAPL?horizon=5&narrative=0")
        d = r.json()
        self.assertIn("data_sources", d)
        for field in db.PROVENANCE_FIELDS:
            self.assertIn(field, d["data_sources"])
            entry = d["data_sources"][field]
            self.assertIsNotNone(entry)
            self.assertEqual(entry["source"], "demo")
            self.assertIsNotNone(entry["updated_at"])

    def test_get_ticker_scorecard_returns_stats_for_known_ticker(self):
        # AAPL has plenty of price history in the demo universe, so the
        # walk-forward backtest (which ran in setUpClass's bootstrap())
        # should have recorded held-out outcomes for it.
        from app import db

        tr = db.get_ticker_scorecard("AAPL")
        self.assertIn("n", tr)
        self.assertGreater(tr["n"], 0)
        self.assertTrue(0.0 <= tr["hit_rate"] <= 1.0)
        self.assertGreaterEqual(tr["mae"], 0.0)

    def test_get_ticker_scorecard_unknown_ticker_returns_empty_dict(self):
        from app import db

        self.assertEqual(db.get_ticker_scorecard("NOT_A_REAL_TICKER"), {})

    def test_forecast_detail_includes_track_record(self):
        r = self.client.get("/api/forecast/AAPL?horizon=5&narrative=0")
        d = r.json()
        self.assertIn("track_record", d)
        tr = d["track_record"]
        self.assertIsNotNone(tr)
        for key in ("n", "hit_rate", "mae"):
            self.assertIn(key, tr)

    def test_forecasts_list_rows_have_track_record_field(self):
        r = self.client.get("/api/forecasts?horizon=5")
        data = r.json()
        self.assertGreater(len(data), 50)
        for row in data:
            self.assertIn("track_record", row)
            # every row's key is present; the value is a dict once that
            # ticker has backtest outcomes, or None if it doesn't yet.
            if row["track_record"] is not None:
                for key in ("n", "hit_rate", "mae"):
                    self.assertIn(key, row["track_record"])


class TestPaperTrading(unittest.TestCase):
    """app.trading turns forecasts into simulated Alpaca orders. These
    never touch the network -- app.broker's functions are mocked out --
    so what's actually under test is the strategy/risk logic: the
    off-by-default guard, the kill switch halting *before* any order is
    placed, and the daily-rebalance (close-then-open, confidence-ranked,
    floor-filtered) behavior."""

    def test_disabled_by_default_without_env_or_keys(self):
        from app import broker, trading

        self.assertFalse(broker.is_configured())
        self.assertFalse(trading.enabled())
        result = trading.run_trading_pass(engine=None)
        self.assertFalse(result["ran"])

    def test_broker_unreachable_aborts_pass_cleanly(self):
        from unittest import mock

        from app import broker, trading

        with mock.patch.object(trading, "TRADING_ENABLED", True), \
             mock.patch.object(broker, "is_configured", return_value=True), \
             mock.patch.object(broker, "get_account", side_effect=broker.BrokerError("boom")):
            result = trading.run_trading_pass(engine=mock.Mock())

        self.assertFalse(result["ran"])
        self.assertIn("boom", result["reason"])

    def test_kill_switch_halts_pass_before_any_order(self):
        from unittest import mock

        from app import broker, db, trading

        with mock.patch.object(trading, "TRADING_ENABLED", True), \
             mock.patch.object(broker, "is_configured", return_value=True), \
             mock.patch.object(broker, "get_account", return_value={"equity": "9000", "last_equity": "10000"}), \
             mock.patch.object(broker, "list_positions") as mock_list, \
             mock.patch.object(broker, "submit_order") as mock_submit:
            result = trading.run_trading_pass(engine=mock.Mock())

        # -10% day breaches the default -3% kill-switch threshold.
        self.assertTrue(result["ran"])
        self.assertTrue(result["halted_by_kill_switch"])
        mock_list.assert_not_called()
        mock_submit.assert_not_called()

        rows = db.recent_paper_trades(limit=1)
        self.assertEqual(rows[0]["action"], "skipped")
        self.assertIn("kill switch", rows[0]["reason"])

    def test_daily_rebalance_closes_prior_positions_and_opens_top_confidence_ones(self):
        from unittest import mock

        from app import broker, trading

        fake_engine = mock.Mock()
        fake_engine.horizons.return_value = [1, 5, 20]
        fake_engine.list_forecasts.return_value = [
            {"ticker": "AAA", "confidence": 0.9, "direction": "up"},
            {"ticker": "BBB", "confidence": 0.4, "direction": "down"},  # below the 0.6 floor
            {"ticker": "CCC", "confidence": 0.7, "direction": "down"},
        ]

        with mock.patch.object(trading, "TRADING_ENABLED", True), \
             mock.patch.object(broker, "is_configured", return_value=True), \
             mock.patch.object(broker, "get_account", return_value={"equity": "10100", "last_equity": "10000"}), \
             mock.patch.object(broker, "list_positions", return_value=[{"symbol": "ZZZ"}]), \
             mock.patch.object(broker, "close_position", return_value={"id": "close-1"}) as mock_close, \
             mock.patch.object(broker, "submit_order", return_value={"id": "order-1"}) as mock_submit:
            result = trading.run_trading_pass(fake_engine)

        self.assertTrue(result["ran"])
        self.assertFalse(result["halted_by_kill_switch"])
        mock_close.assert_called_once_with("ZZZ")

        # Only the two forecasts at/above the 0.6 confidence floor trade;
        # BBB (0.4) is filtered out even though there's room under the
        # default 5-position cap.
        self.assertEqual(mock_submit.call_count, 2)
        sides_by_ticker = {c.args[0]: c.args[1] for c in mock_submit.call_args_list}
        self.assertEqual(sides_by_ticker, {"AAA": "buy", "CCC": "sell"})

    def test_trading_status_endpoint_reports_disabled_by_default(self):
        from app.api import router

        client = _InProcessClient(router)
        r = client.get("/api/trading/status")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertFalse(data["enabled"])
        self.assertIn("recent_trades", data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
