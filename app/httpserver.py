"""A minimal WSGI-free HTTP framework on top of the standard library's
http.server, replacing Starlette so this project has zero third-party web
dependencies. Deliberately small: a router with one dynamic path segment
(`{name}`), a Request with query params + path params + JSON body, and a
JSONResponse/FileResponse pair. Good enough for this project's dozen
routes plus static file serving; not a general-purpose framework.
"""
from __future__ import annotations

import json
import math
import mimetypes
import re
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _json_safe(obj):
    """Recursively replace NaN/Infinity floats with None.

    Python's json.dumps allows these by default and emits the bare (non-
    standard) tokens NaN / Infinity / -Infinity. JavaScript's JSON.parse
    -- which fetch().json() uses under the hood -- rejects those tokens
    outright, so a single stray NaN anywhere in a response (e.g. a model
    metric computed from too little holdout data) silently breaks every
    fetch on the page. Scrubbing here, once, at the response boundary, is
    cheaper than chasing every individual place a NaN could originate.
    """
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _parse_cookies(header_value: str | None) -> dict:
    if not header_value:
        return {}
    out = {}
    for part in header_value.split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = urllib.parse.unquote(v.strip())
    return out


class Request:
    def __init__(
        self,
        method: str,
        path: str,
        query: dict,
        path_params: dict,
        body: bytes,
        headers: dict | None = None,
        client_addr: str | None = None,
    ):
        self.method = method
        self.path = path
        self.query_params = query
        self.path_params = path_params
        self._body = body
        self.headers = headers or {}
        self.cookies = _parse_cookies(self.headers.get("Cookie") or self.headers.get("cookie"))
        # Best-effort caller IP, for the login rate limiter (app/auth.py). If
        # this is deployed behind a reverse proxy, prefer X-Forwarded-For's
        # first hop over the proxy's own socket address.
        forwarded = self.headers.get("X-Forwarded-For") or self.headers.get("x-forwarded-for")
        self.client_addr = (forwarded.split(",")[0].strip() if forwarded else client_addr)

    def json(self):
        return json.loads(self._body or b"{}")


class _CookieMixin:
    """Shared by every response type: lets a handler queue Set-Cookie
    headers regardless of which response class it returns."""

    _cookies: list[str]

    def set_cookie(
        self,
        name: str,
        value: str,
        max_age: int | None = None,
        http_only: bool = True,
        same_site: str = "Lax",
        path: str = "/",
    ):
        if not hasattr(self, "_cookies"):
            self._cookies = []
        parts = [f"{name}={urllib.parse.quote(value)}", f"Path={path}", f"SameSite={same_site}"]
        if max_age is not None:
            parts.append(f"Max-Age={max_age}")
        if http_only:
            parts.append("HttpOnly")
        self._cookies.append("; ".join(parts))
        return self

    def clear_cookie(self, name: str, path: str = "/"):
        return self.set_cookie(name, "", max_age=0, path=path)


class JSONResponse(_CookieMixin):
    def __init__(self, data, status_code: int = 200):
        self.status_code = status_code
        self.body = json.dumps(_json_safe(data)).encode("utf-8")
        self.content_type = "application/json"
        self._cookies = []


class FileResponse(_CookieMixin):
    def __init__(self, path: Path, status_code: int = 200):
        self.status_code = status_code
        self.body = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.content_type = ctype
        self._cookies = []
        # _serve_static() re-reads every file from disk on every request
        # specifically so an edit takes effect on the next load with no
        # server restart -- but browsers apply their own *heuristic* caching
        # to any response with no explicit caching headers at all (which is
        # what this class sent before), so without this, a browser that's
        # already loaded /app.js or /index.html once can keep serving that
        # cached copy for a while even on a plain reload, silently masking
        # updates (and, worse, running mismatched HTML/JS together). "no-
        # cache" (not "no-store") still lets the browser keep a local copy,
        # it just forces a revalidation request every time -- effectively
        # free here since the server has no ETag/Last-Modified to check
        # against anyway, so every request gets the current file.
        self.extra_headers = {"Cache-Control": "no-cache"}


class PlainResponse(_CookieMixin):
    def __init__(self, body: str, status_code: int = 200, content_type: str = "text/plain"):
        self.status_code = status_code
        self.body = body.encode("utf-8")
        self.content_type = content_type
        self._cookies = []


def _compile_path(pattern: str):
    """'/api/forecast/{ticker}' -> compiled regex with named groups."""
    regex = re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", pattern)
    return re.compile(f"^{regex}$")


class Route:
    def __init__(self, path: str, handler, methods: list[str] | None = None):
        self.path = path
        self.handler = handler
        self.methods = methods or ["GET"]
        self.regex = _compile_path(path)


class Router:
    def __init__(self):
        self.routes: list[Route] = []
        self.static_dir: Path | None = None

    def add(self, path: str, handler, methods: list[str] | None = None):
        self.routes.append(Route(path, handler, methods))

    def serve_static(self, directory: Path):
        self.static_dir = directory

    def resolve(self, method: str, path: str):
        for route in self.routes:
            if method not in route.methods:
                continue
            m = route.regex.match(path)
            if m:
                return route, m.groupdict()
        return None, None


def make_app(router: Router):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            print(f"{self.address_string()} - {fmt % args}")

        def _dispatch(self, method: str):
            parsed = urllib.parse.urlsplit(self.path)
            path = urllib.parse.unquote(parsed.path)
            query = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}

            route, path_params = router.resolve(method, path)
            if route is None:
                if method == "GET" and router.static_dir is not None:
                    self._serve_static(path)
                    return
                self._send(JSONResponse({"error": "not found"}, status_code=404))
                return

            length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(length) if length else b""
            req_headers = {k: v for k, v in self.headers.items()}
            req = Request(
                method, path, query, path_params or {}, body,
                headers=req_headers, client_addr=self.client_address[0],
            )
            try:
                resp = route.handler(req)
            except Exception as e:  # pragma: no cover - defensive
                import traceback

                traceback.print_exc()
                resp = JSONResponse({"error": str(e)}, status_code=500)
            self._send(resp)

        def _serve_static(self, path: str):
            rel = path.lstrip("/") or "index.html"
            file_path = (router.static_dir / rel).resolve()
            try:
                file_path.relative_to(router.static_dir.resolve())
            except ValueError:
                self._send(JSONResponse({"error": "forbidden"}, status_code=403))
                return
            if not file_path.exists() or file_path.is_dir():
                if path == "/":
                    file_path = router.static_dir / "index.html"
                else:
                    self._send(JSONResponse({"error": "not found"}, status_code=404))
                    return
            self._send(FileResponse(file_path))

        def _send(self, resp):
            body = resp.body
            self.send_response(resp.status_code)
            self.send_header("Content-Type", resp.content_type)
            self.send_header("Content-Length", str(len(body)))
            # Cookies carry the session, so the wildcard CORS origin above
            # would be unsafe with Access-Control-Allow-Credentials: true --
            # this API is same-origin only (the frontend is served from the
            # same host:port), so no credentialed cross-origin requests are
            # needed and none are allowed.
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            for header_name, header_value in getattr(resp, "extra_headers", {}).items():
                self.send_header(header_name, header_value)
            for cookie in getattr(resp, "_cookies", []):
                self.send_header("Set-Cookie", cookie)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def do_DELETE(self):
            self._dispatch("DELETE")

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Content-Length", "0")
            self.end_headers()

    return Handler


def run(router: Router, host: str, port: int):
    handler_cls = make_app(router)
    server = ThreadingHTTPServer((host, port), handler_cls)
    server.daemon_threads = True
    print(f"Lumina Advisors listening on http://{host}:{port}")
    server.serve_forever()
