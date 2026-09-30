"""Accounts + sessions, stdlib-only (keeps the zero-third-party-dependency
philosophy the rest of this project follows -- see README's "Why zero
dependencies"). Passwords are hashed with PBKDF2-HMAC-SHA256 (100k
iterations, random per-user salt) via hashlib, which is in the standard
library and is a perfectly reasonable choice absent bcrypt/argon2. Sessions
are opaque random tokens (secrets.token_urlsafe) stored server-side in
SQLite and handed to the browser as an HttpOnly cookie -- nothing
cryptographic is done client-side, so there's no JWT-in-localStorage
XSS-exposure surface.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import re
import secrets
import threading
import time

from app import db

log = logging.getLogger("stockgraph.auth")

SESSION_COOKIE = "sg_session"
SESSION_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days
PBKDF2_ITERATIONS = 100_000
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(Exception):
    """Raised for any user-facing auth failure; api.py turns this into a 400."""


class _LoginRateLimiter:
    """Simple in-memory sliding-window brute-force guard for /api/login.

    No extra dependency (in keeping with the rest of this project) and no
    persistence -- a restart clears it, which is fine for what this is
    defending against (a script hammering the login endpoint in one sitting,
    not a long-running distributed attack, which would need real
    infrastructure -- a WAF/reverse-proxy rate limit -- ahead of this
    anyway). Two independent windows: per-email (stops guessing one
    account's password) and per-IP (stops one source spraying many emails).
    """

    WINDOW_SECONDS = 15 * 60
    MAX_PER_EMAIL = 10
    MAX_PER_IP = 30

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_email: dict[str, list[float]] = {}
        self._by_ip: dict[str, list[float]] = {}

    @staticmethod
    def _prune(hits: list[float], now: float) -> list[float]:
        cutoff = now - _LoginRateLimiter.WINDOW_SECONDS
        return [t for t in hits if t > cutoff]

    def check(self, email: str, ip: str | None) -> None:
        """Raises AuthError if this email or IP has too many recent failed
        attempts. Call before verifying the password."""
        now = time.time()
        with self._lock:
            email_hits = self._prune(self._by_email.get(email, []), now)
            if len(email_hits) >= self.MAX_PER_EMAIL:
                raise AuthError(
                    "Too many failed login attempts for this account. Try again in a few minutes."
                )
            if ip:
                ip_hits = self._prune(self._by_ip.get(ip, []), now)
                if len(ip_hits) >= self.MAX_PER_IP:
                    raise AuthError("Too many failed login attempts. Try again in a few minutes.")

    def record_failure(self, email: str, ip: str | None) -> None:
        now = time.time()
        with self._lock:
            self._by_email.setdefault(email, [])
            self._by_email[email] = self._prune(self._by_email[email], now) + [now]
            if ip:
                self._by_ip.setdefault(ip, [])
                self._by_ip[ip] = self._prune(self._by_ip[ip], now) + [now]

    def record_success(self, email: str, ip: str | None) -> None:
        with self._lock:
            self._by_email.pop(email, None)
            # Deliberately do not clear the per-IP bucket on a successful
            # login -- a correct guess for one account on an IP that's
            # actively spraying many emails shouldn't reset that IP's
            # counter and buy it more attempts against other accounts.


_login_limiter = _LoginRateLimiter()


class _SignupRateLimiter:
    """Per-IP sliding-window guard for /api/auth/signup.

    Separate from _LoginRateLimiter above and simpler: signup has no
    existing account to protect (that's what the duplicate-email check is
    for), so the only thing worth defending against here is a script mass-
    creating accounts from one source. Every signup call counts toward the
    window, not just failures -- unlike login, there's no legitimate reason
    for one IP to hit this endpoint dozens of times in 15 minutes.
    """

    WINDOW_SECONDS = 15 * 60
    MAX_PER_IP = 5

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_ip: dict[str, list[float]] = {}

    @staticmethod
    def _prune(hits: list[float], now: float) -> list[float]:
        cutoff = now - _SignupRateLimiter.WINDOW_SECONDS
        return [t for t in hits if t > cutoff]

    def check_and_record(self, ip: str | None) -> None:
        """Raises AuthError if this IP has signed up too many times
        recently; otherwise records this attempt. Call at the very start of
        signup(), before any validation or DB work."""
        if not ip:
            return
        now = time.time()
        with self._lock:
            hits = self._prune(self._by_ip.get(ip, []), now)
            if len(hits) >= self.MAX_PER_IP:
                raise AuthError(
                    "Too many accounts created from this location recently. Try again in a few minutes."
                )
            hits.append(now)
            self._by_ip[ip] = hits


_signup_limiter = _SignupRateLimiter()


def _hash_password(password: str, salt: bytes) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return dk.hex()


def hash_password(password: str) -> tuple[str, str]:
    """Returns (password_hash_hex, salt_hex)."""
    salt = secrets.token_bytes(16)
    return _hash_password(password, salt), salt.hex()


def verify_password(password: str, password_hash_hex: str, salt_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    candidate = _hash_password(password, salt)
    return hmac.compare_digest(candidate, password_hash_hex)


def validate_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not email or not EMAIL_RE.match(email):
        raise AuthError("Enter a valid email address.")
    return email


def validate_password(password: str) -> str:
    if not password or len(password) < 8:
        raise AuthError("Password must be at least 8 characters.")
    return password


def signup(email: str, password: str, display_name: str = "", client_ip: str | None = None) -> dict:
    _signup_limiter.check_and_record(client_ip)
    email = validate_email(email)
    validate_password(password)
    if db.get_user_by_email(email):
        raise AuthError("An account with that email already exists.")
    password_hash, salt = hash_password(password)
    name = (display_name or email.split("@")[0]).strip()[:60]
    user_id = db.create_user(email=email, password_hash=password_hash, salt=salt, display_name=name)
    return db.get_user(user_id)


def login(email: str, password: str, client_ip: str | None = None) -> dict:
    email = validate_email(email)
    _login_limiter.check(email, client_ip)
    user = db.get_user_by_email(email)
    if not user or not verify_password(password, user["password_hash"], user["salt"]):
        _login_limiter.record_failure(email, client_ip)
        raise AuthError("Incorrect email or password.")
    _login_limiter.record_success(email, client_ip)
    return dict(user)


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    db.create_session(token, user_id, ttl_seconds=SESSION_TTL_SECONDS)
    return token


def user_from_token(token: str | None) -> dict | None:
    if not token:
        return None
    row = db.get_session_user(token)
    return dict(row) if row else None


def logout(token: str | None) -> None:
    if token:
        db.delete_session(token)


def public_user(user: dict) -> dict:
    return {
        "id": user["id"],
        "email": user["email"],
        "display_name": user["display_name"],
        "created_at": user["created_at"],
    }
