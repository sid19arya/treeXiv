"""Accounts for the web front-end: users, invites, passwords, sessions.

Web-only, like `web.py` — the CLI and the skill never import it. These two
tables, `users` and `invites`, are the only thing in the repo that outlives a
single run; there are deliberately no saved trees and no run history.

Storage is Postgres for a ``postgres://`` / ``postgresql://`` ``DATABASE_URL``
and SQLite for a ``sqlite:///path`` one (local dev and the test suite). The
SQL is hand-written and kept to what both dialects agree on, and tables are
created on first use — there is no migration tool, because there are two
tables.

Signup is invite-only. An invite is a random token handed out out-of-band;
the database keeps only its SHA-256, and a successful signup consumes it, so
a leaked link works at most once.

Sessions are a stateless signed cookie (``<user id>.<issued at>.<hmac>``),
so there is no sessions table either. It is checked against the ``users``
table on every request, so deleting a user signs them out.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from typing import Any, Literal

SESSION_COOKIE = "treexiv_session"
SESSION_MAX_AGE = 30 * 24 * 3600
MIN_PASSWORD_LENGTH = 10

# scrypt at n=2**14, r=8 needs 16 MiB per hash — well inside OpenSSL's
# default 32 MiB cap and the free tier's 512 MB.
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P, _SCRYPT_LEN = 2**14, 8, 1, 32

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_SCHEMA = {
    "postgres": """
        CREATE TABLE IF NOT EXISTS users (
            id BIGSERIAL PRIMARY KEY,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            last_login_at TIMESTAMPTZ
        );
        CREATE TABLE IF NOT EXISTS invites (
            id BIGSERIAL PRIMARY KEY,
            code_hash TEXT NOT NULL UNIQUE,
            note TEXT,
            created_at TIMESTAMPTZ NOT NULL,
            expires_at TIMESTAMPTZ,
            used_at TIMESTAMPTZ,
            used_by BIGINT REFERENCES users(id) ON DELETE SET NULL
        );
    """,
    # SQLite has no timestamp type; values are ISO-8601 UTC strings.
    "sqlite": """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_login_at TEXT
        );
        CREATE TABLE IF NOT EXISTS invites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code_hash TEXT NOT NULL UNIQUE,
            note TEXT,
            created_at TEXT NOT NULL,
            expires_at TEXT,
            used_at TEXT,
            used_by INTEGER REFERENCES users(id) ON DELETE SET NULL
        );
    """,
}


class StoreUnavailable(Exception):
    """The database couldn't be reached — a 503 for the caller, not a bug."""


class AccountError(Exception):
    """A signup or login the caller should be told about, not a server fault."""


class InvalidInvite(AccountError):
    pass


class EmailTaken(AccountError):
    pass


@dataclass(frozen=True)
class User:
    id: int
    email: str


def utcnow() -> datetime:
    return datetime.now(UTC)


def normalize_email(email: str) -> str:
    """Trim and lowercase; raise AccountError if it doesn't look like an address."""
    cleaned = email.strip().lower()
    if len(cleaned) > 254 or not _EMAIL_RE.match(cleaned):
        raise AccountError("That doesn't look like an email address.")
    return cleaned


def hash_invite(code: str) -> str:
    """What the ``invites`` table stores in place of the code itself."""
    return hashlib.sha256(code.strip().encode("utf-8")).hexdigest()


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(password: str) -> str:
    """``scrypt$n$r$p$salt$hash`` — stdlib scrypt, a fresh 16-byte salt each time."""
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_LEN,
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${_b64(salt)}${_b64(derived)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, expected = stored.split("$")
        if scheme != "scrypt":
            return False
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=_unb64(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(_unb64(expected)),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(derived, _unb64(expected))


@cache
def _dummy_hash() -> str:
    # Verified against when the email is unknown, so a miss costs the same
    # scrypt time as a wrong password and response timing doesn't reveal
    # which emails have accounts.
    return hash_password(secrets.token_urlsafe(16))


def sign_session(user_id: int, secret: str, *, now: float | None = None) -> str:
    payload = f"{user_id}.{int(now if now is not None else time.time())}"
    mac = hmac.new(secret.encode("utf-8"), payload.encode("ascii"), hashlib.sha256)
    return f"{payload}.{mac.hexdigest()}"


def read_session(token: str, secret: str, *, now: float | None = None) -> int | None:
    """The user id a session cookie was signed for, or None if it's forged,
    malformed, or older than SESSION_MAX_AGE."""
    parts = token.split(".")
    if len(parts) != 3 or not parts[0].isdigit() or not parts[1].isdigit():
        return None
    user_id, issued, mac = parts
    expected = hmac.new(
        secret.encode("utf-8"), f"{user_id}.{issued}".encode("ascii"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(mac, expected):
        return None
    age = (now if now is not None else time.time()) - int(issued)
    if age > SESSION_MAX_AGE or age < -60:
        return None
    return int(user_id)


class LoginThrottle:
    """At most ``limit`` failed logins per email per ``window`` seconds.

    In memory, so it resets on restart — it only has to make online guessing
    slow, and scrypt already makes each guess cost something.
    """

    def __init__(self, limit: int = 10, window: float = 15 * 60) -> None:
        self.limit = limit
        self.window = window
        self._failures: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, key: str, now: float) -> deque[float]:
        hits = self._failures.setdefault(key, deque())
        while hits and now - hits[0] > self.window:
            hits.popleft()
        return hits

    def blocked(self, key: str) -> bool:
        with self._lock:
            return len(self._recent(key, time.monotonic())) >= self.limit

    def failed(self, key: str) -> None:
        with self._lock:
            now = time.monotonic()
            self._recent(key, now).append(now)

    def reset(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)


class UserStore:
    """The ``users`` and ``invites`` tables behind one DATABASE_URL."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.dialect: Literal["postgres", "sqlite"]
        if url.startswith(("postgres://", "postgresql://")):
            self.dialect = "postgres"
        elif url.startswith("sqlite:///"):
            self.dialect = "sqlite"
            self._sqlite_path = url.removeprefix("sqlite:///")
        else:
            raise ValueError("DATABASE_URL must be a postgres:// or sqlite:/// URL.")

    @contextmanager
    def connection(self) -> Iterator[Any]:
        """One connection, committed on success and rolled back on error."""
        try:
            if self.dialect == "postgres":
                import psycopg  # the `web` extra; only needed for a Postgres URL

                conn: Any = psycopg.connect(self.url, connect_timeout=10)
            else:
                conn = sqlite3.connect(self._sqlite_path)
                conn.execute("PRAGMA foreign_keys = ON")
        except Exception as exc:  # any driver's "can't connect"
            raise StoreUnavailable(str(exc)) from exc
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def sql(self, query: str) -> str:
        """Queries are written with ``?`` placeholders; psycopg wants ``%s``."""
        return query.replace("?", "%s") if self.dialect == "postgres" else query

    def to_db(self, value: datetime | None) -> Any:
        if value is None or self.dialect == "postgres":
            return value
        return value.astimezone(UTC).isoformat()

    def from_db(self, value: Any) -> datetime | None:
        if value is None or isinstance(value, datetime):
            return value
        return datetime.fromisoformat(value)

    def init_schema(self) -> None:
        with self.connection() as conn:
            if self.dialect == "postgres":
                conn.execute(_SCHEMA["postgres"])
            else:
                conn.executescript(_SCHEMA["sqlite"])

    def get_user(self, user_id: int) -> User | None:
        with self.connection() as conn:
            row = conn.execute(
                self.sql("SELECT id, email FROM users WHERE id = ?"), (user_id,)
            ).fetchone()
        return User(id=row[0], email=row[1]) if row else None

    def authenticate(self, email: str, password: str) -> User | None:
        try:
            email = normalize_email(email)
        except AccountError:
            verify_password(password, _dummy_hash())
            return None
        with self.connection() as conn:
            row = conn.execute(
                self.sql("SELECT id, email, password_hash FROM users WHERE email = ?"),
                (email,),
            ).fetchone()
            if row is None:
                verify_password(password, _dummy_hash())
                return None
            if not verify_password(password, row[2]):
                return None
            conn.execute(
                self.sql("UPDATE users SET last_login_at = ? WHERE id = ?"),
                (self.to_db(utcnow()), row[0]),
            )
        return User(id=row[0], email=row[1])

    def signup(self, invite_code: str, email: str, password: str) -> User:
        """Create an account and consume the invite, in one transaction.

        Raises InvalidInvite (unknown, used, or expired — deliberately one
        error, so the response doesn't tell a guesser which), EmailTaken, or
        AccountError for a bad email or short password.
        """
        email = normalize_email(email)
        if len(password) < MIN_PASSWORD_LENGTH:
            raise AccountError(f"Use a password of at least {MIN_PASSWORD_LENGTH} characters.")
        password_hash = hash_password(password)
        now = utcnow()
        with self.connection() as conn:
            invite = conn.execute(
                self.sql(
                    "SELECT id, expires_at FROM invites WHERE code_hash = ? AND used_at IS NULL"
                ),
                (hash_invite(invite_code),),
            ).fetchone()
            expires_at = self.from_db(invite[1]) if invite else None
            if invite is None or (expires_at is not None and expires_at <= now):
                raise InvalidInvite("This invite link is invalid, expired, or already used.")
            taken = conn.execute(
                self.sql("SELECT 1 FROM users WHERE email = ?"), (email,)
            ).fetchone()
            if taken:
                raise EmailTaken("An account with that email already exists — sign in instead.")
            user_id = conn.execute(
                self.sql(
                    "INSERT INTO users (email, password_hash, created_at, last_login_at) "
                    "VALUES (?, ?, ?, ?) RETURNING id"
                ),
                (email, password_hash, self.to_db(now), self.to_db(now)),
            ).fetchone()[0]
            # `used_at IS NULL` again: if a concurrent signup consumed the
            # invite since the SELECT, nothing updates and this one rolls back.
            consumed = conn.execute(
                self.sql(
                    "UPDATE invites SET used_at = ?, used_by = ? "
                    "WHERE id = ? AND used_at IS NULL"
                ),
                (self.to_db(now), user_id, invite[0]),
            )
            if consumed.rowcount != 1:
                raise InvalidInvite("This invite link is invalid, expired, or already used.")
        return User(id=user_id, email=email)


_stores: dict[str, UserStore] = {}
_stores_lock = threading.Lock()


def open_store(url: str) -> UserStore:
    """The store for ``url``, with its tables created on first open."""
    with _stores_lock:
        store = _stores.get(url)
        if store is None:
            store = UserStore(url)
            store.init_schema()
            _stores[url] = store
        return store
