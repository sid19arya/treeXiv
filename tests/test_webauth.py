"""Tests for web accounts (`treexiv.webauth`), against a throwaway SQLite file."""

from __future__ import annotations

import secrets
from datetime import timedelta
from pathlib import Path

import pytest

from treexiv.webauth import (
    SESSION_MAX_AGE,
    AccountError,
    EmailTaken,
    InvalidInvite,
    LoginThrottle,
    StoreUnavailable,
    UserStore,
    hash_invite,
    hash_password,
    read_session,
    sign_session,
    utcnow,
    verify_password,
)


def add_invite(store: UserStore, code: str, *, expires_in: timedelta | None = None) -> None:
    """Insert an invite the way an operator would: only the code's hash is stored."""
    now = utcnow()
    with store.connection() as conn:
        conn.execute(
            store.sql(
                "INSERT INTO invites (code_hash, note, created_at, expires_at) "
                "VALUES (?, ?, ?, ?)"
            ),
            (
                hash_invite(code),
                "test",
                store.to_db(now),
                store.to_db(now + expires_in) if expires_in is not None else None,
            ),
        )


@pytest.fixture
def store(tmp_path: Path) -> UserStore:
    s = UserStore(f"sqlite:///{tmp_path / 'web.sqlite3'}")
    s.init_schema()
    return s


def test_password_hash_round_trip() -> None:
    stored = hash_password("correct horse battery")
    assert stored.startswith("scrypt$")
    assert verify_password("correct horse battery", stored)
    assert not verify_password("wrong horse battery", stored)


def test_password_hashes_are_salted() -> None:
    assert hash_password("same password") != hash_password("same password")


@pytest.mark.parametrize("stored", ["", "plain", "bcrypt$1$2$3$aa$bb", "scrypt$x$8$1$aa$bb"])
def test_malformed_hash_never_verifies(stored: str) -> None:
    assert not verify_password("anything", stored)


def test_session_round_trip() -> None:
    token = sign_session(42, "secret", now=1_000_000)
    assert read_session(token, "secret", now=1_000_100) == 42


def test_session_rejects_wrong_secret_and_tampering() -> None:
    token = sign_session(42, "secret", now=1_000_000)
    assert read_session(token, "other", now=1_000_100) is None
    user_id, issued, mac = token.split(".")
    assert read_session(f"43.{issued}.{mac}", "secret", now=1_000_100) is None
    assert read_session("garbage", "secret") is None
    assert read_session("a.b.c", "secret") is None


def test_session_expires() -> None:
    token = sign_session(42, "secret", now=1_000_000)
    assert read_session(token, "secret", now=1_000_000 + SESSION_MAX_AGE + 1) is None


def test_signup_creates_user_and_consumes_invite(store: UserStore) -> None:
    add_invite(store, "invite-one")
    user = store.signup("invite-one", "  Ada@Example.com ", "a long password")
    assert user.email == "ada@example.com"
    assert store.get_user(user.id) == user
    with pytest.raises(InvalidInvite):
        store.signup("invite-one", "someone@example.com", "a long password")


def test_invite_code_is_never_stored(store: UserStore) -> None:
    add_invite(store, "invite-secret")
    with store.connection() as conn:
        rows = conn.execute("SELECT code_hash FROM invites").fetchall()
    assert rows == [(hash_invite("invite-secret"),)]


def test_signup_rejects_unknown_and_expired_invites(store: UserStore) -> None:
    with pytest.raises(InvalidInvite):
        store.signup("never-issued", "ada@example.com", "a long password")
    add_invite(store, "old-invite", expires_in=timedelta(days=-1))
    with pytest.raises(InvalidInvite):
        store.signup("old-invite", "ada@example.com", "a long password")


def test_signup_rejects_taken_email_without_burning_the_invite(store: UserStore) -> None:
    add_invite(store, "first")
    add_invite(store, "second")
    store.signup("first", "ada@example.com", "a long password")
    with pytest.raises(EmailTaken):
        store.signup("second", "ADA@example.com", "a long password")
    # "second" is still usable by someone else.
    assert store.signup("second", "bob@example.com", "a long password").email == (
        "bob@example.com"
    )


@pytest.mark.parametrize(
    ("email", "password"),
    [("not-an-email", "a long password"), ("ada@example.com", "short")],
)
def test_signup_validates_email_and_password(
    store: UserStore, email: str, password: str
) -> None:
    add_invite(store, "invite")
    with pytest.raises(AccountError):
        store.signup("invite", email, password)


def test_authenticate(store: UserStore) -> None:
    add_invite(store, "invite")
    user = store.signup("invite", "ada@example.com", "a long password")
    assert store.authenticate("ADA@example.com", "a long password") == user
    assert store.authenticate("ada@example.com", "wrong password") is None
    assert store.authenticate("nobody@example.com", "a long password") is None
    assert store.authenticate("not an email", "a long password") is None


def test_deleted_user_is_gone(store: UserStore) -> None:
    add_invite(store, "invite")
    user = store.signup("invite", "ada@example.com", "a long password")
    with store.connection() as conn:
        conn.execute(store.sql("DELETE FROM users WHERE id = ?"), (user.id,))
    assert store.get_user(user.id) is None


def test_unreachable_database_raises_store_unavailable(tmp_path: Path) -> None:
    store = UserStore(f"sqlite:///{tmp_path / 'missing-dir' / 'web.sqlite3'}")
    with pytest.raises(StoreUnavailable):
        store.init_schema()


def test_unsupported_url() -> None:
    with pytest.raises(ValueError):
        UserStore("mysql://localhost/db")


def test_login_throttle() -> None:
    throttle = LoginThrottle(limit=3, window=60)
    key = secrets.token_hex(4)
    for _ in range(3):
        assert not throttle.blocked(key)
        throttle.failed(key)
    assert throttle.blocked(key)
    throttle.reset(key)
    assert not throttle.blocked(key)
