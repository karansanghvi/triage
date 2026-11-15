"""Passwords, emails, display names and session tokens (app/accounts/credentials.py). No network, no database."""
import time

import pytest
from argon2 import PasswordHasher

from app.accounts import credentials as c
from app.providers.keys import REDACTED, redact

PASSWORD = "correct horse battery"


# ---------- passwords ----------

def test_hash_is_argon2id_with_owasp_parameters_and_verifies():
    h = c.hash_password(PASSWORD)
    assert h.startswith("$argon2id$v=19$m=19456,t=2,p=1$") and PASSWORD not in h
    assert c.verify_password(h, PASSWORD) and not c.verify_password(h, PASSWORD + "!")
    assert c.hash_password(PASSWORD) != h                              # a new salt each time


@pytest.mark.parametrize("password", ["x" * 9, "x" * 129, "", None, 1234567890])
def test_password_length_rules(password):
    with pytest.raises(c.CredentialError) as error:
        c.hash_password(password)
    assert "10 to 128" in str(error.value) and (not isinstance(password, str) or not password
                                                or password not in str(error.value))


@pytest.mark.parametrize("password", ["x" * 10, "x" * 128, "pässwörd-ünïcode"])
def test_passwords_at_the_limits_are_fine(password):
    assert c.verify_password(c.hash_password(password), password)


def test_an_unknown_user_is_checked_against_a_dummy_hash():
    assert c.verify_password(None, PASSWORD) is False
    real = c.hash_password(PASSWORD)
    started = time.perf_counter(); c.verify_password(real, "wrong password!"); real_time = time.perf_counter() - started
    started = time.perf_counter(); c.verify_password(None, "wrong password!"); dummy_time = time.perf_counter() - started
    assert dummy_time > real_time / 4                                  # it really ran argon2, not a shortcut


def test_a_broken_hash_is_a_failed_check_not_a_crash():
    assert c.verify_password("not-a-hash", PASSWORD) is False


def test_old_parameters_need_a_rehash():
    weak = PasswordHasher(time_cost=1, memory_cost=8 * 1024, parallelism=1).hash(PASSWORD)
    assert c.needs_rehash(weak) and not c.needs_rehash(c.hash_password(PASSWORD))
    assert c.verify_password(weak, PASSWORD)                           # still accepted


# ---------- emails and names ----------

@pytest.mark.parametrize("raw, normalized", [
    ("  Alice@Example.COM ", "alice@example.com"),
    ("first.last+tag@example.com", "first.last+tag@example.com"),       # dots and +tags kept
    ("ＡＬＩＣＥ@ｅｘａｍｐｌｅ.com", "alice@example.com"),                    # NFKC: full-width letters
])
def test_email_normalization(raw, normalized):
    assert c.normalize_email(raw) == normalized


@pytest.mark.parametrize("raw", ["", "alice", "alice@", "@example.com", "alice@example", "a b@example.com",
                                 "a@b@example.com", "alice@example.", None, "x" * 250 + "@ex.com"])
def test_invalid_emails_are_refused_without_echoing_them(raw):
    with pytest.raises(c.CredentialError) as error:
        c.normalize_email(raw)
    assert str(error.value) == "Enter a valid email address."


@pytest.mark.parametrize("raw, name", [(None, None), ("", None), ("   ", None), ("  Ada  ", "Ada"),
                                       ("x" * 80, "x" * 80)])
def test_display_names(raw, name):
    assert c.normalize_display_name(raw) == name


def test_a_long_display_name_is_refused():
    with pytest.raises(c.CredentialError, match="at most 80"):
        c.normalize_display_name("x" * 81)


# ---------- session tokens ----------

def test_session_tokens_have_the_prefix_and_256_random_bits():
    tokens = {c.new_session_token() for _ in range(200)}
    assert len(tokens) == 200
    for t in tokens:
        assert t.startswith("trs_") and len(t) == 4 + 43 and c.looks_like_session_token(t)


def test_only_a_sha256_is_stored_and_compared_in_constant_time():
    token = c.new_session_token()
    h = c.token_hash(token)
    assert len(h) == 64 and token not in h and c.hashes_match(h, c.token_hash(token))
    assert not c.hashes_match(h, c.token_hash(c.new_session_token()))


@pytest.mark.parametrize("value", ["trs_short", "sk-abc", "trs_" + "!" * 43, "TRS_" + "a" * 43, ""])
def test_things_that_are_not_session_tokens(value):
    assert not c.looks_like_session_token(value)


def test_session_tokens_and_account_keys_are_redacted():
    token = c.new_session_token()
    text = f"bad token {token} and key trk_{'a' * 40} in an error"
    assert token not in redact(text) and "trk_aaaa" not in redact(text) and redact(text).count(REDACTED) == 2
    assert redact("the trs_ prefix and a trk_ note") == "the trs_ prefix and a trk_ note"   # short: untouched
