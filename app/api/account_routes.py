"""Accounts (Phase 5): sign up, log in, log out, who am I, and account settings. Only when TRIAGE_ACCOUNTS is on.

    POST   /v1/auth/signup      {email, password, display_name?} -> 201 {user, session_token, expires_at}
    POST   /v1/auth/login       {email, password}                -> 200 {user, session_token, expires_at}
    POST   /v1/auth/logout                                       -> 204 (revokes this session)
    GET    /v1/auth/me                                           -> {id, email, display_name, preferred_provider,
                                                                     created_at}
    PATCH  /v1/account          {display_name?, preferred_provider?}            -> the user, as /me
    POST   /v1/account/password {current_password, new_password}               -> 204
    DELETE /v1/account          {password}                                     -> 204

    GET    /v1/account/provider-keys                    -> [{provider, last4, status, verified_at, created_at,
                                                             updated_at}]   (never the key)
    PUT    /v1/account/provider-keys/{provider} {key}   -> the saved key's summary (checked with the provider first)
    POST   /v1/account/provider-keys/{provider}/verify  -> the summary after checking the saved key again
    DELETE /v1/account/provider-keys/{provider}         -> 204

    GET    /v1/account/api-keys          -> [{id, name, public_id, created_at, last_used_at, revoked_at}], newest
                                            first (never the key)
    POST   /v1/account/api-keys {name}   -> 201 {id, name, public_id, created_at, key, note}: the only time the
                                            full key is shown
    DELETE /v1/account/api-keys/{id}     -> 204 (revokes it; unknown or someone else's: 404)

    POST   /v1/auth/password-reset/request  {email}                   -> 202, always the same answer
    POST   /v1/auth/password-reset/verify   {email, code}             -> {reset_token, expires_at}
    POST   /v1/auth/password-reset/complete {reset_token, new_password}
                                                                   -> {user, session_token, expires_at}, as login

Everything except signup, login and password reset needs a session token (Authorization: Bearer trs_...), never a Triage API key.

Never logged or echoed: emails, passwords, tokens. Auth failures log only the visitor's hash and the outcome.
Every database call and password hash runs in a worker thread.
"""
import json
import logging
import unicodedata
import uuid
from typing import Optional

import anyio
from fastapi import APIRouter, BackgroundTasks, Depends, Header, Request, Response
from pydantic import BaseModel, ConfigDict

from app.accounts.credentials import (CredentialError, check_password_rules, hash_password,
                                      looks_like_reset_token, looks_like_session_token, needs_rehash,
                                      normalize_display_name, normalize_email, verify_password)
from app.accounts.email import EmailSendError
from app.accounts.api_keys import MAX_ACTIVE_KEYS, NAME_MAX as API_KEY_NAME_MAX, TooManyKeys
from app.accounts.encryption import DecryptionError
from app.accounts.ratelimit import RateLimited, email_subject
from app.accounts.verify_keys import NAMES
from app.accounts.store import PROVIDERS, EmailTaken
from app.api.demo import client_ip, visitor_id
from app.api.errors import APIError, bad_request

logger = logging.getLogger("triage.auth")

INVALID_CREDENTIALS = "Incorrect email or password."
NEEDS_SESSION = ("This endpoint needs a session token: log in with POST /v1/auth/login and send the token as "
                 "Authorization: Bearer <token>.")
SESSION_ENDED = "Your session has expired or was revoked. Log in again."
INCORRECT_PASSWORD = "That password isn't right."


def _log(event: str, outcome: str, visitor: Optional[str] = None, detail: Optional[str] = None) -> None:
    """One JSON line: the event, the outcome and the visitor's hash (detail: a short fixed kind, e.g. why an
    email failed). Never an email, password, code or token."""
    line = {"event": event, "outcome": outcome, "visitor": visitor}
    if detail is not None:
        line["detail"] = detail
    logger.info(json.dumps(line, separators=(",", ":")))


def _invalid_credentials() -> APIError:
    return APIError(401, INVALID_CREDENTIALS, code="invalid_credentials")


def _too_many(error: RateLimited, what: str) -> APIError:
    minutes = max(1, -(-error.retry_after // 60))
    return APIError(429, f"Too many {what} attempts. Try again in {minutes} minute{'s' if minutes != 1 else ''}.",
                    "rate_limit_error", code="too_many_attempts", headers={"Retry-After": str(error.retry_after)})


def _session_response(user, token: str, session, status: int = 200):
    from fastapi.responses import JSONResponse
    return JSONResponse({"user": user.public(), "session_token": token,
                         "expires_at": session.expires_at.isoformat().replace("+00:00", "Z")}, status_code=status)


def _credential_error(error: CredentialError, param: str) -> APIError:
    return bad_request(str(error), param, "invalid_value")


# ----- request bodies (unknown fields are refused; validation errors never include the values) -----

class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SignupBody(_Body):
    email: str
    password: str
    display_name: Optional[str] = None


class LoginBody(_Body):
    email: str
    password: str


class AccountPatch(_Body):
    display_name: Optional[str] = None
    preferred_provider: Optional[str] = None


class PasswordChange(_Body):
    current_password: str
    new_password: str


class AccountDelete(_Body):
    password: str


class ResetRequestBody(_Body):
    email: str


class ResetVerifyBody(_Body):
    email: str
    code: str


class ResetCompleteBody(_Body):
    reset_token: str
    new_password: str


RESET_SENT = ("If an account exists for that email, we've sent it a 6-digit code. It expires in 10 minutes. "
              "Check your spam folder if it doesn't arrive.")
INVALID_CODE = ("That code isn't right, or it has expired. Use the code from the latest email, or request a new "
                "one.")
INVALID_RESET_TOKEN = "This password reset has expired or was already used. Request a new code to start again."
EMAIL_UNAVAILABLE = "Password reset by email isn't available on this server right now."


class ProviderKeyBody(_Body):
    key: str


class ApiKeyCreate(_Body):
    name: str


ONLY_ONCE = ("This is the only time the full key is shown. Store it somewhere safe: Triage keeps only a hash "
             "and can't show it again. If you lose it, revoke it and create a new one.")


MAX_PROVIDER_KEY_CHARS = 512


def rejected_message(provider: str, verdict) -> str:
    """Why a key wasn't saved. Gemini's 403 is a project-level refusal (billing, verification, restrictions, a
    blocked key), not a typo, so it says so. verdict.detail is already redacted; the key itself is never used."""
    if provider == "gemini" and verdict.http_status == 403:
        text = ("Google refused this Gemini key for a project, billing or permission reason (HTTP 403), so it "
                "wasn't saved. The key itself may be typed correctly: check its Google Cloud project, billing, "
                "verification and API restrictions in Google AI Studio.")
    else:
        text = f"{NAMES[provider]} rejected this key, so it wasn't saved. Check that it was copied correctly."
    return f"{text} ({verdict.detail})" if verdict.detail else text


def account_router() -> APIRouter:
    router = APIRouter()

    def visitor(request: Request) -> str:
        deps = request.app.state.deps
        return visitor_id(client_ip(request, deps.trusted_proxy_hops), deps.hash_secret)

    async def current_session(request: Request, authorization: Optional[str] = Header(None)) -> tuple:
        """(Session, User) for a valid session token. Anything else, including a Triage API key, is a 401."""
        token = authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else ""
        if not looks_like_session_token(token):
            raise APIError(401, NEEDS_SESSION, code="invalid_session")
        found = await anyio.to_thread.run_sync(request.app.state.deps.accounts.session_for_token, token)
        if found is None:
            raise APIError(401, SESSION_ENDED, code="invalid_session")
        return found

    # ----- sign up, log in, log out, me -----

    @router.post("/v1/auth/signup", status_code=201)
    async def signup(body: SignupBody, request: Request, background: BackgroundTasks):
        deps, who = request.app.state.deps, visitor(request)
        try:
            email = normalize_email(body.email)
        except CredentialError as error:
            raise _credential_error(error, "email") from None
        try:
            display_name = normalize_display_name(body.display_name)
        except CredentialError as error:
            raise _credential_error(error, "display_name") from None
        try:
            check_password_rules(body.password)
        except CredentialError as error:
            raise _credential_error(error, "password") from None
        try:
            await anyio.to_thread.run_sync(deps.auth_limiter.hit, "signup_ip", who)
        except RateLimited as error:
            _log("signup", "rate_limited", who)
            raise _too_many(error, "sign-up") from None
        password_hash = await anyio.to_thread.run_sync(hash_password, body.password)
        try:
            user = await anyio.to_thread.run_sync(deps.accounts.create_user, email, password_hash, display_name)
        except EmailTaken:
            _log("signup", "email_taken", who)
            raise APIError(409, "An account with this email already exists. Log in instead.",
                           code="email_taken", param="email") from None
        token, session = await anyio.to_thread.run_sync(deps.accounts.create_session, user.id)
        _log("signup", "ok", who)
        if deps.email_sender is not None and deps.email_sender.available:
            background.add_task(send_welcome, deps, user, who)       # after the response: never slows signup
        return _session_response(user, token, session, status=201)

    @router.post("/v1/auth/login")
    async def login(body: LoginBody, request: Request):
        """Every failure is the same 401: unknown email, wrong password, a malformed email, or too many recent
        failures for this email. Each still runs one argon2 check, so timing doesn't tell them apart."""
        deps, who = request.app.state.deps, visitor(request)
        try:
            await anyio.to_thread.run_sync(deps.auth_limiter.hit, "login_ip", who)
        except RateLimited as error:
            _log("login", "rate_limited", who)
            raise _too_many(error, "login") from None
        try:
            email = normalize_email(body.email)
        except CredentialError:
            email = None
        user, locked, subject = None, False, None
        if email is not None:
            subject = email_subject(email, deps.hash_secret)
            try:
                await anyio.to_thread.run_sync(deps.auth_limiter.check, "login_failure_email", subject)
            except RateLimited:
                locked = True
            if not locked:
                user = await anyio.to_thread.run_sync(deps.accounts.user_by_email, email)
        ok = await anyio.to_thread.run_sync(verify_password, user.password_hash if user else None, body.password)
        if not ok or locked:
            if subject is not None and not locked:
                await anyio.to_thread.run_sync(deps.auth_limiter.record, "login_failure_email", subject)
            _log("login", "email_locked" if locked else "failed", who)
            raise _invalid_credentials()
        if needs_rehash(user.password_hash):            # parameters changed since it was hashed
            new_hash = await anyio.to_thread.run_sync(hash_password, body.password)
            await anyio.to_thread.run_sync(lambda: deps.accounts.update_user(user.id, password_hash=new_hash))
        token, session = await anyio.to_thread.run_sync(deps.accounts.create_session, user.id)
        _log("login", "ok", who)
        return _session_response(user, token, session)

    @router.post("/v1/auth/logout", status_code=204)
    async def logout(request: Request, current=Depends(current_session)):
        session, _ = current
        await anyio.to_thread.run_sync(request.app.state.deps.accounts.revoke_session, session.id)
        return Response(status_code=204)

    @router.get("/v1/auth/me")
    async def me(current=Depends(current_session)):
        return current[1].public()

    # ----- account settings -----

    async def check_password(request: Request, user, password: str, event: str) -> None:
        """The current password, guarded by the same per-email failure limit as login (5 per 15 minutes,
        counted together with failed logins, so switching endpoints buys no extra guesses). Over the limit:
        429 with Retry-After, before any check. Wrong: 403 incorrect_password (not 401: the session is fine,
        so the website shouldn't log the user out), and the failure is counted."""
        deps = request.app.state.deps
        subject = email_subject(user.email, deps.hash_secret)
        try:
            await anyio.to_thread.run_sync(deps.auth_limiter.check, "login_failure_email", subject)
        except RateLimited as error:
            _log(event, "rate_limited", visitor(request))
            raise _too_many(error, "password") from None
        ok = await anyio.to_thread.run_sync(verify_password, user.password_hash, password)
        if not ok:
            await anyio.to_thread.run_sync(deps.auth_limiter.record, "login_failure_email", subject)
            _log(event, "incorrect_password", visitor(request))
            raise APIError(403, INCORRECT_PASSWORD, "permission_error", code="incorrect_password")

    @router.patch("/v1/account")
    async def update_account(body: AccountPatch, request: Request, current=Depends(current_session)):
        """Only the fields sent change; null clears one. preferred_provider: openai, anthropic, groq or gemini
        (any case), the default for X-Triage-Prefer-Provider once account requests exist."""
        _, user = current
        values = {}
        if "display_name" in body.model_fields_set:
            try:
                values["display_name"] = normalize_display_name(body.display_name)
            except CredentialError as error:
                raise _credential_error(error, "display_name") from None
        if "preferred_provider" in body.model_fields_set:
            provider = (body.preferred_provider or "").strip().lower() or None
            if provider is not None and provider not in PROVIDERS:
                raise bad_request(f"preferred_provider must be one of {', '.join(PROVIDERS)}, or null.",
                                  "preferred_provider", "invalid_value")
            values["preferred_provider"] = provider
        if values:
            user = await anyio.to_thread.run_sync(lambda: request.app.state.deps.accounts.update_user(user.id,
                                                                                                    **values))
        return user.public()

    @router.post("/v1/account/password", status_code=204)
    async def change_password(body: PasswordChange, request: Request, current=Depends(current_session)):
        """Every other session of the user is revoked; this one stays logged in."""
        session, user = current
        try:
            check_password_rules(body.new_password)
        except CredentialError as error:
            raise _credential_error(error, "new_password") from None
        await check_password(request, user, body.current_password, "password_change")
        accounts = request.app.state.deps.accounts
        new_hash = await anyio.to_thread.run_sync(hash_password, body.new_password)
        await anyio.to_thread.run_sync(lambda: accounts.update_user(user.id, password_hash=new_hash))
        await anyio.to_thread.run_sync(accounts.revoke_other_sessions, user.id, session.id)
        _log("password_change", "ok", visitor(request))
        return Response(status_code=204)

    @router.delete("/v1/account", status_code=204)
    async def delete_account(body: AccountDelete, request: Request, current=Depends(current_session)):
        """The user and all their sessions, in one transaction. Their request log rows stay, with user_id NULL."""
        _, user = current
        await check_password(request, user, body.password, "delete_account")
        await anyio.to_thread.run_sync(request.app.state.deps.accounts.delete_user, user.id)
        _log("delete_account", "ok", visitor(request))
        return Response(status_code=204)

    # ----- password reset (no session: the user has lost their password) -----

    def email_ready(request: Request) -> None:
        sender = request.app.state.deps.email_sender
        if sender is None or not sender.available:
            raise APIError(503, EMAIL_UNAVAILABLE, "server_error", code="email_unavailable")

    async def limit(request: Request, kind: str, subject: str, event: str, what: str) -> None:
        try:
            await anyio.to_thread.run_sync(request.app.state.deps.auth_limiter.hit, kind, subject)
        except RateLimited as error:
            _log(event, "rate_limited", visitor(request), kind)
            raise _too_many(error, what) from None

    async def send_welcome(deps, user, who: str) -> None:
        """Once, after a signup's 201 has gone out. A failure is logged (outcome and short kind) and not retried."""
        try:
            await anyio.to_thread.run_sync(lambda: deps.email_sender.send_welcome(
                user.email, user.id, str(uuid.uuid4()), display_name=user.display_name))
            _log("welcome_email", "sent", who)
        except EmailSendError as error:
            _log("welcome_email", "send_failed", who, error.kind)
        except Exception as error:             # the type only: its text could quote the email
            _log("welcome_email", "error", who, type(error).__name__)

    async def send_reset_code(deps, email: str, who: str) -> None:
        """After the 202 has gone out, so the response takes as long whether or not the account exists."""
        try:
            user = await anyio.to_thread.run_sync(deps.accounts.user_by_email, email)
            if user is None:
                _log("password_reset_request", "no_account", who)
                return
            issued = await anyio.to_thread.run_sync(deps.password_resets.issue_code, user.id)
            if issued is None:
                _log("password_reset_request", "cooldown", who)
                return
            code, code_id = issued
            await anyio.to_thread.run_sync(lambda: deps.email_sender.send_reset_code(
                user.email, user.id, code, code_id, display_name=user.display_name))
            _log("password_reset_request", "sent", who)
        except EmailSendError as error:
            _log("password_reset_request", "send_failed", who, error.kind)
        except Exception as error:             # the type only: a database error's text can quote the email
            _log("password_reset_request", "error", who, type(error).__name__)

    @router.post("/v1/auth/password-reset/request", status_code=202)
    async def password_reset_request(body: ResetRequestBody, request: Request, background: BackgroundTasks):
        """Always the same 202 (unless malformed or rate limited, which never depends on the account). Finding the
        user, the cooldown, the code and the email all happen in a background task."""
        email_ready(request)
        deps, who = request.app.state.deps, visitor(request)
        try:
            email = normalize_email(body.email)
        except CredentialError as error:
            raise _credential_error(error, "email") from None
        await limit(request, "reset_request_ip", who, "password_reset_request", "password reset")
        await limit(request, "reset_request_email", email_subject(email, deps.hash_secret),
                    "password_reset_request", "password reset")
        background.add_task(send_reset_code, deps, email, who)
        return {"message": RESET_SENT}

    @router.post("/v1/auth/password-reset/verify")
    async def password_reset_verify(body: ResetVerifyBody, request: Request):
        """A wrong code, an unknown email and a dead, used or expired code are the same 400 invalid_code (no
        attempts-left count: it would show the account exists)."""
        email_ready(request)
        deps, who = request.app.state.deps, visitor(request)
        try:
            email = normalize_email(body.email)
        except CredentialError as error:
            raise _credential_error(error, "email") from None
        await limit(request, "reset_verify_ip", who, "password_reset_verify", "reset code")
        subject = email_subject(email, deps.hash_secret)
        try:
            await anyio.to_thread.run_sync(deps.auth_limiter.check, "reset_verify_failure_email", subject)
        except RateLimited as error:
            _log("password_reset_verify", "rate_limited", who, "reset_verify_failure_email")
            raise _too_many(error, "reset code") from None
        user = await anyio.to_thread.run_sync(deps.accounts.user_by_email, email)
        result = await anyio.to_thread.run_sync(deps.password_resets.verify_code, user.id if user else None,
                                                body.code)
        if result is None:
            await anyio.to_thread.run_sync(deps.auth_limiter.record, "reset_verify_failure_email", subject)
            _log("password_reset_verify", "invalid_code", who)
            raise APIError(400, INVALID_CODE, param="code", code="invalid_code")
        token, expires_at = result
        _log("password_reset_verify", "ok", who)
        from fastapi.responses import JSONResponse
        return JSONResponse({"reset_token": token, "expires_at": expires_at.isoformat().replace("+00:00", "Z")},
                            headers={"Cache-Control": "no-store"})

    @router.post("/v1/auth/password-reset/complete")
    async def password_reset_complete(body: ResetCompleteBody, request: Request):
        """Sets the password, revokes every session (account API keys stay valid) and logs the user in."""
        email_ready(request)
        deps, who = request.app.state.deps, visitor(request)
        if not looks_like_reset_token(body.reset_token):
            raise APIError(400, INVALID_RESET_TOKEN, param="reset_token", code="invalid_reset_token")
        try:
            check_password_rules(body.new_password)
        except CredentialError as error:
            raise _credential_error(error, "new_password") from None
        if not await anyio.to_thread.run_sync(deps.password_resets.token_is_live, body.reset_token):
            _log("password_reset_complete", "invalid_token", who)
            raise APIError(400, INVALID_RESET_TOKEN, param="reset_token", code="invalid_reset_token")
        new_hash = await anyio.to_thread.run_sync(hash_password, body.new_password)
        user_id = await anyio.to_thread.run_sync(deps.password_resets.complete, body.reset_token, new_hash)
        if user_id is None:                     # used by a parallel request meanwhile
            _log("password_reset_complete", "invalid_token", who)
            raise APIError(400, INVALID_RESET_TOKEN, param="reset_token", code="invalid_reset_token")
        user = await anyio.to_thread.run_sync(deps.accounts.user_by_id, user_id)
        token, session = await anyio.to_thread.run_sync(deps.accounts.create_session, user.id)
        _log("password_reset_complete", "ok", who)
        return _session_response(user, token, session)

    # ----- saved provider keys -----

    def provider_name(provider: str) -> str:
        name = provider.strip().lower()
        if name not in PROVIDERS:
            raise bad_request(f"provider must be one of {', '.join(PROVIDERS)}.", "provider", "invalid_value")
        return name

    def key_store(request: Request):
        store = request.app.state.deps.provider_keys
        if store.keyring is None:          # only when started without set_up_keyring (it refuses to start)
            raise APIError(503, "Saving provider keys isn't available: the server has no encryption keys.",
                           "server_error", code="encryption_unavailable")
        return store

    async def count_key_check(request: Request, user) -> None:
        try:
            await anyio.to_thread.run_sync(request.app.state.deps.auth_limiter.hit, "provider_key_user", user.id)
        except RateLimited as error:
            _log("provider_key", "rate_limited", visitor(request))
            raise _too_many(error, "provider key") from None

    @router.get("/v1/account/provider-keys")
    async def list_provider_keys(request: Request, current=Depends(current_session)):
        saved = await anyio.to_thread.run_sync(key_store(request).list, current[1].id)
        return [k.public() for k in saved]

    @router.put("/v1/account/provider-keys/{provider}")
    async def save_provider_key(provider: str, body: ProviderKeyBody, request: Request,
                                current=Depends(current_session)):
        """Checked with the provider first: rejected (401/403) is a 400 and nothing is stored; no clear answer
        (timeout, network error, 429, 5xx) is stored as "unverified". Replaces any saved key for the provider."""
        _, user = current
        provider = provider_name(provider)
        store = key_store(request)
        key = body.key.strip()
        if not key or len(key) > MAX_PROVIDER_KEY_CHARS:
            raise bad_request(f"key must be 1 to {MAX_PROVIDER_KEY_CHARS} characters.", "key", "invalid_value")
        await count_key_check(request, user)
        verdict = await anyio.to_thread.run_sync(request.app.state.deps.key_verifier, provider, key)
        if verdict.status == "invalid":
            _log(f"provider_key_save:{provider}", "rejected", visitor(request))
            raise APIError(400, rejected_message(provider, verdict), code="invalid_provider_key", param="key")
        saved = await anyio.to_thread.run_sync(
            lambda: store.save(user.id, provider, key, verdict.status, verdict.detail, verdict.answered))
        key = None                                     # the only plain copy; nothing keeps it
        _log(f"provider_key_save:{provider}", saved.status, visitor(request))
        return saved.public()

    @router.post("/v1/account/provider-keys/{provider}/verify")
    async def verify_provider_key(provider: str, request: Request, current=Depends(current_session)):
        """Check the saved key again and record the result (a rejected key stays saved, marked "invalid")."""
        _, user = current
        provider = provider_name(provider)
        store = key_store(request)
        if await anyio.to_thread.run_sync(store.get, user.id, provider) is None:
            raise APIError(404, f"No saved {NAMES[provider]} key.", param="provider", code="not_found")
        await count_key_check(request, user)

        def check():                                   # decrypt, check and drop, all in the worker thread
            try:
                plain = store.decrypt(user.id, provider)
            except DecryptionError:
                return None
            return request.app.state.deps.key_verifier(provider, plain)

        verdict = await anyio.to_thread.run_sync(check)
        if verdict is None:
            raise APIError(409, f"The saved {NAMES[provider]} key can't be decrypted any more (its encryption key "
                           "was removed). Save it again.", code="key_unreadable", param="provider")
        saved = await anyio.to_thread.run_sync(
            lambda: store.set_status(user.id, provider, verdict.status, verdict.detail, verdict.answered))
        _log(f"provider_key_verify:{provider}", saved.status, visitor(request))
        return saved.public()

    @router.delete("/v1/account/provider-keys/{provider}", status_code=204)
    async def delete_provider_key(provider: str, request: Request, current=Depends(current_session)):
        provider = provider_name(provider)
        await anyio.to_thread.run_sync(key_store(request).delete, current[1].id, provider)
        _log(f"provider_key_delete:{provider}", "ok", visitor(request))
        return Response(status_code=204)

    # ----- account API keys -----

    @router.get("/v1/account/api-keys")
    async def list_api_keys(request: Request, current=Depends(current_session)):
        keys = await anyio.to_thread.run_sync(request.app.state.deps.api_keys.list, current[1].id)
        return [k.public() for k in keys]

    @router.post("/v1/account/api-keys", status_code=201)
    async def create_api_key(body: ApiKeyCreate, request: Request, current=Depends(current_session)):
        _, user = current
        name = unicodedata.normalize("NFKC", body.name).strip()
        if not name or len(name) > API_KEY_NAME_MAX:
            raise bad_request(f"name must be 1 to {API_KEY_NAME_MAX} characters.", "name", "invalid_value")
        deps = request.app.state.deps
        try:
            await anyio.to_thread.run_sync(deps.auth_limiter.hit, "api_key_create_user", user.id)
        except RateLimited as error:
            _log("api_key_create", "rate_limited", visitor(request))
            raise _too_many(error, "API key creation") from None
        try:
            key, created = await anyio.to_thread.run_sync(deps.api_keys.create, user.id, name)
        except TooManyKeys:
            _log("api_key_create", "too_many_keys", visitor(request))
            raise APIError(409, f"You already have {MAX_ACTIVE_KEYS} active API keys, the most allowed. Revoke one "
                           "first.", code="too_many_api_keys") from None
        _log("api_key_create", "ok", visitor(request))
        from fastapi.responses import JSONResponse
        public = created.public()
        return JSONResponse({"id": public["id"], "name": public["name"], "public_id": public["public_id"],
                             "created_at": public["created_at"], "key": key, "note": ONLY_ONCE},
                            status_code=201, headers={"Cache-Control": "no-store"})

    @router.delete("/v1/account/api-keys/{key_id}", status_code=204)
    async def revoke_api_key(key_id: str, request: Request, current=Depends(current_session)):
        """Unknown ids and other users' keys are the same 404, so a key's existence isn't revealed."""
        found = await anyio.to_thread.run_sync(request.app.state.deps.api_keys.revoke, current[1].id, key_id)
        if not found:
            raise APIError(404, "No API key with that id.", param="id", code="not_found")
        _log("api_key_revoke", "ok", visitor(request))
        return Response(status_code=204)

    return router
