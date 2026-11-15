"""Accounts (Phase 5): sign up, log in, log out, who am I, and account settings. Only when TRIAGE_ACCOUNTS is on.

    POST   /v1/auth/signup      {email, password, display_name?} -> 201 {user, session_token, expires_at}
    POST   /v1/auth/login       {email, password}                -> 200 {user, session_token, expires_at}
    POST   /v1/auth/logout                                       -> 204 (revokes this session)
    GET    /v1/auth/me                                           -> {id, email, display_name, preferred_provider,
                                                                     created_at}
    PATCH  /v1/account          {display_name?, preferred_provider?}            -> the user, as /me
    POST   /v1/account/password {current_password, new_password}               -> 204
    DELETE /v1/account          {password}                                     -> 204

Everything except signup and login needs a session token (Authorization: Bearer trs_...), never a Triage API key.

Never logged or echoed: emails, passwords, tokens. Auth failures log only the visitor's hash and the outcome.
Every database call and password hash runs in a worker thread.
"""
import json
import logging
from typing import Optional

import anyio
from fastapi import APIRouter, Depends, Header, Request, Response
from pydantic import BaseModel, ConfigDict

from app.accounts.credentials import (CredentialError, check_password_rules, hash_password,
                                      looks_like_session_token, needs_rehash, normalize_display_name,
                                      normalize_email, verify_password)
from app.accounts.ratelimit import RateLimited, email_subject
from app.accounts.store import PROVIDERS, EmailTaken
from app.api.demo import client_ip, visitor_id
from app.api.errors import APIError, bad_request

logger = logging.getLogger("triage.auth")

INVALID_CREDENTIALS = "Incorrect email or password."
NEEDS_SESSION = ("This endpoint needs a session token: log in with POST /v1/auth/login and send the token as "
                 "Authorization: Bearer <token>.")
SESSION_ENDED = "Your session has expired or was revoked. Log in again."
INCORRECT_PASSWORD = "That password isn't right."


def _log(event: str, outcome: str, visitor: Optional[str] = None) -> None:
    """One JSON line: the event, the outcome and the visitor's hash. Never an email, password or token."""
    logger.info(json.dumps({"event": event, "outcome": outcome, "visitor": visitor}, separators=(",", ":")))


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
    async def signup(body: SignupBody, request: Request):
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
        """403 incorrect_password (not 401: the session is fine, so the website shouldn't log the user out)."""
        ok = await anyio.to_thread.run_sync(verify_password, user.password_hash, password)
        if not ok:
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

    return router
