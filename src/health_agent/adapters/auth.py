"""Bearer-token auth for the public MCP endpoint.

Without this the endpoint is open: anyone who finds the URL can post health
data and spend your Jev credits. Claude's connector UI supports a fixed
header credential (`static_headers`), which is a great deal simpler to run
than OAuth and enough for a single-owner deployment.

The token is compared in constant time and never logged. If it is unset the
server still starts -- refusing to boot would be worse during local
development -- but it says loudly what it is doing, and refuses outright in
production.
"""

from __future__ import annotations

import hmac
import logging
import os

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

#: Paths reachable without a token. Health checks only -- nothing that
#: touches health data or costs money.
PUBLIC_PATHS = frozenset({"/healthz", "/readyz"})


class BearerTokenMiddleware:
    """Rejects anything without the expected bearer token."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self._token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        if request.url.path in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        header = request.headers.get("authorization", "")
        supplied = header[7:].strip() if header.lower().startswith("bearer ") else ""

        # constant time, so a wrong token cannot be narrowed down by timing
        if not supplied or not hmac.compare_digest(supplied, self._token):
            logger.warning("rejected unauthenticated request to %s", request.url.path)
            response = JSONResponse(
                {"error": "unauthorized", "detail": "missing or invalid bearer token"},
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="fitty"'},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


def protect(app, *, require: bool = False):
    """Wrap an ASGI app in token auth if MCP_AUTH_TOKEN is set.

    `require` makes a missing token fatal, which is what any deployment
    reachable from the internet should pass.
    """
    token = os.getenv("MCP_AUTH_TOKEN", "").strip()
    if token:
        if len(token) < 24:
            raise RuntimeError(
                "MCP_AUTH_TOKEN is too short to be safe. Generate one with: "
                "python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )
        return BearerTokenMiddleware(app, token)

    if require:
        raise RuntimeError(
            "MCP_AUTH_TOKEN is not set. A public endpoint without it lets "
            "anyone post health data and spend your API credits. Generate "
            "one with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
        )

    logger.warning(
        "MCP_AUTH_TOKEN is not set -- this server is OPEN. Fine on localhost, "
        "never anywhere reachable."
    )
    return app
