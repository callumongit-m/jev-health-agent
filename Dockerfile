# Multi-stage: the build needs uv and the lock file, the runtime does not.
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependencies first, so a source change does not reinstall the world.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src ./src
COPY README.md ./
RUN uv sync --frozen --no-dev


FROM python:3.12-slim

# Health data should not be processed as root.
RUN useradd --create-home --uid 10001 agent
WORKDIR /app

COPY --from=builder --chown=agent:agent /app/.venv /app/.venv
COPY --from=builder --chown=agent:agent /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    RESPONSE_MODE=data \
    CHECKPOINT_PATH=/data/health_agent.sqlite

# Reminders and sync tokens live here; mount a volume over it in production
# or they vanish on redeploy.
RUN mkdir -p /data && chown agent:agent /data
VOLUME ["/data"]

USER agent
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz',timeout=4).status==200 else 1)"

# --require-auth refuses to boot without MCP_AUTH_TOKEN. An open endpoint
# lets anyone post health data and spend your API credits.
CMD ["python", "-m", "health_agent.adapters.mcp_server", \
     "--transport", "streamable-http", "--host", "0.0.0.0", \
     "--port", "8080", "--require-auth"]
