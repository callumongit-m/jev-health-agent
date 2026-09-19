# Health Risk Agent

A health agent that estimates a person's probability of developing common chronic
conditions, explains why, and proposes ranked changes. **No UI** — it is reachable
only by other agents, over MCP (consumer platforms) and A2A (peer agents).

Educational project. See [SAFETY.md](SAFETY.md) for scope and limits.

## Architecture

```
  Claude / ChatGPT            peer agent
    (consumer)                (A2A client)
         | MCP                     | A2A
         v                         v
    MCP adapter              A2A adapter
    tools + schemas          AgentCard + JSON-RPC
         |                         |
         +------------+------------+
                      v
             CORE HEALTH AGENT
             LangGraph state machine
             intake -> Jev -> reason
                ^               |
                +---- tools <---+
```

**Why two protocols.** MCP exposes *tools with schemas* to a host LLM, which is how
a consumer pastes a URL into Claude and it just works. A2A exposes an *opaque agent*
that owns a task, peer-to-peer. They are not alternatives; they front the same core.

**Why Jev.** A `Noul` question returns a natively calibrated probability (0–1) — the
disease probability, directly, with no LLM. A `Score` question returns a weighted
position over ordered severity levels plus a confidence. All 15 questions batch into
one call. The reasoning LLM runs only after, and only on what Jev found.

## Status

| Phase | | |
|---|---|---|
| 1 | Domain, Jev scoring, eval harness | **done** |
| 2 | LangGraph loop, tools, actuarial calculator | next |
| 3 | MCP adapter | |
| 4 | A2A adapter | |
| 5 | Terra ingest (Apple Health + wearables) | |

## Run it

```sh
uv sync
export PYTHONPATH=.

# score a persona (falls back to the offline backend with no API key)
uv run python scripts/demo_cli.py --persona heavy_smoker_cvd

# the eval suite
uv run python scripts/run_evals.py --k 15 --suite jev
uv run pytest -q
```

Set `TYPESAFE_API_KEY` to use the real Jev model; without it the scorer uses a
seeded rule-based stand-in so everything above it stays testable offline.

## Evals: pass^k, not pass@k

The agent is stochastic, so a single green run proves nothing. Every check runs k
times and is scored on **pass^k** — the fraction of checks where *all* k trials
passed. Per-trial rate is reported beside it: a low per-trial rate is a real bug,
a high per-trial rate with low pass^k is flakiness.

Four check families, all generated from the condition registry so they grow
automatically when a condition is added:

- **monotonicity** — worsening a driver must never lower that condition's probability
- **sensitivity** — driving a marker from healthy to severe must move it *materially*.
  Monotonicity alone cannot catch a model that ignores an input entirely: an ignored
  field produces no drop, so the check passes. This is the one that catches it.
- **ordering** — declared persona pairs must rank in the right order
- **expectations** — per-persona property assertions (thresholds, top factor, sufficiency)

Both monotonicity and sensitivity average `SAMPLES` calls at each end; averaging n
samples cuts noise by √n, which is what separates a real effect from jitter. The
noise floor itself is derived from `backend.noise_sigma` rather than hardcoded.

The suite is fault-injection tested — `tests/test_evals.py` asserts it *fails* when
the backend is made to ignore an input, because a suite that cannot fail proves nothing.

## Layout

```
src/health_agent/
  domain/      profile, condition registry, result shapes
  scoring/     Jev backend, offline fake, scorer
  privacy.py   notice injected into every surface; redaction for logs
evals/         personas, checks, pass^k runner
scripts/       demo_cli.py, run_evals.py
```
