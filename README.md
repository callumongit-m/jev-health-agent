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
| 2 | LangGraph loop, tools, actuarial calculator, red-flag safety | **done** |
| 3 | MCP adapter (streamable HTTP, verified over JSON-RPC) | **done** |
| 4 | A2A adapter (agent card, task lifecycle, peer demo) | **done** |
| 5 | Terra ingest: Apple Health + wearables, signed webhook re-scoring | **done** |

Not yet done: deploying the MCP server somewhere public and adding it as a
custom connector in Claude, and running against the real Jev and Anthropic
APIs. Both need credentials; everything else runs offline today.

## Run it

```sh
uv sync
export PYTHONPATH=.

# score a persona (falls back to the offline backend with no API key)
uv run python scripts/demo_cli.py --persona heavy_smoker_cvd

# run the whole agent: graph, tool cycle, life expectancy, recommendations
uv run python scripts/demo_cli.py --persona heavy_smoker_cvd --agent

# the paths that matter
uv run python scripts/demo_cli.py --persona mid_metabolic_risk --agent \
    --text "I get chest tightness on stairs"        # -> seek_care
uv run python scripts/demo_cli.py --json '{"age": 41}' --agent   # -> needs_input

# the eval suite
uv run python scripts/run_evals.py --k 15 --suite jev
uv run pytest -q
```

Set `TYPESAFE_API_KEY` for the real Jev model and `ANTHROPIC_API_KEY` for the
real reasoner. Without them both fall back to offline stand-ins, so everything
stays runnable and testable with no credentials. The offline reasoner is not a
mock that skips the interesting part: it emits real tool calls and consumes real
`ToolMessage`s, so the reason ⇄ tools cycle is exercised either way.

## The graph

```
ingest -> screen --(red flag)--> END                 status: seek_care
             |
          (clear)
             v
         classify  (Jev, one batched call, no LLM)
             |
           gate --(thin data)--> clarify -> END      status: needs_input
             |
           (ok)
             v
          reason <----------+
             |  |           |
             |  +-> tools --+   actuarial_calc / evidence search / reminders
             v
          respond -> END                             status: complete
```

Three things worth pointing at:

- **The gate is the point of using Jev.** A cheap calibrated sufficiency signal
  decides whether to spend an LLM call at all. Below threshold the agent returns
  targeted questions to the *calling agent* rather than guessing.
- **Red flags run before any model.** Scoring someone describing chest pain would
  be the worst failure this system could have, so `safety.py` short-circuits the
  graph on acute presentations. Deliberately over-inclusive.
- **Life expectancy comes from a tool, not from prose.** The LLM chooses and
  explains; `actuarial_calc` does the arithmetic, so the same profile always
  yields the same number. Factor costs are confidence-weighted (missing data
  must not deduct years) and saturated (overlapping risks do not add up).

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

## Running the adapters

```sh
# MCP -- the consumer front door. Paste the URL into Claude's custom connectors.
uv run python -m health_agent.adapters.mcp_server --transport streamable-http --port 8000
#   -> http://localhost:8000/mcp

# A2A -- the peer front door.
uv run python -m health_agent.adapters.a2a_server --port 9000
curl localhost:9000/.well-known/agent-card.json

# a separate agent that discovers the card and drives a full task,
# including the input-required round trip
uv run python scripts/a2a_peer_demo.py
```

The peer demo exercises all three outcomes:

```
1. Complete profile  SUBMITTED -> WORKING -> COMPLETED
2. Thin profile      SUBMITTED -> WORKING -> INPUT_REQUIRED   + questions
3. Acute symptom     SUBMITTED -> WORKING -> COMPLETED        + seek-care, no scoring
```

## Keeping it fresh

`POST /webhooks/terra` takes a signed Terra payload, resumes the **existing**
thread by `reference_id`, merges the new samples with provenance and recency,
and re-scores. An unsigned or forged payload is rejected — health data that
cannot be verified is not trusted. Device readings never silently overwrite a
fresher lab or clinical value.

## Design notes

Three decisions worth knowing about:

- **Numbers are withheld unless they are earned.** On `needs_input` and
  `seek_care` the response carries no probabilities at all. A calling agent
  handed numbers will present them as final regardless of the caveat attached,
  so the gate withholds rather than qualifies.
- **The privacy notice is structural.** It is injected into the MCP server
  instructions, every MCP tool description, the A2A agent card and every
  response payload, and tests assert all four. Profile values are redacted
  from logs by field name.
- **Missing data cannot cost you years.** Factor costs are weighted by the
  classifier's confidence, and the actuarial calculator refuses outright below
  a sufficiency threshold rather than turning an empty profile into a number.

## Layout

```
src/health_agent/
  domain/      profile, condition registry, result shapes
  scoring/     Jev backend, offline fake, scorer, actuarial calculator
  graph/       state, nodes, tools, reasoner, StateGraph wiring
  adapters/    core (shared), mcp_server, a2a_server, webhook
  ingest/      terra (Apple Health + wearables, one integration)
  store/       reminders -- the only thing that outlives a session
  safety.py    red-flag screen, runs before any model
  privacy.py   notice injected into every surface; redaction for logs
evals/         personas, checks, pass^k runner
scripts/       demo_cli.py, run_evals.py, a2a_peer_demo.py
```
