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
| 3 | MCP adapter | next |
| 4 | A2A adapter | |
| 5 | Terra ingest (Apple Health + wearables) | |

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

## Layout

```
src/health_agent/
  domain/      profile, condition registry, result shapes
  scoring/     Jev backend, offline fake, scorer
  privacy.py   notice injected into every surface; redaction for logs
evals/         personas, checks, pass^k runner
scripts/       demo_cli.py, run_evals.py
```
