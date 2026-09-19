# Health Risk Agent

Most people never find out they were on a trajectory toward something until
they are already on it. The information needed to say so earlier usually
exists — in a wearable, a blood test, a few honest answers — but turning it
into a straight answer costs a consultation most people will not book.

This is an attempt at closing that gap. Give it what you know, and it returns
calibrated probabilities for the conditions you are actually heading toward,
an estimate of how long you have, and a ranked list of what would change it —
with the reasoning shown, not just the numbers.

It has no interface. It is built to be *called* — by Claude, ChatGPT, or any
other agent — so the barrier to using it is a conversation you were having
anyway, not another app to sign up for.

Read [SAFETY.md](SAFETY.md) for what it can and cannot claim. Short version:
it is an estimate, it is honest about its own uncertainty, and it is not a
diagnosis.

## How it works

```
  Claude / ChatGPT            peer agent
    (a conversation)          (A2A client)
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

**A classifier does the classifying.** The probabilities come from
[Jev](https://docs.typesafe.ai/), a decision model that returns a calibrated
probability directly rather than writing a number into a sentence. All fifteen
questions — seven conditions, seven modifiable factors, one data-sufficiency
check — go out in a single call that costs a fraction of a penny and returns
in under a second. No language model is involved in producing a probability.

**A language model does the explaining.** It sees the classifier's findings —
never the raw profile — and its job is to explain them, search for current
evidence where it would otherwise be asserting an effect size from memory, and
turn the ranking into changes someone can actually make this week.

**The classifier decides whether the language model runs at all.** Jev returns
a data-sufficiency signal, and below threshold the agent returns targeted
questions instead of an answer. A cheap calibrated signal gating an expensive
one is the whole reason a model like this belongs in a graph.

**Life expectancy comes from a calculator, not from prose.** National life
tables, adjusted by factor costs that are weighted by the classifier's
confidence and saturated to account for overlap. The language model chooses
and explains; the arithmetic is deterministic, so the same profile always
gives the same number.

## Two front doors

**MCP** — the one people will use. A remote MCP server URL pasted into
Claude, ChatGPT, Perplexity or Le Chat. The tool schema *is* the intake form,
so the host model collects details conversationally and fills it in.

**A2A** — for other agents. An agent card at
`/.well-known/agent-card.json`, JSON-RPC, streaming task updates. Where MCP
exposes tools for a host model to drive, A2A exposes an opaque agent that owns
a task: a peer sends a message and gets a result, never seeing the internals.
When the data is too thin, the task moves to `input-required` with questions
attached rather than guessing.

```sh
uv run python -m health_agent.adapters.mcp_server --transport streamable-http --port 8000
uv run python -m health_agent.adapters.a2a_server --port 9000

curl localhost:9000/.well-known/agent-card.json
uv run python scripts/a2a_peer_demo.py     # a peer agent driving a full task
```

## Keeping it current

A one-off snapshot goes stale, and stale risk estimates are worse than none.
`start_health_sync` returns a private URL and token; pointing an iOS export
automation at it means the picture refreshes on its own.

| Route | Setup | Freshness |
|---|---|---|
| Scheduled push (Health Auto Export, iOS Shortcuts) | one-time, on the phone | daily |
| `export.zip` upload | none | snapshot |
| Just say the numbers | none | whatever you said |

Apple Health already aggregates Apple Watch, Oura, Whoop, Garmin and Fitbit,
so one connection covers all of them. Apple blocks health access while the
phone is locked, so scheduled pushes land when it is next unlocked — freshness
here means *within a day*, not live.

Device readings never silently overwrite a fresher clinical value: every field
carries its source and age, and a blood pressure taken at a GP last week beats
a watch reading from yesterday.

## Run it

```sh
uv sync
export PYTHONPATH=.

cp .env.example .env          # add your keys
uv run python scripts/check_setup.py --live   # prove they work

uv run python scripts/demo_cli.py --persona heavy_smoker_cvd --agent
uv run python scripts/demo_cli.py --apple-health ~/Downloads/export.zip --age 34 --sex male --agent
```

Jev runs either directly (`TYPESAFE_API_KEY`) or through OpenRouter's
Decisions API (`OPENROUTER_API_KEY`, model `~typesafe/jev-latest`) — same
model, same wire format. Reasoning runs on Claude; `claude-haiku-4-5` is the
default and costs about £4 per thousand assessments. With no keys at all,
both fall back to offline stand-ins so everything stays runnable and testable.

## Evals: pass^k, not pass@k

The system is stochastic, so one green run proves nothing. Every check runs k
times and is scored on **pass^k** — the fraction of checks where *all* k trials
passed. Per-trial rate sits beside it: a low per-trial rate is a real bug, a
high one with a low pass^k is flakiness.

```sh
uv run python scripts/run_evals.py --k 20 --suite all
```

Checks are generated from the condition registry, so they grow when a
condition is added:

| Family | What it holds to |
|---|---|
| `monotonicity` | worsening a driver never lowers that condition's probability |
| `sensitivity` | driving a marker healthy→severe moves it *materially* |
| `ordering` | declared persona pairs rank correctly |
| `gate` / `no_unearned_numbers` | thin data returns questions, and no numbers |
| `red_flag` / `no_false_flag` | acute symptoms bypass scoring; ordinary talk does not |
| `evidenced_priority` | the headline recommendation rests on data we have |
| `years_not_oversold` | quoted years match what the calculator returned |
| `le_stability` | identical input gives an identical estimate |

`sensitivity` exists because monotonicity cannot catch a model that *ignores*
an input — an ignored field produces no drop, so the check passes. Both
families average samples at each end and derive their noise floor from the
backend, so a real effect is separated from jitter rather than lost in it.

The suite is fault-injection tested: `tests/test_evals.py` asserts it *fails*
when the backend is made to ignore an input, because a suite that cannot fail
proves nothing. Several of the rules in SAFETY.md exist because a check caught
the system breaking them.

## Layout

```
src/health_agent/
  domain/      profile, condition registry, result shapes
  scoring/     Jev backends, offline fake, scorer, actuarial calculator
  graph/       state, nodes, tools, reasoner, StateGraph wiring
  adapters/    core (shared), mcp_server, a2a_server, health_ingest
  ingest/      apple_health (export), health_sync (scheduled push), terra
  safety.py    red-flag screen, runs before any model
  privacy.py   notice injected into every surface; log redaction
evals/         personas, checks, pass^k runner
scripts/       demo_cli, run_evals, check_setup, a2a_peer_demo
```

Built with [Jev](https://docs.typesafe.ai/), LangChain, LangGraph,
[MCP](https://modelcontextprotocol.io) and [A2A](https://a2a-protocol.org).
