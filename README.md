# Fitty

Using modern technolgies to make fully pesonalised health consulations accesible to 100% of the people who have internet access.

Most people never find out they were on a trajectory toward something until
they are already on it. The information needed to say so earlier usually
exists: in a wearable, a blood test, a few honest answers, but turning it
into a straight answer costs a consultation most people will not book.

This is an attempt at closing that gap. Give it what you know, and it returns
calibrated probabilities for the conditions you are actually heading toward,
an estimate of how long you have, and a ranked list of what would change it —
with the reasoning shown.

It has no interface. It is built to be *called*: by Claude, ChatGPT, or any
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
check, and two acute safety screens — go out in a single call that costs a
fraction of a penny and returns in under a second. No language model is involved in producing a probability.

**It names where the weight sits. It does not tell anyone what to do.**
A questionnaire is not grounds for a treatment plan — there is no
examination, no history, usually no bloods. So the report ranks the areas
carrying the most weight in someone's own picture, attaches the systematic
reviews showing why those areas matter at all, and says plainly that
deciding what to change is a conversation for a clinician. Research comes
from Europe PMC, which is open and needs no key.

**The calling model does the explaining, on its own budget.** Over MCP the
caller is already a capable model sitting in the person's own subscription,
with the conversation context. Running a second model to produce prose the
first will rewrite is cost for no benefit — so the default is to return
findings plus a *presentation contract* and let the caller write it up.

The contract is the safety layer, and a contract is a request rather than a
guarantee. So the parts that must be right — the triage wording, the
non-diagnostic framing, the warning that overlapping factors do not add up —
are supplied as **finished sentences to quote**, not rules to obey. Quoting
is a much lower bar than complying, and it fails gracefully: a model that
ignores the rules but reproduces the text still tells the person the right
thing. An eval asserts the contract never loses a clause.

`RESPONSE_MODE=narrated` writes the report here instead, for callers with no
model of their own. That is the only path that needs `ANTHROPIC_API_KEY`.

**What counts as enough evidence depends on your age.** A flat threshold gets
this wrong in both directions: it refuses 21-year-olds who have never had a
blood test — exactly the people an early warning helps most — and waves
through 55-year-olds whose lifestyle answers no longer discriminate between
them.

| Age | Needed to get an estimate | Ceiling | What raises it |
|---|---|---|---|
| under 30 | body composition + lifestyle | 0.55 | recall answers |
| 30–44 | + three recall answers | 0.70 | blood pressure |
| 45+ | + three recall answers | 0.70 | blood pressure, then bloods |

**Nothing requires a test.** NHS Health Check uptake runs at roughly 46–48%
and covers only ages 40–74, so demanding bloods after 45 would refuse about
half the people the estimate is most useful to. Instead, three questions
anyone can answer from memory — ever prescribed blood pressure medication,
ever told your blood sugar was high, do you eat vegetables most days — carry
enough signal to estimate at any age. That is not a compromise: FINDRISC
reaches AUC 0.71–0.77 for undiagnosed diabetes with no blood test at all.

Blood pressure and bloods raise the confidence ceiling rather than unlock the
door. Below the bar the agent asks for exactly what is missing and says why;
above it, the estimate stands but its confidence is capped by what it rests
on, and the report states the basis — a recall-only estimate should not read
with the authority of one backed by bloods.

## Any unit you like

People know their height in feet and inches and their weight in stone.
Making them convert is friction, and friction is where mistakes happen. So
`5'11"`, `13 stone 4`, `32in`, `5.7%`, `100 mg/dL` all work, alongside cm, kg
and mmol/L. Where a unit is genuinely ambiguous the value is normalised by
plausibility and the assumption is reported back as `units_interpreted`, so a
wrong guess gets corrected rather than quietly shaping the assessment.

## Symptoms: urgency, not diagnosis

Describing symptoms is telling, and the temptation is to name what they point
to. It resists that. A named condition invites self-treatment, and a wrong
keyword match causes real fear for no reason — neither is a good trade for a
guess made without examining anyone.

What it does instead is recognise *combinations* that are unremarkable apart
and time-sensitive together — a headache plus morning vomiting, thirst plus
frequent urination, breathlessness plus swollen ankles — and return how
urgently to be seen, plus the specific details a clinician will want to hear.
That is the part that is both actionable and gettable-right. The risk
assessment still runs alongside it; triage leads the report.

Jev's own sufficiency signal is kept as a veto rather than the gate. It can
see what a field checklist cannot — contradictory values, a profile that does
not hang together — but only a strong objection overrides an age band.

**Life expectancy comes from a calculator, not from prose.** National life
tables, adjusted by factor costs that are weighted by the classifier's
confidence and saturated to account for overlap. The language model chooses
and explains; the arithmetic is deterministic, so the same profile always
gives the same number.

## It asks, rather than handing you a form

A clinician does not work through a questionnaire. They ask something,
listen, and let the answer decide the next question — then stop when the
picture is clear. A form cannot, because it is written before it knows
anything about you.

`next_question` does. At each step every unanswered question is simulated
across its plausible range, and whichever would move *this person's*
estimate most gets asked. Measured against the same questions in a fixed
order, it reaches the same accuracy in about half the questions:

| questions answered | fixed order | adaptive |
|---|---:|---:|
| 1 | 7.0% error | **4.5%** |
| 2 | 3.8% | **1.8%** |
| 3 | 4.0% | **1.8%** |

Knowing when to *stop* matters as much as what to ask. It ends when
nothing left would move the estimate by more than three points, rather
than marching through a list — so a fit 23-year-old is not asked for a
cholesterol panel that would change nothing.

"I don't know" is a real answer and moves things on. Without that the
same question comes back forever, which is what it did first time.

## And it says what is worth measuring next

The one thing here a language model cannot do by reasoning. Asked which
test someone should get, a model produces a sensible list — the same list
for everyone. This scores *this* person across each unknown's plausible
range and reports how far the answer actually moves, weighted by how easy
the thing is to obtain:

```
blood pressure     moves it 18%   Free at most pharmacies, no appointment.
kidney function    moves it 19%   A blood test, ordered alongside others.
waist measurement  moves it  5%   A tape measure, at the navel.
```

Blood pressure ranks above the eGFR that swings slightly further, because
the useful answer is not the most informative test in principle but the
most informative thing someone could go and do today.

## The host renders the UI

There is no interface, so the calling model is the interface. The contract
hands it finished components rather than instructions to build them:

- **`risk_table`** — every condition with its probability, band and
  certainty, as rows. Supplied as data because a figure recomputed in prose
  is a figure that drifts.
- **`projection`** — cumulative risk by age if nothing changes, one series
  per condition, ready to plot. The classifier answers a *ten-year*
  question, so asking it again at an older age produces a flat line that
  says nothing; composing successive decades survival-style is what turns
  it into the actual story. For a fit 21-year-old with diabetes in the
  family, a 21% decade risk compounds to 81% by 81.

That costs one classifier call per decade, run concurrently — about a
second, and affordable only because the classifier is cheap. The same sweep
on a language model would be slow and pointless.

The assumptions travel with the curve and the contract requires them to be
shown: it assumes nothing changes, treats decades as independent, and
ignores competing mortality. Each is a reason the line overstates, and a
rising line someone cannot contextualise is just frightening.

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
model, same wire format, a fraction of a penny per assessment. That is the
only API cost in the default path: `ANTHROPIC_API_KEY` is needed only for
`RESPONSE_MODE=narrated`. With no keys at all, everything falls back to
offline stand-ins so it stays runnable and testable.

## Deploying

[DEPLOY.md](DEPLOY.md) covers both: testing with Claude Desktop in five
minutes with no deployment at all, and a Fly.io deploy for a public endpoint.

The public endpoint requires `MCP_AUTH_TOKEN` and refuses to boot without it
— an open MCP server lets anyone post health data and spend your Jev credits.

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
| `age_band` / `no_test_required` | no age is refused for having had no tests |
| `contract_guarantees` | the presentation contract never loses a safety clause |
| `diagnostic_values_stated` | a diagnostic lab value is reported as fact, not probability |
| `evidence_caps_certainty` | thin evidence cannot claim full confidence |
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
