# Deploying, and testing with a real LLM

Two separate things. You can test the whole system with Claude in about five
minutes without deploying anything; deploying is for when you want other
people to reach it.

---

## 1. Test it today — Claude Desktop, no deploy, no auth

The fastest honest test of the whole loop. Claude runs the server locally
over stdio, so there is no URL, no token, and no hosting.

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "health-agent": {
      "command": "uv",
      "args": [
        "run", "--directory", "/Users/callumsaccount/Documents/jev-health-agent",
        "python", "-m", "health_agent.adapters.mcp_server"
      ],
      "env": {
        "PYTHONPATH": "/Users/callumsaccount/Documents/jev-health-agent",
        "OPENROUTER_API_KEY": "sk-or-...",
        "RESPONSE_MODE": "data"
      }
    }
  }
}
```

Restart Claude Desktop, then say:

> I'm 21, male, 5'11.5", 85kg. I don't smoke, barely drink, train five times
> a week. Assess my health risk.

What to watch for, because this is the first time the presentation contract
faces a model that has not been told to obey it:

- Does Claude reproduce the `must_include_verbatim` sentences, or paraphrase
  them? Paraphrasing is the failure mode that matters.
- Does it respect `must_not` — no summed years, no named conditions?
- Does it ask for the recall questions when they are missing?

That is a real eval of the contract, and one no offline test can give you.

---

## 2. Deploy it — Fly.io

Fly suits this: scale-to-zero (so it costs pennies between assessments),
sub-second machine starts that an MCP call absorbs unnoticed, and a volume
for the SQLite that holds reminders and wearable sync tokens.

Roughly $2–5/month with the machine stopped most of the time.

```sh
brew install flyctl && fly auth login

# creates the app without deploying, so secrets land before first boot
fly launch --no-deploy --name jev-health-agent --region lhr

fly volumes create health_agent_data --size 1 --region lhr

# The container refuses to start without this. That is deliberate: an open
# endpoint lets anyone post health data and spend your Jev credits.
fly secrets set \
  MCP_AUTH_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" \
  OPENROUTER_API_KEY="sk-or-..."

fly deploy
fly logs
```

Check it:

```sh
curl https://jev-health-agent.fly.dev/healthz
# {"ok":true,"response_mode":"data","scoring":"openrouter"}
```

Then read the token back for the connector setup:

```sh
fly ssh console -C "printenv MCP_AUTH_TOKEN"
```

### Why not Cloudflare Workers

Workers runs Python on Pyodide/WASM. This tree has 15 compiled C extensions
(`pydantic_core`, protobuf's `upb`, `jiter`, `orjson`, `cryptography`…) in a
100MB virtualenv, against a 10MB bundle limit. Cloudflare **Containers**
would run it, and Cloud Run or Railway work equally well — but Cloud Run's
ephemeral filesystem means moving reminders and sync tokens to Postgres
first.

---

## 3. Connect it to Claude

Settings → Connectors → Add custom connector.

- **URL**: `https://jev-health-agent.fly.dev/mcp`
- **Authentication**: static header — `Authorization: Bearer <MCP_AUTH_TOKEN>`

Never put the token in the URL. A connector URL ending `?token=...` leaks
through server logs, proxy logs, browser history and screenshots.

ChatGPT, Perplexity, Grok and Le Chat take the same URL.

---

## 4. Before anyone else uses it

Honest list of what is not production-ready yet:

- **One shared token.** Fine for you. It does not identify *who* is asking,
  so multi-user needs OAuth (Claude supports `oauth_dcr` and `oauth_cimd`)
  and per-user thread isolation.
- **No rate limiting.** One enthusiastic caller can run up your Jev bill.
- **SQLite on one volume.** Correct for a single machine, wrong the moment
  you scale past one. Postgres before you do.
- **`noise_sigma` is a placeholder.** Score one profile 50 times against
  real Jev, take the worst per-condition standard deviation, and set it —
  the monotonicity and sensitivity checks derive their tolerances from it.
- **Not validated against outcomes.** See [SAFETY.md](SAFETY.md). The suite
  shows the numbers behave correctly, which is not the same thing.
