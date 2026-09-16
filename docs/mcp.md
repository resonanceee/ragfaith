# ragfaith MCP server

First and foremost: **connecting the ragfaith MCP server is not enough — the model
will not call it on its own reliably. You must instruct the model to invoke it.**

- **If this should hold for all sessions:** put the instruction in the **system prompt**
  (or equivalent persistent instruction file, e.g. `AGENTS.md` in opencode, Claude
  Code's memory, your client's custom instructions).
- **If this should apply to one conversation only:** put the instruction in the
  **first message** of that session.

The MCP tools only fire when the model decides to call them. Suggested system
prompt addition:

```text
You have access to the ragfaith MCP server, which live-checks answers against
their sources. Whenever your answer is grounded in retrieved documents or
sources, you MUST call the ragfaith tool check_faithfulness(question, passages,
answer) with the user's question, the exact passages you relied on, and your
draft answer BEFORE finalizing the reply. If any claim comes back "unfaithful"
or "unverifiable", correct or remove it and state the correction plainly. If
check_faithfulness is unavailable or errors, say so instead of silently
skipping the check. You may call judge_status to confirm the judge is
configured.
```

Shorter variant for a per-session message:

```text
Ground every sourced claim: call the ragfaith MCP check_faithfulness(question,
passages, draft answer) before you answer, then fix or drop anything flagged
unfaithful or unverifiable.
```

## Install & configure

```sh
pip install ragfaith-mcp
export OPENROUTER_API_KEY=...   # required for check_faithfulness
```

Environment variables (same as ragfaith-proxy):

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | — (required) | OpenRouter key used by the judge |
| `RFE_OPENROUTER_MAIN_MODEL` | `z-ai/glm-5.3-flash` | judge model |
| `RFE_OPENROUTER_BASE` | `https://openrouter.ai/api/v1` | provider base URL |

Transport is **stdio** (the MCP default for local servers): the host launches
`ragfaith-mcp` as a subprocess.

### Host config examples

opencode (`~/.config/opencode/opencode.json`):

```json
{
  "mcp": {
    "ragfaith": {
      "type": "local",
      "command": ["ragfaith-mcp"],
      "environment": { "OPENROUTER_API_KEY": "{env:OPENROUTER_API_KEY}" }
    }
  }
}
```

Claude Code / Claude Desktop and other hosts take the same command-line entry:
`ragfaith-mcp`, with `OPENROUTER_API_KEY` in its environment.

## Tools

### `check_faithfulness(question, passages, answer)`

- `question` (str): the user's question.
- `passages` (list[str]): the exact retrieved passages the answer relies on.
- `answer` (str): the draft answer to check.

Decomposes `answer` into atomic claims (same `split_claims` cascade as
ragfaith-proxy) and judges each against `QUESTION + PASSAGES` with the
configured LLM judge. Returns:

```json
{
  "claims": [{"claim": "...", "verdict": "faithful|unfaithful|unverifiable"}],
  "summary": {"faithful": 3, "unverifiable": 1},
  "all_faithful": false,
  "model": "z-ai/glm-5.3-flash"
}
```

Raises if `OPENROUTER_API_KEY` is unset — by design, so a missing key is
visible instead of silently skipping the check.

### `judge_status()`

Reports `{model, provider, base_url, api_key_configured}`. Never returns the
key itself. Call it to confirm the judge is configured before relying on
`check_faithfulness`.

## Development (from this repo)

```sh
pip install ./integrations/proxy './integrations/mcp[dev]'
pytest integrations/mcp/tests -q
```

Tests are stub-based (no API key, no network): a fake judge is injected and a
real MCP client exercises the server in-memory, including assertions that the
invocation instructions above stay in this document and in the server's
`instructions`. CI runs them on every PR (`.github/workflows/ci.yml`,
job `mcp-server`).
