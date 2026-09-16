# ragfaith-proxy

Universal OpenAI-compatible sidecar proxy with a **live ragfaith faithfulness
cascade**. Any host tool without a plugin system — chat GUIs, agent harnesses,
scripts — just points its OpenAI base URL at this proxy. Streaming replies pass
through byte-for-byte (**first token never delayed**) while claims in the reply
are judged against the sources actually pulled in the conversation
(`role: "tool"` results). Flagged claims are corrected with a nudge, in-band
or on the next turn.

Pure Python stdlib. No third-party runtime dependencies.

## Install & run

```sh
pip install ragfaith-proxy         # from PyPI
ragfaith-proxy                     # listens on 127.0.0.1:8787
```

Registry page: [pypi.org/project/ragfaith-proxy](https://pypi.org/project/ragfaith-proxy/) ·
also attached to [GitHub releases](https://github.com/resonanceee/ragfaith/releases).

From a checkout instead: `pip install .`

Tests (no network): `pip install .[dev] && pytest -q`

## Authentication: client-auth-only

The proxy holds **no API key of its own**. Every client request must carry its
own `Authorization: Bearer <provider key>` header, which the proxy forwards to
the upstream verbatim. The client's token also authenticates the judge calls
that request's cascade triggers. Requests without an `Authorization` header
get 401. Consequence: the proxy can be hosted openly — anyone who connects
needs and spends only their own provider key.

## FlowDown (reference example)

FlowDown is an Apple AI chat client that accepts a custom OpenAI-compatible
endpoint. To use it with the cascade:

1. Start the proxy: `ragfaith-proxy` (default `http://127.0.0.1:8787`).
2. In FlowDown, add a custom OpenAI-compatible provider with:
   - Base URL: `http://127.0.0.1:8787/v1`
   - API key: your real provider key (e.g. a synthetic key) — it is forwarded
     upstream and reused for that session's judge calls.
3. Chat as usual. When a reply misstates the retrieved sources, the next turn
   automatically carries a corrective nudge — or, in chain mode, the correction
   streams right after the flagged reply.

Any other client works the same way: set its base URL to
`http://127.0.0.1:8787/v1`.

## How it works

```
client -> proxy -> upstream LLM
             |-> stream back immediately, buffer a copy
             |-> on finish: decompose claims (spaCy/regex sentences)
             |-> judge claims vs pulled tool-result premises
             |-> flagged? nudge
```

- **Model detection**: the request's `model` field is the active model.
- **Judge selection** (never self-judge): the active model is compared against
  the configured main judge model (provider prefixes and `:` variants ignored).
  A match is judged by the configured fallback model; anything else is judged
  by the main model. Both models are configurable for any
  OpenAI-compatible endpoint via `RFE_JUDGE_MAIN_MODEL` /
  `RFE_JUDGE_FALLBACK_MODEL` — no `hf:`-style id shape required.
- **Premises**: content the conversation actually pulled — `role: "tool"`
  messages and `tool_result` content blocks, capped to the most recent
  24k chars (`RFE_PREMISE_CAP`). A small per-conversation store covers clients
  that do not resend full history (LRU-capped at 1000 conversations);
  conversations are keyed by `X-Conversation-Id`, else a hash of the first
  user message (full message list if none), else Host header.

## Nudge modes (`RFE_NUDGE_MODE`)

- **`chain` (default)**: after a streaming reply with flagged claims finishes,
  the proxy withholds the stream close, issues ONE internal continuation call
  (original messages + assistant reply + aggregated nudge), streams it down the
  same SSE stream, then terminates with a single `[DONE]`. Non-streaming
  requests can't be chained, so they fall back to `next` behavior (logged).
  **Spend caveat**: every flagged reply costs one extra upstream completion —
  chain mode roughly doubles the upstream cost of flagged replies.
- **`next`**: the nudge is stashed per conversation and injected as a user
  message right after the flagged assistant message in the next request's
  history.
- **`regen`**: the whole reply is buffered, claims are judged, and the client
  sees exactly one assistant response — the original when faithful, a
  regenerated one (from the internal nudged call) when flagged. The nudge and
  the original flawed text never reach the client. **Latency caveat**: the
  client waits for generation + judging before anything is shown; there is no
  stream-through in this mode. The regen nudge uses a directive template
  (`RFE_REGEN_NUDGE`, per-host key `regen_template`): the model is told to
  keep the re-check inside its own reasoning and output only a direct answer
  to the user's original query — no tables, no meta-commentary, no references
  to the judge. This keeps false-positive flags from turning into visible
  reconciliation essays.

All-faithful replies are fully silent: no extra call, no injection.

Judging parallelism: per-claim judge calls within one cascade run concurrently
(`RFE_JUDGE_WORKERS`, default 8) — one judge round-trip per reply instead of
one per claim. 429/5xx judge errors are retried with backoff + jitter
(4 attempts); claims that still fail are skipped fail-open and counted in the
`judge-skipped` log line.

Flag set: only `unfaithful` verdicts nudge by default — `unverifiable` marks
derivation/drift, not fabrication. Set `RFE_STRICTNESS=strict` to fire on both
verdicts with verdict-aware nudge arms (unfaithful → reconciliation text;
unverifiable → each claim must gain a cited source or an explicit
provenance disclosure: internal knowledge or context inference), or set
`RFE_FLAG_VERDICTS=unfaithful,unverifiable` (or a per-host `flag_verdicts`
decorator) to surface drift with the normal template. When premises exceed the
per-claim budget, the judge is told to prefer `unverifiable` over `unfaithful`
so premise filtering/truncation can't read as fabrication.

Stream timing: in `chain` mode `[DONE]` is withheld until the cascade has
ruled (the correction appends to the same stream). In `next` mode — and for
faithful replies in any mode — the stream closes immediately after the
upstream reply and the cascade runs in the background.

Auditability: `RFE_JUDGE_LOG` rows include the claim text, verdict, verdict
cache key, and conversation id; nudge stash/delivery events are logged
(`nudge-stash` / `nudge-delivered`, nudge text truncated to 500 chars).
Client-visible failures are logged too (issue #85): every relayed non-200
(`passthrough-status` with a redacted body excerpt), every proxy-side request
rejection (`request-rejected`: `invalid-json`, `bad-messages`, `bad-model`,
`missing-auth`, `bad-content-length`, `payload-too-large`), and the nudge
next-mode path (`nudge-inject` payload size, `nudge-request-failed` upstream
status when a nudged request fails).

Default nudge template:

```
ragfaith judge (<model>): N claim(s) in your previous reply were flagged
<verdict>: <claims>. Re-check against the sources actually pulled in this
conversation and reconcile; do not invent corrections.
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `RFE_PROXY_HOST` | `127.0.0.1` | Listen address |
| `RFE_PROXY_PORT` | `8787` | Listen port |
| `RFE_UPSTREAM_BASE` | `https://api.synthetic.new/v1` | Upstream OpenAI-compatible base |
| `RFE_JUDGE_PROVIDER` | `synthetic` | Preset: `synthetic` or `openrouter` (free-form with generic vars) |
| `RFE_JUDGE_BASE` | preset | Generic judge base URL; overrides preset |
| `RFE_JUDGE_MAIN_MODEL` | preset | Generic main judge model; overrides preset (e.g. `inclusionai/ling-3.0-flash` on OpenRouter — fastest/cheapest judge, but highest parse-error rate; GLM default stays accuracy-first) |
| `RFE_JUDGE_FALLBACK_MODEL` | preset | Generic fallback judge model; overrides preset |
| `RFE_SYNTHETIC_BASE` | upstream base | Judge base URL (synthetic preset) |
| `RFE_SYNTHETIC_MAIN_MODEL` | `hf:zai-org/GLM-5.3-Flash` | Main judge model (synthetic preset) |
| `RFE_SYNTHETIC_FALLBACK_MODEL` | `hf:deepseek-ai/DeepSeek-V4.1-Flash` | Judge when the main judge is the active model (synthetic preset) |
| `RFE_OPENROUTER_BASE` | `https://openrouter.ai/api/v1` | Judge base URL (openrouter preset) |
| `RFE_OPENROUTER_MAIN_MODEL` | `z-ai/glm-5.3-flash` | OpenRouter main judge |
| `RFE_OPENROUTER_FALLBACK_MODEL` | `deepseek/deepseek-v4.1-flash` | OpenRouter fallback judge |
| `RFE_NUDGE_MODE` | `chain` | `chain`, `next`, or `regen` |
| `RFE_NUDGE_TEMPLATE` | chain/next template | Override the visible nudge text |
| `RFE_REGEN_NUDGE` | directive template | Override the regen-mode internal nudge |
| `RFE_JUDGE_WORKERS` | `8` | Concurrent per-claim judge calls (lower this for rate-limited upstreams; 429/5xx are retried with backoff) |
| `RFE_FLAG_VERDICTS` | `unfaithful` | CSV of verdicts that trigger a nudge (e.g. `unfaithful,unverifiable` to also surface drift); wins over the strictness preset |
| `RFE_STRICTNESS` | `normal` | `normal` = flag `unfaithful` only; `strict` = flag both verdicts + verdict-aware nudge arms |
| `RFE_NUDGE_TEMPLATE_UNVERIFIABLE` | provenance template | Strict-mode nudge arm for `unverifiable` claims (per-host key: `unverifiable_template`) |
| `RFE_PREMISE_CAP` | `24000` | Max chars of premises per verdict |
| `RFE_HOST_DECORATORS` | — | JSON map of per-host overrides |
| `RFE_HOST_CONFIG` | — | Path to same JSON map in a file |
| `RFE_CACHE_DIR` | — | Append-only JSONL verdict cache dir |
| `RFE_JUDGE_LOG` | stderr | File for session token log lines |
| `RFE_MAX_BODY` | `10485760` | Max request body bytes (larger gets 413) |

Precedence: generic `RFE_JUDGE_*` vars → preset vars (`RFE_SYNTHETIC_*` /
`RFE_OPENROUTER_*`, selected by `RFE_JUDGE_PROVIDER`) → built-in defaults.
Example — judge with any OpenAI-compatible endpoint:

```sh
export RFE_JUDGE_PROVIDER=my-endpoint        # preset name is free-form
export RFE_JUDGE_BASE=http://llm.internal/v1
export RFE_JUDGE_MAIN_MODEL=openai/gpt-oss-120b
export RFE_JUDGE_FALLBACK_MODEL=mistral/magistral-small
```

Per-host decorators are matched by the `X-RFE-Host` header (exact key), else
by User-Agent substring, else `default`:

```json
{"flowdown": {"template": "...", "nudge_mode": "next"}, "default": {}}
```

## Logging & cache

Token log lines go to stderr (or `RFE_JUDGE_LOG` file), one JSON per judge
call — tokens only, no cost math:

```json
{"ts": 1757680000.0, "kind": "judge", "model": "...", "prompt_tokens": 0, "completion_tokens": 0, "conversation": "..."}
```

With `RFE_CACHE_DIR` set, verdicts persist to
`<dir>/proxy-cache-<judgeModelSanitized>.jsonl` (`{"key","verdict"}` rows,
sha256 keys over model/context/claim). Unset: in-memory dict only.

## spaCy (optional)

Claim decomposition uses `split_claims`, a vendored copy of
`rag_faithfulness_eval.decompose` (copied, not imported). With
`pip install ragfaith-proxy[spacy]` you get the benchmarked spaCy sentencizer
path. Without it the proxy falls back to a regex sentence splitter and logs a
prominent warning — claim boundaries may differ from benchmark results in
edge cases.

## Failure behavior

The proxy never blocks or corrupts the stream: any upstream/judge/cascade
failure is logged and the response passes through untouched. Client
disconnects mid-stream abort cleanly. Responses are requested
`Accept-Encoding: identity` from upstream so the buffered copy stays
parseable.
