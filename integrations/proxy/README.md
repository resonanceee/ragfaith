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
pip install .
ragfaith-proxy                    # listens on 127.0.0.1:8787
```

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
- **Judge selection** (never self-judge): if the active model is
  GLM-5.3-Flash, DeepSeek-V4.1-Flash judges; anything else is judged by
  GLM-5.3-Flash.
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

All-faithful replies are fully silent: no extra call, no injection.

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
| `RFE_JUDGE_PROVIDER` | `synthetic` | `synthetic` or `openrouter` |
| `RFE_SYNTHETIC_BASE` | upstream base | Judge base URL (synthetic) |
| `RFE_SYNTHETIC_GLM_MODEL` | `hf:zai-org/GLM-5.3-Flash` | Primary judge model |
| `RFE_SYNTHETIC_DEEPSEEK_MODEL` | `hf:deepseek-ai/DeepSeek-V4.1-Flash` | Judge when active model is GLM |
| `RFE_OPENROUTER_BASE` | `https://openrouter.ai/api/v1` | Judge base URL (openrouter) |
| `RFE_OPENROUTER_GLM_MODEL` | `z-ai/glm-5.3-flash` | OpenRouter primary judge |
| `RFE_OPENROUTER_DEEPSEEK_MODEL` | `deepseek/deepseek-v4.1-flash` | OpenRouter GLM-active judge |
| `RFE_NUDGE_MODE` | `chain` | `chain` or `next` |
| `RFE_PREMISE_CAP` | `24000` | Max chars of premises per verdict |
| `RFE_HOST_DECORATORS` | — | JSON map of per-host overrides |
| `RFE_HOST_CONFIG` | — | Path to same JSON map in a file |
| `RFE_CACHE_DIR` | — | Append-only JSONL verdict cache dir |
| `RFE_JUDGE_LOG` | stderr | File for session token log lines |
| `RFE_MAX_BODY` | `10485760` | Max request body bytes (larger gets 413) |

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
