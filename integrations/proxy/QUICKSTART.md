# ragfaith proxy — setup quickstart

Get the faithfulness proxy running in front of any OpenAI-compatible client
in about two minutes.

## 1. Install

Requires Python 3.11+. No third-party runtime dependencies.

```sh
cd integrations/proxy
pip install .
```

## 2. Start

```sh
ragfaith-proxy
```

Default listen address: `http://127.0.0.1:8787`. Change with
`RFE_PROXY_HOST` / `RFE_PROXY_PORT`, or `--host` / `--port`.

## 3. Point your client at it

In any OpenAI-compatible client (FlowDown, your own code, curl):

- **Base URL:** `http://127.0.0.1:8787/v1`
- **API key:** your real provider key (e.g. a synthetic key). The proxy
  holds no key of its own — your `Authorization` header is forwarded
  upstream and reused for the judge calls of that request.

Smoke check:

```sh
curl http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer $SYNTHETIC_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"<your-model>","messages":[{"role":"user","content":"hi"}]}'
```

## 4. Give it something to verify

The cascade judges each assistant reply against what was actually pulled.
Pull content via a tool call (or include `role: "tool"` messages in the
request), then ask the model a question. If a reply claims something the
pulled sources don't support, you'll see the nudge:

- **chain mode** (default): a self-correction streams into the same reply.
- **next mode** (`RFE_NUDGE_MODE=next`): the nudge rides in with your next
  message.

Faithful replies stay completely silent.

## 5. Watch it work

```sh
RFE_JUDGE_LOG=judge.jsonl ragfaith-proxy   # token-only judge cost log
```

## Next

- Full config (env vars, per-host decorators, cache): [README.md](README.md)
- Layout follows the universal-adapter idea — one default nudge template,
  host-specific overrides by `X-RFE-Host` header or User-Agent match.
