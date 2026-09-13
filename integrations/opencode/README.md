# @resonanceee/opencode-ragfaith

opencode plugin implementing the **ragfaith live faithfulness cascade**: after
each assistant reply, claims are decomposed and judged against the sources
actually pulled in the session. Unfaithful/unverifiable claims produce a single
aggregated nudge injected back into the session. Faithful replies are silent.

## Install

### Option A — local file copy (working today)

The whole plugin is one self-contained TS file (type-only imports are erased
at load, so no runtime deps). It uses the opencode v1 plugin module shape
(`default export { id, server }`), which the loader requires for path-based
plugins:

```sh
mkdir -p ~/.config/opencode/plugins
cp src/index.ts ~/.config/opencode/plugins/ragfaith.ts
```

opencode loads `~/.config/opencode/plugins/*.ts` automatically via Bun.

### Option B — npm (after publish)

The package is not published to npm yet; this will 404 until then. Once
published:

```sh
npm install @resonanceee/opencode-ragfaith
```

```jsonc
// ~/.config/opencode/opencode.json (or project .opencode/opencode.json)
{
  "plugin": ["@resonanceee/opencode-ragfaith"]
}
```

## Behavior

1. **Premise capture** — on `tool.execute.after` for content-pull tools
   (`read`, `webfetch`, `websearch`, anything matching the premise-tools
   regex), the tool output is appended to the session premise buffer, capped
   to the most recent ~24k chars. Sensitive paths (`.env*`, `*.pem`, `*.key`,
   `id_rsa`, `id_ed25519`, `*.p12`, `.npmrc`, `.netrc`, credentials, `.ssh/`,
   `.aws/credentials`) are never captured, and secret-looking strings are
   redacted before storage (see Privacy below).
2. **Doc-pull check** (free, deterministic) — on `tool.execute.before`, if a
   package is invoked (npm/bun/pip install args, import/require tokens) without
   a fetched-docs premise mentioning it, a non-blocking toast warns:
   "doc-pull check: X used without fetched docs". Advisory only, never blocks.
3. **Active model detection** — captured from `chat.params` /
   assistant `message.updated` (`providerID/modelID`); overridable via
   `RFE_ACTIVE_MODEL`.
4. **Judge selection** — active model matching the precise GLM-5.3-Flash
   marker `glm-5.3-flash` (case-insensitive) → judge = DeepSeek-V4.1-Flash;
   otherwise judge = GLM-5.3-Flash. A model never judges itself (a plain
   `glm-5-flash` is not treated as GLM-5.3-Flash).
5. **Per reply** — on assistant `message.updated` with `time.completed`, reply
   text (accumulated from `message.part.updated`) is segmented into sentence
   claims via `Intl.Segmenter` and each claim is judged against session
   premises, capped at `RFE_MAX_CLAIMS` per reply (default 50; skipped count is
   logged). Everything is async fire-and-forget; the reply is never blocked.
6. **Verdicts** — `faithful` passes silently. Any `unfaithful`/`unverifiable`
   claims are aggregated into ONE nudge, injected as a follow-up user message
   (`client.session.prompt`, `noReply: true`, `synthetic: true` so it never
   auto-triggers a new turn and can't loop). If injection fails, falls back to
   a `tui.toast.show` warning plus a structured log line. Claims are never
   auto-corrected.
7. **Judge call** — port of `rag_faithfulness_eval/llm_judge.py`:
   `POST {base}/chat/completions`, `temperature 0`, `reasoning: {exclude: true}`,
   `max_tokens 256` (one retry with 512 on parse failure; final parse fallback
   = `unverifiable`, counted as a parse error). Bilingual EN/DE system prompt
   copied verbatim. Verdict parsed from `"verdict"\s*:\s*"(\w+)"`. Exponential
   backoff on 429/5xx/network errors; other HTTP statuses (401, 400, ...) fail
   fast without retrying. Call failures and parse failures are counted
   separately and a failed call is never cached.
8. **Caching + logging** — verdict key `sha256(model + "\x00" + context +
   "\x00" + claim)`. With `RFE_CACHE_DIR`: append-only JSONL at
   `<RFE_CACHE_DIR>/opencode-cache-<model>.jsonl`; otherwise memory-only. One
   cache per judge model is loaded once per plugin lifetime (repeated replies
   reuse it). Every judge API call emits a token log line (tokens only, no USD)
   to stderr or `RFE_JUDGE_LOG`.

## Privacy — what leaves the machine

The judge runs on an external API (`api.synthetic.new` or OpenRouter), so
premise text must be sent there to get a verdict. Per judge call, the plugin
sends: the session premise buffer (content actually pulled by content-pull
tools, most recent ~24k chars) and the single claim being judged. It does not
send the full conversation, your config, or unrelated files.

Mitigations: premises from sensitive paths are skipped entirely, and
secret-looking strings (API keys, bearer tokens, AWS keys, private-key blocks,
quoted `api_key`/`token`/`secret`/`password` assignments) are redacted before
the buffer is stored or sent. Redaction is best-effort regex matching — do not
rely on it as your only secret boundary; keep credentials out of files the
agent reads.

## Configuration

| Env var                        | Default                              | Purpose |
|--------------------------------|--------------------------------------|---------|
| `RFE_JUDGE_PROVIDER`           | `synthetic`                          | `synthetic` or `openrouter` judge API |
| `SYNTHETIC_API_KEY`            | —                                    | key for `https://api.synthetic.new/v1` |
| `OPENROUTER_API_KEY`           | —                                    | key for `https://openrouter.ai/api/v1` |
| `RFE_SYNTHETIC_GLM_MODEL`      | `hf:zai-org/GLM-5.3-Flash`           | synthetic GLM judge model id |
| `RFE_SYNTHETIC_DEEPSEEK_MODEL` | `hf:deepseek-ai/DeepSeek-V4.1-Flash` | synthetic DeepSeek judge model id |
| `RFE_OPENROUTER_GLM_MODEL`     | `z-ai/glm-5.3-flash`                 | openrouter GLM judge model id |
| `RFE_OPENROUTER_DEEPSEEK_MODEL`| `deepseek/deepseek-v4.1-flash`       | openrouter DeepSeek judge model id |
| `RFE_ACTIVE_MODEL`             | auto-detect                          | override active-model detection |
| `RFE_PREMISE_TOOLS`            | `read\|fetch\|web\|doc\|search`       | regex (case-insensitive) for premise-capture tool names |
| `RFE_PREMISE_CAP`              | `24000`                              | max chars kept in premise buffer (most recent) |
| `RFE_MAX_CLAIMS`               | `50`                                 | max claims judged per reply (rest skipped + logged) |
| `RFE_CACHE_DIR`                | unset (memory-only)                  | persistent verdict cache directory |
| `RFE_JUDGE_LOG`                | unset (stderr)                       | file for judge token/error log lines |

## Cost notes

- Judge calls are priced by the judge provider's own token billing; this plugin
  logs **tokens only** (no USD estimates — price tables rot).
- Verdict caching (sha256 of model+context+claim) dedupes repeated judgments;
  set `RFE_CACHE_DIR` for persistence across restarts.
- `reasoning: {exclude: true}` still bills ~100–250 hidden reasoning tokens per
  call on GLM-family models (same as the Python judge).
- Replies judged only when premises exist in-session; replies to ragfaith's own
  nudges are skipped (loop guard).

## Development

```sh
bun install
bun test          # unit tests (no network calls)
bunx tsc --noEmit # strict typecheck
```
