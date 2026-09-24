# @resonanceee/opencode-ragfaith

opencode plugin implementing the **ragfaith live faithfulness cascade**: after
each assistant reply, claims are decomposed and judged against the sources
actually pulled in the session. Unfaithful/unverifiable claims produce a single
aggregated nudge injected back into the session. Faithful replies are silent.

## Install

### Option A — npm

```sh
npm install @resonanceee/opencode-ragfaith
```

Zero runtime dependencies — the package ships as a single self-contained TS
file (type-only imports are erased when opencode loads it), so nothing besides
the plugin itself is installed. It targets the opencode v1 plugin API
(`@opencode-ai/plugin` `>=1.0.0`, tested against 1.18.x); compatibility is
covered by the host, not by npm, since no peer dependency is declared.

Registry pages: [npmjs](https://www.npmjs.com/package/@resonanceee/opencode-ragfaith) ·
[GitHub Packages](https://github.com/resonanceee/ragfaith/pkgs/npm/opencode-ragfaith)

The package is also published to GitHub Packages under the same name and
versioning:

```sh
# one-time: auth with a token that has read:packages
npm login --scope=@resonanceee --registry=https://npm.pkg.github.com
npm install @resonanceee/opencode-ragfaith --registry=https://npm.pkg.github.com
```

```jsonc
// ~/.config/opencode/opencode.json (or project .opencode/opencode.json)
{
  "plugin": ["@resonanceee/opencode-ragfaith"]
}
```

### Option B — local file copy

The whole plugin is one self-contained TS file (type-only imports are erased
at load, so no runtime deps). It uses the opencode v1 plugin module shape
(`default export { id, server }`), which the loader requires for path-based
plugins:

```sh
mkdir -p ~/.config/opencode/plugins
cp src/index.ts ~/.config/opencode/plugins/ragfaith.ts
```

opencode loads `~/.config/opencode/plugins/*.ts` automatically via Bun.

## Behavior

1. **Premise capture** — on `tool.execute.after` for content-pull tools
   (`read`, `webfetch`, `websearch`, anything matching the premise-tools
   regex), the tool output is appended to the session premise buffer, capped
   to the most recent ~24k chars. Sensitive paths (`.env*`, `*.pem`, `*.key`,
   `id_rsa`, `id_ed25519`, `*.p12`, `.npmrc`, `.netrc`, credentials, `.ssh/`,
   `.aws/credentials`) are never captured, and secret-looking strings are
   redacted before storage (see Privacy below).
2. **Doc-pull check** (free, deterministic) — on `tool.execute.before`, if a
   package is invoked (npm/bun/pnpm/yarn/pip install args — `install`, `i`,
   and `add` aliases all match — plus import/require tokens) without
   a fetched-docs premise mentioning it, a non-blocking toast warns:
   "doc-pull check: X used without fetched docs". Advisory only, never blocks.
3. **Active model detection** — captured from `chat.params` /
   assistant `message.updated` (`providerID/modelID`); overridable via
   `RFE_ACTIVE_MODEL`.
4. **Judge selection** — the active model is compared against the *configured*
   main judge model (`RFE_JUDGE_MAIN_MODEL` / provider preset), ignoring provider
   prefixes and `:` variants: a match → judge = the configured fallback model;
   otherwise judge = the configured main model. With the default synthetic preset
   that means a GLM-5.3-Flash active model is judged by DeepSeek-V4.1-Flash (a
   plain `glm-5-flash` is not treated as the judge model). Because the check
   uses your configured id, custom GLM-family ids still get never-self-judge
   protection. Both judge models are settable for any OpenAI-compatible
   provider via `RFE_JUDGE_MAIN_MODEL` / `RFE_JUDGE_FALLBACK_MODEL` — no
   synthetic-style `hf:` id shape required.
5. **Per turn, at idle** — when the session goes idle (`session.idle`, i.e. the
    turn is truly over), the LAST assistant reply (text accumulated from
    `message.part.updated`) is segmented into sentence claims via
    `Intl.Segmenter` and each claim is judged against session premises, capped
    at `RFE_MAX_CLAIMS` per reply (default 50; skipped count is logged).
    Intermediate progress replies mid-turn are never judged, and tool CALLS
    ("ran read X") are captured as premises so process claims ("I read the
    handoff") are judgeable. Everything is async fire-and-forget; the reply is
    never blocked.
6. **Verdicts** — `faithful` passes silently. Default flag set is `unfaithful`
   only; `RFE_STRICTNESS=strict` adds `unverifiable` (explicit
   `RFE_FLAG_VERDICTS` CSV wins). Flagged claims are aggregated into ONE nudge,
   injected as a follow-up user message
    (`client.session.prompt`, `synthetic: true`) that triggers an immediate
    reconciliation turn — the agent acts on the nudge right away, not at the
    next user prompt. The nudge tells the agent it is
    from an automated judge, not the user: never mention the judge/flags/
    verdicts; the reconcile reply must contain ONLY the corrected claims as
    ordinary content — no apologies, no process narration. In strict mode the
    nudge gains a provenance arm: each
   unverifiable claim must gain a cited source or an explicit disclosure
   (internal knowledge / context inference), stated to the user as ordinary
   content. If injection fails, falls back to
   a `tui.toast.show` warning plus a structured log line. Claims are never
   auto-corrected.
   **Cascade bound:** the nudge-child reply is judged again (verification
   round) until `RFE_NUDGE_DEPTH` auto-rounds per user turn (default 2) are
   used up; claims already nudged in the current turn are deduped, so an
   unfixed claim can never re-fire the cascade. A real user message resets
   the depth; synthetic user messages (nudges, system-injected) do not.
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
   to `RFE_JUDGE_LOG` — silent by default (`stderr` value for explicit debug;
   unset would otherwise leak raw JSONL into the TUI).

## Privacy — what leaves the machine

The judge runs on an external API (the configured `RFE_JUDGE_BASE_URL`,
`synthetic` / `openrouter` presets by default), so
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
| `RFE_JUDGE_PROVIDER`           | `synthetic`                          | provider preset; any string works with the generic vars below (`synthetic` / `openrouter` presets come with defaults) |
| `RFE_JUDGE_BASE_URL`           | provider preset                      | generic judge API base URL (overrides preset) |
| `RFE_JUDGE_API_KEY`            | provider preset key                  | generic judge API key (overrides `SYNTHETIC_API_KEY` / `OPENROUTER_API_KEY`) |
| `RFE_JUDGE_MAIN_MODEL`          | provider preset                      | generic main judge model id — use this for non-synthetic id shapes (e.g. `inclusionai/ling-3.0-flash` on OpenRouter: fastest/cheapest judge, but highest parse-error rate; GLM default stays accuracy-first) |
| `RFE_JUDGE_FALLBACK_MODEL`     | provider preset                      | generic fallback judge model id — use this for non-synthetic id shapes |
| `SYNTHETIC_API_KEY`            | —                                    | key for `https://api.synthetic.new/v1` |
| `OPENROUTER_API_KEY`           | —                                    | key for `https://openrouter.ai/api/v1` |
| `RFE_SYNTHETIC_MAIN_MODEL`      | `hf:zai-org/GLM-5.3-Flash`           | synthetic preset main judge model id |
| `RFE_SYNTHETIC_FALLBACK_MODEL` | `hf:deepseek-ai/DeepSeek-V4.1-Flash` | synthetic preset fallback judge model id |
| `RFE_OPENROUTER_MAIN_MODEL`     | `z-ai/glm-5.3-flash`                 | openrouter preset main judge model id |
| `RFE_OPENROUTER_FALLBACK_MODEL`| `deepseek/deepseek-v4.1-flash`       | openrouter preset fallback judge model id |
| `RFE_ACTIVE_MODEL`             | auto-detect                          | override active-model detection |
| `RFE_STRICTNESS`               | `normal`                             | `normal` = flag `unfaithful` only; `strict` = flag both verdicts + provenance arm in the nudge |
| `RFE_FLAG_VERDICTS`            | unset                                 | CSV of verdicts that trigger a nudge (e.g. `unfaithful,unverifiable`); explicit CSV wins over `RFE_STRICTNESS` |
| `RFE_EVIDENCE_SCOPE`           | `web`                                 | evidence thoroughness: `web` = web fetches/searches + file reads; `all` = additionally command-execution output (bash, exec, shell, …). An explicit `RFE_PREMISE_TOOLS` regex wins over both presets. Premises from on-machine tools (reads, commands) are judged with the inference-tolerant machine prompt; web premises with the standard prompt. |
| `RFE_PREMISE_TOOLS`            | `read\|fetch\|web\|doc\|search`       | regex (case-insensitive) for premise-capture tool names (overrides `RFE_EVIDENCE_SCOPE`) |
| `RFE_PREMISE_CAP`              | `24000`                              | max chars kept in premise buffer (most recent) |
| `RFE_MAX_CLAIMS`               | `50`                                 | max claims judged per reply (rest skipped + logged) |
| `RFE_NUDGE_DEPTH`              | `2`                                  | max nudge-triggered auto-rounds per user turn; nudge-child replies beyond the cap are not judged |
| `RFE_CACHE_DIR`                | unset (memory-only)                  | persistent verdict cache directory |
| `RFE_JUDGE_LOG`                | unset (silent)                       | log sink: file path, or `stderr` for debug |

Precedence: generic `RFE_JUDGE_*` vars → provider-specific vars → preset
defaults. Example — judge with any OpenAI-compatible endpoint:

```sh
export RFE_JUDGE_PROVIDER=my-endpoint        # preset name is free-form
export RFE_JUDGE_BASE_URL=https://llm.internal/v1
export RFE_JUDGE_API_KEY=...
export RFE_JUDGE_MAIN_MODEL=openai/gpt-oss-120b
export RFE_JUDGE_FALLBACK_MODEL=mistral/magistral-small
```

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
