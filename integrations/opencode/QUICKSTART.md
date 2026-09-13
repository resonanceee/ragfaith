# ragfaith opencode plugin — setup quickstart

Install the faithfulness cascade into [opencode](https://opencode.ai) in
about two minutes.

## 1. Install

**Option A — npm (preferred):**

```sh
npm install @resonanceee/opencode-ragfaith
```

```jsonc
// ~/.config/opencode/opencode.json (or .opencode/opencode.json in a project)
{
  "plugin": ["@resonanceee/opencode-ragfaith"]
}
```

**Option B — single file:** copy `src/index.ts` to
`~/.config/opencode/plugins/ragfaith.ts`. It's self-contained; opencode's
Bun loader picks it up automatically.

## 2. Give the judge a key

The judge is a separate model, so it needs its own credentials in your
environment:

```sh
export SYNTHETIC_API_KEY=...          # default provider
# or: export RFE_JUDGE_PROVIDER=openrouter
#      export OPENROUTER_API_KEY=...
```

Never-self-judge is automatic: if your active model is GLM-5.3-Flash, the
plugin switches to DeepSeek-V4.1-Flash as judge (and vice versa).

## 3. Use opencode normally

Start (or restart) opencode. In any session:

1. Pull some content — `read` a file, `webfetch` a URL, run a search.
2. Ask the model to answer using it.

Each completed reply is decomposed into claims and judged against the
pulled sources (async — replies are never blocked or delayed).

## 4. What you'll see

- **Faithful reply:** nothing. No annotation, no noise.
- **Flagged reply:** one follow-up user message listing the flagged claims
  and asking the model to re-check its sources — the model reconciles on
  its next turn. Claims are never auto-corrected.
- **Missing docs:** a dependency invoked without a prior docs fetch draws a
  non-blocking toast (advisory only).

## 5. Watch it work

```sh
export RFE_JUDGE_LOG=judge.jsonl   # token-only judge cost log
opencode                            # start a session, check the file
```

Optional persistence: `RFE_CACHE_DIR=/path/to/dir` caches verdicts as
append-only JSONL; unset keeps them in memory.

## Next

- Full behavior + env var table: [README.md](README.md)
- The plugin ignores work it can't verify: no premises in the session means
  no judging, by design.
