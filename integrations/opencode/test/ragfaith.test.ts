import { describe, expect, test, beforeEach, afterEach } from "bun:test";
import { mkdtempSync, readFileSync, existsSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Event } from "@opencode-ai/sdk";
import {
  segmentClaims,
  selectJudgeModel,
  resolveProvider,
  parseVerdict,
  buildNudge,
  PremiseStore,
  Judge,
  makeCache,
  makeCacheRegistry,
  verdictKey,
  extractPackages,
  isGlmFlash,
  isActiveJudgeModel,
  isSensitivePath,
  redactSecrets,
  maxClaims,
  selectPremise,
  RagfaithPlugin,
  logLine,
} from "../src/index";

const GLM = "test/glm-judge";
const DS = "test/ds-judge";

function fakeFetch(body: unknown, status = 200): typeof fetch {
  const resp = {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
  return (async () => resp) as unknown as typeof fetch;
}

function okResp(verdict: string, fenced = false) {
  const content = fenced
    ? `\`\`\`json\n{"verdict": "${verdict}"}\n\`\`\``
    : `{"verdict": "${verdict}"}`;
  return {
    choices: [{ message: { content } }],
    usage: { prompt_tokens: 10, completion_tokens: 5 },
  };
}

function noSleep(): Promise<void> {
  return Promise.resolve();
}

describe("claim segmentation", () => {
  test("3-sentence EN reply -> 3 claims", () => {
    const claims = segmentClaims(
      "The sky is blue. Water boils at 100 degrees. Paris is in France.",
    );
    expect(claims.length).toBe(3);
  });

  test("3-sentence DE reply -> 3 claims", () => {
    const claims = segmentClaims(
      "Der Himmel ist blau. Wasser kocht bei 100 Grad. Paris liegt in Frankreich.",
    );
    expect(claims.length).toBe(3);
  });

  test("empty/whitespace drops out", () => {
    expect(segmentClaims("   ").length).toBe(0);
  });
  test("markdown furniture dropped, prose kept (issue 76)", () => {
    const claims = segmentClaims(
      [
        "Alzheon reported Phase 2 data in April 2025.",
        "| **Alzheon (ALZH)** | ALZ-801 / APOLLOE4 | **Already read out Apr 2025 — missed primary endpoint** overall, positive only in MCI subgroup ([Alzheon](https://example.com)) | The big catalyst already happened and was a miss; now in long-term extension.",
        "^4] |",
        '[^6]: BioCosm, "Remternetug — Eli Lilly," updated 30 May 2026.',
        "|---|---|---|",
        "[Alzheon]: https://example.com/apolloe4",
        "Remternetug is currently in Phase 3.",
        "---",
      ].join("\n"),
    );
    expect(claims).toEqual([
      "Alzheon reported Phase 2 data in April 2025.",
      "Remternetug is currently in Phase 3.",
    ]);
  });

  test("prose containing inline links kept", () => {
    const claims = segmentClaims(
      "The ALZ-801 trial ([Alzheon](https://example.com)) missed its endpoint. It is now in extension.",
    );
    expect(claims.length).toBe(2);
    expect(claims[0]).toContain("ALZ-801 trial");
  });

  test("footnote / meta / confidence-interval (issue 76 reopened)", () => {
    expect(segmentClaims('[^6]: BioCosm, "Remternetug — Eli Lilly," updated 30 May 2026.')).toEqual(
      [],
    );
    expect(segmentClaims("Overall confidence: High.")).toEqual([]);
    expect(segmentClaims("certainty - medium")).toEqual([]);
    expect(segmentClaims("tl;dr: everything above was wrong")).toEqual([]);
    expect(segmentClaims("Spoiler: the butler did it")).toEqual([]);
    expect(segmentClaims("The confidence interval was 95%.")).toEqual([
      "The confidence interval was 95%.",
    ]);
  });

  test("headings dropped, never fused with following text", () => {
    expect(
      segmentClaims(
        "## The ticker and the options problem (read this first)\nActual prose follows here.",
      ),
    ).toEqual(["Actual prose follows here."]);
    expect(
      segmentClaims("## What it is\nFixed-dose combo of tenofovir and emtricitabine."),
    ).toEqual(["Fixed-dose combo of tenofovir and emtricitabine."]);
    expect(segmentClaims("#trending topic line here.")).toEqual(["#trending topic line here."]);
  });

  test("mega block never fuses (production Estelle case)", () => {
    const claims = segmentClaims(
      [
        "Yo twin, here's the rundown on Estelle (born 1980 in Hammersmith, London):",
        "",
        "**Who she is**",
        "- British singer blending R&B, soul and grime ([Wikipedia](https://example.com))",
        "- Started out in London's Deal Real record store; John Legend became her mentor",
        "- Debut album The 18th Day dropped in 2005",
      ].join("\n"),
    );
    const markers = ["R&B", "John Legend", "18th Day"];
    for (const c of claims) {
      expect(markers.filter((m) => c.includes(m)).length).toBeLessThanOrEqual(1);
      expect(c).not.toContain("**Who she is**");
      expect(c.length).toBeLessThanOrEqual(300);
    }
    expect(claims.some((c) => c.startsWith("Yo twin"))).toBe(true);
    expect(claims.some((c) => c.includes("Deal Real record store"))).toBe(true);
    expect(claims.some((c) => c.startsWith("Debut album"))).toBe(true);
  });

  test("bold label dropped, bold-leading claim kept", () => {
    expect(segmentClaims("**Who she is**")).toEqual([]);
    expect(segmentClaims("*Summary*")).toEqual([]);
    expect(segmentClaims("**Alzheon** reported Phase 2 data in April 2025.")).toEqual([
      "**Alzheon** reported Phase 2 data in April 2025.",
    ]);
  });

  test("overlong sentences resplit at clauses, unsplittable dropped", () => {
    const src =
      "The committee reviewed the full dossier over several weeks " +
      "and interviewed witnesses ".repeat(8) +
      "; then it voted to release the findings; and the chair signed the final report.";
    const claims = segmentClaims(src);
    expect(claims.length).toBeGreaterThan(1);
    for (const c of claims) expect(c.length).toBeLessThanOrEqual(300);
    expect(claims.some((c) => c.includes("voted to release"))).toBe(true);
    expect(claims.some((c) => c.includes("signed the final report"))).toBe(true);
    for (const c of claims) {
      expect(c.includes("reviewed the full dossier") && c.includes("voted")).toBe(false);
    }
    expect(segmentClaims(("word ".repeat(120)).trim())).toEqual([]);
  });

  test("code fence / blockquote / numbered list / hr / bold-start", () => {
    expect(segmentClaims("```python\nprint('hello world')\n```\nThe sky is blue.")).toEqual([
      "The sky is blue.",
    ]);
    expect(segmentClaims("> The tower is in Paris.")).toEqual(["The tower is in Paris."]);
    expect(segmentClaims("1. Cats cannot fly.\n2) Dogs bark.")).toEqual([
      "Cats cannot fly.",
      "Dogs bark.",
    ]);
    expect(
      segmentClaims(["---", "***", "___", ":---", "***Bold emphasis opens a real claim about Paris."].join("\n")),
    ).toEqual(["***Bold emphasis opens a real claim about Paris."]);
  });

  test("corpus invariants over production-shaped reply", () => {
    const corpus = [
      "## Quick takes",
      "Here is the summary you asked for:",
      "",
      "| Ticker | Status |",
      "|---|---|",
      "| ALZH | missed endpoint |",
      "[^1]: Source, title, 2026.",
      "```",
      "const x = 1;",
      "```",
      "**Who she is**",
      "- Estelle was born in 1980 in Hammersmith, London.",
      "- She blends R&B, soul, reggae, grime and dance.",
      "Overall confidence: High.",
      "Remternetug is currently in Phase 3 trials.",
      "The committee reviewed the full dossier over several weeks " +
        "and interviewed witnesses ".repeat(8) +
        "; then it voted to release the findings.",
    ].join("\n");
    const claims = segmentClaims(corpus);
    expect(claims.length).toBeGreaterThan(0);
    for (const c of claims) {
      expect(c.length).toBeLessThanOrEqual(300);
      expect(c.trimStart().startsWith("|") || c.trimStart().startsWith("[^")).toBe(false);
      expect(c).not.toContain("Overall confidence");
      expect(c).not.toContain("const x");
    }
    expect(claims.some((c) => c.startsWith("Estelle was born in 1980"))).toBe(true);
    expect(claims.some((c) => c.startsWith("She blends"))).toBe(true);
    expect(claims.some((c) => c.includes("Remternetug"))).toBe(true);
    expect(claims.some((c) => c.includes("voted to release"))).toBe(true);
    expect(claims.every((c) => !c.includes("Ticker"))).toBe(true);
  });
});

describe("judge selection", () => {
  const cfg = resolveProvider({
    RFE_JUDGE_GLM_MODEL: GLM,
    RFE_JUDGE_DEEPSEEK_MODEL: DS,
  } as Record<string, string>);

  test("glm-flash active -> deepseek judge", () => {
    expect(selectJudgeModel(GLM, cfg)).toBe(DS);
    expect(selectJudgeModel("z-ai/glm-5.3-flash", cfg)).toBe(DS);
    expect(selectJudgeModel("hf:zai-org/GLM-5.3-Flash", cfg)).toBe(DS);
    expect(selectJudgeModel("z-ai/glm-5.3-flash:free", cfg)).toBe(DS);
  });

  test("anything else -> glm judge", () => {
    expect(selectJudgeModel("anthropic/claude-sonnet-4", cfg)).toBe(GLM);
    expect(selectJudgeModel("openai/gpt-5", cfg)).toBe(GLM);
    expect(selectJudgeModel("unknown", cfg)).toBe(GLM);
  });

  test("self-judge impossible when active model deliberately matches", () => {
    const judge = selectJudgeModel(GLM, cfg);
    expect(judge).not.toBe(GLM);
    expect(judge).toBe(DS);
    // and if the deepseek judge itself were active, judge flips back to glm
    expect(selectJudgeModel(DS, cfg)).toBe(GLM);
  });

  test("provider-specific env overrides win", () => {
    const c = resolveProvider({
      RFE_JUDGE_PROVIDER: "openrouter",
      RFE_OPENROUTER_DEEPSEEK_MODEL: "custom/ds",
      RFE_OPENROUTER_GLM_MODEL: "custom/glm",
    } as Record<string, string>);
    expect(c.glmModel).toBe("custom/glm");
    expect(c.deepseekModel).toBe("custom/ds");
    expect(c.baseUrl).toBe("https://openrouter.ai/api/v1");
  });

  test("generic env overrides win over provider-specific and defaults", () => {
    const c = resolveProvider({
      RFE_JUDGE_PROVIDER: "openrouter",
      RFE_OPENROUTER_GLM_MODEL: "preset/glm",
      RFE_JUDGE_GLM_MODEL: "any-provider/glm",
      RFE_JUDGE_DEEPSEEK_MODEL: "any-provider/ds",
    } as Record<string, string>);
    expect(c.glmModel).toBe("any-provider/glm");
    expect(c.deepseekModel).toBe("any-provider/ds");
  });

  test("any OpenAI-compatible provider + custom ids works (no hf/ shape needed)", () => {
    const c = resolveProvider({
      RFE_JUDGE_PROVIDER: "my-endpoint",
      RFE_JUDGE_BASE_URL: "https://llm.internal/v1",
      RFE_JUDGE_API_KEY: "k",
      RFE_JUDGE_GLM_MODEL: "openai/gpt-oss-120b",
      RFE_JUDGE_DEEPSEEK_MODEL: "mistral/magistral-small",
    } as Record<string, string>);
    expect(c.baseUrl).toBe("https://llm.internal/v1");
    expect(c.apiKey).toBe("k");
    const active = selectJudgeModel("openai/gpt-oss-120b", c);
    expect(active).toBe("mistral/magistral-small");
    expect(selectJudgeModel("other/model", c)).toBe("openai/gpt-oss-120b");
  });

  test("synthetic preset defaults", () => {
    const c = resolveProvider({
      RFE_JUDGE_PROVIDER: "synthetic",
    } as Record<string, string>);
    expect(c.glmModel).toBe("hf:zai-org/GLM-5.3-Flash");
    expect(c.deepseekModel).toBe("hf:deepseek-ai/DeepSeek-V4.1-Flash");
  });

  test("openrouter preset defaults", () => {
    const c = resolveProvider({
      RFE_JUDGE_PROVIDER: "openrouter",
    } as Record<string, string>);
    expect(c.glmModel).toBe("z-ai/glm-5.3-flash");
    expect(c.deepseekModel).toBe("deepseek/deepseek-v4.1-flash");
  });

  test("active-model check references the configured glm model, not a fixed id", () => {
    expect(isGlmFlash("z-ai/glm-5-flash")).toBe(false);
    expect(selectJudgeModel("z-ai/glm-5-flash", cfg)).toBe(GLM);
    const c = resolveProvider({
      RFE_JUDGE_PROVIDER: "custom",
      RFE_JUDGE_GLM_MODEL: "local/glm-5.3-flash",
      RFE_JUDGE_DEEPSEEK_MODEL: "local/ds",
    } as Record<string, string>);
    expect(isActiveJudgeModel("local/glm-5.3-flash", c)).toBe(true);
    expect(isActiveJudgeModel("local/glm-5.3-flash:free", c)).toBe(true);
    expect(isActiveJudgeModel("local/glm-4.6-flash", c)).toBe(false);
    expect(selectJudgeModel("local/glm-5.3-flash", c)).toBe("local/ds");
    expect(selectJudgeModel("local/glm-4.6-flash", c)).toBe("local/glm-5.3-flash");
  });
});

describe("verdict parsing", () => {
  test("valid JSON", () => {
    expect(parseVerdict('{"verdict": "faithful"}')).toBe("faithful");
  });

  test("markdown-fenced JSON", () => {
    expect(parseVerdict('```json\n{"verdict": "unfaithful"}\n```')).toBe(
      "unfaithful",
    );
  });

  test("garbage -> throw (judge falls back to unverifiable + parseErrors++)", async () => {
    expect(() => parseVerdict("total garbage")).toThrow();
    let calls = 0;
    const judge = new Judge({
      baseUrl: "https://x.test",
      apiKey: "k",
      model: GLM,
      session: "s",
      fetchImpl: (async () => {
        calls++;
        return {
          ok: true,
          status: 200,
          json: async () => okResp("nonsense"),
        } as Response;
      }) as unknown as typeof fetch,
      sleepImpl: noSleep,
    });
    const v = await judge.verdict("ctx", "claim");
    expect(v).toBe("unverifiable");
    expect(judge.parseErrors).toBe(1);
    expect(judge.callFailures).toBe(0);
    expect(calls).toBe(2); // retry once with 2x max_tokens on parse failure
  });
});

describe("judge audit rows (issue 82)", () => {
  test("parse-error row carries context_chars + retry tokens; judge rows too", async () => {
    const lines: string[] = [];
    const origWrite = process.stderr.write.bind(process.stderr);
    (process.stderr as { write: unknown }).write = (chunk: unknown) => {
      lines.push(String(chunk));
      return true;
    };
    try {
      const judge = new Judge({
        baseUrl: "https://x.test",
        apiKey: "k",
        model: GLM,
        session: "s",
        logFile: "stderr",
        fetchImpl: (async () => ({
          ok: true,
          status: 200,
          json: async () => okResp("nonsense"),
        })) as unknown as typeof fetch,
        sleepImpl: noSleep,
      });
      const ctx = "Cats cannot fly.";
      expect(await judge.verdict(ctx, "Cats can fly.")).toBe("unverifiable");
      const errRow = JSON.parse(lines.find((l) => l.includes('"judge-parse-error"'))!);
      expect(errRow.context_chars).toBe(ctx.length);
      expect(errRow.retry_prompt_tokens).toBeGreaterThan(0); // 2 attempts x 10
      expect(errRow.retry_completion_tokens).toBeGreaterThan(0); // 2 attempts x 5
      const judgeRow = JSON.parse(lines.find((l) => l.includes('"kind":"judge"'))!);
      expect(judgeRow.context_chars).toBe(ctx.length);
    } finally {
      process.stderr.write = origWrite;
    }
  });
});

describe("cache", () => {
  let dir: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "rfe-cache-"));
  });
  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
    delete process.env["RFE_CACHE_DIR"];
  });

  test("same (model,context,claim) twice -> one fetch call", async () => {
    let calls = 0;
    const judge = new Judge({
      baseUrl: "https://x.test",
      apiKey: "k",
      model: GLM,
      session: "s",
      cache: makeCache(GLM, {}),
      fetchImpl: (async () => {
        calls++;
        return { ok: true, status: 200, json: async () => okResp("faithful") } as Response;
      }) as unknown as typeof fetch,
      sleepImpl: noSleep,
    });
    expect(await judge.verdict("c", "p")).toBe("faithful");
    expect(await judge.verdict("c", "p")).toBe("faithful");
    expect(calls).toBe(1);
  });

  test("RFE_CACHE_DIR set -> file written", async () => {
    process.env["RFE_CACHE_DIR"] = dir;
    const judge = new Judge({
      baseUrl: "https://x.test",
      apiKey: "k",
      model: GLM,
      session: "s",
      cache: makeCache(GLM),
      fetchImpl: fakeFetch(okResp("faithful")),
      sleepImpl: noSleep,
    });
    await judge.verdict("ctx", "claim");
    const file = join(dir, "opencode-cache-test_glm-judge.jsonl");
    expect(existsSync(file)).toBe(true);
    const row = JSON.parse(readFileSync(file, "utf8").trim());
    expect(row.key).toBe(verdictKey(GLM, "ctx", "claim"));
    expect(row.verdict).toBe("faithful");
  });

  test("unset -> memory only", async () => {
    delete process.env["RFE_CACHE_DIR"];
    const cache = makeCache(GLM);
    const key = verdictKey(GLM, "a", "b");
    cache.store(key, "faithful");
    expect(cache.get(key)).toBe("faithful");
    expect(existsSync(join(dir, "nothing.jsonl"))).toBe(false);
  });

  test("failed call not cached; callFailures separate from parseErrors", async () => {
    const cache = makeCache(GLM, {});
    const judge = new Judge({
      baseUrl: "https://x.test",
      apiKey: "k",
      model: GLM,
      session: "s",
      retries: 1,
      cache,
      fetchImpl: (async () => {
        throw new Error("network down");
      }) as unknown as typeof fetch,
      sleepImpl: noSleep,
    });
    expect(await judge.verdict("ctx", "claim")).toBe("unverifiable");
    expect(cache.get(verdictKey(GLM, "ctx", "claim"))).toBeUndefined();
    expect(judge.callFailures).toBe(1);
    expect(judge.parseErrors).toBe(0);
  });

  test("transient failure then success -> second call judges, cached after", async () => {
    let calls = 0;
    const cache = makeCache(GLM, {});
    const judge = new Judge({
      baseUrl: "https://x.test",
      apiKey: "k",
      model: GLM,
      session: "s",
      retries: 1,
      cache,
      fetchImpl: (async () => {
        calls++;
        if (calls === 1) throw new Error("transient");
        return { ok: true, status: 200, json: async () => okResp("faithful") } as Response;
      }) as unknown as typeof fetch,
      sleepImpl: noSleep,
    });
    expect(await judge.verdict("ctx", "claim")).toBe("unverifiable");
    expect(await judge.verdict("ctx", "claim")).toBe("faithful");
    expect(await judge.verdict("ctx", "claim")).toBe("faithful");
    expect(calls).toBe(2);
  });

  test("registry memoizes cache per judge model (no re-read)", () => {
    process.env["RFE_CACHE_DIR"] = dir;
    const key = verdictKey(GLM, "a", "b");
    const file = join(dir, "opencode-cache-test_glm-judge.jsonl");
    writeFileSync(file, JSON.stringify({ key, verdict: "faithful" }) + "\n");
    const registry = makeCacheRegistry();
    expect(registry(GLM).get(key)).toBe("faithful");
    rmSync(file);
    expect(registry(GLM).get(key)).toBe("faithful");
    expect(registry(DS).get(key)).toBeUndefined();
  });
});

describe("http retry policy", () => {
  test("401 is not retried", async () => {
    let calls = 0;
    let sleeps = 0;
    const judge = new Judge({
      baseUrl: "https://x.test",
      apiKey: "bad",
      model: GLM,
      session: "s",
      retries: 5,
      fetchImpl: (async () => {
        calls++;
        return { ok: false, status: 401, json: async () => ({}) } as Response;
      }) as unknown as typeof fetch,
      sleepImpl: async () => {
        sleeps++;
      },
    });
    expect(await judge.verdict("ctx", "claim")).toBe("unverifiable");
    expect(calls).toBe(1);
    expect(sleeps).toBe(0);
    expect(judge.callFailures).toBe(1);
  });

  test("503 is retried", async () => {
    let calls = 0;
    const judge = new Judge({
      baseUrl: "https://x.test",
      apiKey: "k",
      model: GLM,
      session: "s",
      retries: 3,
      fetchImpl: (async () => {
        calls++;
        return { ok: false, status: 503, json: async () => ({}) } as Response;
      }) as unknown as typeof fetch,
      sleepImpl: noSleep,
    });
    expect(await judge.verdict("ctx", "claim")).toBe("unverifiable");
    expect(calls).toBe(3);
  });
});

describe("privacy redaction", () => {
  test("sensitive paths detected", () => {
    expect(isSensitivePath({ filePath: "/home/u/.env" })).toBe(true);
    expect(isSensitivePath({ filePath: "/home/u/.env.local" })).toBe(true);
    expect(isSensitivePath({ filePath: "/home/u/.ssh/id_rsa" })).toBe(true);
    expect(isSensitivePath({ filePath: "/x/server.pem" })).toBe(true);
    expect(isSensitivePath({ filePath: "/x/.npmrc" })).toBe(true);
    expect(isSensitivePath({ filePath: "/src/index.ts" })).toBe(false);
    expect(isSensitivePath({ filePath: "/x/.environment" })).toBe(false);
  });

  test("secret patterns redacted", () => {
    const blob = [
      "OPENAI_KEY=sk-abcdefghijklmnopqrstuvwx",
      "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig",
      "AKIAIOSFODNN7EXAMPLE",
      "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
      'aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"',
      'api_key: "supersecretvalue123"',
    ].join("\n");
    const out = redactSecrets(blob);
    expect(out).not.toContain("sk-abcdefghijklmnopqrstuvwx");
    expect(out).not.toContain("eyJhbGciOiJIUzI1NiJ9");
    expect(out).not.toContain("AKIAIOSFODNN7EXAMPLE");
    expect(out).not.toContain("ghp_abcdefghijklmnopqrstuvwxyz");
    expect(out).not.toContain("wJalrXUtnFEMI");
    expect(out).not.toContain("supersecretvalue123");
    expect(out).toContain("api_key=[REDACTED]");
  });

  test("private key block redacted", () => {
    const key =
      "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n-----END RSA PRIVATE KEY-----";
    const out = redactSecrets(`here: ${key}`);
    expect(out).not.toContain("MIIEowIBAAKCAQEA");
    expect(out).toContain("[REDACTED PRIVATE KEY]");
  });
});

describe("claim cap", () => {
  test("default 50", () => {
    expect(maxClaims({})).toBe(50);
  });

  test("env override", () => {
    expect(maxClaims({ RFE_MAX_CLAIMS: "3" })).toBe(3);
  });

  test("invalid -> default", () => {
    expect(maxClaims({ RFE_MAX_CLAIMS: "-1" })).toBe(50);
  });
});

describe("premise cap", () => {
  test("30k chars in -> most recent 24k retained", () => {
    const p = new PremiseStore(24_000);
    p.append("x".repeat(30_000));
    expect(p.length).toBe(24_000);
    p.append("A".repeat(10_000));
    p.append("B".repeat(20_000));
    expect(p.length).toBe(24_000);
    expect(p.text.startsWith("A".repeat(4000 - 1))).toBe(true); // older tail kept
    expect(p.text.endsWith("B".repeat(20_000))).toBe(true); // newest kept
  });
});

describe("premise selection (issue 82)", () => {
  test("short blob returned whole", () => {
    const src = "Alpha is A.\n\nBeta is B.";
    expect(selectPremise(src, "What is Alpha?")).toBe(src);
  });

  test("relevant passage kept, filler dropped, within budget", () => {
    const passages = Array.from(
      { length: 20 },
      (_, i) => `Filler document number ${i} discusses unrelated matters. ${"y".repeat(700)}`,
    );
    const relevant = "Alpha concentration measured in serum samples was elevated.";
    passages[3] = relevant;
    const src = passages.join("\n\n");
    const out = selectPremise(src, "Was the Alpha concentration elevated?");
    expect(out).toContain(relevant);
    expect(out).not.toContain("Filler document number 19");
    expect(out.length).toBeLessThanOrEqual(12_000);
  });

  test("oversized relevant passage head-truncated", () => {
    const src = ["zz".repeat(50), `Alpha serum levels rose sharply. ${"q".repeat(20000)}`].join(
      "\n\n",
    );
    const out = selectPremise(src, "Did Alpha serum levels rise?");
    expect(out.startsWith("Alpha serum levels rose sharply.")).toBe(true);
    expect(out.length).toBeLessThanOrEqual(12_000);
  });

  test("no overlap falls back to most recent chars", () => {
    const src = Array.from({ length: 10 }, (_, i) => `${String(i).padStart(2, "0")} ${"z".repeat(2000)}`).join("\n\n");
    const out = selectPremise(src, "Completely unrelated xylophone question?");
    expect(out).toBe(src.slice(-4000));
  });
});

describe("nudge aggregation", () => {
  test("2 flagged claims -> exactly one nudge message", () => {
    const nudge = buildNudge(GLM, [
      { claim: "Sky is green.", verdict: "unfaithful" },
      { claim: "Moon is cheese.", verdict: "unverifiable" },
    ]);
    expect(nudge).toContain("2 claim(s)");
    expect(nudge).toContain("test/glm-judge");
    expect(nudge).toContain("unfaithful (1)");
    expect(nudge).toContain("unverifiable (1)");
    expect(nudge).toContain("do not invent corrections");
    expect((nudge.match(/ragfaith judge/g) ?? []).length).toBe(1);
  });
});

describe("doc-pull heuristic", () => {
  test("install args captured", () => {
    const pkgs = extractPackages("bun install lodash @types/node -D");
    expect(pkgs.has("lodash")).toBe(true);
    expect(pkgs.has("@types/node")).toBe(true);
  });

  test("imports captured", () => {
    const pkgs = extractPackages(`import x from "express"; const y = require("zod");`);
    expect(pkgs.has("express")).toBe(true);
    expect(pkgs.has("zod")).toBe(true);
  });
});

describe("hooks never throw", () => {
  let stderrLines: string[];

  test("judge fetch rejects -> unverifiable, no exception; errors only when sink opted in", async () => {
    const origWrite = process.stderr.write.bind(process.stderr);
    stderrLines = [];
    (process.stderr as { write: unknown }).write = (chunk: unknown) => {
      stderrLines.push(String(chunk));
      return true;
    };
    const mk = (logFile?: string) =>
      new Judge({
        baseUrl: "https://x.test",
        apiKey: "k",
        model: GLM,
        session: "s",
        logFile,
        fetchImpl: (async () => {
          throw new Error("network down");
        }) as unknown as typeof fetch,
        sleepImpl: noSleep,
      });
    try {
      // default: silent — failing judge must not print to stderr
      const silent = await mk(undefined).verdict("ctx", "claim");
      expect(silent).toBe("unverifiable");
      expect(stderrLines.length).toBe(0);
      // explicit stderr sink: the error row is emitted
      const v = await mk("stderr").verdict("ctx", "claim");
      expect(v).toBe("unverifiable");
      const errLog = stderrLines.find((l) => l.includes('"kind":"error"'));
      expect(errLog).toBeDefined();
    } finally {
      (process.stderr as { write: unknown }).write = origWrite;
    }
  });
});

describe("plugin hooks", () => {
  interface PromptCall {
    path: { id: string };
    body: { parts: Array<{ type: string; text: string; synthetic?: boolean }> };
  }

  function fakeClient(): {
    client: {
      tui: { showToast: (arg: { body: { message: string } }) => Promise<void> };
      session: { prompt: (arg: PromptCall) => Promise<void> };
    };
    toasts: string[];
    prompts: PromptCall[];
  } {
    const toasts: string[] = [];
    const prompts: PromptCall[] = [];
    return {
      client: {
        tui: {
          showToast: async (arg) => {
            toasts.push(arg.body.message);
          },
        },
        session: {
          prompt: async (arg) => {
            prompts.push(arg);
          },
        },
      },
      toasts,
      prompts,
    };
  }

  async function pluginHooks() {
    const c = fakeClient();
    const hooks = await RagfaithPlugin(
      { client: c.client } as unknown as Parameters<typeof RagfaithPlugin>[0],
    );
    return { ...c, hooks };
  }

  test("session.deleted drops session state", async () => {
    const { hooks, toasts } = await pluginHooks();
    await hooks["tool.execute.after"]!(
      { tool: "read", sessionID: "s1", callID: "c1", args: { filePath: "/src/a.ts" } },
      { title: "a", output: "lodash documentation", metadata: {} },
    );
    await hooks["tool.execute.before"]!(
      { tool: "bash", sessionID: "s1", callID: "c2" },
      { args: { command: "bun install lodash" } },
    );
    expect(toasts.length).toBe(0);
    await hooks.event!({
      event: {
        type: "session.deleted",
        properties: { info: { id: "s1" } },
      } as unknown as Event,
    });
    await hooks["tool.execute.before"]!(
      { tool: "bash", sessionID: "s1", callID: "c3" },
      { args: { command: "bun install lodash" } },
    );
    expect(toasts.length).toBe(1);
  });

  test("sensitive path premise never captured or sent", async () => {
    const { hooks, prompts } = await pluginHooks();
    const origFetch = globalThis.fetch;
    process.env["SYNTHETIC_API_KEY"] = "test-key";
    let calls = 0;
    globalThis.fetch = (async () => {
      calls++;
      return { ok: true, status: 200, json: async () => okResp("faithful") } as Response;
    }) as unknown as typeof fetch;
    try {
      await hooks["tool.execute.after"]!(
        { tool: "read", sessionID: "s2", callID: "c1", args: { filePath: "/home/u/.env" } },
        { title: ".env", output: "OPENAI_API_KEY=sk-secretsecretsecret", metadata: {} },
      );
      await hooks.event!({
        event: {
          type: "message.part.updated",
          properties: {
            part: { id: "p1", sessionID: "s2", messageID: "m1", type: "text", text: "A claim." },
          },
        } as unknown as Event,
      });
      await hooks.event!({
        event: {
          type: "message.updated",
          properties: {
            info: {
              id: "m1",
              sessionID: "s2",
              role: "assistant",
              time: { completed: 1 },
              modelID: "m",
              providerID: "p",
              parentID: "u1",
            },
          },
        } as unknown as Event,
      });
      await Bun.sleep(20);
      expect(calls).toBe(0);
      expect(prompts.length).toBe(0);
    } finally {
      globalThis.fetch = origFetch;
      delete process.env["SYNTHETIC_API_KEY"];
    }
  });

  test("secrets redacted before premise reaches judge", async () => {
    const { hooks } = await pluginHooks();
    const origFetch = globalThis.fetch;
    process.env["SYNTHETIC_API_KEY"] = "test-key";
    let sent = "";
    globalThis.fetch = (async (_url: unknown, init: { body?: string }) => {
      sent = String(init?.body ?? "");
      return {
        ok: true,
        status: 200,
        json: async () => okResp("faithful"),
      } as Response;
    }) as unknown as typeof fetch;
    try {
      await hooks["tool.execute.after"]!(
        { tool: "read", sessionID: "s4", callID: "c1", args: { filePath: "/src/a.ts" } },
        { title: "a", output: "OPENAI_API_KEY=sk-secretsecretsecret", metadata: {} },
      );
      await hooks.event!({
        event: {
          type: "message.part.updated",
          properties: {
            part: { id: "p1", sessionID: "s4", messageID: "m1", type: "text", text: "A claim." },
          },
        } as unknown as Event,
      });
      await hooks.event!({
        event: {
          type: "message.updated",
          properties: {
            info: {
              id: "m1",
              sessionID: "s4",
              role: "assistant",
              time: { completed: 1 },
              modelID: "m",
              providerID: "p",
              parentID: "u1",
            },
          },
        } as unknown as Event,
      });
      for (let i = 0; i < 100 && !sent; i++) await Bun.sleep(5);
      expect(sent).toContain("A claim.");
      expect(sent).not.toContain("sk-secretsecretsecret");
    } finally {
      globalThis.fetch = origFetch;
      delete process.env["SYNTHETIC_API_KEY"];
    }
  });

  test("unfaithful reply -> one synthetic nudge, claims capped", async () => {
    const { hooks, prompts } = await pluginHooks();
    const origFetch = globalThis.fetch;
    process.env["SYNTHETIC_API_KEY"] = "test-key";
    process.env["RFE_MAX_CLAIMS"] = "1";
    globalThis.fetch = (async () => ({
      ok: true,
      status: 200,
      json: async () => okResp("unfaithful"),
    })) as unknown as typeof fetch;
    try {
      await hooks["tool.execute.after"]!(
        { tool: "read", sessionID: "s3", callID: "c1", args: { filePath: "/src/a.ts" } },
        { title: "a", output: "some context", metadata: {} },
      );
      await hooks.event!({
        event: {
          type: "message.part.updated",
          properties: {
            part: {
              id: "p1",
              sessionID: "s3",
              messageID: "m1",
              type: "text",
              text: "One claim. Two claim.",
            },
          },
        } as unknown as Event,
      });
      await hooks.event!({
        event: {
          type: "message.updated",
          properties: {
            info: {
              id: "m1",
              sessionID: "s3",
              role: "assistant",
              time: { completed: 1 },
              modelID: "m",
              providerID: "p",
              parentID: "u1",
            },
          },
        } as unknown as Event,
      });
      for (let i = 0; i < 100 && prompts.length === 0; i++) await Bun.sleep(5);
      expect(prompts.length).toBe(1);
      const part = prompts[0]!.body.parts[0]!;
      expect(part.synthetic).toBe(true);
      expect((part.text.match(/\[unfaithful\]/g) ?? []).length).toBe(1);
    } finally {
      globalThis.fetch = origFetch;
      delete process.env["SYNTHETIC_API_KEY"];
      delete process.env["RFE_MAX_CLAIMS"];
    }
  });
});

// ---------------------------------------------------------------- logging sink

describe("logging sink", () => {
  const origWrite = process.stderr.write;

  afterEach(() => {
    process.stderr.write = origWrite;
  });

  function capture(): string[] {
    const lines: string[] = [];
    process.stderr.write = ((s: unknown) => {
      lines.push(String(s));
      return true;
    }) as typeof process.stderr.write;
    return lines;
  }

  test("default: silent — nothing on stderr without RFE_JUDGE_LOG", () => {
    const lines = capture();
    logLine({ kind: "judge", prompt_tokens: 1 }, undefined);
    expect(lines.length).toBe(0);
  });

  test("RFE_JUDGE_LOG=stderr: explicit debug output on stderr", () => {
    const lines = capture();
    logLine({ kind: "judge" }, "stderr");
    expect(lines.length).toBe(1);
    expect(lines[0]!.startsWith("{")).toBe(true);
  });

  test("file path: JSONL appended, stderr untouched", () => {
    const lines = capture();
    const dir = mkdtempSync(join(tmpdir(), "rfe-log-"));
    const file = join(dir, "judge.jsonl");
    logLine({ kind: "judge" }, file);
    logLine({ kind: "judge" }, file);
    expect(lines.length).toBe(0);
    expect(readFileSync(file, "utf8").trim().split("\n").length).toBe(2);
    rmSync(dir, { recursive: true, force: true });
  });
});
