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
  isSensitivePath,
  redactSecrets,
  maxClaims,
  RagfaithPlugin,
} from "../src/index";

const GLM = "hf:zai-org/GLM-5.3-Flash";
const DS = "hf:deepseek-ai/DeepSeek-V4.1-Flash";

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
});

describe("judge selection", () => {
  const cfg = resolveProvider({
    RFE_JUDGE_PROVIDER: "synthetic",
  } as Record<string, string>);

  test("glm-flash active -> deepseek judge", () => {
    expect(selectJudgeModel(GLM, cfg)).toBe(DS);
    expect(selectJudgeModel("z-ai/glm-5.3-flash", cfg)).toBe(DS);
  });

  test("anything else -> glm judge", () => {
    expect(selectJudgeModel("anthropic/claude-sonnet-4", cfg)).toBe(GLM);
    expect(selectJudgeModel("openai/gpt-5", cfg)).toBe(GLM);
    expect(selectJudgeModel("unknown", cfg)).toBe(GLM);
  });

  test("self-judge impossible when active model deliberately matches", () => {
    const judge = selectJudgeModel(GLM, cfg);
    expect(judge).not.toBe(GLM);
    expect(judge.toLowerCase()).toContain("deepseek");
    // and if the deepseek judge itself were active, judge flips back to glm
    expect(selectJudgeModel(DS, cfg)).toBe(GLM);
  });

  test("env overrides win", () => {
    const c = resolveProvider({
      RFE_JUDGE_PROVIDER: "openrouter",
      RFE_OPENROUTER_DEEPSEEK_MODEL: "custom/ds",
      RFE_OPENROUTER_GLM_MODEL: "custom/glm",
    } as Record<string, string>);
    expect(c.glmModel).toBe("custom/glm");
    expect(c.deepseekModel).toBe("custom/ds");
    expect(c.baseUrl).toBe("https://openrouter.ai/api/v1");
  });

  test("synthetic provider-specific overrides win", () => {
    const c = resolveProvider({
      RFE_SYNTHETIC_GLM_MODEL: "s/glm",
      RFE_SYNTHETIC_DEEPSEEK_MODEL: "s/ds",
    } as Record<string, string>);
    expect(c.glmModel).toBe("s/glm");
    expect(c.deepseekModel).toBe("s/ds");
  });

  test("openrouter defaults match proxy model ids", () => {
    const c = resolveProvider({
      RFE_JUDGE_PROVIDER: "openrouter",
    } as Record<string, string>);
    expect(c.glmModel).toBe("z-ai/glm-5.3-flash");
    expect(c.deepseekModel).toBe("deepseek/deepseek-v4.1-flash");
  });

  test("non-5.3 glm-flash is not treated as self-judging", () => {
    expect(isGlmFlash("z-ai/glm-5-flash")).toBe(false);
    expect(selectJudgeModel("z-ai/glm-5-flash", cfg)).toBe(GLM);
    expect(isGlmFlash("hf:zai-org/GLM-5.3-Flash")).toBe(true);
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
    const file = join(dir, "opencode-cache-hf_zai-org_GLM-5.3-Flash.jsonl");
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
    const file = join(dir, "opencode-cache-hf_zai-org_GLM-5.3-Flash.jsonl");
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

describe("nudge aggregation", () => {
  test("2 flagged claims -> exactly one nudge message", () => {
    const nudge = buildNudge(GLM, [
      { claim: "Sky is green.", verdict: "unfaithful" },
      { claim: "Moon is cheese.", verdict: "unverifiable" },
    ]);
    expect(nudge).toContain("2 claim(s)");
    expect(nudge).toContain(GLM);
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

  test("judge fetch rejects -> unverifiable, no exception, log emitted", async () => {
    const origWrite = process.stderr.write.bind(process.stderr);
    stderrLines = [];
    (process.stderr as { write: unknown }).write = (chunk: unknown) => {
      stderrLines.push(String(chunk));
      return true;
    };
    try {
      const judge = new Judge({
        baseUrl: "https://x.test",
        apiKey: "k",
        model: GLM,
        session: "s",
        fetchImpl: (async () => {
          throw new Error("network down");
        }) as unknown as typeof fetch,
        sleepImpl: noSleep,
      });
      const v = await judge.verdict("ctx", "claim");
      expect(v).toBe("unverifiable");
      expect(judge.callFailures).toBe(1);
      expect(judge.parseErrors).toBe(0);
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
