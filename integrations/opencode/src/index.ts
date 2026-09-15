import type { Plugin } from "@opencode-ai/plugin";
import type { Event, AssistantMessage } from "@opencode-ai/sdk";
import { createHash } from "node:crypto";
import { appendFileSync, mkdirSync, readFileSync } from "node:fs";
import { dirname } from "node:path";

// ---------------------------------------------------------------------------
// ragfaith opencode plugin: live faithfulness cascade.
// Port of rag_faithfulness_eval/llm_judge.py verdict call (semantics verbatim).
// All failures degrade to log lines; hooks never throw; nothing blocks replies.
// ---------------------------------------------------------------------------

export const VERDICTS = ["faithful", "unfaithful", "unverifiable"] as const;
export type Verdict = (typeof VERDICTS)[number];

export const SYSTEM_PROMPT =
  "You are a RAG faithfulness judge. / Du bist ein RAG-Treuerichter.\n" +
  "Decide if the CLAIM is fully supported by the CONTEXT alone (never use " +
  "outside knowledge). Answer with ONLY one JSON object, no other text:\n" +
  '{"verdict": "faithful"} - every fact in the claim is supported by the context\n' +
  '{"verdict": "unfaithful"} - at least one fact contradicts or is unsupported ' +
  "by the context (wrong entity, number, date, or fabricated detail)\n" +
  '{"verdict": "unverifiable"} - the context does not address the claim at all';

const VERDICT_RE = /"verdict"\s*:\s*"(\w+)"/;
const DEFAULT_PREMISE_CAP = 24_000;
const DEFAULT_PREMISE_TOOLS = "read|fetch|web|doc|search";
const DEFAULT_MAX_CLAIMS = 50;
const PACKAGE_RE = /@?[a-z0-9][a-z0-9._\/-]*/gi;

// ---------------------------------------------------------------------------
// privacy: sensitive paths + secret redaction (no deps, applied before store)
// ---------------------------------------------------------------------------

const SENSITIVE_PATH_RE =
  /(?:^|[\\/"'\s:=])(?:\.env(?:\.[\w.-]+)?|\.npmrc|\.netrc|\.ssh[\\/]|\.aws[\\/]credentials|id_rsa|id_ed25519|credentials|[\w.-]+\.(?:pem|key|p12))(?=$|[\\/"'\s,}\]])/i;

const SECRET_PATTERNS: Array<[RegExp, string]> = [
  [
    /-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g,
    "[REDACTED PRIVATE KEY]",
  ],
  [/-----BEGIN [A-Z ]*PRIVATE KEY-----/g, "[REDACTED PRIVATE KEY]"],
  [/\bsk-[A-Za-z0-9_-]{8,}/g, "sk-[REDACTED]"],
  [/\bBearer\s+[A-Za-z0-9._~+/=-]{8,}/g, "Bearer [REDACTED]"],
  [/\bAKIA[0-9A-Z]{16}\b/g, "[REDACTED AWS KEY]"],
  [/\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b/g, "[REDACTED GITHUB TOKEN]"],
  [
    /\baws_secret_access_key\s*[:=]\s*["']?[A-Za-z0-9/+=]{16,}["']?/gi,
    "aws_secret_access_key=[REDACTED]",
  ],
  [
    /\b(api[_-]?key|token|secret|password)\s*[:=]\s*["'][^"'\n]{8,}["']/gi,
    "$1=[REDACTED]",
  ],
];

export function isSensitivePath(args: unknown): boolean {
  const s = typeof args === "string" ? args : JSON.stringify(args ?? "");
  return SENSITIVE_PATH_RE.test(s);
}

export function redactSecrets(text: string): string {
  let out = text;
  for (const [re, repl] of SECRET_PATTERNS) out = out.replace(re, repl);
  return out;
}

// ---------------------------------------------------------------------------
// env config (read lazily so tests can set env per case)
// ---------------------------------------------------------------------------

export interface JudgeProviderConfig {
  provider: string;
  baseUrl: string;
  apiKey: string;
  glmModel: string;
  deepseekModel: string;
}

export function resolveProvider(
  env: Record<string, string | undefined> = process.env,
): JudgeProviderConfig {
  const provider = env["RFE_JUDGE_PROVIDER"] ?? "synthetic";
  const isOpenRouter = provider === "openrouter";
  const baseUrl =
    env["RFE_JUDGE_BASE_URL"] ??
    (isOpenRouter ? "https://openrouter.ai/api/v1" : "https://api.synthetic.new/v1");
  const apiKey =
    env["RFE_JUDGE_API_KEY"] ?? (isOpenRouter ? env["OPENROUTER_API_KEY"] : env["SYNTHETIC_API_KEY"]) ?? "";
  // generic model overrides take precedence over provider-specific vars, so any
  // OpenAI-compatible endpoint works without synthetic-style model ids
  const glmModel =
    env["RFE_JUDGE_GLM_MODEL"] ??
    (isOpenRouter
      ? env["RFE_OPENROUTER_GLM_MODEL"] ?? "z-ai/glm-5.3-flash"
      : env["RFE_SYNTHETIC_GLM_MODEL"] ?? "hf:zai-org/GLM-5.3-Flash");
  const deepseekModel =
    env["RFE_JUDGE_DEEPSEEK_MODEL"] ??
    (isOpenRouter
      ? env["RFE_OPENROUTER_DEEPSEEK_MODEL"] ?? "deepseek/deepseek-v4.1-flash"
      : env["RFE_SYNTHETIC_DEEPSEEK_MODEL"] ?? "hf:deepseek-ai/DeepSeek-V4.1-Flash");
  return { provider, baseUrl, apiKey, glmModel, deepseekModel };
}

function premiseToolsRegex(
  env: Record<string, string | undefined> = process.env,
): RegExp {
  const src = env["RFE_PREMISE_TOOLS"] ?? DEFAULT_PREMISE_TOOLS;
  try {
    return new RegExp(src, "i");
  } catch {
    return new RegExp(DEFAULT_PREMISE_TOOLS, "i");
  }
}

function premiseCap(
  env: Record<string, string | undefined> = process.env,
): number {
  const n = Number.parseInt(env["RFE_PREMISE_CAP"] ?? "", 10);
  return Number.isFinite(n) && n > 0 ? n : DEFAULT_PREMISE_CAP;
}

export function maxClaims(
  env: Record<string, string | undefined> = process.env,
): number {
  const n = Number.parseInt(env["RFE_MAX_CLAIMS"] ?? "", 10);
  return Number.isFinite(n) && n > 0 ? n : DEFAULT_MAX_CLAIMS;
}

/** Verdicts that trigger a nudge (issue #86): normal = unfaithful only;
 *  RFE_STRICTNESS=strict adds unverifiable; explicit RFE_FLAG_VERDICTS wins. */
export function flagVerdicts(
  env: Record<string, string | undefined> = process.env,
): Verdict[] {
  const csv = env["RFE_FLAG_VERDICTS"];
  if (csv) {
    const items = csv
      .split(",")
      .map((v) => v.trim())
      .filter((v): v is Verdict => (VERDICTS as readonly string[]).includes(v));
    if (items.length) return items;
  }
  return env["RFE_STRICTNESS"] === "strict"
    ? ["unfaithful", "unverifiable"]
    : ["unfaithful"];
}

// ---------------------------------------------------------------------------
// per-claim premise selection (issue #82)
// ---------------------------------------------------------------------------

// English glue words excluded from premise/claim overlap scoring; content
// words carry the signal (also covers DE/IT reasonably via unicode \p{L})
const STOPWORDS = new Set(
  ("the a an and or of to in on for with is are was were be been it this that " +
    "these those as at by from not but if then than so such into over under " +
    "about their its his her they them you your we our can could will would " +
    "should may might must do does did done have has had what which who whom " +
    "when where why how").split(" "),
);

const DEFAULT_PREMISE_BUDGET = 12_000;
const DEFAULT_PREMISE_FALLBACK = 4_000;

// conservative label under premise filtering (issue #86): support missing only
// because of truncation must not read as fabrication
export const TRUNCATION_NOTE =
  "\n\n[NOTE: this context is a relevance-filtered excerpt of the pulled " +
  'sources; if the claim\'s support is missing only because of that filtering ' +
  'or truncation, answer "unverifiable", not "unfaithful".]';

function claimTokens(text: string): Set<string> {
  const out = new Set<string>();
  for (const w of text.toLowerCase().matchAll(/[\p{L}\p{N}]{4,}/gu)) {
    const t = w[0];
    if (t && !STOPWORDS.has(t)) out.add(t);
  }
  return out;
}

/** Most relevant premise passages for one claim, within `budget` chars
 *  (lexical overlap, no embeddings). Oversized relevant passages are
 *  head-truncated to the remaining room; with no overlap at all, the most
 *  recent `fallback` chars are sent so the judge can answer unverifiable. */
export function selectPremise(
  premises: string,
  claim: string,
  budget = DEFAULT_PREMISE_BUDGET,
  fallback = DEFAULT_PREMISE_FALLBACK,
): string {
  if (premises.length <= budget) return premises;
  const claimToks = claimTokens(claim);
  if (claimToks.size === 0) return premises.slice(-fallback);
  const passages = premises.split("\n\n");
  const scored = passages.map((p, i) => {
    const toks = claimTokens(p);
    let score = 0;
    for (const t of claimToks) if (toks.has(t)) score++;
    return { score, i, p };
  });
  scored.sort((a, b) => b.score - a.score || a.i - b.i);
  const picked: Array<{ i: number; text: string }> = [];
  let used = 0;
  for (const s of scored) {
    if (s.score <= 0 || used >= budget) continue;
    const room = budget - used;
    const text = s.p.length <= room ? s.p : s.p.slice(0, room);
    picked.push({ i: s.i, text });
    used += text.length + 2;
  }
  if (picked.length === 0) return premises.slice(-fallback);
  picked.sort((a, b) => a.i - b.i);
  return picked.map((p) => p.text).join("\n\n");
}

// ---------------------------------------------------------------------------
// claim segmentation
// ---------------------------------------------------------------------------

// Longest claim handed to the judge (nudge display truncates at 200; longer
// blobs degrade judge JSON compliance and make nudges un-actionable).
const MAX_CLAIM_CHARS = 300;

// heading lines (## ... ) — labels, not assertions
const HEADING_RE = /^\s*#{1,6}\s+\S/;

// table rows/fragments, footnotes, reference-style link definitions
const FURNITURE_RE = /^\s*(?:\||\[\^|\^\S|\[[^\]\n]*\]:\s*<?https?:\/\/)/;

// thematic breaks: --- *** ___ :--- (must be the whole line)
const HR_RE = /^\s*:?[*_-]{3,}:?\s*$/;

// meta-confidence lines: "Overall confidence: High." / "tl;dr: ..." /
// "spoiler: ..." — never assertions. Requires : or - right after the label
// so "Confidence interval was 95%" survives.
const META_RE =
  /^\s*(?:tl\s*;\s*dr|spoiler(?:\s+alert)?|(?:overall\s+)?confidence|certainty)\s*[:\-]/i;

// whole-line bold/italic labels: "**Who she is**" — sub-headers, not claims
// (content after the label keeps the line as a claim candidate)
const LABEL_RE = /^\s*\*{1,3}[^*\n]{1,60}\*{1,3}\s*[:.]?\s*$/;

// list bullets / numbered markers: content becomes its own atomic candidate
const LIST_RE = /^\s*(?:[-*+]|\d{1,3}[.)])\s+/;

// blockquote markers
const QUOTE_RE = /^\s*>\s?/;

// clause boundaries for over-long claims: after ; : — –
const CLAUSE_RE = /(?<=[;:\u2014\u2013])\s+/;

/** Markdown-structural pre-pass (issue #76): one candidate per source line
 *  so whole blocks never fuse into mega-claims. */
function claimSegments(text: string): string[] {
  const segs: string[] = [];
  let fence = false;
  for (const line of text.split("\n")) {
    const s = line.trim();
    if (!s) continue; // blank line = hard boundary; candidates never fuse
    if (fence) {
      if (s.startsWith("```") || s.startsWith("~~~")) fence = false;
      continue; // code lines are not claims
    }
    if (s.startsWith("```") || s.startsWith("~~~")) {
      fence = true;
      continue;
    }
    if (HR_RE.test(s) || HEADING_RE.test(s) || FURNITURE_RE.test(s)) continue;
    if (META_RE.test(s) || LABEL_RE.test(s)) continue;
    let t = s.replace(LIST_RE, "");
    t = t.replace(QUOTE_RE, "");
    if (t) segs.push(t);
  }
  return segs;
}

/** Enforce MAX_CLAIM_CHARS: re-split at clause boundaries, drop remnants
 *  that are still over-long (never pass a blob to the judge). */
function claimTexts(sentence: string): string[] {
  if (sentence.length <= MAX_CLAIM_CHARS) return [sentence];
  return sentence
    .split(CLAUSE_RE)
    .map((p) => p.trim())
    .filter((p) => p && p.length <= MAX_CLAIM_CHARS);
}

/** Split reply text into sentence-granularity claims via Intl.Segmenter.
 *  Markdown furniture/labels/headings/code are never claims; list items and
 *  blockquotes are atomic; no claim exceeds MAX_CLAIM_CHARS (issue #76). */
export function segmentClaims(text: string): string[] {
  // sentence-boundary drift vs spaCy sentencizer; swap in a real
  // segmenter lib if parity matters
  const seg = new Intl.Segmenter(undefined, { granularity: "sentence" });
  const claims: string[] = [];
  for (const cand of claimSegments(text)) {
    for (const s of seg.segment(cand)) {
      const t = s.segment.trim();
      if (t) claims.push(...claimTexts(t));
    }
  }
  return claims;
}

// ---------------------------------------------------------------------------
// judge selection
// ---------------------------------------------------------------------------

export function isGlmFlash(model: string, pattern = "glm-5.3-flash"): boolean {
  return normalizeModelId(model).includes(pattern.toLowerCase());
}

// provider prefixes ("hf:org/", "z-ai/") carry no identity; strip to the base id
function normalizeModelId(model: string): string {
  return (model.toLowerCase().split("/").pop() ?? model).split(":")[0] ?? model;
}

/** True when the active model is the configured GLM judge (or the built-in
 *  GLM-5.3-Flash preset), ignoring provider prefix / ":" variants, so a custom
 *  GLM model id still gets never-self-judge protection. */
export function isActiveJudgeModel(activeModel: string, cfg: JudgeProviderConfig): boolean {
  const active = normalizeModelId(activeModel);
  const candidates = [normalizeModelId(cfg.glmModel)];
  if (cfg.glmModel !== "glm-5.3-flash") candidates.push("glm-5.3-flash");
  return candidates.some((c) => c !== "" && active.includes(c));
}

/** Never self-judge: active model is the GLM judge -> DeepSeek judge; else GLM. */
export function selectJudgeModel(
  activeModel: string,
  cfg: JudgeProviderConfig,
): string {
  return isActiveJudgeModel(activeModel, cfg) ? cfg.deepseekModel : cfg.glmModel;
}

// ---------------------------------------------------------------------------
// verdict parsing (port of llm_judge._parse_verdict)
// ---------------------------------------------------------------------------

export function parseVerdict(text: string): Verdict {
  const m = VERDICT_RE.exec(text);
  const verdict = (m?.[1] ?? "") as Verdict;
  if (!VERDICTS.includes(verdict)) {
    throw new Error(`unparseable verdict: ${text.slice(0, 200)}`);
  }
  return verdict;
}

// ---------------------------------------------------------------------------
// verdict key + cache (port of llm_judge._verdict_key / JSONL store)
// ---------------------------------------------------------------------------

export function verdictKey(model: string, context: string, claim: string): string {
  return createHash("sha256")
    .update(`${model}\x00${context}\x00${claim}`)
    .digest("hex");
}

function sanitizeModel(model: string): string {
  return model.replace(/[^a-zA-Z0-9.-]/g, "_");
}

export interface VerdictCache {
  get(key: string): Verdict | undefined;
  store(key: string, verdict: Verdict): void;
}

export function makeCache(
  judgeModel: string,
  env: Record<string, string | undefined> = process.env,
): VerdictCache {
  const dir = env["RFE_CACHE_DIR"];
  const mem = new Map<string, Verdict>();
  let file: string | undefined;
  if (dir) {
    file = `${dir}/opencode-cache-${sanitizeModel(judgeModel)}.jsonl`;
    try {
      for (const line of readFileSync(file, "utf8").split("\n")) {
        if (!line.trim()) continue;
        const row = JSON.parse(line) as { key: string; verdict: Verdict };
        mem.set(row.key, row.verdict);
      }
    } catch {
      /* fresh cache */
    }
  }
  return {
    get: (key) => mem.get(key),
    store: (key, verdict) => {
      mem.set(key, verdict);
      if (file) {
        try {
          mkdirSync(dirname(file), { recursive: true });
          appendFileSync(file, JSON.stringify({ key, verdict }) + "\n");
        } catch (e) {
          logErr("cache-store", e);
        }
      }
    },
  };
}

/** Plugin-scope cache memo: one loaded cache per judge model, reused replies. */
export function makeCacheRegistry(
  env: Record<string, string | undefined> = process.env,
): (judgeModel: string) => VerdictCache {
  const caches = new Map<string, VerdictCache>();
  return (judgeModel) => {
    let c = caches.get(judgeModel);
    if (!c) {
      c = makeCache(judgeModel, env);
      caches.set(judgeModel, c);
    }
    return c;
  };
}

// ---------------------------------------------------------------------------
// logging: token lines only, never USD
// ---------------------------------------------------------------------------

export function logLine(obj: Record<string, unknown>, logFile?: string): void {
  // silent by default: opencode surfaces plugin stderr in the TUI, and raw
  // JSONL rows are noise for users. Opt in via RFE_JUDGE_LOG=<path> for a
  // file, or RFE_JUDGE_LOG=stderr for explicit debug output.
  if (!logFile) return;
  const line = JSON.stringify(obj) + "\n";
  try {
    if (logFile === "stderr") process.stderr.write(line);
    else appendFileSync(logFile, line);
  } catch {
    /* logging must never throw */
  }
}

function logErr(kind: string, e: unknown, logFile?: string): void {
  logLine(
    {
      ts: new Date().toISOString(),
      kind: "error",
      where: kind,
      error: e instanceof Error ? e.message : String(e),
    },
    logFile,
  );
}

// ---------------------------------------------------------------------------
// judge (port of llm_judge.OpenRouterJudge._call / .verdict / ._account)
// ---------------------------------------------------------------------------

type FetchImpl = typeof fetch;
type SleepImpl = (ms: number) => Promise<void>;

export interface JudgeOptions {
  baseUrl: string;
  apiKey: string;
  model: string;
  session: string;
  cache?: VerdictCache;
  logFile?: string;
  maxTokens?: number;
  retries?: number;
  fetchImpl?: FetchImpl;
  sleepImpl?: SleepImpl;
}

interface ChatResp {
  choices?: Array<{ message?: { content?: string | null } }>;
  usage?: { prompt_tokens?: number; completion_tokens?: number };
}

class NonRetryableHttpError extends Error {}

export class Judge {
  readonly checkpoint: string;
  parseErrors = 0;
  callFailures = 0;
  private readonly o: Required<Omit<JudgeOptions, "cache">> & {
    cache?: VerdictCache;
  };

  constructor(opts: JudgeOptions) {
    this.o = {
      maxTokens: 256,
      retries: 6,
      fetchImpl: globalThis.fetch.bind(globalThis),
      sleepImpl: (ms) => new Promise((r) => setTimeout(r, ms)),
      logFile: "",
      ...opts,
    };
    this.checkpoint = opts.model;
  }

  private async call(userMsg: string, maxTokens: number): Promise<ChatResp> {
    const body = JSON.stringify({
      model: this.checkpoint,
      temperature: 0,
      max_tokens: maxTokens,
      reasoning: { exclude: true }, // hide reasoning; still billed ~100-250 tok
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: userMsg },
      ],
    });
    const { retries } = this.o;
    let lastErr: unknown;
    for (let attempt = 0; attempt < retries; attempt++) {
      try {
        const resp = await this.o.fetchImpl(
          `${this.o.baseUrl}/chat/completions`,
          {
            method: "POST",
            headers: {
              Authorization: `Bearer ${this.o.apiKey}`,
              "Content-Type": "application/json",
            },
            body,
          },
        );
        const retryable = resp.status === 429 || resp.status >= 500;
        if (retryable && attempt < retries - 1) {
          await this.o.sleepImpl(2 ** attempt * 1000);
          continue;
        }
        if (!resp.ok) {
          throw retryable
            ? new Error(`judge HTTP ${resp.status}`)
            : new NonRetryableHttpError(`judge HTTP ${resp.status}`);
        }
        return (await resp.json()) as ChatResp;
      } catch (e) {
        if (e instanceof NonRetryableHttpError) throw e;
        lastErr = e;
        if (attempt < retries - 1) {
          // network drop: wait it out (up to 1 min per try)
          await this.o.sleepImpl(Math.min(60, 5 * 2 ** attempt) * 1000);
          continue;
        }
      }
    }
    throw lastErr instanceof Error ? lastErr : new Error(String(lastErr));
  }

  private account(resp: ChatResp, contextChars: number): void {
    const usage = resp.usage ?? {};
    logLine(
      {
        ts: new Date().toISOString(),
        session: this.o.session,
        kind: "judge",
        model: this.checkpoint,
        prompt_tokens: usage.prompt_tokens ?? 0,
        completion_tokens: usage.completion_tokens ?? 0,
        context_chars: contextChars,
      },
      this.o.logFile || undefined,
    );
  }

  /** Judge one claim against context. Never throws; fallback = unverifiable. */
  async verdict(context: string, claim: string): Promise<Verdict> {
    const key = verdictKey(this.checkpoint, context, claim);
    const hit = this.o.cache?.get(key);
    if (hit) return hit;
    let verdict: Verdict = "unverifiable";
    const msg = `CONTEXT:\n${context}\n\nCLAIM:\n${claim}`;
    try {
      let parsed: Verdict | undefined;
      let retryPrompt = 0;
      let retryCompletion = 0;
      for (const maxTokens of [this.o.maxTokens, this.o.maxTokens * 2]) {
        const resp = await this.call(msg, maxTokens);
        this.account(resp, context.length);
        retryPrompt += resp.usage?.prompt_tokens ?? 0;
        retryCompletion += resp.usage?.completion_tokens ?? 0;
        const content = resp.choices?.[0]?.message?.content ?? "";
        try {
          parsed = parseVerdict(content);
          break;
        } catch {
          continue;
        }
      }
      if (parsed === undefined) {
        this.parseErrors += 1; // parse failure: conservative fallback, counted
        logLine(
          {
            ts: new Date().toISOString(),
            session: this.o.session,
            kind: "judge-parse-error",
            model: this.checkpoint,
            claim,
            context_chars: context.length,
            retry_prompt_tokens: retryPrompt,
            retry_completion_tokens: retryCompletion,
          },
          this.o.logFile || undefined,
        );
      } else {
        verdict = parsed;
      }
      this.o.cache?.store(key, verdict);
    } catch (e) {
      // network/HTTP exhausted retries: conservative, logged, never cached
      this.callFailures += 1;
      logErr("judge-call", e, this.o.logFile || undefined);
    }
    return verdict;
  }
}

// ---------------------------------------------------------------------------
// premise store (cap keeps most recent ~cap chars)
// ---------------------------------------------------------------------------

export class PremiseStore {
  private buf = "";
  constructor(private readonly cap = DEFAULT_PREMISE_CAP) {}

  append(text: string): void {
    this.buf = (this.buf + "\n" + text).slice(-this.cap);
  }

  get text(): string {
    return this.buf;
  }

  get length(): number {
    return this.buf.length;
  }
}

// ---------------------------------------------------------------------------
// doc-pull check (free deterministic heuristic)
// ---------------------------------------------------------------------------

const INSTALL_RE =
  /(?:npm|bun|pnpm|yarn|pip)\s+(?:install|add|i)\b((?:\s+(?!--)[^\s;&|]+)*)/gi;
const IMPORT_RE =
  /(?:import\s+(?:[\w*{}\s,]+\s+from\s+)?|require\s*\()\s*["']([@\w][@\w._\/-]*)["']/g;

/** Heuristic package tokens referenced by a command/file snippet. */
export function extractPackages(text: string): Set<string> {
  const out = new Set<string>();
  for (const m of text.matchAll(INSTALL_RE)) {
    const tail = m[1] ?? "";
    for (const tok of tail.split(/\s+/)) {
      const t = tok.trim();
      if (!t || t.startsWith("-")) continue;
      const name = t.match(PACKAGE_RE)?.[0];
      if (name && name.length >= 2) out.add(name.toLowerCase());
    }
  }
  for (const m of text.matchAll(IMPORT_RE)) {
    const name = m[1];
    if (name && name.length >= 2 && !name.startsWith(".")) {
      out.add(name.toLowerCase());
    }
  }
  return out;
}

/**
 * Tokens "documented this session": package names mentioned by any fetched-docs
 * premise. Bare-word match keeps false-positive warnings (annoying) rare at the
 * cost of false negatives (silent) — the check is advisory anyway.
 */
export function extractDocTokens(premiseText: string): Set<string> {
  const out = new Set<string>();
  for (const m of premiseText.toLowerCase().matchAll(PACKAGE_RE)) {
    const t = m[0];
    if (t.length >= 3) out.add(t);
  }
  for (const p of extractPackages(premiseText)) out.add(p);
  return out;
}

// ---------------------------------------------------------------------------
// nudge
// ---------------------------------------------------------------------------

export interface FlaggedClaim {
  claim: string;
  verdict: "unfaithful" | "unverifiable";
}

export function buildNudge(judgeModel: string, flagged: FlaggedClaim[]): string {
  const counts = new Map<string, number>();
  for (const f of flagged) counts.set(f.verdict, (counts.get(f.verdict) ?? 0) + 1);
  const verdicts = [...counts.entries()]
    .map(([v, n]) => `${v} (${n})`)
    .join(", ");
  const claims = flagged
    .map((f, i) => `${i + 1}. [${f.verdict}] ${f.claim.slice(0, 200)}`)
    .join("\n");
  let out =
    `ragfaith judge (${judgeModel}): ${flagged.length} claim(s) in your last ` +
    `reply were flagged ${verdicts}.\nClaims:\n${claims}\n` +
    "Re-check against the sources actually pulled in this session and " +
    "reconcile; do not invent corrections.";
  if (counts.has("unverifiable")) {
    // strict arm (issue #86): an unverifiable claim is unattributed, not wrong
    out +=
      "\n\nUnverifiable claims are not in the pulled sources; resolve each " +
      "visibly in your next reply in exactly one of these ways: (1) back it " +
      "with further searches or fetches and cite the newly pulled source; " +
      "(2) openly disclose that it comes from your internal (training) " +
      "knowledge, not the pulled sources; (3) openly disclose that it was " +
      "inferred from data inside the context. No silent assertions.";
  }
  return out;
}

// ---------------------------------------------------------------------------
// plugin
// ---------------------------------------------------------------------------

interface SessionState {
  premises: PremiseStore;
  docTokens: Set<string>;
  activeModel: string;
  parts: Map<string, Map<string, string>>; // messageID -> partID -> text
  judged: Set<string>; // messageIDs already judged
  nudgeIds: Set<string>; // our own injected message ids (loop guard)
}

function makeState(): SessionState {
  return {
    premises: new PremiseStore(premiseCap()),
    docTokens: new Set(),
    activeModel: "",
    parts: new Map(),
    judged: new Set(),
    nudgeIds: new Set(),
  };
}

export const RagfaithPlugin: Plugin = async ({ client }) => {
  const states = new Map<string, SessionState>();
  const cacheOf = makeCacheRegistry();
  const logFile = process.env["RFE_JUDGE_LOG"] || undefined;
  const toolsRe = premiseToolsRegex();

  const stateOf = (sessionID: string): SessionState => {
    let s = states.get(sessionID);
    if (!s) {
      s = makeState();
      states.set(sessionID, s);
    }
    return s;
  };

  const toast = (message: string, title = "ragfaith"): void => {
    // fire-and-forget; TUI may not be connected (headless) — fine
    client.tui
      .showToast({ body: { title, message, variant: "warning" } })
      .then(
        () => {},
        (e) => logErr("toast", e, logFile),
      );
  };

  /** Judge a completed reply; aggregate flagged claims into ONE nudge. */
  const judgeReply = (sessionID: string, messageID: string): void => {
    void (async () => {
      const st = stateOf(sessionID);
      if (st.judged.has(messageID)) return;
      st.judged.add(messageID);
      const text = [...(st.parts.get(messageID)?.values() ?? [])].join("\n");
      const context = st.premises.text;
      if (!text.trim() || !context.trim()) return;
      const active = process.env["RFE_ACTIVE_MODEL"] || st.activeModel || "unknown";
      const provider = resolveProvider();
      const judgeModel = selectJudgeModel(active, provider);
      if (!provider.apiKey) {
        logLine(
          {
            ts: new Date().toISOString(),
            session: sessionID,
            kind: "skip",
            reason: "no api key for judge provider",
            provider: provider.provider,
          },
          logFile,
        );
        return;
      }
      const judge = new Judge({
        baseUrl: provider.baseUrl,
        apiKey: provider.apiKey,
        model: judgeModel,
        session: sessionID,
        cache: cacheOf(judgeModel),
        logFile: logFile ?? "",
      });
      const claims = segmentClaims(text);
      const kept = claims.slice(0, maxClaims());
      if (kept.length < claims.length) {
        logLine(
          {
            ts: new Date().toISOString(),
            session: sessionID,
            kind: "claims-skipped",
            judged: kept.length,
            skipped: claims.length - kept.length,
          },
          logFile,
        );
      }
      const allowed = flagVerdicts();
      const flagged: FlaggedClaim[] = [];
      for (const claim of kept) {
        const sel = selectPremise(context, claim);
        // premise filtering dropped content: prefer the conservative label so
        // truncation can't read as fabrication (issue #86)
        const ctx =
          context.length > DEFAULT_PREMISE_BUDGET
            ? sel + TRUNCATION_NOTE
            : sel;
        const v = await judge.verdict(ctx, claim);
        if (v !== "faithful" && allowed.includes(v)) {
          flagged.push({ claim, verdict: v });
        }
        // faithful: silent pass, no annotation
      }
      if (flagged.length === 0) return;
      const nudge = buildNudge(judgeModel, flagged);
      // opencode message ids must start with "msg" (server schema enforces)
      const nudgeId = `msg_rfe_nudge_${Date.now().toString(36)}`;
      st.nudgeIds.add(nudgeId);
      try {
        await client.session.prompt({
          path: { id: sessionID },
          body: {
            messageID: nudgeId,
            noReply: true,
            parts: [{ type: "text", text: nudge, synthetic: true }],
          },
        });
      } catch (e) {
        // SDK injection failed: toast + structured log fallback
        logErr("nudge-inject", e, logFile);
        logLine(
          {
            ts: new Date().toISOString(),
            session: sessionID,
            kind: "nudge",
            model: judgeModel,
            flagged: flagged.map((f) => ({ claim: f.claim.slice(0, 200), verdict: f.verdict })),
          },
          logFile,
        );
        toast(nudge.slice(0, 500));
      }
    })().catch((e) => logErr("judge-reply", e, logFile));
  };

  return {
    "tool.execute.before": async (input, output) => {
      try {
        // doc-pull process check: package invoked without fetched docs?
        const argsText =
          typeof output.args === "string"
            ? output.args
            : JSON.stringify(output.args ?? "");
        if (!argsText) return;
        const pkgs = extractPackages(argsText);
        if (pkgs.size === 0) return;
        const st = stateOf(input.sessionID);
        const missing = [...pkgs].filter((p) => !st.docTokens.has(p));
        if (missing.length > 0) {
          toast(
            `doc-pull check: ${missing.join(", ")} used without fetched docs`,
          );
        }
      } catch (e) {
        logErr("doc-pull-check", e, logFile);
      }
    },

    "tool.execute.after": async (input, output) => {
      try {
        if (!toolsRe.test(input.tool)) return;
        if (isSensitivePath(input.args)) {
          logLine(
            {
              ts: new Date().toISOString(),
              session: input.sessionID,
              kind: "skip",
              reason: "sensitive path, premise not captured",
              tool: input.tool,
            },
            logFile,
          );
          return;
        }
        const text =
          typeof output.output === "string"
            ? output.output
            : JSON.stringify(output.output ?? "");
        if (!text) return;
        const safe = redactSecrets(text);
        const st = stateOf(input.sessionID);
        st.premises.append(safe);
        for (const t of extractDocTokens(safe)) st.docTokens.add(t);
      } catch (e) {
        logErr("premise-capture", e, logFile);
      }
    },

    "chat.params": async (input) => {
      try {
        const st = stateOf(input.sessionID);
        // provider-qualified id normalizes fine: match is substring-based
        st.activeModel = `${input.model.providerID}/${input.model.id}`;
      } catch (e) {
        logErr("chat-params", e, logFile);
      }
    },

    event: async ({ event }: { event: Event }) => {
      try {
        if (event.type === "session.deleted") {
          states.delete(event.properties.info.id);
          return;
        }
        if (event.type === "message.part.updated") {
          const part = event.properties.part;
          if (part.type === "text" && part.text) {
            const st = stateOf(part.sessionID);
            let msg = st.parts.get(part.messageID);
            if (!msg) {
              msg = new Map();
              st.parts.set(part.messageID, msg);
            }
            msg.set(part.id, part.text);
          }
          return;
        }
        if (event.type === "message.updated") {
          const info = event.properties.info;
          if (info.role !== "assistant") return;
          const a = info as AssistantMessage;
          if (!a.time.completed || a.error) return;
          const st = stateOf(info.sessionID);
          if (st.nudgeIds.has(a.parentID)) return; // loop guard
          if (!st.activeModel && a.modelID) {
            st.activeModel = `${a.providerID}/${a.modelID}`;
          }
          judgeReply(info.sessionID, info.id);
          // part text accumulates unboundedly otherwise
          if (st.parts.size > 64) {
            const first = st.parts.keys().next().value;
            if (first !== undefined) st.parts.delete(first);
          }
        }
      } catch (e) {
        logErr("event", e, logFile);
      }
    },
  };
};

// opencode v1 plugin module shape: path-based plugins must default-export an
// object with id + server() — the legacy path rejects non-function exports.
export default { id: "ragfaith", server: RagfaithPlugin };
