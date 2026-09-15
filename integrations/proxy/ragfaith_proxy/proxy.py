"""ragfaith-proxy: universal OpenAI-compatible sidecar with a live faithfulness cascade.

Point any OpenAI-compatible host tool's base URL at this proxy. Streaming
replies stream through byte-for-byte (first token never delayed) while a copy
is buffered; once the reply completes, claims are judged in parallel against
the premises actually pulled in the conversation (tool results). Flagged
claims trigger one aggregated corrective nudge, delivered per RFE_NUDGE_MODE:

- chain (default): withhold the stream close, issue one internal continuation
  call (original messages + assistant reply + nudge), stream it down the same
  SSE stream, then terminate with a single [DONE].
- next: stash the nudge and inject it into the next request from the same
  conversation, right after the flagged assistant message. Non-streaming
  requests always fall back to this (chain needs a stream to append to).
- regen: buffer the whole reply, judge, then show the client exactly one
  assistant response — the original when faithful, a regenerated one (from
  the internal nudged call) when flagged. The nudge itself never reaches the
  client. Costs first-token latency: the client waits for generation plus
  judging before anything is shown.

Every cascade failure degrades to clean passthrough. Stdlib only.
"""

import argparse
import hashlib
import http.client
import json
import logging
import os
import re
import sys
import threading
import time
import urllib.parse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .decompose import split_claims
from .judge import VERDICTS, Judge

logger = logging.getLogger(__name__)

DEFAULT_UPSTREAM = "https://api.synthetic.new/v1"
DEFAULT_NUDGE = (
    "ragfaith judge ({model}): {n} claim(s) in your previous reply were flagged "
    "{verdict}: {claims}. Re-check against the sources actually pulled in this "
    "conversation and reconcile; do not invent corrections."
)
# regen mode: the nudge never reaches the client, so it can be directive —
# the model must keep the re-check internal and output only a clean answer
DEFAULT_REGEN_NUDGE = (
    "Do not reply to this message, and do not mention it, judges, flags, "
    "verification, or any re-checking process in your output. {n} claim(s) in "
    "your draft reply were flagged as not directly supported by the pulled "
    "sources: {claims}. Silently re-check them against the sources actually "
    "pulled in this conversation. Keep all reconciliation inside your own "
    "reasoning only. Then write your final answer as if directly answering "
    "the user's original query: correct or drop any unsupported claims, keep "
    "the supported ones unchanged, and output ONLY that answer — no tables, "
    "no meta-commentary, no references to this correction."
)
DEFAULT_UNVERIFIABLE_NUDGE = (
    "ragfaith judge ({model}): {n} claim(s) in your previous reply were flagged "
    "unverifiable - the sources actually pulled in this conversation do not "
    "contain them: {claims}. Resolve each flagged claim visibly in your next "
    "reply in exactly one of these ways: (1) back it with further searches or "
    "fetches and cite the newly pulled source; (2) openly disclose that it "
    "comes from your internal (training) knowledge, not the pulled sources; "
    "(3) openly disclose that it was inferred from data inside the context. "
    "No silent assertions."
)
# conservative label under premise filtering (issue #86): support missing only
# because of truncation must not read as fabrication
_TRUNCATION_NOTE = (
    "\n\n[NOTE: this context is a relevance-filtered excerpt of the pulled "
    "sources; if the claim's support is missing only because of that filtering "
    'or truncation, answer "unverifiable", not "unfaithful".]'
)
SSE_DONE = object()

# per-conversation store bounded so long sessions can't grow memory without limit
CONV_MESSAGE_CAP = 128
DEFAULT_PREMISE_CAP = 24000

# per-claim premise budget: relevant passages only (issue #82 - one shared
# 24k+ blob per claim cost 822k prompt tokens and 177s judging for one reply)
PREMISE_BUDGET = 12000
PREMISE_FALLBACK = 4000

# English glue words excluded from premise/claim overlap scoring; content
# words carry the signal (also covers DE/IT reasonably via \w unicode)
_STOPWORDS = frozenset(
    "the a an and or of to in on for with is are was were be been it this that "
    "these those as at by from not but if then than so such into over under "
    "about their its his her they them you your we our can could will would "
    "should may might must do does did done have has had what which who whom "
    "when where why how".split()
)
CONV_STORE_MAX = 1000
JUDGE_STORE_MAX = 100

# concurrent per-claim judge calls inside one cascade evaluation
JUDGE_WORKERS = int(os.environ.get("RFE_JUDGE_WORKERS", "8"))


def _parse_flag_verdicts(raw: str) -> tuple:
    """CSV of verdicts that trigger a nudge. Default: unfaithful only —
    'unverifiable' marks derivation/drift, not fabrication (issue #74)."""
    items = tuple(v.strip() for v in raw.split(",") if v.strip() in VERDICTS)
    return items or ("unfaithful",)


MAX_BODY = int(os.environ.get("RFE_MAX_BODY", str(10 * 1024 * 1024)))


@dataclass
class Config:
    proxy_host: str = "127.0.0.1"
    proxy_port: int = 8787
    upstream_base: str = DEFAULT_UPSTREAM
    judge_provider: str = "synthetic"
    judge_base: str = DEFAULT_UPSTREAM
    glm_model: str = "hf:zai-org/GLM-5.3-Flash"
    deepseek_model: str = "hf:deepseek-ai/DeepSeek-V4.1-Flash"
    # char cap on judge-prompt premises (most recent kept). Static only: the
    # 0.1.7 dynamic half-context-window default regressed badly (issue #82).
    premise_cap: int = DEFAULT_PREMISE_CAP
    nudge_mode: str = "chain"
    nudge_template: str = DEFAULT_NUDGE
    regen_template: str = DEFAULT_REGEN_NUDGE
    flag_verdicts: tuple = ("unfaithful",)
    # RFE_STRICTNESS preset (issue #86): normal = unfaithful only; strict also
    # fires on unverifiable and switches to verdict-aware nudge arms
    strictness: str = "normal"
    unverifiable_template: str = DEFAULT_UNVERIFIABLE_NUDGE
    cache_dir: str | None = None
    judge_log: str | None = None
    host_decorators: dict = field(default_factory=lambda: {"default": {}})

    @classmethod
    def from_env(cls, env=None) -> "Config":
        env = os.environ if env is None else env
        provider = env.get("RFE_JUDGE_PROVIDER", "synthetic")
        if provider == "openrouter":
            judge_base = env.get("RFE_OPENROUTER_BASE", "https://openrouter.ai/api/v1")
            glm = env.get("RFE_OPENROUTER_GLM_MODEL", "z-ai/glm-5.3-flash")
            deepseek = env.get("RFE_OPENROUTER_DEEPSEEK_MODEL", "deepseek/deepseek-v4.1-flash")
        else:
            judge_base = env.get(
                "RFE_SYNTHETIC_BASE", env.get("RFE_UPSTREAM_BASE", DEFAULT_UPSTREAM)
            )
            glm = env.get("RFE_SYNTHETIC_GLM_MODEL", "hf:zai-org/GLM-5.3-Flash")
            deepseek = env.get("RFE_SYNTHETIC_DEEPSEEK_MODEL", "hf:deepseek-ai/DeepSeek-V4.1-Flash")
        # generic overrides take precedence: any OpenAI-compatible judge works
        judge_base = env.get("RFE_JUDGE_BASE", judge_base)
        glm = env.get("RFE_JUDGE_GLM_MODEL", glm)
        deepseek = env.get("RFE_JUDGE_DEEPSEEK_MODEL", deepseek)
        # RFE_STRICTNESS is a preset; an explicit RFE_FLAG_VERDICTS wins (issue #86)
        strictness = env.get("RFE_STRICTNESS", "normal")
        if "RFE_FLAG_VERDICTS" in env:
            flag_verdicts = _parse_flag_verdicts(env["RFE_FLAG_VERDICTS"])
        elif strictness == "strict":
            flag_verdicts = ("unfaithful", "unverifiable")
        else:
            flag_verdicts = ("unfaithful",)
        decorators = {"default": {}}
        if env.get("RFE_HOST_CONFIG"):
            decorators.update(json.loads(Path(env["RFE_HOST_CONFIG"]).read_text()))
        if env.get("RFE_HOST_DECORATORS"):
            decorators.update(json.loads(env["RFE_HOST_DECORATORS"]))
        return cls(
            proxy_host=env.get("RFE_PROXY_HOST", "127.0.0.1"),
            proxy_port=int(env.get("RFE_PROXY_PORT", "8787")),
            upstream_base=env.get("RFE_UPSTREAM_BASE", DEFAULT_UPSTREAM),
            judge_provider=provider,
            judge_base=judge_base,
            glm_model=glm,
            deepseek_model=deepseek,
            premise_cap=int(env.get("RFE_PREMISE_CAP", str(DEFAULT_PREMISE_CAP))),
            nudge_mode=env.get("RFE_NUDGE_MODE", "chain"),
            nudge_template=env.get("RFE_NUDGE_TEMPLATE", DEFAULT_NUDGE),
            regen_template=env.get("RFE_REGEN_NUDGE", DEFAULT_REGEN_NUDGE),
            unverifiable_template=env.get(
                "RFE_NUDGE_TEMPLATE_UNVERIFIABLE", DEFAULT_UNVERIFIABLE_NUDGE
            ),
            flag_verdicts=flag_verdicts,
            strictness=strictness,
            cache_dir=env.get("RFE_CACHE_DIR") or None,
            judge_log=env.get("RFE_JUDGE_LOG") or None,
            host_decorators=decorators,
        )


def _bare_model_id(model: str) -> str:
    """Last path segment without the ':' variant suffix ('hf:zai-org/GLM-5.3-Flash'
    -> 'glm-5.3-flash', 'z-ai/glm-5.3-flash:free' -> 'glm-5.3-flash')."""
    return model.rsplit("/", 1)[-1].split(":", 1)[0].lower()


def select_judge(active_model: str, cfg: Config) -> str:
    """Never self-judge: active model matching the configured GLM judge is judged
    by the configured DeepSeek model; anything else by the configured GLM model."""
    active = _bare_model_id(active_model)
    glm = _bare_model_id(cfg.glm_model)
    if active and (glm == active or glm in active):
        return cfg.deepseek_model
    return cfg.glm_model


def _content_text(content) -> str:
    """Flatten message content (str, or text/tool_result blocks) to text; skip
    non-string values defensively."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        return ""
    out = []
    for block in content:
        if not isinstance(block, dict):
            continue
        inner = block.get("content")
        if isinstance(inner, (str, list)):
            out.append(_content_text(inner))
        elif isinstance(block.get("text"), str):
            out.append(block["text"])
    return " ".join(x for x in out if x)


def harvest_premises(messages: list[dict], cap: int) -> str:
    """Concatenate pulled content (tool messages + tool_result blocks), keep the
    most recent `cap` chars."""
    parts = []
    for m in messages:
        content = m.get("content")
        if m.get("role") == "tool":
            text = _content_text(content)
        elif isinstance(content, list):
            text = " ".join(
                _content_text(b)
                for b in content
                if isinstance(b, dict) and b.get("type") == "tool_result"
            )
        else:
            continue
        if text:
            parts.append(text)
    return "\n\n".join(parts)[-cap:]


def apply_tool_deltas(acc: list[dict], deltas: list[dict]) -> list[dict]:
    """Assemble streamed OpenAI tool_calls deltas into complete tool call objects."""
    for d in deltas:
        idx = d.get("index", 0)
        while len(acc) <= idx:
            acc.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
        slot = acc[idx]
        if d.get("id"):
            slot["id"] = d["id"]
        if d.get("type"):
            slot["type"] = d["type"]
        fn = d.get("function") or {}
        slot["function"]["name"] += fn.get("name") or ""
        slot["function"]["arguments"] += fn.get("arguments") or ""
    return acc


def parse_sse_line(raw: bytes):
    """Return parsed event dict, SSE_DONE sentinel, or None for noise lines."""
    line = raw.strip()
    if not line.startswith(b"data:"):
        return None
    payload = line[len(b"data:") :].strip()
    if payload == b"[DONE]":
        return SSE_DONE
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def consume_delta(event: dict, text_parts: list[str], tool_calls: list[dict]):
    """Apply one streamed chunk to the buffers; return finish_reason if present."""
    if not isinstance(event, dict):
        return None
    choice = (event.get("choices") or [{}])[0]
    delta = choice.get("delta") or {}
    if delta.get("content"):
        text_parts.append(delta["content"])
    if delta.get("tool_calls"):
        apply_tool_deltas(tool_calls, delta["tool_calls"])
    return choice.get("finish_reason")


def render_template(template: str, **kw) -> str:
    try:
        return template.format(**kw)
    except (KeyError, IndexError, ValueError):
        return DEFAULT_NUDGE.format(**kw)


def resolve_host(cfg: Config, headers) -> dict:
    """Host decorator entry: explicit X-RFE-Host, else User-Agent substring."""
    explicit = headers.get("x-rfe-host")
    if explicit and explicit in cfg.host_decorators:
        return cfg.host_decorators[explicit]
    ua = (headers.get("user-agent") or "").lower()
    for name, entry in cfg.host_decorators.items():
        if name != "default" and name.lower() in ua:
            return entry
    return cfg.host_decorators.get("default", {})


def inject_nudge(messages: list[dict], nudge: str) -> list[dict]:
    """Insert the nudge right after the most recent assistant message."""
    msgs = list(messages)
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "assistant":
            msgs.insert(i + 1, {"role": "user", "content": nudge})
            return msgs
    msgs.append({"role": "user", "content": nudge})
    return msgs


def upstream_request(
    cfg: Config, method: str, suffix: str, body: bytes | None = None, auth: str = ""
):
    parts = urllib.parse.urlsplit(cfg.upstream_base.rstrip("/") + "/")
    if parts.scheme == "https":
        conn_cls = http.client.HTTPSConnection
    else:
        conn_cls = http.client.HTTPConnection
    conn = conn_cls(parts.hostname, parts.port, timeout=120)
    headers = {
        # client-auth-only: the proxy holds no key of its own
        "Authorization": auth,
        "Accept-Encoding": "identity",  # body must stay parseable; never gzip upstream
        "Connection": "close",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    conn.request(method, parts.path.rstrip("/") + suffix, body=body, headers=headers)
    return conn.getresponse(), conn


def select_premise(
    premises: str, claim: str, budget: int = PREMISE_BUDGET, fallback: int = PREMISE_FALLBACK
) -> str:
    """Most relevant premise passages for one claim, within `budget` chars
    (lexical overlap, no embeddings). Oversized relevant passages are
    head-truncated to the remaining room; with no overlap at all, the most
    recent `fallback` chars are sent so the judge can answer unverifiable."""
    if len(premises) <= budget:
        return premises
    claim_toks = {w for w in re.findall(r"\w{4,}", claim.lower()) if w not in _STOPWORDS}
    if not claim_toks:
        return premises[-fallback:]
    scored = []
    for i, p in enumerate(premises.split("\n\n")):
        toks = {w for w in re.findall(r"\w{4,}", p.lower()) if w not in _STOPWORDS}
        scored.append((len(claim_toks & toks), i, p))
    scored.sort(key=lambda t: (-t[0], t[1]))
    picked: list[tuple[int, str]] = []
    used = 0
    for score, i, p in scored:
        if score <= 0 or used >= budget:
            continue
        room = budget - used
        chunk = p if len(p) <= room else p[:room]
        picked.append((i, chunk))
        used += len(chunk) + 2
    if not picked:
        return premises[-fallback:]
    picked.sort(key=lambda t: t[0])
    return "\n\n".join(chunk for _, chunk in picked)


class Cascade:
    """Thread-safe cascade state: judges, verdict cache wiring, conversation store."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._judges: OrderedDict[str, Judge] = OrderedDict()
        self._conv: OrderedDict[str, dict] = OrderedDict()
        self._end_count = 0
        self.events: list[dict] = []
        self._log_fh = open(cfg.judge_log, "a", encoding="utf-8") if cfg.judge_log else None

    def _state(self, conv: str) -> dict:
        """Get/create per-conversation state, evicting the LRU entry past the
        cap. Caller holds self._lock."""
        st = self._conv.get(conv)
        if st is None:
            st = {"messages": [], "nudge": None}
            self._conv[conv] = st
            if len(self._conv) > CONV_STORE_MAX:
                self._conv.popitem(last=False)
        else:
            self._conv.move_to_end(conv)
        return st

    def close(self) -> None:
        with self._lock:
            if self._log_fh:
                self._log_fh.close()
                self._log_fh = None

    def log(self, record: dict) -> None:
        line = json.dumps({"ts": round(time.time(), 3), **record}, ensure_ascii=False)
        if self._log_fh:
            with self._lock:
                self._log_fh.write(line + "\n")
                self._log_fh.flush()
        else:
            print(line, file=sys.stderr, flush=True)

    def note_end(self, reason: str, conv: str) -> None:
        with self._lock:
            self._end_count += 1
            self.events.append({"kind": "stream-end", "reason": reason, "conversation": conv})

    @property
    def end_count(self) -> int:
        with self._lock:
            return self._end_count

    def conv_key(self, headers, messages: list[dict]) -> str:
        cid = headers.get("x-conversation-id")
        if cid:
            return cid
        for m in messages:  # first user message: stable across turns, unique per chat
            if isinstance(m, dict) and m.get("role") == "user":
                seed = json.dumps(m, sort_keys=True, ensure_ascii=False)
                return hashlib.sha256(seed.encode()).hexdigest()[:16]
        if messages:
            seed = json.dumps(messages, sort_keys=True, ensure_ascii=False)
            return hashlib.sha256(seed.encode()).hexdigest()[:16]
        return headers.get("host", "anon")

    def record_messages(self, conv: str, messages: list[dict]) -> None:
        with self._lock:
            st = self._state(conv)
            stored = st["messages"]
            if len(messages) >= len(stored) and messages[: len(stored)] == stored:
                merged = list(messages)  # incoming extends the known history
            else:
                overlap = 0  # incremental client: splice onto the stored tail
                for k in range(min(len(stored), len(messages)), 0, -1):
                    if stored[-k:] == messages[:k]:
                        overlap = k
                        break
                merged = stored + list(messages)[overlap:]
            st["messages"] = merged[-CONV_MESSAGE_CAP:]

    def record_reply(self, conv: str, message: dict) -> None:
        with self._lock:
            st = self._state(conv)
            st["messages"] = (st["messages"] + [dict(message)])[-CONV_MESSAGE_CAP:]

    def premises(self, conv: str) -> str:
        with self._lock:
            st = self._conv.get(conv)
            if st is not None:
                self._conv.move_to_end(conv)
            messages = list(st["messages"]) if st is not None else []
        return harvest_premises(messages, self.cfg.premise_cap)

    def set_nudge(self, conv: str, nudge: str | None) -> None:
        with self._lock:
            self._state(conv)["nudge"] = nudge
        if nudge is not None:
            # audit trail: the nudge text carries the flagged claims verbatim
            self.log({"kind": "nudge-stash", "conversation": conv, "nudge": nudge[:500]})

    def pop_nudge(self, conv: str) -> str | None:
        with self._lock:
            st = self._state(conv)
            nudge = st["nudge"]
            st["nudge"] = None
        if nudge is not None:
            self.log({"kind": "nudge-delivered", "conversation": conv, "nudge": nudge[:500]})
        return nudge

    def judge(self, model: str, token: str) -> Judge:
        # judges are keyed by (model, client token digest): verdicts are the same
        # for any caller, but each caller's judge authenticates with its own key
        cache_key = f"{model}\x00{hashlib.sha256(token.encode()).hexdigest()}"
        with self._lock:
            j = self._judges.get(cache_key)
            if j is None:
                cache_path = None
                if self.cfg.cache_dir:
                    sane = re.sub(r"[^A-Za-z0-9]+", "-", model).strip("-")
                    cache_path = Path(self.cfg.cache_dir) / f"proxy-cache-{sane}.jsonl"
                j = Judge(
                    model,
                    self.cfg.judge_base,
                    token,
                    cache_path=cache_path,
                    log=self.log,
                )
                self._judges[cache_key] = j
                if len(self._judges) > JUDGE_STORE_MAX:
                    self._judges.popitem(last=False)
            else:
                self._judges.move_to_end(cache_key)
        return j

    def evaluate(
        self,
        conv: str,
        judge_model: str,
        text: str,
        token: str = "",
        flag_verdicts: tuple | None = None,
    ) -> list[tuple[str, str]]:
        """Return [(claim, verdict)] for claims whose verdict is in the flag
        set (default: unfaithful only). Fail-open: judge errors are logged and
        the claim is skipped, never blocks the stream."""
        claims = [t for *_, t in split_claims(text)] if text else []
        if not claims:
            return []
        premises = self.premises(conv)
        if not premises:
            return []
        judge = self.judge(judge_model, token)
        flag_set = flag_verdicts if flag_verdicts is not None else self.cfg.flag_verdicts

        def _verdict(claim: str):
            try:
                context = select_premise(premises, claim)
                if len(premises) > PREMISE_BUDGET:
                    # premise filtering dropped content: prefer the conservative
                    # label so truncation can't read as fabrication (issue #86)
                    context += _TRUNCATION_NOTE
                return claim, judge.verdict(context, claim, conversation=conv)
            except Exception as e:  # noqa: BLE001 - cascade must never break passthrough
                self.log({"kind": "judge-error", "error": str(e), "conversation": conv})
                return claim, None

        # claims are independent: judge them concurrently (one judge round-trip
        # instead of one per claim); pool.map keeps result order stable
        with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as pool:
            results = list(pool.map(_verdict, claims))
        skipped = sum(1 for _, v in results if v is None)
        if skipped:
            self.log(
                {"kind": "judge-skipped", "n": skipped, "total": len(results), "conversation": conv}
            )
        return [(c, v) for c, v in results if v in flag_set]


def _norm(path: str) -> str:
    return path.split("?", 1)[0].rstrip("/")


def _bearer_token(auth: str) -> str:
    """Raw token from an Authorization header value (judge sends its own Bearer prefix)."""
    return auth.split(None, 1)[1] if auth.lower().startswith("bearer ") else auth


def make_handler(cascade: Cascade):
    cfg = cascade.cfg

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "ragfaith-proxy"
        timeout = 60  # a stalled client cannot pin a handler thread forever

        def log_message(self, fmt, *args):
            logger.debug(fmt, *args)

        def end_headers(self):
            self._response_started = True
            super().end_headers()

        def do_GET(self):
            if _norm(self.path) != "/v1/models":
                self.send_error(404)
                return
            status = None
            raw = b""
            ctype = "application/json"
            try:
                resp, conn = upstream_request(
                    cfg, "GET", "/v1/models", auth=self.headers.get("Authorization", "")
                )
                status = resp.status
                raw = resp.read()
                ctype = resp.getheader("Content-Type") or "application/json"
                conn.close()
            except OSError as e:
                cascade.log({"kind": "upstream-error", "error": str(e)})
            if status in (404, 405, None):
                # upstream has no model listing (or is unreachable): serve a
                # minimal static list so clients that require enumeration
                # before allowing manual model IDs can still onboard
                fallback = {
                    "object": "list",
                    "data": [
                        {"id": cfg.glm_model, "object": "model", "owned_by": "ragfaith-proxy"},
                        {
                            "id": cfg.deepseek_model,
                            "object": "model",
                            "owned_by": "ragfaith-proxy",
                        },
                    ],
                }
                status, raw, ctype = 200, json.dumps(fallback).encode(), "application/json"
            elif status != 200:
                cascade.log(
                    {
                        "kind": "passthrough-status",
                        "status": status,
                        "body": _excerpt(raw),
                    }
                )
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            try:
                self.wfile.write(raw)
            except OSError:
                pass

        def do_POST(self):
            self._response_started = False
            if _norm(self.path) != "/v1/chat/completions":
                self.send_error(404)
                return
            try:
                self._handle_chat()
            except Exception:  # noqa: BLE001 - proxy must answer, never wedge
                logger.exception("handler error")
                cascade.log({"kind": "handler-error"})
                if not self._response_started:
                    try:
                        self.send_error(500, "internal proxy error")
                    except OSError:
                        pass
                self.close_connection = True

        # -- chat completions -------------------------------------------------

        def _reject(self, status: int, reason: str, message: str, messages=None):
            """Client-visible validation reject: log then answer (issue #85 —
            a 400 the client sees must be diagnosable from the log)."""
            # messages is unvalidated client input; only use it when it's the
            # shape conv_key expects
            msgs = messages if isinstance(messages, list) else []
            cascade.log(
                {
                    "kind": "request-rejected",
                    "reason": reason,
                    "conversation": cascade.conv_key(self.headers, msgs),
                }
            )
            self.send_error(status, message)

        def _handle_chat(self):
            length_raw = self.headers.get("Content-Length", "0") or "0"
            try:
                length = int(length_raw)
            except (TypeError, ValueError):
                self.close_connection = True
                self._reject(400, "bad-content-length", "invalid Content-Length")
                return
            if length < 0:
                self.close_connection = True
                self._reject(400, "bad-content-length", "invalid Content-Length")
                return
            if length > MAX_BODY:
                self.close_connection = True
                self._reject(413, "payload-too-large", "request body too large")
                return
            body = self.rfile.read(length)
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                self._reject(400, "invalid-json", "invalid JSON body")
                return
            if not isinstance(payload, dict):
                self._reject(400, "invalid-json", "invalid JSON body")
                return
            messages = payload.get("messages")
            if not isinstance(messages, list):
                self._reject(400, "bad-messages", "messages must be a list", messages)
                return
            model = payload.get("model")
            if not isinstance(model, str):
                self._reject(400, "bad-model", "model must be a string", messages)
                return
            auth = self.headers.get("Authorization", "")
            if not auth:
                # client-auth-only: proxy cannot reach upstream or judge without it
                self._reject(401, "missing-auth", "Authorization header required", messages)
                return
            conv = cascade.conv_key(self.headers, messages)
            deco = resolve_host(cfg, self.headers)
            mode = deco.get("nudge_mode", cfg.nudge_mode)
            template = deco.get("template", cfg.nudge_template)
            regen_template = deco.get("regen_template", cfg.regen_template)
            raw_flags = deco.get("flag_verdicts", cfg.flag_verdicts)
            flag_set = (
                _parse_flag_verdicts(raw_flags) if isinstance(raw_flags, str) else tuple(raw_flags)
            )
            cascade.record_messages(conv, list(messages))

            pending = cascade.pop_nudge(conv)  # next-mode delivery
            if pending is not None:
                payload["messages"] = messages = inject_nudge(messages, pending)

            judge_model = select_judge(model, cfg)
            upstream_body = json.dumps(payload).encode()
            if pending is not None:
                # the injected payload is the likeliest 400 source; log its size
                cascade.log(
                    {"kind": "nudge-inject", "conversation": conv, "bytes": len(upstream_body)}
                )
            try:
                resp, conn = upstream_request(
                    cfg, "POST", "/chat/completions", upstream_body, auth=auth
                )
            except OSError as e:
                if pending is not None:  # never lose a stashed nudge on upstream failure
                    cascade.set_nudge(conv, pending)
                cascade.log({"kind": "upstream-error", "error": str(e)})
                self.send_error(502, "upstream connect failed")
                return
            if resp.status != 200:
                if pending is not None:
                    cascade.log(
                        {
                            "kind": "nudge-request-failed",
                            "status": resp.status,
                            "conversation": conv,
                        }
                    )
                    cascade.set_nudge(conv, pending)
                raw = resp.read()
                conn.close()
                cascade.log(
                    {
                        "kind": "passthrough-status",
                        "status": resp.status,
                        "conversation": conv,
                        "body": _excerpt(raw),
                    }
                )
                self.send_response(resp.status)
                ctype = resp.getheader("Content-Type") or "application/json"
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                try:
                    self.wfile.write(raw)
                except OSError:
                    pass
                return
            if mode == "regen":
                self._regen(
                    resp, conn, conv, payload, messages, judge_model, regen_template, auth, flag_set
                )
            elif payload.get("stream"):
                self._stream(
                    resp, conn, conv, payload, messages, judge_model, mode, template, auth, flag_set
                )
            else:
                self._plain(resp, conn, conv, judge_model, mode, auth, flag_set)

        def _plain(self, resp, conn, conv, judge_model, mode, auth, flag_set):
            raw = resp.read()
            conn.close()
            self.send_response(200)
            self.send_header("Content-Type", resp.getheader("Content-Type") or "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            try:
                self.wfile.write(raw)
            except OSError:
                cascade.note_end("disconnected", conv)
                return
            reason = "done"
            try:  # cascade runs after the passthrough; failures only log
                data = json.loads(raw)
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                text = message.get("content") or ""
                finish = choice.get("finish_reason")
                if text and finish == "stop":
                    cascade.record_reply(conv, message)
                    flagged = cascade.evaluate(
                        conv, judge_model, text, _bearer_token(auth), flag_set
                    )
                    if flagged:
                        if mode == "chain":
                            cascade.log(
                                {
                                    "kind": "chain-fallback",
                                    "reason": "non-streaming",
                                    "conversation": conv,
                                }
                            )
                        cascade.set_nudge(
                            conv,
                            build_nudge(
                                self._template,
                                self._uv_template,
                                self._strict,
                                judge_model,
                                flagged,
                            ),
                        )
                        reason = "stashed"
                    else:
                        reason = "faithful"
            except Exception as e:  # noqa: BLE001
                cascade.log({"kind": "cascade-error", "error": str(e), "conversation": conv})
            cascade.note_end(reason, conv)

        @property
        def _template(self):
            return resolve_host(cfg, self.headers).get("template", cfg.nudge_template)

        @property
        def _uv_template(self):
            return resolve_host(cfg, self.headers).get(
                "unverifiable_template", cfg.unverifiable_template
            )

        @property
        def _strict(self):
            return cfg.strictness == "strict"

        def _stream(
            self, resp, conn, conv, payload, messages, judge_model, mode, template, auth, flag_set
        ):
            # send response head before reading any upstream body: first token
            # latency must never wait on the cascade
            self.send_response(200)
            self.send_header("Content-Type", resp.getheader("Content-Type") or "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            text_parts: list[str] = []
            tool_calls: list[dict] = []
            finish = None
            held_done = False
            dead = False
            buf = b""
            # only chain mode needs [DONE] withheld (it may append the
            # self-correction to the same stream); every other mode flushes
            # [DONE] immediately so the client's spinner stops on time, and
            # the cascade runs in the background (issue #72)
            hold_done = mode == "chain"
            try:
                while True:
                    block = resp.read1(8192)
                    if not block:
                        break
                    buf += block
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        event = parse_sse_line(line)
                        if event is SSE_DONE:  # withheld until the cascade ruled
                            held_done = True
                            continue
                        self.wfile.write(line + b"\n")
                        self.wfile.flush()
                        if isinstance(event, dict):
                            finish = consume_delta(event, text_parts, tool_calls) or finish
            except OSError:  # client (or upstream) went away mid-stream
                dead = True
            finally:
                conn.close()

            if not dead and buf:
                try:
                    if parse_sse_line(buf) is SSE_DONE:
                        held_done = True
                    else:
                        self.wfile.write(buf)
                        self.wfile.flush()
                except OSError:
                    dead = True

            if not hold_done and not dead:
                # close the stream now; judge + stash in a background thread
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except OSError:
                    dead = True
                if not dead:
                    self._cascade_bg(
                        conv,
                        payload,
                        messages,
                        judge_model,
                        template,
                        auth,
                        flag_set,
                        text_parts,
                        tool_calls,
                        finish,
                    )
                    return

            reason = "disconnected" if dead else "done"
            try:
                if not dead and (text_parts or tool_calls):
                    reply = {"role": "assistant", "content": "".join(text_parts) or None}
                    if tool_calls:
                        reply["tool_calls"] = tool_calls
                    cascade.record_reply(conv, reply)
                assistant_text = "".join(text_parts)
                if not dead and assistant_text and finish == "stop":
                    flagged = cascade.evaluate(
                        conv, judge_model, assistant_text, _bearer_token(auth), flag_set
                    )
                    if flagged:
                        nudge = build_nudge(
                            template, self._uv_template, self._strict, judge_model, flagged
                        )
                        if mode == "chain":
                            try:
                                chained_ok = self._chain(
                                    conv, payload, messages, assistant_text, nudge, auth
                                )
                                if chained_ok:
                                    reason = "chained"
                                else:  # chain failed: keep the nudge for next-mode delivery
                                    cascade.set_nudge(conv, nudge)
                                    reason = "stashed"
                            except OSError:
                                dead = True
                                reason = "disconnected"
                        else:
                            cascade.set_nudge(conv, nudge)
                            reason = "stashed"
                    else:
                        reason = "faithful"
            except Exception as e:  # noqa: BLE001 - cascade failures never corrupt the stream
                cascade.log({"kind": "cascade-error", "error": str(e), "conversation": conv})

            if not dead and (held_done or reason == "chained"):
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except OSError:
                    pass
            cascade.note_end(reason, conv)

        def _cascade_bg(
            self,
            conv,
            payload,
            messages,
            judge_model,
            template,
            auth,
            flag_set,
            text_parts,
            tool_calls,
            finish,
        ):
            """Background cascade for stream-through modes (chain not held):
            record the reply, judge, stash the nudge. Errors only log — the
            client stream is already closed."""

            def _run():
                reason = "done"
                try:
                    if text_parts or tool_calls:
                        reply = {"role": "assistant", "content": "".join(text_parts) or None}
                        if tool_calls:
                            reply["tool_calls"] = tool_calls
                        cascade.record_reply(conv, reply)
                    assistant_text = "".join(text_parts)
                    if assistant_text and finish == "stop":
                        flagged = cascade.evaluate(
                            conv, judge_model, assistant_text, _bearer_token(auth), flag_set
                        )
                        if flagged:
                            nudge = build_nudge(
                                template, self._uv_template, self._strict, judge_model, flagged
                            )
                            cascade.set_nudge(conv, nudge)
                            reason = "stashed"
                        else:
                            reason = "faithful"
                except Exception as e:  # noqa: BLE001
                    cascade.log({"kind": "cascade-error", "error": str(e), "conversation": conv})
                cascade.note_end(reason, conv)

            threading.Thread(target=_run, daemon=True, name="ragfaith-cascade").start()

        def _regen(
            self, resp, conn, conv, payload, messages, judge_model, template, auth, flag_set
        ):
            """regen mode: buffer the whole upstream reply, judge in parallel,
            then show the client exactly one assistant response — the original
            when faithful, the regenerated one when flagged. The nudge itself
            never reaches the client. Fail-open: any regen failure replays the
            original reply."""
            client_stream = bool(payload.get("stream"))
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            finish = None
            sse_lines: list[bytes] = []
            plain_raw = b""
            dead = False
            try:
                if client_stream:
                    buf = b""
                    while True:
                        block = resp.read1(8192)
                        if not block:
                            break
                        buf += block
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            event = parse_sse_line(line)
                            if event is SSE_DONE:
                                continue
                            sse_lines.append(line + b"\n")
                            if isinstance(event, dict):
                                finish = consume_delta(event, text_parts, tool_calls) or finish
                    if buf and parse_sse_line(buf) is not SSE_DONE:
                        sse_lines.append(buf)
                else:
                    plain_raw = resp.read()
            except OSError:
                dead = True
            finally:
                conn.close()

            if dead:
                cascade.note_end("disconnected", conv)
                return

            original = {"role": "assistant", "content": "".join(text_parts) or None}
            if not client_stream:
                try:
                    data = json.loads(plain_raw)
                    choice = (data.get("choices") or [{}])[0]
                    original = dict(choice.get("message") or original)
                    finish = choice.get("finish_reason")
                except (json.JSONDecodeError, IndexError):
                    pass  # malformed upstream payload: passthrough untouched
            if tool_calls and not original.get("tool_calls"):
                original["tool_calls"] = tool_calls

            assistant_text = original.get("content") or ""
            tool_turn = bool(original.get("tool_calls")) and not assistant_text.strip()

            if tool_turn or finish != "stop" or not assistant_text.strip():
                # nothing judgeable: replay the original untouched
                try:
                    if client_stream:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        for line in sse_lines:
                            self.wfile.write(line)
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    else:
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(plain_raw)))
                        self.end_headers()
                        self.wfile.write(plain_raw)
                        self.wfile.flush()
                    cascade.record_reply(conv, original)
                    cascade.note_end("passthrough", conv)
                except OSError:
                    cascade.note_end("disconnected", conv)
                return

            reason = "faithful"
            seen = original
            regen_raw = b""
            try:
                flagged = cascade.evaluate(
                    conv, judge_model, assistant_text, _bearer_token(auth), flag_set
                )
                if flagged:
                    nudge = build_nudge(
                        template, self._uv_template, self._strict, judge_model, flagged
                    )
                    regen_payload = dict(
                        payload,
                        messages=[
                            *messages,
                            {"role": "assistant", "content": assistant_text},
                            {"role": "user", "content": nudge},
                        ],
                        stream=False,
                    )
                    regen_raw = b""
                    try:
                        r2, c2 = upstream_request(
                            cfg,
                            "POST",
                            "/chat/completions",
                            json.dumps(regen_payload).encode(),
                            auth=auth,
                        )
                        status2 = r2.status
                        regen_raw = r2.read() if status2 == 200 else b""
                        c2.close()
                        if status2 == 200:
                            data2 = json.loads(regen_raw)
                            msg2 = (data2.get("choices") or [{}])[0].get("message") or {}
                            if (msg2.get("content") or "").strip():
                                seen, reason = msg2, "regenerated"
                            else:
                                cascade.log(
                                    {"kind": "regen-error", "error": "empty", "conversation": conv}
                                )
                        else:
                            cascade.log(
                                {"kind": "regen-error", "status": status2, "conversation": conv}
                            )
                    except (OSError, json.JSONDecodeError) as e:
                        cascade.log({"kind": "regen-error", "error": str(e), "conversation": conv})

                # deliver exactly one assistant response
                if reason == "regenerated":
                    if client_stream:
                        rid = f"ragfaith-regen-{int(time.time() * 1000)}"
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        first = {
                            "id": rid,
                            "object": "chat.completion.chunk",
                            "model": payload.get("model", ""),
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": seen.get("content")},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        last = {
                            "id": rid,
                            "object": "chat.completion.chunk",
                            "model": payload.get("model", ""),
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }
                        self.wfile.write(b"data: " + json.dumps(first).encode() + b"\n\n")
                        self.wfile.write(b"data: " + json.dumps(last).encode() + b"\n\n")
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    else:
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(regen_raw)))
                        self.end_headers()
                        self.wfile.write(regen_raw)
                        self.wfile.flush()
                else:
                    if client_stream:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        for line in sse_lines:
                            self.wfile.write(line)
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    else:
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(plain_raw)))
                        self.end_headers()
                        self.wfile.write(plain_raw)
                        self.wfile.flush()
            except OSError:
                cascade.note_end("disconnected", conv)
                return
            except Exception as e:  # noqa: BLE001 - cascade failures never break delivery
                cascade.log({"kind": "cascade-error", "error": str(e), "conversation": conv})
                reason = "cascade-error"
            cascade.record_reply(conv, seen)
            cascade.note_end(reason, conv)

        def _chain(self, conv, payload, messages, assistant_text, nudge, auth):
            """One internal continuation call with the aggregated nudge, streamed
            onto the same SSE stream (its own [DONE] swallowed). Returns False on
            upstream failure so the caller can re-stash the nudge."""
            chained = dict(
                payload,
                messages=[
                    *messages,
                    {"role": "assistant", "content": assistant_text},
                    {"role": "user", "content": nudge},
                ],
                stream=True,
            )
            chained_body = json.dumps(chained).encode()
            try:
                resp, conn = upstream_request(
                    cfg, "POST", "/chat/completions", chained_body, auth=auth
                )
            except OSError:
                cascade.log(
                    {"kind": "chain-error", "error": "connect-failed", "conversation": conv}
                )
                return False
            if resp.status != 200:
                cascade.log({"kind": "chain-error", "status": resp.status, "conversation": conv})
                resp.read()
                conn.close()
                return False
            buf = b""
            try:
                while True:
                    block = resp.read1(8192)
                    if not block:
                        break
                    buf += block
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if parse_sse_line(line) is SSE_DONE:
                            continue
                        self.wfile.write(line + b"\n")
                        self.wfile.flush()
                if buf and parse_sse_line(buf) is not SSE_DONE:
                    self.wfile.write(buf)
                    self.wfile.flush()
            finally:
                conn.close()
            return True

    return Handler


def _nudge_args(judge_model: str, flagged: list[tuple[str, str]]) -> dict:
    return {
        "model": judge_model,
        "n": len(flagged),
        "verdict": "/".join(sorted({v for _, v in flagged})),
        "claims": "; ".join(c for c, _ in flagged),
    }


def build_nudge(template, uv_template, strict: bool, judge_model: str, flagged) -> str:
    """Strict mode renders per-verdict arms (issue #86): the unfaithful arm
    keeps the reconciliation text, the unverifiable arm demands provenance.
    Normal keeps the single aggregated template."""
    if not strict:
        return render_template(template, **_nudge_args(judge_model, flagged))
    parts = []
    for tmpl, verdict in ((template, "unfaithful"), (uv_template, "unverifiable")):
        arm = [(c, v) for c, v in flagged if v == verdict]
        if arm:
            parts.append(render_template(tmpl, **_nudge_args(judge_model, arm)))
    return "\n\n".join(parts)


# body excerpts in logs must never carry secrets (same core patterns as the
# opencode plugin's redactSecrets)
_SECRET_RES = (
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"), "sk-[REDACTED]"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE), "Bearer [REDACTED]"),
    (
        re.compile(
            r"\b(api[_-]?key|token|secret|password)\s*[:=]\s*[\"'][^\"'\n]{8,}[\"']",
            re.IGNORECASE,
        ),
        r"\1=[REDACTED]",
    ),
)


def _excerpt(raw: bytes, cap: int = 400) -> str:
    """Redacted, truncated body excerpt for log lines."""
    text = raw.decode("utf-8", "replace")
    for pattern, repl in _SECRET_RES:
        text = pattern.sub(repl, text)
    return text[:cap]


def make_server(cfg: Config, cascade: Cascade | None = None):
    cascade = cascade or Cascade(cfg)
    server = ThreadingHTTPServer((cfg.proxy_host, cfg.proxy_port), make_handler(cascade))
    server.daemon_threads = True
    return server, cascade


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="ragfaith-proxy", description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=None, help="override RFE_PROXY_HOST")
    parser.add_argument("--port", type=int, default=None, help="override RFE_PROXY_PORT")
    args = parser.parse_args()
    cfg = Config.from_env()
    if args.host:
        cfg.proxy_host = args.host
    if args.port:
        cfg.proxy_port = args.port
    server, cascade = make_server(cfg)
    logger.info(
        "ragfaith-proxy on http://%s:%d/v1 -> %s (judges: %s / %s via %s, nudge=%s)",
        cfg.proxy_host,
        server.server_address[1],
        cfg.upstream_base,
        cfg.glm_model,
        cfg.deepseek_model,
        cfg.judge_provider,
        cfg.nudge_mode,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        cascade.close()


if __name__ == "__main__":
    main()
