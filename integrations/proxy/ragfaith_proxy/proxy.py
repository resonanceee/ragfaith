"""ragfaith-proxy: universal OpenAI-compatible sidecar with a live faithfulness cascade.

Point any OpenAI-compatible host tool's base URL at this proxy. Streaming
replies stream through byte-for-byte (first token never delayed) while a copy
is buffered; once the reply completes, claims are judged against the premises
actually pulled in the conversation (tool results). Flagged claims trigger one
aggregated corrective nudge, delivered per RFE_NUDGE_MODE:

- chain (default): withhold the stream close, issue one internal continuation
  call (original messages + assistant reply + nudge), stream it down the same
  SSE stream, then terminate with a single [DONE].
- next: stash the nudge and inject it into the next request from the same
  conversation, right after the flagged assistant message. Non-streaming
  requests always fall back to this (chain needs a stream to append to).

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
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .decompose import split_claims
from .judge import Judge

logger = logging.getLogger(__name__)

DEFAULT_UPSTREAM = "https://api.synthetic.new/v1"
DEFAULT_NUDGE = (
    "ragfaith judge ({model}): {n} claim(s) in your previous reply were flagged "
    "{verdict}: {claims}. Re-check against the sources actually pulled in this "
    "conversation and reconcile; do not invent corrections."
)
GLM_FLASH_MARK = "glm-5.3-flash"
SSE_DONE = object()

# per-conversation store bounded so long sessions can't grow memory without limit
CONV_MESSAGE_CAP = 128
CONV_STORE_MAX = 1000
JUDGE_STORE_MAX = 100
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
    premise_cap: int = 24000
    nudge_mode: str = "chain"
    nudge_template: str = DEFAULT_NUDGE
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
            premise_cap=int(env.get("RFE_PREMISE_CAP", "24000")),
            nudge_mode=env.get("RFE_NUDGE_MODE", "chain"),
            cache_dir=env.get("RFE_CACHE_DIR") or None,
            judge_log=env.get("RFE_JUDGE_LOG") or None,
            host_decorators=decorators,
        )


def select_judge(active_model: str, cfg: Config) -> str:
    """Never self-judge: GLM-5.3-Flash replies are judged by DeepSeek-V4.1-Flash."""
    if GLM_FLASH_MARK in active_model.lower():
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

    def set_nudge(self, conv: str, nudge: str) -> None:
        with self._lock:
            self._state(conv)["nudge"] = nudge

    def pop_nudge(self, conv: str) -> str | None:
        with self._lock:
            st = self._state(conv)
            nudge = st["nudge"]
            st["nudge"] = None
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
        self, conv: str, judge_model: str, text: str, token: str = ""
    ) -> list[tuple[str, str]]:
        """Return [(claim, verdict)] for claims not judged faithful. Fail-open:
        judge errors are logged and the claim is skipped, never blocks the stream."""
        claims = [t for *_, t in split_claims(text)] if text else []
        if not claims:
            return []
        premises = self.premises(conv)
        if not premises:
            return []
        judge = self.judge(judge_model, token)
        flagged = []
        for claim in claims:
            try:
                verdict = judge.verdict(premises, claim, conversation=conv)
            except Exception as e:  # noqa: BLE001 - cascade must never break passthrough
                self.log({"kind": "judge-error", "error": str(e), "conversation": conv})
                continue
            if verdict != "faithful":
                flagged.append((claim, verdict))
        return flagged


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
            try:
                resp, conn = upstream_request(
                    cfg, "GET", "/v1/models", auth=self.headers.get("Authorization", "")
                )
                raw = resp.read()
                conn.close()
            except OSError as e:
                cascade.log({"kind": "upstream-error", "error": str(e)})
                self.send_error(502, "upstream connect failed")
                return
            self.send_response(resp.status)
            self.send_header("Content-Type", resp.getheader("Content-Type") or "application/json")
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

        def _handle_chat(self):
            length_raw = self.headers.get("Content-Length", "0") or "0"
            try:
                length = int(length_raw)
            except (TypeError, ValueError):
                self.close_connection = True
                self.send_error(400, "invalid Content-Length")
                return
            if length < 0:
                self.close_connection = True
                self.send_error(400, "invalid Content-Length")
                return
            if length > MAX_BODY:
                self.close_connection = True
                self.send_error(413, "request body too large")
                return
            body = self.rfile.read(length)
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                self.send_error(400, "invalid JSON body")
                return
            if not isinstance(payload, dict):
                self.send_error(400, "invalid JSON body")
                return
            messages = payload.get("messages")
            if not isinstance(messages, list):
                self.send_error(400, "messages must be a list")
                return
            model = payload.get("model")
            if not isinstance(model, str):
                self.send_error(400, "model must be a string")
                return
            auth = self.headers.get("Authorization", "")
            if not auth:
                # client-auth-only: proxy cannot reach upstream or judge without it
                self.send_error(401, "Authorization header required")
                return
            conv = cascade.conv_key(self.headers, messages)
            deco = resolve_host(cfg, self.headers)
            mode = deco.get("nudge_mode", cfg.nudge_mode)
            template = deco.get("template", cfg.nudge_template)
            cascade.record_messages(conv, list(messages))

            pending = cascade.pop_nudge(conv)  # next-mode delivery
            if pending is not None:
                payload["messages"] = messages = inject_nudge(messages, pending)

            judge_model = select_judge(model, cfg)
            upstream_body = json.dumps(payload).encode()
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
                    cascade.set_nudge(conv, pending)
                raw = resp.read()
                conn.close()
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
            if payload.get("stream"):
                self._stream(resp, conn, conv, payload, messages, judge_model, mode, template, auth)
            else:
                self._plain(resp, conn, conv, judge_model, mode, auth)

        def _plain(self, resp, conn, conv, judge_model, mode, auth):
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
                    flagged = cascade.evaluate(conv, judge_model, text, _bearer_token(auth))
                    if flagged:
                        if mode == "chain":
                            cascade.log(
                                {
                                    "kind": "chain-fallback",
                                    "reason": "non-streaming",
                                    "conversation": conv,
                                }
                            )
                        deco_args = _nudge_args(judge_model, flagged)
                        cascade.set_nudge(conv, render_template(self._template, **deco_args))
                        reason = "stashed"
                    else:
                        reason = "faithful"
            except Exception as e:  # noqa: BLE001
                cascade.log({"kind": "cascade-error", "error": str(e), "conversation": conv})
            cascade.note_end(reason, conv)

        @property
        def _template(self):
            return resolve_host(cfg, self.headers).get("template", cfg.nudge_template)

        def _stream(self, resp, conn, conv, payload, messages, judge_model, mode, template, auth):
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
                        conv, judge_model, assistant_text, _bearer_token(auth)
                    )
                    if flagged:
                        nudge = render_template(template, **_nudge_args(judge_model, flagged))
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
