"""LLM judge for ragfaith-proxy (OpenAI-compatible chat API, stdlib only).

Adapted from rag_faithfulness_eval/llm_judge.py: provider-pluggable base URL
and key supplied by the caller (env-only, no .env loading), token-only log
lines instead of USD accounting.
"""

import hashlib
import json
import logging
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger(__name__)

VERDICTS = ("faithful", "unfaithful", "unverifiable")
VCACHE_MAX = 10000
RETRIES = 3  # streaming handler must not stall minutes on an upstream outage

SYSTEM_PROMPT = (
    "You are a RAG faithfulness judge. / Du bist ein RAG-Treuerichter.\n"
    "Decide if the CLAIM is fully supported by the CONTEXT alone (never use "
    "outside knowledge). Answer with ONLY one JSON object, no other text:\n"
    '{"verdict": "faithful"} - every fact in the claim is supported by the context\n'
    '{"verdict": "unfaithful"} - at least one fact contradicts or is unsupported '
    "by the context (wrong entity, number, date, or fabricated detail)\n"
    '{"verdict": "unverifiable"} - the context does not address the claim at all'
)


def _verdict_key(model: str, context: str, claim: str) -> str:
    return hashlib.sha256(f"{model}\x00{context}\x00{claim}".encode()).hexdigest()


def _parse_verdict(text: str) -> str:
    m = re.search(r'"verdict"\s*:\s*"(\w+)"', text)
    verdict = m.group(1) if m else ""
    if verdict not in VERDICTS:
        raise ValueError(f"unparseable verdict: {text[:200]!r}")
    return verdict


class Judge:
    """Chat-model judge against any OpenAI-compatible base URL."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        *,
        max_tokens: int = 256,
        cache_path: Path | None = None,
        log=None,
        vcache_max: int = VCACHE_MAX,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.cache_path = cache_path
        self._log = log or (lambda r: print(json.dumps(r), file=sys.stderr, flush=True))
        self._lock = threading.Lock()
        self._vcache_max = vcache_max
        self._vcache: OrderedDict[str, str] = OrderedDict()
        if cache_path and cache_path.exists():
            for line in cache_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._vcache[row["key"]] = row["verdict"]

    def _call(self, user_msg: str, retries: int = RETRIES, max_tokens: int | None = None) -> dict:
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "max_tokens": max_tokens or self.max_tokens,
                "reasoning": {"exclude": True},  # hide reasoning; still billed ~100-250 tok
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            }
        ).encode()
        url = f"{self.base_url}/chat/completions"
        for attempt in range(retries):
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                    time.sleep(2**attempt)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as e:
                # network drop: bounded retry (<=15 s total sleep at 3 attempts)
                if attempt < retries - 1:
                    logger.warning("network error (%s); retry %d/%d", e, attempt + 1, retries)
                    time.sleep(min(60, 5 * 2**attempt))
                    continue
                raise
        raise RuntimeError("unreachable: retry loop exhausted")

    def _account(self, resp: dict, conversation: str) -> None:
        usage = resp.get("usage") or {}
        self._log(
            {
                "kind": "judge",
                "model": self.model,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "conversation": conversation,
            }
        )

    def verdict(self, context: str, claim: str, conversation: str = "") -> str:
        key = _verdict_key(self.model, context, claim)
        with self._lock:
            if key in self._vcache:
                self._vcache.move_to_end(key)
                return self._vcache[key]
        msg = f"CONTEXT:\n{context}\n\nCLAIM:\n{claim}"
        for max_tokens in (self.max_tokens, self.max_tokens * 2):
            resp = self._call(msg, max_tokens=max_tokens)
            self._account(resp, conversation)
            content = resp["choices"][0]["message"].get("content")
            try:
                verdict = _parse_verdict(content or "")
                self._store(key, verdict)
                return verdict
            except ValueError:
                continue
        verdict = "unverifiable"  # parse failure after 256 -> 2x tokens: conservative
        self._store(key, verdict)
        return verdict

    def _store(self, key: str, verdict: str) -> None:
        with self._lock:
            self._vcache[key] = verdict
            self._vcache.move_to_end(key)
            while len(self._vcache) > self._vcache_max:
                self._vcache.popitem(last=False)
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a") as f:
                    f.write(json.dumps({"key": key, "verdict": verdict}) + "\n")
