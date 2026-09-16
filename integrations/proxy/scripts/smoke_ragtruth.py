"""Merge-gate smoke test: 50 cached RAGTruth claims through the synthetic endpoint.

GLM (hf:zai-org/GLM-5.3-Flash) verdicts are compared against the cached
openrouter z-ai/glm-5.3-flash verdicts (results/exp4/llm_cache_B.jsonl, keys
sha256(model + "\\x00" + premise + "\\x00" + claim), premise = Exp4 query-mode).
DeepSeek (hf:deepseek-ai/DeepSeek-V4.1-Flash) must parse >=95%.

Key from env SYNTHETIC_API_KEY only. Idempotent: no on-disk verdict cache;
re-running re-judges the same deterministic 50. Writes smoke_results.json
(verdicts + summary only, no claims, no keys).
"""

import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragfaith_proxy.judge import Judge, _parse_verdict, _verdict_key


def _results_dir() -> Path:
    env = os.environ.get("RFE_RESULTS_DIR")
    if env:
        return Path(env)
    relative = Path(__file__).resolve().parents[3] / "results"  # <repo>/results
    if relative.is_dir():
        return relative
    return Path("/Users/res/Code/ragfaith/results")  # last-resort legacy fallback


RESULTS = _results_dir()
CACHE_MODEL = "z-ai/glm-5.3-flash"  # openrouter id used when the cache was written
N = 50
SEED = 0


class SmokeJudge(Judge):
    """Judge that records raw capture + parse-failure fallback separately."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.parse_failures = 0

    def verdict_with_flag(self, context, claim):
        msg = f"CONTEXT:\n{context}\n\nCLAIM:\n{claim}"
        for mt in (self.max_tokens, self.max_tokens * 2):
            resp = self._call(msg, max_tokens=mt)
            self._account(resp, "smoke")
            content = resp["choices"][0]["message"].get("content")
            try:
                return _parse_verdict(content or ""), False
            except ValueError:
                continue
        self.parse_failures += 1
        return "unverifiable", True  # conservative fallback, counted against the parse bar


def load_pairs():
    cache = {}
    for line in (RESULTS / "exp4/llm_cache_B.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            cache[r["key"]] = r["verdict"]
    pairs = []
    for line in (RESULTS / "exp5/sample.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = _verdict_key(CACHE_MODEL, r["premise"], r["claim"])
        if key in cache:  # hard validation: reconstruction must hit the cache
            pairs.append(
                {"id": r["id"], "context": r["premise"], "claim": r["claim"], "cache": cache[key]}
            )
    if not pairs:
        raise SystemExit("reconstruction yielded zero cache hits; aborting before any spend")
    return pairs


def pick50(pairs):
    """Stratified proportional pick (largest remainder), seeded."""
    rng = random.Random(SEED)
    by_verdict = {}
    for p in pairs:
        by_verdict.setdefault(p["cache"], []).append(p)
    quotas = {v: len(ps) * N / len(pairs) for v, ps in by_verdict.items()}
    alloc = {v: int(q) for v, q in quotas.items()}
    remainders = sorted(quotas, key=lambda v: quotas[v] % 1, reverse=True)
    for v in remainders[: N - sum(alloc.values())]:
        alloc[v] += 1
    picked = []
    for v, k in alloc.items():
        rng.shuffle(by_verdict[v])
        picked += by_verdict[v][:k]
    rng.shuffle(picked)
    return picked


def judge_all(judge, pairs):
    out = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(judge.verdict_with_flag, p["context"], p["claim"]): p for p in pairs}
        for fut in futs:
            p = futs[fut]
            verdict, parse_failed = fut.result()
            out[p["id"]] = {"verdict": verdict, "parse_failed": parse_failed}
    return out


def main():
    api_key = os.environ["SYNTHETIC_API_KEY"]  # env only, never printed/stored
    base = os.environ.get("RFE_SYNTHETIC_BASE", "https://api.synthetic.new/v1")

    pairs = load_pairs()
    picked = pick50(pairs)
    print(f"cache hits: {len(pairs)}, picked {len(picked)}: {Counter(p['cache'] for p in picked)}")

    results = {"pairs": {}, "summary": {}}
    for label, model in (
        ("main", "hf:zai-org/GLM-5.3-Flash"),
        ("fallback", "hf:deepseek-ai/DeepSeek-V4.1-Flash"),
    ):
        tokens = {"prompt": 0, "completion": 0}

        def log(rec, t=tokens):
            t["prompt"] += rec["prompt_tokens"]
            t["completion"] += rec["completion_tokens"]

        env_override = os.environ.get(
            "RFE_SYNTHETIC_MAIN_MODEL" if label == "main" else "RFE_SYNTHETIC_FALLBACK_MODEL"
        )
        judge = SmokeJudge(env_override or model, base_url=base, api_key=api_key, log=log)
        verdicts = judge_all(judge, picked)
        for p in picked:
            row = results["pairs"].setdefault(p["id"], {"cache": p["cache"]})
            row[f"{label}_verdict"] = verdicts[p["id"]]["verdict"]
            if verdicts[p["id"]]["parse_failed"]:
                row[f"{label}_parse_failed"] = True
        results["summary"][label] = {
            "n": len(picked),
            "dist": dict(Counter(v["verdict"] for v in verdicts.values())),
            "parse_failures": judge.parse_failures,
            "prompt_tokens": tokens["prompt"],
            "completion_tokens": tokens["completion"],
        }
        print(label, results["summary"][label])

    # main-judge agreement vs cache
    main = [(p["cache"], results["pairs"][p["id"]]["main_verdict"]) for p in picked]
    exact = sum(c == g for c, g in main)
    results["summary"]["main"]["exact_match"] = f"{exact}/{len(main)} ({exact / len(main):.1%})"
    non_fallback = [(c, g) for c, g in main if c != "unverifiable" and g != "unverifiable"]
    exact_nf = sum(c == g for c, g in non_fallback)
    results["summary"]["main"]["exact_match_non_unverifiable"] = (
        f"{exact_nf}/{len(non_fallback)} ({exact_nf / max(1, len(non_fallback)):.1%})"
    )
    confusion = Counter(f"cache={c} syn={g}" for c, g in main if c != g)
    results["summary"]["main"]["confusion"] = dict(confusion)

    out = Path(__file__).with_name("smoke_results.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
