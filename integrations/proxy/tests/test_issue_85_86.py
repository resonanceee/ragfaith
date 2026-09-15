"""Issues #85 (client-visible 4xx logging) and #86 (RFE_STRICTNESS +
verdict-aware nudges + conservative truncation label)."""

import json
import tempfile
from pathlib import Path

import pytest
from conftest import full_msg, post, raw_post, wait_ends
from ragfaith_proxy.judge import SYSTEM_PROMPT
from ragfaith_proxy.proxy import (
    DEFAULT_NUDGE,
    DEFAULT_UNVERIFIABLE_NUDGE,
    Config,
    build_nudge,
)

TOOL_EXCHANGE = [
    {"role": "user", "content": "can cats fly?"},
    {"role": "tool", "content": "Cats cannot fly."},
]

LOGFILE = Path(tempfile.gettempdir()) / "rfe-4xx-test.jsonl"


def _rows():
    return [json.loads(x) for x in LOGFILE.read_text().splitlines() if x.strip()]


# ---------------------------------------------------------------- issue #85


@pytest.mark.parametrize("rig", [str(LOGFILE)], indirect=True)
def test_passthrough_4xx_logged(rig):
    open(LOGFILE, "w").close()
    rig.state["fail_status"] = 400
    _, resp = post(rig, {"model": "m", "messages": TOOL_EXCHANGE}, {"X-Conversation-Id": "pt"})
    assert resp.status == 400
    resp.read()
    rows = _rows()
    row = next(r for r in rows if r.get("kind") == "passthrough-status")
    assert row["status"] == 400 and row["conversation"] == "pt"
    assert "upstream boom" in row["body"]


def test_body_excerpt_redacted():
    # secrets in relayed upstream error bodies never reach the log
    from ragfaith_proxy.proxy import _excerpt

    raw = json.dumps(
        {
            "error": "key sk-secret123456789 bearer Bearer abcdefghi123 "
            "cfg api_key='supersecret99' rejected"
        }
    ).encode()
    out = _excerpt(raw)
    assert "sk-secret123456789" not in out and "abcdefghi123" not in out
    assert "supersecret99" not in out and "[REDACTED]" in out


@pytest.mark.parametrize("rig", [str(LOGFILE)], indirect=True)
@pytest.mark.parametrize(
    "payload,headers,reason",
    [
        ({"model": "m", "messages": TOOL_EXCHANGE}, {"Authorization": ""}, "missing-auth"),
        ("{broken", {"Authorization": "Bearer t"}, "invalid-json"),
        ({"model": "m", "messages": "nope"}, {}, "bad-messages"),
        ({"model": 3, "messages": TOOL_EXCHANGE}, {}, "bad-model"),
    ],
)
def test_request_rejected_logged(rig, payload, headers, reason):
    open(LOGFILE, "w").close()
    body = json.dumps(payload).encode() if not isinstance(payload, str) else payload.encode()
    _, resp = raw_post(rig, {"Content-Type": "application/json", **headers}, body)
    assert resp.status in (400, 401)
    resp.read()
    rows = [r for r in _rows() if r.get("kind") == "request-rejected"]
    assert rows and rows[-1]["reason"] == reason


@pytest.mark.parametrize("rig", [str(LOGFILE)], indirect=True)
def test_nudge_inject_and_failed_request_logged(rig):
    # hardest remote-debug case: a 400 on a request that carried a nudge
    open(LOGFILE, "w").close()
    rig.cfg.nudge_mode = "next"
    rig.state["script"] = [{"json": full_msg("Cats can fly.")}, {"json": full_msg("ok.")}]
    rig.state["verdicts"] = ["unfaithful"]
    _, resp = post(rig, {"model": "m", "messages": TOOL_EXCHANGE}, {"X-Conversation-Id": "nj"})
    resp.read()
    assert wait_ends(rig.cascade, 1)
    rig.state["fail_status"] = 400
    msgs = TOOL_EXCHANGE + [
        {"role": "assistant", "content": "Cats can fly."},
        {"role": "user", "content": "continue"},
    ]
    _, resp2 = post(rig, {"model": "m", "messages": msgs}, {"X-Conversation-Id": "nj"})
    assert resp2.status == 400
    resp2.read()
    rows = _rows()
    inject = next(r for r in rows if r.get("kind") == "nudge-inject")
    assert inject["conversation"] == "nj" and inject["bytes"] > 0
    failed = next(r for r in rows if r.get("kind") == "nudge-request-failed")
    assert failed["status"] == 400 and failed["conversation"] == "nj"
    assert rig.cascade.pop_nudge("nj") is not None  # nudge survived the failure


# ---------------------------------------------------------------- issue #86


def test_strictness_preset_flag_sets():
    assert Config.from_env({}).flag_verdicts == ("unfaithful",)
    assert Config.from_env({"RFE_STRICTNESS": "strict"}).flag_verdicts == (
        "unfaithful",
        "unverifiable",
    )
    # explicit CSV wins over the preset
    assert Config.from_env(
        {"RFE_STRICTNESS": "strict", "RFE_FLAG_VERDICTS": "unfaithful"}
    ).flag_verdicts == ("unfaithful",)


def test_build_nudge_normal_single_template():
    flagged = [("Cats purr.", "unverifiable"), ("Cats fly.", "unfaithful")]
    n = build_nudge(DEFAULT_NUDGE, DEFAULT_UNVERIFIABLE_NUDGE, False, "jm", flagged)
    assert "internal (training) knowledge" not in n
    assert "reconcile" in n and "unfaithful/unverifiable" in n


def test_build_nudge_strict_verdict_arms():
    flagged = [("Cats purr.", "unverifiable"), ("Cats fly.", "unfaithful")]
    n = build_nudge(DEFAULT_NUDGE, DEFAULT_UNVERIFIABLE_NUDGE, True, "jm", flagged)
    assert "reconcile" in n  # unfaithful arm
    assert "internal (training) knowledge" in n  # unverifiable arm
    assert "No silent assertions" in n
    # verdict-scoped counts, not the aggregate
    assert "1 claim(s)" in n


def test_build_nudge_strict_unverifiable_only():
    n = build_nudge(
        DEFAULT_NUDGE,
        DEFAULT_UNVERIFIABLE_NUDGE,
        True,
        "jm",
        [("Cats purr.", "unverifiable")],
    )
    assert "reconcile" not in n
    assert "No silent assertions" in n


def test_strict_mode_flags_unverifiable_end_to_end(rig):
    rig.cfg.strictness = "strict"
    rig.cfg.flag_verdicts = ("unfaithful", "unverifiable")
    rig.cfg.nudge_mode = "next"
    rig.state["script"] = [{"json": full_msg("Cats purr loudly.")}, {"json": full_msg("ok.")}]
    rig.state["verdicts"] = ["unverifiable"]
    _, resp = post(rig, {"model": "m", "messages": TOOL_EXCHANGE}, {"X-Conversation-Id": "st"})
    resp.read()
    assert wait_ends(rig.cascade, 1)
    nudge = rig.cascade.pop_nudge("st")
    assert nudge is not None and "internal (training) knowledge" in nudge


def test_truncation_note_when_premises_filtered(rig):
    rig.cfg.nudge_mode = "next"
    rig.state["script"] = [{"json": full_msg("Zebra quantum flux.")}, {"json": full_msg("ok.")}]
    rig.state["verdicts"] = ["unfaithful"]
    long_premise = "irrelevant filler sentence. " * 600  # > PREMISE_BUDGET chars
    msgs = [{"role": "user", "content": "q"}, {"role": "tool", "content": long_premise}]
    _, resp = post(rig, {"model": "m", "messages": msgs}, {"X-Conversation-Id": "tr"})
    resp.read()
    assert wait_ends(rig.cascade, 1)
    user_msg = rig.state["judge_calls"][0]["messages"][-1]["content"]
    assert "relevance-filtered excerpt" in user_msg


def test_no_truncation_note_small_premises(rig):
    rig.state["script"] = [{"json": full_msg("Cats cannot fly.")}]
    rig.state["verdicts"] = ["faithful"]
    _, resp = post(rig, {"model": "m", "messages": TOOL_EXCHANGE}, {"X-Conversation-Id": "sm"})
    resp.read()
    assert wait_ends(rig.cascade, 1)
    user_msg = rig.state["judge_calls"][0]["messages"][-1]["content"]
    assert "relevance-filtered excerpt" not in user_msg


def test_judge_prompt_taxonomy_intact():
    # prerequisite of #86: unfaithful = contradiction/fabrication, unverifiable
    # = not addressed (incl. derivation); #74 prompt conflation stays fixed
    assert "absent from every" in SYSTEM_PROMPT
    assert "extrapolate" in SYSTEM_PROMPT
