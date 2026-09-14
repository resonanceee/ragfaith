"""Fixes for issues 71-74: flag set, auditability, [DONE] timing, 429 retry."""

import json
import tempfile
import time
from pathlib import Path

import pytest
from test_proxy import (
    DONE_LINE,
    chunk,
    full_msg,
    iter_sse_lines,
    post,
    wait_ends,
)

TOOL_EXCHANGE = [
    {"role": "user", "content": "can cats fly?"},
    {"role": "tool", "content": "Cats cannot fly."},
]

LOGFILE = Path(tempfile.gettempdir()) / "rfe-audit-test.jsonl"


def test_unverifiable_not_flagged_by_default(rig):
    # issue #74: derivation/drift must not trigger nudges; only unfaithful does
    rig.cfg.nudge_mode = "next"
    rig.state["script"] = [{"json": full_msg("Cats purr.")}, {"json": full_msg("ok.")}]
    rig.state["verdicts"] = ["unverifiable", "faithful"]
    _, resp = post(rig, {"model": "m", "messages": TOOL_EXCHANGE}, {"X-Conversation-Id": "flag-d"})
    resp.read()
    assert wait_ends(rig.cascade, 1)
    assert rig.cascade.pop_nudge("flag-d") is None


def test_flag_verdicts_config_includes_unverifiable(rig):
    rig.cfg.nudge_mode = "next"
    rig.cfg.flag_verdicts = ("unfaithful", "unverifiable")
    rig.state["script"] = [{"json": full_msg("Cats purr.")}, {"json": full_msg("ok.")}]
    rig.state["verdicts"] = ["unverifiable", "faithful"]
    _, resp = post(rig, {"model": "m", "messages": TOOL_EXCHANGE}, {"X-Conversation-Id": "flag-b"})
    resp.read()
    assert wait_ends(rig.cascade, 1)
    nudge = rig.cascade.pop_nudge("flag-b")
    assert nudge is not None and "unverifiable" in nudge


def test_judge_prompt_separates_unsupported_from_unaddressed():
    # issue #74, root cause 2: prompt must not conflate the two predicates
    from ragfaith_proxy.judge import SYSTEM_PROMPT

    assert "absent from every" in SYSTEM_PROMPT
    assert "extrapolate" in SYSTEM_PROMPT


def test_done_flushed_before_cascade_in_next_mode(rig):
    # issue #72: [DONE] must not wait for the judge cascade
    rig.cfg.nudge_mode = "next"
    rig.state["judge_delay"] = 1.0
    rig.state["script"] = [{"stream": [chunk("Cats can fly.", finish="stop"), DONE_LINE]}]
    rig.state["verdicts"] = ["unfaithful"]
    start = time.monotonic()
    conn, resp = post(
        rig, {"model": "m", "stream": True, "messages": TOOL_EXCHANGE}, {"X-Conversation-Id": "dt"}
    )
    body = "".join(iter_sse_lines(resp))
    done_at = time.monotonic() - start
    conn.close()
    assert "[DONE]" in body
    assert done_at < 1.0  # cascade takes ~1s; the stream closed before it
    assert wait_ends(rig.cascade, 1)  # background cascade still completes
    assert rig.cascade.pop_nudge("dt") is not None


def test_chain_mode_still_holds_done(rig):
    # chain appends to the stream, so [DONE] stays withheld until judged
    rig.state["judge_delay"] = 0.5
    rig.state["script"] = [
        {"stream": [chunk("Cats can fly.", finish="stop"), DONE_LINE]},
        {"stream": [chunk("They cannot.", finish="stop"), DONE_LINE]},
    ]
    rig.state["verdicts"] = ["unfaithful"]
    start = time.monotonic()
    conn, resp = post(
        rig, {"model": "m", "stream": True, "messages": TOOL_EXCHANGE}, {"X-Conversation-Id": "ch"}
    )
    body = "".join(iter_sse_lines(resp))
    took = time.monotonic() - start
    conn.close()
    assert "They cannot." in body
    assert took >= 0.5  # [DONE] withheld through the cascade
    assert body.count("[DONE]") == 1


def test_judge_429_retried_then_verdict_recorded(rig):
    # issue #71: transient 429 must not silently drop a claim
    rig.cfg.nudge_mode = "next"
    rig.state["judge_429_left"] = 1
    rig.state["script"] = [{"json": full_msg("Cats can fly.")}, {"json": full_msg("ok.")}]
    rig.state["verdicts"] = ["unfaithful"]
    _, resp = post(
        rig, {"model": "m", "messages": TOOL_EXCHANGE}, {"X-Conversation-Id": "retry-429"}
    )
    resp.read()
    assert wait_ends(rig.cascade, 1)
    with rig.state["lock"]:
        assert len(rig.state["judge_calls"]) == 2  # first call 429'd, retry landed
    assert rig.cascade.pop_nudge("retry-429") is not None


@pytest.mark.parametrize("rig", [str(LOGFILE)], indirect=True)
def test_judge_log_has_claim_verdict_and_nudge_events(rig):
    # issue #73: auditability — verdict rows carry claim + verdict + key,
    # nudge stash/delivery events are logged (file wired via rig fixture)
    open(LOGFILE, "w").close()  # fresh log; cascade fh is append-mode
    rig.cfg.nudge_mode = "next"
    rig.state["script"] = [{"json": full_msg("Cats can fly.")}, {"json": full_msg("ok.")}]
    rig.state["verdicts"] = ["unfaithful"]
    _, resp = post(rig, {"model": "m", "messages": TOOL_EXCHANGE}, {"X-Conversation-Id": "audit"})
    resp.read()
    assert wait_ends(rig.cascade, 1)  # stash happens post-response in _plain
    # second request in the same conversation delivers the stashed nudge
    msgs = TOOL_EXCHANGE + [
        {"role": "assistant", "content": "Cats can fly."},
        {"role": "user", "content": "again"},
    ]
    _, resp2 = post(rig, {"model": "m", "messages": msgs}, {"X-Conversation-Id": "audit"})
    resp2.read()
    assert wait_ends(rig.cascade, 2)
    rows = [json.loads(x) for x in LOGFILE.read_text().splitlines() if x.strip()]
    judge_rows = [r for r in rows if r.get("kind") == "judge"]
    assert judge_rows and all("claim" in r and "verdict" in r and "key" in r for r in judge_rows)
    assert any(r["verdict"] == "unfaithful" and "Cats can fly." in r["claim"] for r in judge_rows)
    kinds = {r["kind"] for r in rows}
    assert "nudge-stash" in kinds and "nudge-delivered" in kinds
