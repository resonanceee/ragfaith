"""Stub-based MCP integration tests for the ragfaith server (no network, no key).

A fake Judge is injected; a real MCP client exercises the server in-memory
(mcp.Client(mcp)), asserting tool listing, check_faithfulness verdict flow, the
missing-key error path, judge_status, and that the "instruct the model to invoke
ragfaith" guidance stays present in docs/mcp.md and the server instructions.
"""

import asyncio
import json
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp", reason="mcp SDK not installed (pip install integrations/mcp)")
from mcp import Client  # noqa: E402
from ragfaith_mcp import server  # noqa: E402

DOCS = Path(__file__).resolve().parents[3] / "docs" / "mcp.md"


class StubJudge:
    def __init__(self):
        self.model = "stub-judge"

    def verdict(self, context: str, claim: str, conversation: str = "") -> str:
        if "unsupported" in claim:
            return "unfaithful"
        if "maybe" in claim:
            return "unverifiable"
        return "faithful"


@pytest.fixture(autouse=True)
def stub_judge(monkeypatch):
    monkeypatch.setattr(server, "_judge", StubJudge())
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


def run(coro):
    return asyncio.run(coro)


def result_dict(result) -> dict:
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


def test_tools_listed():
    async def go():
        async with Client(server.mcp) as client:
            tools = await client.list_tools()
            return {t.name for t in tools.tools}

    assert run(go()) == {"check_faithfulness", "judge_status"}


def test_check_faithfulness_verdicts():
    async def go():
        async with Client(server.mcp) as client:
            result = await client.call_tool(
                "check_faithfulness",
                {
                    "question": "What color is the sky?",
                    "passages": ["The sky is blue."],
                    "answer": (
                        "The sky is blue. The sky is unsupported purple. The sky is maybe green."
                    ),
                },
            )
            return result_dict(result)

    out = run(go())
    verdicts = [c["verdict"] for c in out["claims"]]
    assert set(verdicts) == {"faithful", "unfaithful", "unverifiable"}
    assert out["summary"]["faithful"] >= 1
    assert out["summary"]["unfaithful"] == 1
    assert out["summary"]["unverifiable"] == 1
    assert out["all_faithful"] is False
    assert out["model"] == "stub-judge"


def test_check_faithfulness_all_faithful():
    async def go():
        async with Client(server.mcp) as client:
            result = await client.call_tool(
                "check_faithfulness",
                {
                    "question": "Q",
                    "passages": ["The sky is blue."],
                    "answer": "The sky is blue.",
                },
            )
            return result_dict(result)

    assert run(go())["all_faithful"] is True


def test_check_faithfulness_real_judge_without_key_errors():
    async def go():
        async with Client(server.mcp) as client:
            result = await client.call_tool(
                "check_faithfulness",
                {"question": "Q", "passages": ["P"], "answer": "The sky is blue."},
            )
            return result

    server._judge = None  # bypass stub: hit the real lazy-builder path
    try:
        result = run(go())
        assert result.is_error
        assert "OPENROUTER_API_KEY" in result.content[0].text
    finally:
        server._judge = StubJudge()


def test_judge_status_reports_key_presence(monkeypatch):
    async def go():
        async with Client(server.mcp) as client:
            result = await client.call_tool("judge_status", {})
            return result_dict(result)

    out = run(go())
    assert out["api_key_configured"] is False
    assert out["provider"] == "openrouter"

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-dummy")
    out = run(go())
    assert out["api_key_configured"] is True
    assert "sk-test-dummy" not in json.dumps(out)


def test_judge_status_model_env(monkeypatch):
    monkeypatch.setenv("RFE_OPENROUTER_MAIN_MODEL", "some/other-model")

    async def go():
        async with Client(server.mcp) as client:
            return result_dict(await client.call_tool("judge_status", {}))

    assert run(go())["model"] == "some/other-model"


def test_invocation_instructions_present():
    """The model must be told to invoke ragfaith: guidance in server
    instructions AND docs, with the suggested system-prompt wording."""
    assert "check_faithfulness" in server.INSTRUCTIONS
    assert "MUST call" in server.INSTRUCTIONS

    docs = DOCS.read_text()
    assert "check_faithfulness(question, passages, answer)" in docs
    assert "system prompt" in docs
    assert "MUST call the ragfaith tool check_faithfulness" in docs
