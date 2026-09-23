"""ragfaith MCP server: live faithfulness checks for RAG answers.

Exposes two tools over stdio (the MCP default transport for local servers):

- check_faithfulness(question, passages, answer): decomposes the answer into
  atomic claims (ragfaith_proxy.decompose.split_claims) and judges each against
  QUESTION + PASSAGES with the configured OpenRouter judge (ragfaith_proxy.judge.Judge).
- judge_status(): model/base URL/OPENROUTER_API_KEY presence, never the key.

The model must be instructed to call check_faithfulness (see docs/mcp.md); that
guidance also ships in the server's MCP `instructions` below.
"""

import os

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from ragfaith_proxy.decompose import split_claims
from ragfaith_proxy.judge import Judge

DEFAULT_MODEL = "z-ai/glm-5.3-flash"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

INSTRUCTIONS = (
    "ragfaith live-checks whether an answer is faithful to its sources. "
    "Whenever an answer is grounded in retrieved documents, you MUST call "
    "check_faithfulness(question, passages, answer) with the user's question, the "
    "exact passages relied on, and the draft answer BEFORE finalizing the reply. "
    "If any claim returns 'unfaithful' or 'unverifiable', correct or remove it. "
    "The check is internal: never mention the check, the judge, verdicts, or "
    "flags in your user-visible reply - just answer the user with the corrected "
    "content. If a claim can only be disclosed rather than corrected, state that "
    "to the user as ordinary content. If the tool is unavailable or errors, say so "
    "instead of silently skipping the check. Call judge_status to confirm the judge "
    "is configured."
)

mcp = MCPServer("ragfaith", instructions=INSTRUCTIONS)

_judge: Judge | None = None


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


def _get_judge() -> Judge:
    global _judge
    if _judge is None:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            # ToolError: the message reaches the model instead of a masked crash
            raise ToolError(
                "OPENROUTER_API_KEY is not set; the ragfaith judge cannot run. "
                "Set it in the MCP server environment and retry."
            )
        _judge = Judge(
            model=_env("RFE_OPENROUTER_MAIN_MODEL", DEFAULT_MODEL),
            base_url=_env("RFE_OPENROUTER_BASE", DEFAULT_BASE_URL),
            api_key=key,
        )
    return _judge


@mcp.tool()
def check_faithfulness(question: str, passages: list[str], answer: str) -> dict:
    """Check an answer's claims against the passages it should be grounded in.

    Splits the answer into atomic claims and returns a verdict per claim:
    faithful / unfaithful / unverifiable, plus a summary. Anything not
    "faithful" must be corrected or removed before answering the user. The
    check is internal: never mention the check, the judge, verdicts, or
    flags in the user-visible reply - just answer with the corrected content.
    """
    context = f"QUESTION:\n{question}\n\nPASSAGES:\n" + "\n---\n".join(passages)
    judge = _get_judge()
    claims = [
        {"claim": c, "verdict": judge.verdict(context, c)} for _, _, c in split_claims(answer)
    ]
    summary: dict[str, int] = {}
    for c in claims:
        summary[c["verdict"]] = summary.get(c["verdict"], 0) + 1
    return {
        "claims": claims,
        "summary": summary,
        "all_faithful": all(c["verdict"] == "faithful" for c in claims),
        "model": judge.model,
    }


@mcp.tool()
def judge_status() -> dict:
    """Report judge configuration: model, provider base URL, and whether
    OPENROUTER_API_KEY is set. Never returns the key itself."""
    return {
        "model": _env("RFE_OPENROUTER_MAIN_MODEL", DEFAULT_MODEL),
        "provider": "openrouter",
        "base_url": _env("RFE_OPENROUTER_BASE", DEFAULT_BASE_URL),
        "api_key_configured": bool(os.environ.get("OPENROUTER_API_KEY")),
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
