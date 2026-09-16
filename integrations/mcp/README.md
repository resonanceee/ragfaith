# ragfaith-mcp

MCP server for ragfaith: live faithfulness checks for RAG answers over stdio,
reusing the ragfaith-proxy judge and claim decomposition.

Tools: `check_faithfulness(question, passages, answer)` and `judge_status()`.

**Important:** the model will not call these on its own — instruct it in the
system prompt (all sessions) or first message (one session). Suggested wording,
install, host config, and dev/test instructions: **[docs/mcp.md](../../docs/mcp.md)**.

```sh
pip install ragfaith-mcp
export OPENROUTER_API_KEY=...
```
