"""ragfaith-proxy tests. No network: a local fake upstream serves completions
and judge verdicts; the proxy under test runs on an ephemeral port."""

import http.client
import json
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
import ragfaith_proxy.judge as judge_mod
import ragfaith_proxy.proxy as proxy_mod
from ragfaith_proxy.decompose import split_claims
from ragfaith_proxy.judge import Judge
from ragfaith_proxy.proxy import (
    Config,
    apply_tool_deltas,
    harvest_premises,
    inject_nudge,
    make_server,
    select_judge,
)

DONE_LINE = "data: [DONE]\n\n"


def chunk(content=None, finish=None, tool_calls=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    obj = {"id": "c1", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return f"data: {json.dumps(obj)}\n\n"


def full_msg(content, finish="stop"):
    return {
        "id": "c1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 5},
    }


class FakeHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        with self.server.state["lock"]:
            self.server.state["auths"].append(self.headers.get("Authorization", ""))
        self._send_json({"object": "list", "data": [{"id": "fake-model"}]})

    def do_POST(self):
        state = self.server.state
        with state["lock"]:
            state["auths"].append(self.headers.get("Authorization", ""))
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("model") in (state["glm"], state["deepseek"]):
            self._judge(state, body)
        else:
            self._completions(state, body)

    def _judge(self, state, body):
        with state["lock"]:
            state["judge_calls"].append(body)
            fail = state["judge_fail_status"]
            verdict = state["verdicts"].pop(0) if state["verdicts"] else "faithful"
        if fail:
            self._send_json({"error": "judge upstream boom"}, status=fail)
            return
        resp = {
            "choices": [
                {"message": {"content": json.dumps({"verdict": verdict})}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        }
        self._send_json(resp)

    def _completions(self, state, body):
        with state["lock"]:
            state["requests"].append(body)
            idx = len(state["requests"]) - 1
        fail = state["fail_status"]
        fail_from = state["fail_from"]
        if fail and (fail_from is None or idx >= fail_from):
            self._send_json({"error": "upstream boom"}, status=fail)
            return
        script = state["script"]
        entry = script[min(idx, len(script) - 1)]
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for line in entry["stream"]:
                if state["delay"]:
                    time.sleep(state["delay"])
                self.wfile.write(line.encode())
                self.wfile.flush()
        else:
            self._send_json(entry["json"])

    def _send_json(self, obj, status=200):
        raw = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture()
def rig():
    state = {
        "lock": threading.Lock(),
        "requests": [],
        "judge_calls": [],
        "verdicts": [],
        "script": [],
        "delay": 0.0,
        "auths": [],
        "fail_status": None,
        "fail_from": None,
        "judge_fail_status": None,
        "glm": "hf:zai-org/GLM-5.3-Flash",
        "deepseek": "hf:deepseek-ai/DeepSeek-V4.1-Flash",
    }
    fake = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
    fake.daemon_threads = True
    fake.state = state
    threading.Thread(target=fake.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{fake.server_address[1]}/v1"
    cfg = Config(
        proxy_port=0,
        upstream_base=base,
        judge_base=base,
    )
    server, cascade = make_server(cfg)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield SimpleNamespace(
        state=state, cfg=cfg, cascade=cascade, base=base, port=server.server_address[1]
    )
    server.shutdown()
    fake.shutdown()
    server.server_close()
    fake.server_close()


def post(rig, payload, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", rig.port, timeout=30)
    base_headers = {"Content-Type": "application/json", "Authorization": "Bearer test-token"}
    base_headers.update(headers or {})
    conn.request(
        "POST",
        "/v1/chat/completions",
        body=json.dumps(payload),
        headers=base_headers,
    )
    return conn, conn.getresponse()


def raw_post(rig, headers, body=b""):
    conn = http.client.HTTPConnection("127.0.0.1", rig.port, timeout=10)
    base_headers = {"Content-Type": "application/json", "Authorization": "Bearer test-token"}
    base_headers.update(headers)
    conn.request("POST", "/v1/chat/completions", body=body, headers=base_headers)
    return conn, conn.getresponse()


def iter_sse_lines(resp):
    buf = b""
    while True:
        block = resp.read1(4096)
        if not block:
            return
        buf += block
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            yield line.decode()


def wait_ends(cascade, n, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cascade.end_count >= n:
            return True
        time.sleep(0.02)
    return False


# ---------------------------------------------------------------- stream passthrough


def test_sse_stream_through(rig):
    chunks = [chunk("Hello "), chunk("world"), chunk("!", finish="stop"), DONE_LINE]
    rig.state["script"] = [{"stream": chunks}]
    rig.state["delay"] = 0.25
    payload = {"model": "gpt-4o", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    _, resp = post(rig, payload)
    assert resp.status == 200
    t0 = time.monotonic()
    events, t_first = [], None
    for line in iter_sse_lines(resp):
        if line.strip():
            if t_first is None:
                t_first = time.monotonic() - t0
            events.append(line)
    total = time.monotonic() - t0
    assert t_first < 1.0, "first token must not wait on the cascade"
    assert total > 0.5, "chunks must arrive as upstream sends them, not batched"
    assert events == [c.splitlines()[0] for c in chunks]  # byte-for-byte, in order


# ---------------------------------------------------------------- chain mode


def test_chain_mode(rig):
    rig.state["script"] = [
        {"stream": [chunk("The sky is green."), chunk(None, finish="stop"), DONE_LINE]},
        {"stream": [chunk("Correction: sources say blue."), chunk(None, finish="stop"), DONE_LINE]},
    ]
    rig.state["verdicts"] = ["unfaithful"]
    _, resp = post(
        rig,
        {
            "model": "gpt-4o",
            "stream": True,
            "messages": [
                {"role": "user", "content": "what color is the sky?"},
                {"role": "tool", "content": "Source: the sky is blue."},
            ],
        },
    )
    events = [line for line in iter_sse_lines(resp) if line.strip()]
    dones = [line for line in events if "[DONE]" in line]
    assert len(dones) == 1, "exactly one terminal [DONE]"
    assert events[-1] == dones[0]
    payloads = [line[len("data: ") :] for line in events if line.startswith("data:")]
    deltas = [json.loads(p)["choices"][0]["delta"].get("content") for p in payloads[:-1]]
    assert "The sky is green." in deltas
    assert "Correction: sources say blue." in deltas
    # one internal continuation call: original messages + assistant + nudge user
    assert len(rig.state["requests"]) == 2
    msgs = rig.state["requests"][1]["messages"]
    assert msgs[-2] == {"role": "assistant", "content": "The sky is green."}
    assert msgs[-1]["role"] == "user"
    assert "1 claim(s)" in msgs[-1]["content"] and "The sky is green." in msgs[-1]["content"]
    assert len(rig.state["judge_calls"]) == 1
    assert rig.state["judge_calls"][0]["model"] == rig.cfg.glm_model


def test_chain_fallback_non_streaming(rig):
    """chain needs a stream; non-streaming falls back to next-mode delivery."""
    rig.state["script"] = [
        {"json": full_msg("Cats can fly.")},
        {"json": full_msg("ok.")},
        {"json": full_msg("fine.")},
    ]
    rig.state["verdicts"] = ["unverifiable", "faithful"]
    headers = {"X-Conversation-Id": "conv-fb"}
    messages = [
        {"role": "user", "content": "can cats fly?"},
        {"role": "tool", "content": "Cats cannot fly."},
    ]
    _, resp = post(rig, {"model": "m", "messages": messages}, headers)
    assert json.loads(resp.read())["choices"][0]["message"]["content"] == "Cats can fly."
    assert wait_ends(rig.cascade, 1)
    assert rig.cascade.pop_nudge("conv-fb") is not None  # stashed, not chained
    rig.cascade.set_nudge("conv-fb", None)  # reset for cleanliness


# ---------------------------------------------------------------- next mode


def test_next_mode(rig):
    rig.cfg.nudge_mode = "next"
    rig.state["script"] = [{"json": full_msg("Cats can fly.")}, {"json": full_msg("noted.")}]
    rig.state["verdicts"] = ["unverifiable", "faithful"]
    headers = {"X-Conversation-Id": "conv-1"}
    history = [
        {"role": "user", "content": "can cats fly?"},
        {"role": "tool", "content": "Cats cannot fly."},
    ]
    _, resp = post(rig, {"model": "m", "messages": history}, headers)
    resp.read()
    assert wait_ends(rig.cascade, 1)
    next_messages = [
        *history,
        {"role": "assistant", "content": "Cats can fly."},
        {"role": "user", "content": "contenu? no, continue"},
    ]
    _, resp2 = post(rig, {"model": "m", "messages": next_messages}, headers)
    resp2.read()
    assert wait_ends(rig.cascade, 2)
    msgs = rig.state["requests"][-1]["messages"]
    idx = msgs.index({"role": "assistant", "content": "Cats can fly."})
    nudge = msgs[idx + 1]
    assert nudge["role"] == "user"
    assert "unverifiable" in nudge["content"] and "Cats can fly." in nudge["content"]
    # delivered once only
    assert rig.state["requests"][-1]["messages"].count(nudge) == 1


def test_faithful_is_silent(rig):
    rig.state["script"] = [{"stream": [chunk("2 + 2 = 4."), chunk(None, finish="stop"), DONE_LINE]}]
    rig.state["verdicts"] = ["faithful"]
    _, resp = post(
        rig,
        {
            "model": "m",
            "stream": True,
            "messages": [
                {"role": "user", "content": "math?"},
                {"role": "tool", "content": "2 + 2 = 4."},
            ],
        },
    )
    events = [line for line in iter_sse_lines(resp) if line.strip()]
    assert events[-1] == "data: [DONE]"
    assert len(rig.state["requests"]) == 1  # no second upstream call
    assert rig.cascade.pop_nudge("__none__") is None
    assert wait_ends(rig.cascade, 1)
    conv = rig.cascade.events[0]["conversation"]
    assert rig.cascade._conv[conv]["nudge"] is None


# ---------------------------------------------------------------- client disconnect


def test_client_disconnect_mid_chain_no_hang(rig):
    rig.state["delay"] = 0.05
    rig.state["script"] = [
        {"stream": [chunk("Berlin is in France."), chunk(None, finish="stop"), DONE_LINE]},
        {"stream": [chunk(f"long correction part {i}.\n" * 10) for i in range(50)] + [DONE_LINE]},
    ]
    rig.state["verdicts"] = ["unfaithful"]
    conn, resp = post(
        rig,
        {
            "model": "m",
            "stream": True,
            "messages": [
                {"role": "user", "content": "where is berlin?"},
                {"role": "tool", "content": "Berlin is in Germany."},
            ],
        },
    )
    gen = iter_sse_lines(resp)
    next(gen)  # first chunk arrived
    conn.close()  # client bails while the chain is about to stream
    # proxy stays alive and serves the next request promptly
    rig.state["delay"] = 0.0
    rig.state["script"] = [{"json": full_msg("alive.")}]
    _, resp2 = post(rig, {"model": "m", "messages": [{"role": "user", "content": "ping"}]})
    assert resp2.status == 200
    assert json.loads(resp2.read())["choices"][0]["message"]["content"] == "alive."
    assert wait_ends(rig.cascade, 2, timeout=15)


# ---------------------------------------------------------------- judge behavior


def test_select_judge():
    cfg = Config()
    assert select_judge("hf:zai-org/GLM-5.3-Flash", cfg) == cfg.deepseek_model
    assert select_judge("glm-5.3-flash-instruct", cfg) == cfg.deepseek_model
    assert select_judge("z-ai/glm-5.3-flash:free", cfg) == cfg.deepseek_model
    assert select_judge("hf:deepseek-ai/DeepSeek-V4.1-Flash", cfg) == cfg.glm_model
    assert select_judge("gpt-4o", cfg) == cfg.glm_model


def test_verdict_garbage_falls_back_to_unverifiable(rig):
    rig.state["verdicts"] = ["word-salad", "still-word-salad"]
    judge = Judge(rig.cfg.glm_model, rig.base, "k")
    assert judge.verdict("ctx", "claim") == "unverifiable"  # 256 -> 2x tokens, then fallback
    assert len(rig.state["judge_calls"]) == 2


def test_verdict_cache_hit_and_persistence(rig, tmp_path):
    rig.state["verdicts"] = ["unfaithful"]
    cache = tmp_path / "proxy-cache-x.jsonl"
    j1 = Judge(rig.cfg.glm_model, rig.base, "k", cache_path=cache)
    assert j1.verdict("ctx", "claim") == "unfaithful"
    assert j1.verdict("ctx", "claim") == "unfaithful"  # in-memory hit
    assert len(rig.state["judge_calls"]) == 1
    j2 = Judge(rig.cfg.glm_model, rig.base, "k", cache_path=cache)
    assert j2.verdict("ctx", "claim") == "unfaithful"  # persisted JSONL hit, no network
    assert len(rig.state["judge_calls"]) == 1


def test_cache_dir_wiring(rig, tmp_path):
    rig.cfg.cache_dir = str(tmp_path)
    rig.state["script"] = [{"json": full_msg("Cats can fly.")}]
    rig.state["verdicts"] = ["unfaithful"]
    _, resp = post(
        rig,
        {
            "model": "m",
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "tool", "content": "cats cannot fly"},
            ],
        },
    )
    resp.read()
    assert wait_ends(rig.cascade, 1)
    files = list(tmp_path.glob("proxy-cache-*.jsonl"))
    assert len(files) == 1
    rows = [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]
    assert rows and rows[0]["verdict"] == "unfaithful"


# ---------------------------------------------------------------- unit helpers


def test_premise_cap_keeps_most_recent():
    msgs = [{"role": "tool", "content": "A" * 20000}, {"role": "tool", "content": "B" * 20000}]
    out = harvest_premises(msgs, 24000)
    assert len(out) <= 24000
    assert out.endswith("B" * 500)  # most recent content survives


def test_premises_include_tool_result_blocks():
    msgs = [{"role": "user", "content": [{"type": "tool_result", "content": "fetched data"}]}]
    assert harvest_premises(msgs, 24000) == "fetched data"


def test_tool_call_delta_assembly():
    acc = []
    apply_tool_deltas(
        acc,
        [
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_", "arguments": ""},
            }
        ],
    )
    apply_tool_deltas(acc, [{"index": 0, "function": {"name": "weather", "arguments": '{"ci'}}])
    apply_tool_deltas(acc, [{"index": 0, "function": {"arguments": 'ty":"SF"}'}}])
    assert acc[0]["id"] == "call_1"
    assert acc[0]["function"] == {"name": "get_weather", "arguments": '{"city":"SF"}'}


def test_split_claims_regex_fallback():
    claims = split_claims("Cats can fly. Dogs bark, loudly!")
    texts = [t for *_, t in claims]
    assert texts == ["Cats can fly.", "Dogs bark, loudly!"]
    src = "Cats can fly. Dogs bark, loudly!"
    for start, end, text in claims:
        assert src[start:end] == text


def test_inject_nudge_after_last_assistant():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    out = inject_nudge(msgs, "NUDGE")
    assert out[2] == {"role": "user", "content": "NUDGE"}
    assert len(out) == 4


def test_config_from_env():
    env = {
        "RFE_UPSTREAM_BASE": "http://x/v1",
        "RFE_PREMISE_CAP": "100",
        "RFE_NUDGE_MODE": "next",
        "RFE_JUDGE_PROVIDER": "openrouter",
        "RFE_HOST_DECORATORS": json.dumps(
            {"flowdown": {"nudge_mode": "next", "template": "t {n}"}}
        ),
    }
    cfg = Config.from_env(env)
    assert cfg.upstream_base == "http://x/v1"
    assert cfg.premise_cap == 100 and cfg.nudge_mode == "next"
    assert cfg.judge_base == "https://openrouter.ai/api/v1"
    assert cfg.glm_model == "z-ai/glm-5.3-flash"
    assert cfg.host_decorators["flowdown"]["nudge_mode"] == "next"
    # client-auth-only: no key fields exist on the config
    assert not hasattr(cfg, "upstream_key") and not hasattr(cfg, "judge_key")


def test_models_passthrough(rig):
    conn = http.client.HTTPConnection("127.0.0.1", rig.port, timeout=10)
    conn.request("GET", "/v1/models", headers={"Authorization": "Bearer test-token"})
    resp = conn.getresponse()
    assert resp.status == 200
    assert json.loads(resp.read())["data"][0]["id"] == "fake-model"


def test_auth_required(rig):
    conn = http.client.HTTPConnection("127.0.0.1", rig.port, timeout=10)
    conn.request(
        "POST",
        "/v1/chat/completions",
        body=json.dumps({"model": "gpt-4o", "messages": []}),
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    resp.read()
    assert resp.status == 401


def test_client_auth_forwarded(rig):
    rig.state["script"] = [{"stream": [chunk("hi", finish="stop"), DONE_LINE]}]
    conn, resp = post(
        rig,
        {"model": "gpt-4o", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Conversation-Id": "auth-test"},
    )
    assert resp.status == 200
    list(iter_sse_lines(resp))
    conn.close()
    with rig.state["lock"]:
        auths = list(rig.state["auths"])
    assert auths and all(a == "Bearer test-token" for a in auths)


# ---------------------------------------------------------------- conversation identity


def test_conv_key_separates_chats_with_shared_system_prompt(rig):
    system = {"role": "system", "content": "You are a helpful assistant."}
    chat_a = [system, {"role": "user", "content": "chat A question"}]
    chat_b = [system, {"role": "user", "content": "chat B question"}]
    key_a = rig.cascade.conv_key({}, chat_a)
    key_b = rig.cascade.conv_key({}, chat_b)
    assert key_a != key_b, "same system prompt must not collapse distinct chats"
    later_a = chat_a + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "more"}]
    assert rig.cascade.conv_key({}, later_a) == key_a, "same chat later turn keeps the key"
    assert rig.cascade.conv_key({"x-conversation-id": "conv-x"}, chat_a) == "conv-x"
    # state never bleeds across the two chats
    rig.cascade.record_messages(key_a, [{"role": "tool", "content": "premise A"}])
    rig.cascade.record_messages(key_b, [{"role": "tool", "content": "premise B"}])
    assert "premise A" in rig.cascade.premises(key_a)
    assert "premise B" not in rig.cascade.premises(key_a)


def test_conversation_store_lru_evicted(rig, monkeypatch):
    monkeypatch.setattr(proxy_mod, "CONV_STORE_MAX", 2)
    rig.cascade.record_messages("c1", [{"role": "user", "content": "1"}])
    rig.cascade.record_messages("c2", [{"role": "user", "content": "2"}])
    rig.cascade.premises("c1")  # refresh c1
    rig.cascade.record_messages("c3", [{"role": "user", "content": "3"}])
    assert set(rig.cascade._conv) == {"c1", "c3"}


def test_judge_store_lru_evicted(rig, monkeypatch):
    monkeypatch.setattr(proxy_mod, "JUDGE_STORE_MAX", 2)
    j1 = rig.cascade.judge("m", "secret-token-1")
    j2 = rig.cascade.judge("m", "secret-token-2")
    assert rig.cascade.judge("m", "secret-token-1") is j1  # refresh j1
    rig.cascade.judge("m", "secret-token-3")
    assert j1 in rig.cascade._judges.values()
    assert j2 not in rig.cascade._judges.values()
    assert all("secret-token" not in k for k in rig.cascade._judges), "raw tokens never keys"


def test_verdict_cache_lru_evicted(rig):
    judge = Judge(rig.cfg.glm_model, rig.base, "k", vcache_max=2)
    judge._store("k1", "faithful")
    judge._store("k2", "unfaithful")
    judge._store("k3", "faithful")
    assert list(judge._vcache) == ["k2", "k3"]


def test_incremental_client_accumulates_premises(rig):
    conv = "incremental"
    rig.cascade.record_messages(conv, [{"role": "tool", "content": "premise one"}])
    rig.cascade.record_messages(conv, [{"role": "tool", "content": "premise two"}])
    out = rig.cascade.premises(conv)
    assert "premise one" in out, "shorter request must not drop stored history"
    assert "premise two" in out, "new tool message must be stored"


# ---------------------------------------------------------------- request body limits & validation


def test_oversized_body_413(rig, monkeypatch):
    monkeypatch.setattr(proxy_mod, "MAX_BODY", 64)
    _, resp = raw_post(rig, {"Content-Length": "100"}, b"x" * 100)
    resp.read()
    assert resp.status == 413


def test_malformed_content_length_400(rig):
    _, resp = raw_post(rig, {"Content-Length": "abc"}, b"x")
    resp.read()
    assert resp.status == 400


def test_negative_content_length_400(rig):
    _, resp = raw_post(rig, {"Content-Length": "-1"}, b"x")
    resp.read()
    assert resp.status == 400


def test_messages_wrong_type_400(rig):
    _, resp = post(rig, {"model": "m", "messages": 42})
    resp.read()
    assert resp.status == 400


def test_model_wrong_type_400(rig):
    _, resp = post(rig, {"model": 42, "messages": [{"role": "user", "content": "x"}]})
    resp.read()
    assert resp.status == 400


def test_internal_error_gets_500(rig, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(rig.cascade, "conv_key", boom)
    _, resp = post(rig, {"model": "m", "messages": [{"role": "user", "content": "x"}]})
    resp.read()
    assert resp.status == 500


# ---------------------------------------------------------------- nudge never lost


def test_nudge_restashed_on_upstream_non_200(rig):
    rig.cascade.set_nudge("conv-fail", "pending nudge")
    rig.state["fail_status"] = 500
    _, resp = post(
        rig,
        {"model": "m", "messages": [{"role": "user", "content": "x"}]},
        {"X-Conversation-Id": "conv-fail"},
    )
    resp.read()
    assert resp.status == 500
    assert rig.cascade.pop_nudge("conv-fail") == "pending nudge"


def test_nudge_restashed_on_upstream_connect_error(rig, monkeypatch):
    rig.cascade.set_nudge("conv-conn", "pending nudge")

    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(proxy_mod, "upstream_request", boom)
    _, resp = post(
        rig,
        {"model": "m", "messages": [{"role": "user", "content": "x"}]},
        {"X-Conversation-Id": "conv-conn"},
    )
    resp.read()
    assert resp.status == 502
    assert rig.cascade.pop_nudge("conv-conn") == "pending nudge"


def test_nudge_restashed_when_chain_fails(rig):
    rig.state["script"] = [
        {"stream": [chunk("The sky is green."), chunk(None, finish="stop"), DONE_LINE]}
    ]
    rig.state["verdicts"] = ["unfaithful"]
    rig.state["fail_from"] = 1
    rig.state["fail_status"] = 502
    _, resp = post(
        rig,
        {
            "model": "m",
            "stream": True,
            "messages": [
                {"role": "user", "content": "color?"},
                {"role": "tool", "content": "The sky is blue."},
            ],
        },
        {"X-Conversation-Id": "conv-chain-fail"},
    )
    events = [line for line in iter_sse_lines(resp) if line.strip()]
    assert events[-1] == "data: [DONE]"
    assert wait_ends(rig.cascade, 1)
    nudge = rig.cascade.pop_nudge("conv-chain-fail")
    assert nudge is not None and "The sky is green." in nudge


# ---------------------------------------------------------------- premises flattening


def test_premises_flatten_nested_tool_result_content():
    msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "content": [
                        {"type": "text", "text": "part one"},
                        {"type": "text", "text": "part two"},
                    ],
                },
                {"type": "tool_result", "content": 123},  # defensive: non-text skipped
            ],
        }
    ]
    out = harvest_premises(msgs, 24000)
    assert "part one" in out and "part two" in out


# ---------------------------------------------------------------- judge retry budget


def test_judge_retry_budget_bounded(rig, monkeypatch):
    sleeps = []
    monkeypatch.setattr(judge_mod.time, "sleep", lambda s: sleeps.append(s))
    rig.state["judge_fail_status"] = 500
    judge = Judge(rig.cfg.glm_model, rig.base, "k")
    with pytest.raises(urllib.error.HTTPError):
        judge.verdict("ctx", "claim")
    assert len(rig.state["judge_calls"]) == judge_mod.RETRIES
    assert sum(sleeps) <= 15, "retry sleeps must stay bounded"
