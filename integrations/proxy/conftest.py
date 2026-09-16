# Shared test infrastructure: fake upstream + proxy rig fixture.
# Present so pytest prepends this directory to sys.path (ragfaith_proxy import).

import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from ragfaith_proxy.proxy import Config, make_server

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
        forced = self.server.state.get("models_status")
        if forced is not None:
            self.send_response(forced)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")
            return
        self._send_json({"object": "list", "data": [{"id": "fake-model"}]})

    def do_POST(self):
        state = self.server.state
        with state["lock"]:
            state["auths"].append(self.headers.get("Authorization", ""))
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("model") in (state["main"], state["fallback"]):
            self._judge(state, body)
        else:
            self._completions(state, body)

    def _judge(self, state, body):
        with state["lock"]:
            state["judge_calls"].append(body)
            state["judge_active"] = state.get("judge_active", 0) + 1
            state["judge_max_conc"] = max(state.get("judge_max_conc", 0), state["judge_active"])
            fail = state["judge_fail_status"]
            rate_left = state.get("judge_429_left", 0)
            if rate_left > 0:
                state["judge_429_left"] = rate_left - 1
                self._send_json({"error": "slow down"}, status=429)
                return
            verdict = state["verdicts"].pop(0) if state["verdicts"] else "faithful"
        time.sleep(state.get("judge_delay", 0.0))
        with state["lock"]:
            state["judge_active"] -= 1
        if fail:
            self._send_json({"error": "judge upstream boom"}, status=fail)
            return
        resp = {
            "choices": [
                {
                    "message": {
                        "content": (
                            state["judge_content"]
                            if state.get("judge_content") is not None
                            else json.dumps({"verdict": verdict})
                        )
                    },
                    "finish_reason": "stop",
                }
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
def rig(request):
    judge_log = getattr(request, "param", None)
    state = {
        "lock": threading.Lock(),
        "requests": [],
        "judge_calls": [],
        "verdicts": [],
        "script": [],
        "delay": 0.0,
        "judge_delay": 0.0,
        "judge_429_left": 0,
        "judge_content": None,
        "auths": [],
        "fail_status": None,
        "fail_from": None,
        "judge_fail_status": None,
        "main": "test/glm-judge",
        "fallback": "test/ds-judge",
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
        main_model=state["main"],
        fallback_model=state["fallback"],
        judge_log=judge_log,
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


# ---------------------------------------------------------------- parallel judging


def test_judge_calls_run_in_parallel(rig):
    rig.state["judge_delay"] = 0.4
    rig.state["verdicts"] = ["unfaithful"] * 3
    rig.state["script"] = [{"stream": [chunk("Alpha. Beta. Gamma.", finish="stop"), DONE_LINE]}]
    conn, resp = post(
        rig,
        {
            "model": "gpt-4o",
            "stream": True,
            "messages": [
                {"role": "assistant", "content": "", "tool_calls": []},
                {
                    "role": "tool",
                    "tool_call_id": "t1",
                    "content": "Alpha is A. Beta is B. Gamma is G.",
                },
                {"role": "user", "content": "q"},
            ],
        },
        headers={"X-Conversation-Id": "par"},
    )
    start = time.monotonic()
    list(iter_sse_lines(resp))
    conn.close()
    elapsed = time.monotonic() - start
    with rig.state["lock"]:
        n = len(rig.state["judge_calls"])
        max_conc = rig.state.get("judge_max_conc", 0)
    assert n == 3
    assert max_conc >= 2  # overlapped, not sequential
    assert elapsed < 3 * 0.4  # sequential would be >= 1.2s


# ---------------------------------------------------------------- regen mode


def _regen_payload(stream=True):
    return {
        "model": "gpt-4o",
        "stream": stream,
        "messages": [
            {"role": "user", "content": "Where is the tower? Use the source."},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "tool_call_id": "t1", "content": "The tower is in Paris, France."},
            {"role": "user", "content": "Answer using the source."},
        ],
    }


def test_regen_faithful_replays_original_only(rig):
    rig.cfg.nudge_mode = "regen"
    rig.state["script"] = [{"stream": [chunk("It is in Paris.", finish="stop"), DONE_LINE]}]
    conn, resp = post(rig, _regen_payload(), headers={"X-Conversation-Id": "regen-f"})
    body = "".join(iter_sse_lines(resp))
    conn.close()
    assert "It is in Paris." in body
    with rig.state["lock"]:
        assert len(rig.state["requests"]) == 1  # no internal regen call
        assert len(rig.state["judge_calls"]) == 1


def test_regen_flagged_client_sees_only_regen(rig):
    rig.cfg.nudge_mode = "regen"
    rig.state["verdicts"] = ["unfaithful"]
    rig.state["script"] = [
        {"stream": [chunk("It is in Munich.", finish="stop"), DONE_LINE]},
        {"json": full_msg("It is in Paris, France.")},  # internal regen call
    ]
    conn, resp = post(rig, _regen_payload(), headers={"X-Conversation-Id": "regen-u"})
    body = "".join(iter_sse_lines(resp))
    conn.close()
    assert "It is in Paris, France." in body  # regenerated text
    assert "Munich" not in body  # original never shown
    assert "ragfaith judge" not in body  # nudge never shown
    assert body.count("[DONE]") == 1
    with rig.state["lock"]:
        assert len(rig.state["requests"]) == 2
        internal = rig.state["requests"][1]
        assert internal["messages"][-1]["role"] == "user"
        nudge = internal["messages"][-1]["content"]
        # regen template: directive, silent-correction contract
        assert "Do not reply to this message" in nudge
        assert "original query" in nudge
        assert "ONLY that answer" in nudge
        assert not internal.get("stream")


def test_regen_false_positive_gets_direct_answer_only(rig):
    # judge false positive: model must still output one clean answer to the
    # user, never a defense against the nudge (no tables, no meta)
    rig.state["verdicts"] = ["unfaithful"]
    rig.state["script"] = [
        {"stream": [chunk("It is in Paris, France.", finish="stop"), DONE_LINE]},
        {
            "json": full_msg(
                "It is in Paris, France. The sources pulled in this "
                "conversation place the tower in Paris."
            )
        },
    ]
    conn, resp = post(rig, _regen_payload(), headers={"X-Conversation-Id": "regen-fp"})
    body = "".join(iter_sse_lines(resp))
    conn.close()
    # no judge/verification meta anywhere in the client-visible output
    assert "ragfaith" not in body.lower()
    assert "judge" not in body.lower()
    assert "claim" not in body.lower()
    assert "re-check" not in body.lower()
    assert "It is in Paris, France." in body


def test_regen_plain_client_flagged(rig):
    rig.cfg.nudge_mode = "regen"
    rig.state["verdicts"] = ["unfaithful"]
    rig.state["script"] = [
        {"json": full_msg("It is in Munich.")},
        {"json": full_msg("It is in Paris, France.")},
    ]
    conn, resp = post(rig, _regen_payload(stream=False), headers={"X-Conversation-Id": "regen-p"})
    data = json.loads(resp.read())
    conn.close()
    assert data["choices"][0]["message"]["content"] == "It is in Paris, France."


# pytest patches judge_mod.RETRIES in some tests; keep a handle importable
