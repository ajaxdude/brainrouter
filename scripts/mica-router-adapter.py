#!/usr/bin/env python3
"""mica-router-adapter — OpenAI-compatible routing-decision endpoint backed by Mica.

brainrouter's decision slot (kind: openai-compatible, generic prompt profile)
POSTs /v1/chat/completions whose user message is the features JSON
(route|tier|confidence rationale grammar). Mica answers only /v1/systemone
probability questions, so this adapter maps:

    features.last_user_excerpt + feature summary  ->  systemone `state`
    "needs the cloud model" noul question         ->  route  (p >= threshold -> cloud)
    reasoning-depth score question                ->  tier   (0 fast / 1 balanced / 2 deep)
    noul margin                                   ->  confidence

Run under systemd as mica-router-adapter.service; points at the mica runtime
on MICA_UPSTREAM (default http://127.0.0.1:8010). CPU-only by construction:
both processes are llama.cpp CPU builds; no GPU/VRAM is touched.

Config via env:
  MICA_UPSTREAM   upstream base URL (default http://127.0.0.1:8010)
  MICA_PORT       listen port (default 9210)
  MICA_CLOUD_THRESHOLD  noul probability at/above which the route is cloud
                        (default 0.55; deliberately >0.5 — "must pick local
                        unless the task is very difficult")
"""
import json
import os
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("MICA_UPSTREAM", "http://127.0.0.1:8010")
PORT = int(os.environ.get("MICA_PORT", "9210"))
CLOUD_THRESHOLD = float(os.environ.get("MICA_CLOUD_THRESHOLD", "0.55"))
TIMEOUT_S = float(os.environ.get("MICA_TIMEOUT_S", "30"))

TIER_BY_LEVEL = {0: "fast", 1: "balanced", 2: "deep"}


MAX_BODY_BYTES = 1 << 20  # 1 MiB


def _post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return json.load(resp)


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI multimodal parts
        return "".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _features_from_messages(body: dict) -> dict:
    """Generic profile puts the features JSON in the user message content."""
    for msg in reversed(body.get("messages", [])):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, dict):
            return content
        text = _content_to_text(content)
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {"last_user_excerpt": text}
        except json.JSONDecodeError:
            return {"last_user_excerpt": text}
    raise ValueError("no user message")


def _decide(features: dict) -> dict:
    excerpt = str(features.get("last_user_excerpt", ""))[:2000]
    facts = {k: features.get(k) for k in
             ("role", "approx_prompt_tokens", "msg_count", "has_tools", "code_density")}
    state = f"User request: {excerpt!r}\nContext: {json.dumps(facts)}"
    out = _post(f"{UPSTREAM}/v1/systemone", {
        "state": state,
        "questions": {
            "q": {"type": "noul",
                  "instructions": "Does this task require the cloud model to be handled well?"},
            "depth": {"type": "score",
                      "instructions": "How much reasoning depth does this task need?",
                      "criteria": ["simple one-step work",
                                   "moderate multi-step work",
                                   "deep architectural reasoning"]},
        },
    })
    ans = out.get("answers", {})
    q = ans.get("q")
    noul = q.get("noul") if isinstance(q, dict) else None
    if not isinstance(noul, (int, float)):
        # No usable probability: raise -> 503 -> slot treats it Unavailable
        # and falls back to the heuristic. Never emit a confident decision
        # from missing data (would defeat min_confidence gating).
        raise ValueError("mica response missing noul probability")
    p_cloud = float(noul)
    depth = ans.get("depth") if isinstance(ans.get("depth"), dict) else {}
    # Schema carries both: discrete `level` (int) and `expected_level` (float).
    raw = depth.get("level")
    if raw is None:
        raw = depth.get("expected_level")
    try:
        level = int(round(float(raw))) if raw is not None else 1
    except (TypeError, ValueError):
        level = 1
    tier = TIER_BY_LEVEL.get(max(0, min(2, level)), "balanced")
    route = "cloud" if p_cloud >= CLOUD_THRESHOLD else "local"
    # Confidence = probability mass of the CHOSEN route, so it is comparable
    # with the decision slot's min_confidence (default 0.4) on both sides.
    confidence = p_cloud if route == "cloud" else 1.0 - p_cloud
    return {
        "route": route,
        "tier": tier,
        "confidence": round(min(max(confidence, 0.0), 1.0), 3),
        "rationale": (f"mica p(cloud)={p_cloud:.3f} threshold={CLOUD_THRESHOLD} "
                      f"depth={depth.get('expected_level', depth.get('level', '?'))}"),
    }


log = sys.stderr


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_error(404, "only /v1/chat/completions")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 <= length <= MAX_BODY_BYTES:
                self.send_error(413, "invalid content length")
                return
            body = json.loads(self.rfile.read(length))
            decision = _decide(_features_from_messages(body))
            content = json.dumps(decision)
            envelope = {
                "id": "mica-adapter", "object": "chat.completion", "created": 0,
                "model": "mica-router-adapter",
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": content},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
            raw = json.dumps(envelope).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            try:
                self.wfile.write(raw)
            except OSError as exc:  # headers already sent; cannot send_error
                log.write(f"mica adapter write failed: {exc!r}\n")
        except Exception as exc:  # 503 -> slot sees Unavailable -> heuristic
            log.write(f"mica adapter failure: {exc!r}\n")
            log.flush()
            self.send_error(503, "mica adapter failure")

    def log_message(self, *args):  # quiet
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
