import os

from mitmproxy import http
import json
import time
import os

ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROXY_LOG_DIR = os.path.join(ROOT_DIR, "proxy_logs")
os.makedirs(PROXY_LOG_DIR, exist_ok=True)

MAX_BODY_BYTES = 32_000  # keep logs small; adjust as needed


def _safe_text(content: bytes | None) -> str | None:
    if not content:
        return None
    try:
        return json.dumps(content.decode("utf-8", errors="replace"))[:MAX_BODY_BYTES]
    except Exception:
        return None


class ReverseLogger:
    def __init__(self):
        path = os.path.join(PROXY_LOG_DIR, "http_traffic.jsonl")
        self.out = open(path, "a", buffering=1)

    def request(self, flow: http.HTTPFlow) -> None:
        r = flow.request
        # openai hack
        if r.path == "/chat/completions" and "openai" in r.host:
            r.path = "/v1/chat/completions"
        entry = {
            "ts": time.time(),
            "event": "request",
            "method": r.method,
            "scheme": r.scheme,
            "host": r.host,
            "port": r.port,
            "path": r.path,
            "http_version": r.http_version,
            "headers": dict(r.headers),
            "body_sample": _safe_text(r.raw_content),
        }
        self.out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def response(self, flow: http.HTTPFlow) -> None:
        if not flow.response:
            return
        s = flow.response
        entry = {
            "ts": time.time(),
            "event": "response",
            "status_code": s.status_code,
            "reason": s.reason,
            "http_version": s.http_version,
            "headers": dict(s.headers),
            "body_sample": _safe_text(s.raw_content),
        }
        self.out.write(json.dumps(entry, ensure_ascii=False) + "\n")



addons = [ReverseLogger()]
