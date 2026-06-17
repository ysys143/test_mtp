#!/usr/bin/env python3
# enable_thinking 주입 프록시: 모든 /chat/completions 요청에 chat_template_kwargs.enable_thinking 강제.
# ENABLE_THINK 환경변수 = true|false (기본 false). lm-eval -> 이 프록시(:8001) -> vLLM(:8000).
import json, os, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
UP = "http://localhost:8000"
THINK = os.environ.get("ENABLE_THINK", "false").lower() == "true"
class H(BaseHTTPRequestHandler):
    def _fwd(self, method):
        l = int(self.headers.get("Content-Length", 0)); body = self.rfile.read(l) if l else None
        if body and self.path.endswith("/chat/completions"):
            try:
                o = json.loads(body); k = o.setdefault("chat_template_kwargs", {})
                k["enable_thinking"] = THINK; body = json.dumps(o).encode()
            except Exception: pass
        req = urllib.request.Request(UP + self.path, data=body, headers={"Content-Type": "application/json"}, method=method)
        try:
            r = urllib.request.urlopen(req, timeout=3600); data = r.read(); code = r.status
        except urllib.error.HTTPError as e:
            data = e.read(); code = e.code
        except Exception as e:
            data = json.dumps({"error": str(e)}).encode(); code = 502
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_POST(self): self._fwd("POST")
    def do_GET(self): self._fwd("GET")
    def log_message(self, *a): pass
ThreadingHTTPServer(("0.0.0.0", 8001), H).serve_forever()
