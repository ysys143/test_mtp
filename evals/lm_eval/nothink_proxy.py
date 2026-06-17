import json, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
UP="http://localhost:8000"
class H(BaseHTTPRequestHandler):
    def _fwd(self, method):
        l=int(self.headers.get("Content-Length",0)); body=self.rfile.read(l) if l else None
        if body and self.path.endswith("/chat/completions"):
            try:
                o=json.loads(body); k=o.setdefault("chat_template_kwargs",{}); k["enable_thinking"]=False; body=json.dumps(o).encode()
            except Exception: pass
        req=urllib.request.Request(UP+self.path, data=body, headers={"Content-Type":"application/json"}, method=method)
        try:
            r=urllib.request.urlopen(req, timeout=1800); data=r.read(); code=r.status
        except urllib.error.HTTPError as e:
            data=e.read(); code=e.code
        except Exception as e:
            data=json.dumps({"error":str(e)}).encode(); code=502
        self.send_response(code); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_POST(self): self._fwd("POST")
    def do_GET(self): self._fwd("GET")
    def log_message(self,*a): pass
ThreadingHTTPServer(("0.0.0.0",8001),H).serve_forever()
