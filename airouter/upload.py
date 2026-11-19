import hmac
import os
import secrets
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import unquote

from . import attach

PAGE = """<!doctype html>
<html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>sela · lampirkan</title>
<style>
body { margin: 0; background: #1F1F1F; color: #CCCCCC; font: 16px system-ui, sans-serif; }
main { max-width: 480px; margin: 0 auto; padding: 24px 16px; }
h1 { font-size: 20px; color: #FFFFFF; margin: 0 0 4px; }
p { color: #9D9D9D; margin: 0 0 20px; }
label { display: block; padding: 18px; text-align: center; border-radius: 14px; background: #0078D4;
        color: #FFFFFF; font-weight: 600; font-size: 17px; }
input { display: none; }
ul { list-style: none; padding: 0; margin: 20px 0 0; }
li { padding: 12px 14px; margin-bottom: 8px; border-radius: 10px; background: #2B2B2B; overflow-wrap: anywhere; }
li.ok { color: #7EE787; } li.bad { color: #F85149; }
textarea { box-sizing: border-box; width: 100%; min-height: 96px; margin-top: 12px; padding: 12px;
           border: 1px solid #454545; border-radius: 12px; background: #2B2B2B; color: #FFFFFF;
           font: 16px system-ui, sans-serif; }
button { width: 100%; margin-top: 12px; padding: 16px; border: 0; border-radius: 14px; background: #2EA043;
         color: #FFFFFF; font-weight: 600; font-size: 17px; }
button:disabled { background: #454545; color: #9D9D9D; }
#done { display: none; text-align: center; padding-top: 40px; }
</style></head><body><main>
<div id="form">
<h1>✻ sela</h1>
<p>Pilih file, tulis instruksi, lalu kirim. Popup di sela tertutup sendiri.</p>
<label for="files">Pilih file atau foto</label>
<input id="files" type="file" multiple>
<ul id="list"></ul>
<textarea id="text" placeholder="Instruksi untuk file ini (boleh kosong)"></textarea>
<button id="send">Kirim ke sela</button>
</div>
<div id="done"><h1>✓ Terkirim</h1><p>Kembali ke Termius untuk melihat jawabannya.</p></div>
</main><script>
const list = document.getElementById("list");
const base = location.pathname.replace(/\\/$/, "");
let pending = Promise.resolve();
document.getElementById("files").addEventListener("change", (event) => {
  const files = Array.from(event.target.files);
  event.target.value = "";
  pending = pending.then(() => upload(files));
});
document.getElementById("send").addEventListener("click", async (event) => {
  const button = event.target;
  button.disabled = true;
  await pending;
  try {
    const reply = await fetch(base + "/done", {method: "POST", body: document.getElementById("text").value});
    if (!reply.ok) throw new Error(await reply.text());
  } catch (error) {
    button.disabled = false;
    button.textContent = "Gagal, coba lagi";
    return;
  }
  document.getElementById("form").style.display = "none";
  document.getElementById("done").style.display = "block";
  window.close();
});
async function upload(files) {
  for (const file of files) {
    const row = document.createElement("li");
    row.textContent = "Mengirim " + file.name + "…";
    list.prepend(row);
    try {
      const reply = await fetch(base + "/file", {
        method: "POST", body: file, headers: {"X-Name": encodeURIComponent(file.name)}});
      const text = await reply.text();
      row.textContent = (reply.ok ? "✓ " : "✕ ") + file.name + (reply.ok ? "" : " · " + text);
      row.className = reply.ok ? "ok" : "bad";
    } catch (error) {
      row.textContent = "✕ " + file.name + " · koneksi putus";
      row.className = "bad";
    }
  }
}
</script></body></html>
"""


def reachable_address() -> str:
    parts = os.environ.get("SSH_CONNECTION", "").split()
    if len(parts) == 4:
        return parts[2]
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


class Server:
    def __init__(self, on_file: Callable[[Path], None], host: str | None = None, port: int = 0,
                 on_done: Callable[[str], None] | None = None):
        self.token = secrets.token_urlsafe(9)
        self.on_file = on_file
        self.on_done = on_done
        self.host = host or reachable_address()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _allowed(self, suffix: str) -> bool:
                path = self.path.split("?", 1)[0].rstrip("/")
                return hmac.compare_digest(path.encode(), f"/{server.token}{suffix}".encode())

            def _reply(self, code: int, body: str, kind: str = "text/plain; charset=utf-8"):
                data = body.encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self._allowed(""):
                    self._reply(200, PAGE, "text/html; charset=utf-8")
                else:
                    self._reply(404, "tidak ditemukan")

            def do_POST(self):
                if self._allowed("/done"):
                    size = int(self.headers.get("Content-Length") or 0)
                    if size > 100_000:
                        self._reply(413, "instruksi terlalu panjang")
                        return
                    text = self.rfile.read(size).decode("utf-8", "replace")
                    if server.on_done:
                        server.on_done(text)
                    self._reply(200, "ok")
                    return
                if not self._allowed("/file"):
                    self._reply(404, "tidak ditemukan")
                    return
                try:
                    size = int(self.headers.get("Content-Length", ""))
                except ValueError:
                    self._reply(411, "ukuran file tidak diketahui")
                    return
                if size > attach.MAX_BYTES:
                    self._reply(413, "lebih dari 20 MB")
                    return
                name = Path(unquote(self.headers.get("X-Name", "") or "file")).name
                try:
                    path = attach.save_bytes(name, self.rfile.read(size))
                    server.on_file(path)
                except (OSError, ValueError) as exc:
                    self._reply(400, str(exc))
                    return
                self._reply(200, "ok")

        self.httpd = ThreadingHTTPServer((self.host, port), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.httpd.server_address[1]}/{self.token}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
