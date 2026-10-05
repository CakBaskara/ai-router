import json
import os
import queue
import subprocess
import threading

from . import attach, dispatch


class CodexProc:
    def __init__(self, model, effort, session):
        self.key = (model, effort)
        self.session = session
        self.proc = subprocess.Popen(dispatch.codex_server_cmd(), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.lock = threading.RLock()
        self.events = queue.Queue()
        self.requests = {}
        self.serial = 0
        self.active = False
        self.turn_id = None
        self.pending = {}
        self.deferred = []
        self.cancelled = False
        self.errors = []
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._drain, daemon=True).start()
        try:
            self._rpc("initialize", {"clientInfo": {"name": "ai-router", "title": "ai-router",
                                                   "version": "0.1.0"}})
            self._write({"method": "initialized", "params": {}})
            params = {"model": model, "cwd": os.getcwd(), "sandbox": "workspace-write",
                      "config": {"model_reasoning_effort": effort}}
            if session:
                params["threadId"] = session
            result = self._rpc("thread/resume" if session else "thread/start", params)
            self.session = result["thread"]["id"]
        except Exception:
            self.close()
            raise

    def _drain(self):
        for raw in self.proc.stderr:
            self.errors = (self.errors + [raw.decode("utf-8", "replace").strip()])[-5:]

    def _write(self, message):
        with self.lock:
            self.proc.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))
            self.proc.stdin.flush()

    def _request(self, method, params, wait=True):
        with self.lock:
            self.serial += 1
            request_id = self.serial
            response = {"ready": threading.Event()} if wait else None
            self.requests[request_id] = response
            try:
                self._write({"id": request_id, "method": method, "params": params})
            except (OSError, ValueError):
                self.requests.pop(request_id, None)
                raise
        return request_id, response

    def _rpc(self, method, params):
        request_id, response = self._request(method, params)
        if not response["ready"].wait(30):
            with self.lock:
                self.requests.pop(request_id, None)
            raise OSError(f"Codex {method} timed out")
        message = response["message"]
        if "error" in message:
            raise OSError(message["error"].get("message", "Codex request failed"))
        return message["result"]

    def _read(self):
        try:
            for raw in self.proc.stdout:
                try:
                    message = json.loads(raw.decode("utf-8", "replace"))
                except (json.JSONDecodeError, UnicodeError):
                    continue
                if "id" in message and "method" not in message:
                    with self.lock:
                        response = self.requests.pop(message["id"], None)
                        if response is not None:
                            response["message"] = message
                            response["ready"].set()
                        else:
                            self.events.put(message)
                elif "id" in message:
                    self._write({"id": message["id"], "error": {
                        "code": -32601, "message": "ai-router does not support this client request"}})
                else:
                    self.events.put(message)
        finally:
            with self.lock:
                for response in self.requests.values():
                    if response is not None:
                        response["message"] = {"error": {"message": "Codex app-server disconnected"}}
                        response["ready"].set()
                self.requests.clear()
            self.events.put(None)

    def alive(self):
        return self.proc.poll() is None

    @staticmethod
    def _input(prompt, files):
        inputs = [{"type": "text", "text": prompt}]
        inputs += [{"type": "localImage", "path": str(file)} for file in files if attach.is_image(file)]
        return inputs

    def _steer(self, inputs):
        request_id, _ = self._request("turn/steer", {"threadId": self.session,
                                                   "expectedTurnId": self.turn_id,
                                                   "input": inputs}, wait=False)
        self.pending[request_id] = inputs

    def _start(self, inputs):
        result = self._rpc("turn/start", {"threadId": self.session, "input": inputs,
                                          "model": self.key[0], "effort": self.key[1]})
        with self.lock:
            self.turn_id = result["turn"]["id"]
            messages, self.deferred = self.deferred, []
            for message in messages:
                self._steer(message)
            if self.cancelled:
                self._request("turn/interrupt", {"threadId": self.session,
                                                 "turnId": self.turn_id}, wait=False)

    def ask(self, prompt, turn, show, files=()):
        with self.lock:
            self.active = True
        turn.session = self.session
        completed = None
        try:
            self._start(self._input(prompt, files))
            while True:
                message = self.events.get()
                if message is None:
                    raise OSError(self.errors[-1] if self.errors else "Codex app-server disconnected")
                if "id" in message:
                    with self.lock:
                        inputs = self.pending.pop(message["id"], None)
                        if inputs and "error" in message:
                            error = message["error"].get("message", "Codex steering failed")
                            if "active turn" in error.lower() or "turn id" in error.lower():
                                self.deferred.append(inputs)
                            else:
                                raise OSError(error)
                else:
                    method, params = message.get("method"), message.get("params", {})
                    if params.get("threadId") != self.session:
                        continue
                    if params.get("turnId") and params["turnId"] != self.turn_id:
                        continue
                    if method == "turn/completed":
                        if params["turn"]["id"] != self.turn_id:
                            continue
                        with self.lock:
                            completed = params["turn"]
                            self.turn_id = None
                    else:
                        shown = turn.feed_app(method, params)
                        if shown:
                            show(*shown)
                with self.lock:
                    if completed is None or self.pending:
                        continue
                    if self.deferred and completed["status"] == "completed" and not self.cancelled:
                        inputs = [item for message_input in self.deferred for item in message_input]
                        self.deferred = []
                    else:
                        self.active = False
                        inputs = None
                if inputs is not None:
                    completed = None
                    self._start(inputs)
                    continue
                status = completed["status"]
                if status == "failed":
                    turn.error = (completed.get("error") or {}).get("message", "Codex turn failed")
                return 130 if status == "interrupted" else 1 if status == "failed" else 0
        except (OSError, ValueError, KeyError) as exc:
            turn.error = str(exc)
            self.close()
            return 1
        finally:
            with self.lock:
                self.active = False
                self.turn_id = None
                self.pending.clear()
                self.deferred = []

    def inject(self, prompt, files=()):
        with self.lock:
            if not self.active or self.cancelled or not self.alive():
                return False
            try:
                inputs = self._input(prompt, files)
                if self.turn_id is None:
                    self.deferred.append(inputs)
                else:
                    self._steer(inputs)
            except (OSError, ValueError):
                return False
            return True

    def cancel(self):
        with self.lock:
            self.cancelled = True
            if self.active and self.turn_id:
                self._request("turn/interrupt", {"threadId": self.session,
                                                 "turnId": self.turn_id}, wait=False)

    def close(self):
        if self.alive():
            self.proc.kill()
