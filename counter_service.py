        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(json.dumps(obj).encode(), "application/json", code)

    def do_POST(self):
        self.do_GET()

    def do_GET(self):
        path = self.path.split("?")[0]
        p = self._params()
        svc = SVC

        if path == "/status":
            self._json(svc.status())

        elif path == "/snapshot.jpg":
            cam = svc.camera()
            got = cam.wait_frame(-1, timeout=1.5) if cam and cam.online else None
            self._send(got[0] if got else _placeholder("NO CAMERA", "nothing attached"),
                       "image/jpeg")

        elif path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            seq = -1
            try:
                while True:
                    cam = svc.camera()          # re-resolved: the worker can be replaced
                    jpg = None
                    if cam is not None:
                        got = cam.wait_frame(seq, timeout=1.0)
                        if got:
                            jpg, seq = got
                    if jpg is None:
                        jpg = _placeholder("NO SIGNAL", "camera not responding")
                        time.sleep(0.5)
                    self.wfile.write(b"--FRAME\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(b"Content-Length: %d\r\n\r\n" % len(jpg))
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        elif path == "/run/start":
            self._json({"ok": True, "run": svc.start_run(p.get("product", "")[:80],
                                                          p.get("recipe", "")[:80])})
        elif path == "/run/stop":
            self._json({"ok": True, "run": svc.run.stop()})
        elif path == "/run/reset":
            svc.reader.clear()
            svc.transit = None
            self._json({"ok": True, "run": svc.run.reset()})

        elif path == "/gate/get":
            self._json(svc.gate)
        elif path == "/gate/set":
            self._json({"ok": True, "gate": svc.set_gate(p)})

        elif path == "/belt/learn":
            svc.counter.start_learning()
            self._json({"ok": True})

        elif path == "/config":
            if p:
                self._json({"ok": True, "thresholds": svc.set_thresholds(p)})
            else:
                self._json({"enter": svc.counter.enter, "leave": svc.counter.leave})

        elif path == "/code/set":
            self._json(svc.set_code_window(p))
        elif path == "/code/clear":
            self._json(svc.set_code_window({"clear": "1"}))

        elif path == "/camera/focus":
            self._json({"ok": svc.focus_sweep(), "focusing": True})
        elif path == "/camera/set":
            self._json({"ok": True, "controls": svc.set_camera(p)})
        elif path == "/camera/get":
            self._json(dict(svc.camera_controls))

        elif path == "/ocr/last.jpg":
            n = p.get("n")
            try:
                n = int(n) if n is not None else None
            except ValueError:
                n = None
            jpg = svc.reader.thumb(n)
            self._send(jpg if jpg else _placeholder("NO READ", "nothing read yet"), "image/jpeg")

        elif path == "/ocr/pack.jpg":
            with svc.reader.lock:
                lp = svc.reader.last_pack
            self._send(lp["jpg"] if lp else _placeholder("NO PACK", "pass a pack first"), "image/jpeg")

        elif path == "/ocr/frame.jpg":
            with svc.reader.lock:
                jpg = svc.reader.frame_jpg
            self._send(jpg if jpg else _placeholder("NO READ", "nothing read yet"), "image/jpeg")

        elif path == "/":
            self._send(PAGE, "text/html; charset=utf-8")
        else:
            self._json({"ok": False, "error": "not found"}, 404)


if __name__ == "__main__":
    SVC = Service()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.daemon_threads = True
    print("[counter] serving on http://0.0.0.0:%d/" % PORT, flush=True)
    srv.serve_forever()
