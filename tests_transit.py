"""
Drive the service's own frame handler with a synthetic transit at the camera's
real resolution, so the count -> best-frame -> reader -> thumbnail path is
exercised exactly as the capture thread exercises it. No camera needed.

What must hold: one count, a read fired while the pack was still in view, the
read matches the expected code, and a thumbnail is served for that pack.

Run:  python3 tests_transit.py
"""
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/home/jark/proinspect")
import counter_service as cs  # noqa: E402

# Never touch the live appliance's learned belt or its debug ring from a test.
cs.BELT_FILE = cs.CONFIG / "belt_transit_test.png"
cs.DEBUG_ON = False

W, H = cs.CAM_W, cs.CAM_H
CODE = "623304515A  10:20 B\nAUG/26 - JUL/27"
rng = np.random.default_rng(9)


def belt():
    img = np.full((H, W, 3), 205, np.uint8)
    return np.clip(img + rng.normal(0, 3, (H, W, 1)), 0, 255).astype(np.uint8)


def pack_at(cx):
    """A dark pouch with a white batch patch, centred at cx, sized for this resolution."""
    img = belt()
    pw, ph = int(W * 0.30), int(H * 0.36)
    x0, y0 = int(cx - pw / 2), (H - ph) // 2
    body = np.full((ph, pw, 3), (95, 70, 60), np.uint8)
    body = np.clip(body + rng.normal(0, 6, (ph, pw, 1)), 0, 255).astype(np.uint8)
    cv2.rectangle(body, (int(pw * .08), int(ph * .1)), (int(pw * .4), int(ph * .35)), (40, 120, 200), -1)
    px, py, pwid, phgt = int(pw * .5), int(ph * .5), int(pw * .45), int(ph * .3)
    cv2.rectangle(body, (px, py), (px + pwid, py + phgt), (246, 246, 246), -1)
    sc = phgt / 110.0
    cv2.putText(body, "623304515A 10:20 B", (px + int(8 * sc), py + int(34 * sc)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * sc, (25, 25, 25), max(1, int(1.5 * sc)), cv2.LINE_AA)
    cv2.putText(body, "AUG/26 - JUL/27", (px + int(8 * sc), py + int(72 * sc)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * sc, (25, 25, 25), max(1, int(1.5 * sc)), cv2.LINE_AA)
    # paste with clipping at the frame edges
    sx0, sx1 = max(0, -x0), min(pw, W - x0)
    if sx1 > sx0:
        img[y0:y0 + ph, x0 + sx0:x0 + sx1] = body[:, sx0:sx1]
    return img


def jpeg(img):
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return enc.tobytes()


# A Service without its camera supervisor: the pieces the frame handler needs.
svc = cs.Service.__new__(cs.Service)
svc.gate = {"x": 0.30, "y": 0.25, "w": 0.40, "h": 0.50}
svc.counter = cs.ProductCounter(svc.gate)
svc.run = cs.RunState(cs.CONFIG / "run_transit_test.json")
svc.counter.on_count = svc.run.bump
svc.reader = cs.Reader(svc.run)
try:
    import ppocr
    if ppocr.available():
        svc.reader.gpu = ppocr.PPOCR()
        svc.reader.engine_name = "gpu"
        print("  (reader: PP-OCRv4 on TensorRT)")
except Exception as exc:
    print("  (reader: tesseract - %s)" % exc)
svc.reader.start()
svc.transit = None
svc.cam = None
svc.cam_lock = __import__("threading").Lock()
svc.run.start("PRD-1002", "RCP-1201", expected=CODE)

results = []


def check(label, ok, detail=""):
    results.append(ok)
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", label, ("  (" + detail + ")") if detail else ""))


t0 = time.time()
svc.counter.start_learning()
for _ in range(cs.LEARN_FRAMES + 2):
    svc._on_frame(jpeg(belt()), time.time())
check("belt learned from synthetic frames", svc.counter.ref is not None and not svc.counter.learning())

# one pack crossing left to right in 30 frames, then empty belt
fired_at = None
n_frames = 30
for i in range(n_frames):
    cx = -W * 0.2 + i * (W * 1.4) / (n_frames - 1)
    svc._on_frame(jpeg(pack_at(cx)), time.time())
    if fired_at is None and svc.reader.status()["ocr"].get("state") in ("reading", "done"):
        fired_at = i
for _ in range(12):
    svc._on_frame(jpeg(belt()), time.time())
check("exactly one count", svc.run.snapshot()["count"] == 1, "count %d" % svc.run.snapshot()["count"])
check("read fired while the pack was in view", fired_at is not None and fired_at < n_frames - 4,
      "fired at transit frame %s of %d" % (fired_at, n_frames))

for _ in range(300):
    st = svc.reader.status()["ocr"]
    if st.get("state") == "done" and svc.reader.status()["ocr_pending"] == 0 and svc.transit is None:
        break
    time.sleep(0.1)
st = svc.reader.status()["ocr"]
check("read matches the expected code", st.get("state") == "done" and (st.get("match") or {}).get("ok") is True,
      "%r conf %s rot %s in %s ms (%s reads), latency %s ms" % (
          st.get("text"), st.get("conf"), st.get("rotate"), st.get("ms"), st.get("reads"), st.get("latency_ms")))
check("thumbnail served for pack 1", svc.reader.thumb(1) is not None)
if st.get("engine") == "gpu":
    check("GPU read under 250 ms", (st.get("ms") or 9999) < 250, "%s ms, latency %s ms" % (st.get("ms"), st.get("latency_ms")))
else:
    check("glyphs measured at this resolution", (st.get("char_px") or 0) >= 12, "char_px %s upscale %s" % (st.get("char_px"), st.get("upscale")))
print("  (whole test %.1f s)" % (time.time() - t0))

for f in (cs.CONFIG / "run_transit_test.json", cs.BELT_FILE):
    try:
        f.unlink()
    except OSError:
        pass
print("\n%d/%d checks passed" % (sum(results), len(results)))
sys.exit(0 if all(results) else 1)
