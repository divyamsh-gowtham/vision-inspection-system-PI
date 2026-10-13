"""
Validate the reader on a synthetic pack: a dark pouch on a light belt with a
white batch patch printed in three lines, presented in each orientation.

What must hold:
  1. the patch is located on the pack (not the belt, not the artwork);
  2. the batch code is read at a usable level in every orientation, and the
     winning rotation is reported so the next pack is read first time;
  3. compare() accepts the right batch and rejects a one-character-off date;
  4. a pack with no print reports an error rather than inventing a code.

Run:  python3 tests_ocr.py
"""
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/home/jark/proinspect")
from ocr import read_pack, compare, find_codes  # noqa: E402

W, H = 1280, 720
CODE = "623304515A  10:20 B\nAUG/26 - JUL/27"
rng = np.random.default_rng(5)


def belt():
    img = np.full((H, W, 3), 205, np.uint8)
    noise = rng.normal(0, 3, (H, W, 1))
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def pouch(with_patch=True, rotate=0):
    """A metallised pouch (mid-dark, textured) with a white batch patch."""
    img = belt()
    pw, ph = 420, 300
    x0, y0 = (W - pw) // 2, (H - ph) // 2
    body = np.full((ph, pw, 3), (95, 70, 60), np.uint8)
    body = np.clip(body + rng.normal(0, 6, (ph, pw, 1)), 0, 255).astype(np.uint8)
    # some artwork: a logo block and stripes, so the patch is not the only feature
    cv2.rectangle(body, (30, 30), (170, 110), (40, 120, 200), -1)
    cv2.putText(body, "EVERYDAY", (36, 85), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    for i in range(0, pw, 28):
        cv2.line(body, (i, ph - 40), (i + 14, ph - 10), (120, 90, 80), 2)
    if with_patch:
        px, py, pwid, phgt = 210, 150, 190, 110
        cv2.rectangle(body, (px, py), (px + pwid, py + phgt), (246, 246, 246), -1)
        cv2.putText(body, "623304515A 10:20 B", (px + 8, py + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 25), 1, cv2.LINE_AA)
        cv2.putText(body, "AUG/26 - JUL/27", (px + 8, py + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 25), 1, cv2.LINE_AA)
    if rotate:
        turns = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}[rotate]
        body = cv2.rotate(body, turns)
    bh, bw = body.shape[:2]
    x0, y0 = (W - bw) // 2, (H - bh) // 2
    img[y0:y0 + bh, x0:x0 + bw] = body
    return img, (x0 - 20, y0 - 20, bw + 40, bh + 40)


results = []


def check(label, ok, detail=""):
    results.append(ok)
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", label, ("  (" + detail + ")") if detail else ""))


# 1 + 2. read in every orientation ------------------------------------------
for rot in (0, 90, 180, 270):
    frame, pack = pouch(rotate=rot)
    t0 = time.time()
    r = read_pack(frame, pack, expected=CODE)
    ms = int((time.time() - t0) * 1000)
    f = r["fields"]
    got_lot = f.get("lot", "").startswith("623304515")
    got_dates = len(f.get("dates", [])) == 2
    check("rotation %3d: code read (lot %s, dates %s) via %s in %d ms, %d reads, rot %d" % (
        rot, f.get("lot"), f.get("dates"), r["via"], ms, r["reads"], r["rotate"]),
        got_lot and got_dates and r["match"]["ok"],
        repr(r["text"]) + " ratio %.2f" % r["match"]["ratio"])
    px, py, pw, ph = pack
    inside = (r["box"] is not None
              and px <= r["box"]["x"] * W and (r["box"]["x"] + r["box"]["w"]) * W <= px + pw
              and py <= r["box"]["y"] * H and (r["box"]["y"] + r["box"]["h"]) * H <= py + ph)
    check("rotation %3d: patch located on the pack" % rot, inside, str(r["box"]))

# rotation hint makes the next read a single Tesseract call
frame, pack = pouch(rotate=90)
r1 = read_pack(frame, pack, expected=CODE)
r2 = read_pack(frame, pack, rotations=[r1["rotate"]] + [x for x in (0, 90, 180, 270) if x != r1["rotate"]], expected=CODE)
check("rotation hint -> one read", r2["reads"] == 1 and r2["match"]["ok"], "%d reads" % r2["reads"])

# 3. compare ------------------------------------------------------------------
check("right batch accepted", compare("623304515A 10:20 B\nAUG/26 - JUL/27", CODE)["ok"])
check("one-digit-off lot rejected", not compare("623304516A 10:20 B\nAUG/26 - JUL/27", CODE)["ok"])
check("wrong month rejected", not compare("623304515A 10:20 B\nSEP/26 - JUL/27", CODE)["ok"])
check("nothing read rejected", not compare("", CODE)["ok"])
check("no expected -> not checked", not compare("anything", "")["checked"])
check("fields parsed", find_codes(CODE)["lot"] == "623304515A" and find_codes(CODE)["time"] == "10:20")

# 4. a pack with no print ----------------------------------------------------
frame, pack = pouch(with_patch=False)
r = read_pack(frame, pack, expected=CODE)
check("no batch patch -> no code-shaped read", r["code_score"] == 0 and not r["match"]["ok"],
      "text %r error %r" % (r["text"], r["error"]))

# 5. the reader thread end to end: submit a JPEG, get a result and a thumbnail
import counter_service as cs  # noqa: E402
cs.DEBUG_ON = False                       # keep test packs out of the live debug ring
frame, pack = pouch(rotate=90)
ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
run = cs.RunState(cs.CONFIG / "run_test.json")
run.start("PRD-1002", "RCP-1201", expected=CODE)
rd = cs.Reader(run)
rd.start()
track = {"x": pack[0] / W, "y": pack[1] / H, "w": pack[2] / W, "h": pack[3] / H}
rd.submit(1, 1, enc.tobytes(), track, 1.0)
for _ in range(100):
    st = rd.status()["ocr"]
    if st.get("state") == "done":
        break
    time.sleep(0.1)
check("reader thread: result published", st.get("state") == "done" and st.get("match", {}).get("ok") is True,
      "%s conf %s in %s ms, latency %s ms" % (repr(st.get("text")), st.get("conf"), st.get("ms"), st.get("latency_ms")))
check("reader thread: thumbnail available", rd.thumb(1) is not None and len(rd.thumb(1)) > 500)
check("reader thread: rotation hint learned", rd.rotate_hint == 90, "hint %s" % rd.rotate_hint)
frame2, pack2 = pouch(with_patch=False)
ok, enc2 = cv2.imencode(".jpg", frame2, [cv2.IMWRITE_JPEG_QUALITY, 85])
rd.submit(2, 1, enc2.tobytes(), track, 1.0)
for _ in range(400):                      # a gibberish pack costs up to MAX_READS reads
    st = rd.status()["ocr"]
    if st.get("state") == "done" and st.get("n") == 2:
        break
    time.sleep(0.1)
check("reader thread: gibberish is reported as no read, not as a code",
      st.get("n") == 2 and st.get("text") == "" and bool(st.get("error")),
      "text %r error %r raw conf %s" % (st.get("text"), st.get("error"), st.get("conf")))
try:
    (cs.CONFIG / "run_test.json").unlink()
except OSError:
    pass

# 6. the GPU pipeline, when the engines have been built ------------------------
try:
    import ppocr
    GPU = ppocr.available()
except Exception:
    GPU = False
if GPU:
    from ocr import read_pack_gpu
    eng = ppocr.PPOCR()
    for rot in (0, 90, 180, 270):
        frame, pack = pouch(rotate=rot)
        t0 = time.time()
        r = read_pack_gpu(eng, frame, pack, expected=CODE)
        ms = int((time.time() - t0) * 1000)
        check("GPU rotation %3d: code read in %d ms (det %s cls %s rec %s ms, %s boxes)" % (
            rot, ms, r.get("timing", {}).get("det_ms"), r.get("timing", {}).get("cls_ms"),
            r.get("timing", {}).get("rec_ms"), r.get("timing", {}).get("boxes")),
            r["match"]["ok"], "%r conf %s via %s all=%s" % (r["text"], r["conf"], r["via"],
                                                            [l["text"] for l in r.get("all_lines", [])]))
    frame, pack = pouch(with_patch=False)
    r = read_pack_gpu(eng, frame, pack, expected=CODE)
    check("GPU: pack without a batch patch does not match", not r["match"]["ok"],
          "%r via %s" % (r["text"], r["via"]))
else:
    print("  (GPU engines not built; skipping the TensorRT checks)")

print("\n%d/%d checks passed" % (sum(results), len(results)))
sys.exit(0 if all(results) else 1)
