#!/usr/bin/env python3
"""
Find the printed code on a pack and read it.

Two jobs, in the order a person would do them:

  1. LOCATE. Packaging codes are printed on a light patch - a white panel, a
     label, a blank area of film - because that is what makes them legible.
     Inside the pack, find the bright regions, then the dense small dark marks
     inside them. That is print. Rank by how much print and how tightly it
     clusters, then let the READ decide between the strongest candidates,
     because ink density alone picked the nutrition table over the batch panel
     on the real pack.
  2. READ. Tesseract through its binary: there is no pip on this appliance and
     no python binding in apt. A subprocess call is a few milliseconds against
     the ~170 ms the read itself takes.

Everything here was measured on the real Nestle EVERYDAY pouch in the first
build: the batch panel is three short lines and comes out nearly square, so
orientation cannot be guessed from its shape - it is read in the orientations
allowed and the one that yields code-shaped fields wins. The winning rotation
is returned so the caller can try it first on the next pack.

    from ocr import read_pack
    r = read_pack(frame_bgr, pack_box_px, rotations=(0, 90, 180, 270))
    r["text"], r["fields"], r["conf"], r["rotate"], r["box"], r["ms"]
"""

import difflib
import os
import re
import subprocess
import tempfile
import time

import cv2
import numpy as np

TESSERACT = "/usr/bin/tesseract"

# The character set a packaging code actually uses. Constraining it stops
# Tesseract inventing punctuation out of pouch creases and foil glare.
WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/:.-() "

# Tesseract wants roughly 30 px of cap height; small panels are upscaled.
TARGET_CAP_PX = 34
MAX_PREP_PX = 700_000              # cap on the upscaled image handed to Tesseract
NLMEANS_MAX_PX = 120_000           # non-local means only on small crops; median above
READ_TIMEOUT_S = 4.0               # one Tesseract call
READ_BUDGET_S = 6.0                # all reads for one pack

# A code region has to be at least this fraction of the frame to be worth
# reading, and no more than this much or we have found the whole pack.
MIN_AREA_FRAC = 0.0006
MAX_AREA_FRAC = 0.25
# Print is small, dark and repeated. Blobs outside this range are artwork,
# barcodes, or noise.
MIN_BLOB_PX = 12
MAX_BLOB_FRAC = 0.02


# ---------------------------------------------------------------------- read
def prepare(gray, rotate=0, upscale=None):
    """Rotate upright, upscale, flatten the lighting, binarise."""
    img = gray
    if rotate:
        turns = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(int(rotate) % 360)
        if turns is not None:
            img = cv2.rotate(img, turns)
    if upscale is None:
        upscale = max(1.0, min(4.0, TARGET_CAP_PX / max(img.shape[0] / 6.0, 1.0)))
    # Bound the work: a x4 upscale of a big panel made a 2400x1200 image and the
    # denoiser below took seconds on it. Tesseract needs ~30 px glyphs, not
    # megapixels; anything past MAX_PREP_PX buys nothing.
    area = img.shape[0] * img.shape[1] * upscale * upscale
    if area > MAX_PREP_PX:
        upscale = max(1.0, (MAX_PREP_PX / (img.shape[0] * img.shape[1])) ** 0.5)
    if upscale > 1.01:
        img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    # Divide out the slow-varying illumination: keeps glare from eating strokes.
    bg = cv2.GaussianBlur(img, (0, 0), sigmaX=max(img.shape[1] // 14, 3))
    flat = cv2.divide(img, bg, scale=255)
    # Non-local means is worth it on small crops (foil texture) but is O(pixels
    # x window) and crawls past ~250k px; a median does the job on big ones.
    if flat.shape[0] * flat.shape[1] <= NLMEANS_MAX_PX:
        flat = cv2.fastNlMeansDenoising(flat, None, 7, 7, 21)
    else:
        flat = cv2.medianBlur(flat, 3)
    return cv2.adaptiveThreshold(flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 11)


def read(gray, rotate=0, psm=6, whitelist=WHITELIST, upscale=None):
    """
    Read a window. Returns {"text", "lines", "conf", "words", "error"}.

    Never raises: OCR failing must not stop the line.
    """
    out = {"text": "", "lines": [], "conf": 0.0, "words": [], "error": ""}
    try:
        img = prepare(gray, rotate=rotate, upscale=upscale)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as fh:
            cv2.imwrite(fh.name, img)
            cmd = [TESSERACT, fh.name, "stdout", "--psm", str(psm), "-l", "eng",
                   "--dpi", "300",
                   "-c", "tessedit_char_whitelist=%s" % whitelist,
                   "-c", "classify_bln_numeric_mode=0", "tsv"]
            # One thread: the OpenMP pool gains nothing on a crop this size
            # and fights the capture thread and the panel for cores.
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=READ_TIMEOUT_S,
                               env=dict(os.environ, OMP_THREAD_LIMIT="1"))
        if p.returncode != 0:
            out["error"] = (p.stderr or "tesseract failed").strip().splitlines()[-1][:120]
            return out
        rows, confs = {}, []
        for line in p.stdout.splitlines()[1:]:
            f = line.split("\t")
            if len(f) < 12:
                continue
            try:
                conf = float(f[10])
            except ValueError:
                continue
            word = f[11].strip()
            if not word or conf < 0:
                continue
            key = (f[2], f[3], f[4])                 # block, paragraph, line
            rows.setdefault(key, []).append(word)
            confs.append(conf)
            out["words"].append({"text": word, "conf": round(conf, 1)})
        out["lines"] = [" ".join(v) for v in rows.values()]
        out["text"] = "\n".join(out["lines"])
        out["conf"] = round(sum(confs) / len(confs), 1) if confs else 0.0
    except subprocess.TimeoutExpired:
        # Tesseract crawls on a patch full of texture and no print; a read that
        # takes this long was never going to be a code.
        out["error"] = "read timed out (%.0fs): no legible print" % READ_TIMEOUT_S
    except Exception as exc:
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
    return out


# ------------------------------------------------------------------- compare
def normalise(s):
    """Case and run-length whitespace never distinguish one batch from another."""
    return re.sub(r"\s+", " ", (s or "").strip()).upper()


MIN_TOKEN_LEN = 4


def compare(read_text, expected, min_ratio=0.85, min_token=0.88):
    """
    Does the read match the batch we are meant to be running?

    Whole-string similarity AND the worst significant token must agree: a date
    of 26SEP26 against 27SEP26 differs by one character in one token and the
    matching tokens would dilute it above any sane whole-string threshold.
    Tokens shorter than MIN_TOKEN_LEN are skipped because OCR merges and splits
    them freely ("10:20 B" came back as "10:208" on a good pack).
    """
    if not expected:
        return {"checked": False, "ok": True, "ratio": 1.0, "worst_token": 1.0,
                "worst_field": "", "bad_fields": [], "read": normalise(read_text), "expected": ""}
    a, b = normalise(read_text), normalise(expected)
    # Whole-string agreement is judged with spaces removed on both sides, and
    # with the lines in canonical order too: a pack read upside down lists its
    # lines bottom-up, which is not a different code.
    def canon(txt):
        return "".join(sorted(squash(l) for l in (txt or "").split("\n") if l.strip()))
    ratio = max(difflib.SequenceMatcher(None, squash(a).replace("\n", ""), squash(b).replace("\n", "")).ratio(),
                difflib.SequenceMatcher(None, canon(read_text), canon(expected)).ratio())
    ta, tb = a.split(), b.split()
    joined = a.replace(" ", "")
    worst, culprit = 1.0, ""
    if not ta:
        worst, culprit = 0.0, "(nothing read)"
    else:
        for tok in tb:
            if len(tok) < MIN_TOKEN_LEN:
                continue
            best = max((difflib.SequenceMatcher(None, tok, x).ratio() for x in ta), default=0.0)
            m = difflib.SequenceMatcher(None, tok, joined)
            blk = m.find_longest_match(0, len(tok), 0, len(joined))
            if blk.size:
                best = max(best, blk.size / float(len(tok)))
            if best < worst:
                worst, culprit = best, tok
    # The fields that identify a batch - lot, dates, time - must match
    # EXACTLY. Token similarity alone lets a one-digit-off lot through: a
    # 10-character lot with one wrong digit still scores 0.90. A misread digit
    # therefore shows as a mismatch, with the read on the panel to judge by;
    # that is the right bias for a machine whose job is catching wrong batches.
    fe, fr = find_codes(expected), find_codes(read_text)      # raw: line breaks matter here
    bad_fields = []
    if fe["lot"] and fe["lot"] not in fr.get("lots", [fr["lot"]]):
        bad_fields.append("lot")      # a licence number is also a long number; any match will do
    if fe.get("bn") and fe["bn"] not in fr.get("bns", []):
        bad_fields.append("batch")
    if fe["dates"] and sorted(fe["dates"]) != sorted(fr["dates"]):
        bad_fields.append("dates")
    if fe["time"] and fe["time"] != fr["time"]:
        bad_fields.append("time")
    if bad_fields and worst > 0.0:
        worst, culprit = min(worst, 0.5), culprit or bad_fields[0]
    has_fields = bool(fe["lot"] or fe["dates"] or fe["time"] or fe.get("bn"))
    ok = (not bad_fields) if has_fields else (ratio >= min_ratio and worst >= min_token)
    return {"checked": True, "ok": ok, "ratio": round(ratio, 3),
            "worst_token": round(worst, 3), "worst_field": culprit,
            "bad_fields": bad_fields, "read": a, "expected": b}


def squash(s):
    """Upper case with spaces removed, line breaks kept: the GPU recogniser
    drops the spaces inside a code and a printer's spacing is not part of a
    batch's identity, but two lines must not run into one number."""
    return re.sub(r"[ \t]+", "", (s or "")).upper()


def find_codes(text):
    """Name the fields in a free-form read: lot, MON/YY dates, HH:MM time, MRP."""
    t = squash(text)
    lots = re.findall(r"(?<![0-9A-Z/])[0-9]{6,}[A-Z]?", t)
    tm = re.search(r"(?<![0-9])[0-9]{1,2}:[0-9]{2}(?![0-9])", t)
    mrp = re.search(r"RS\.?[0-9]+", t)
    # Dates: DD/MON/YY first (Amul cartons), then MON/YY (Nestle) on what is left.
    long_dates = re.findall(r"(?<![0-9])[0-9]{2}/[A-Z]{3}/[0-9]{2}(?![0-9])", t)
    rest = re.sub(r"(?<![0-9])[0-9]{2}/[A-Z]{3}/[0-9]{2}(?![0-9])", " ", t)
    dates = long_dates + re.findall(r"[A-Z]{3}/[0-9]{2}", rest)
    # Batch ids like KIF2071: three letters and four digits, as printed after "BN."
    # "BN. KIF2071" squashes to "BN.KIF2071" or "BNKIF2071": the id may follow BN directly.
    bns = re.findall(r"(?:(?<=BN)|(?<=BN\.)|(?<![A-Z]))([A-Z]{3}[0-9]{4})(?![0-9])", t)
    return {"lot": lots[0] if lots else "", "lots": lots, "bn": bns[0] if bns else "", "bns": bns,
            "dates": dates,
            "time": tm.group(0) if tm else "",
            "mrp": mrp.group(0) if mrp else ""}


def code_score(text):
    """How much does this read like a packaging code rather than prose?"""
    f = find_codes(text)
    score = 0.0
    if f["lot"] or f.get("bn"):
        score += 3.0
    score += 2.0 * len(f["dates"])
    if f["time"]:
        score += 1.5
    if f["mrp"]:
        score += 0.5
    words = len(normalise(text).split())
    if words > 45:
        score *= 0.25
    elif words > 25:
        score *= 0.6
    return score, f


# -------------------------------------------------------------------- locate
def bright_panels(gray, region):
    """Light patches on the product that a code would be printed on, largest first."""
    rx, ry, rw, rh = region
    ref = gray[ry:ry + rh, rx:rx + rw]
    if ref.size < 64:
        ref = gray
    # Percentiles over the PRODUCT, not the frame: on a white belt the belt is
    # the brightest thing in view and a frame-wide threshold loses the label.
    hi = float(np.percentile(ref, 96))
    lo = float(np.percentile(ref, 40))
    thr = max(lo + 0.45 * (hi - lo), 90.0)
    m = (gray > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    n, _lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    H, W = gray.shape
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < MIN_AREA_FRAC * H * W or a > MAX_AREA_FRAC * H * W:
            continue
        if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:
            continue                    # touching the frame edge: belt, not pack
        cx, cy = x + w / 2.0, y + h / 2.0
        if not (rx <= cx <= rx + rw and ry <= cy <= ry + rh):
            continue                    # not on the product
        out.append((int(x), int(y), int(w), int(h), int(a)))
    out.sort(key=lambda b: -b[4])
    return out[:6]


def print_in(gray, box, pad=6):
    """Bounding box of dense dark print inside `box`, plus how much there is."""
    x, y, w, h = box[:4]
    sub = gray[y:y + h, x:x + w]
    if sub.size < 64:
        return None
    bg = cv2.GaussianBlur(sub, (0, 0), sigmaX=max(w // 12, 3))
    flat = cv2.divide(sub, bg, scale=255)
    dark = (flat < 190).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _lab, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    cap = MAX_BLOB_FRAC * sub.size
    keep = [st[i] for i in range(1, n) if MIN_BLOB_PX <= st[i][4] <= cap]
    if len(keep) < 4:
        return None
    xs0 = min(b[0] for b in keep); ys0 = min(b[1] for b in keep)
    xs1 = max(b[0] + b[2] for b in keep); ys1 = max(b[1] + b[3] for b in keep)
    bw, bh = xs1 - xs0, ys1 - ys0
    if bw < 8 or bh < 8:
        return None
    ink = sum(b[4] for b in keep) / float(max(bw * bh, 1))
    # Typical glyph height: the median blob extent, which is what the reader
    # scales to Tesseract's comfortable size. Blobs are characters or bits of
    # characters, so the median is a fair estimate and outliers do not skew it.
    char_px = float(np.median([min(b[2], b[3]) for b in keep]))
    return {"x": int(x + xs0 - pad), "y": int(y + ys0 - pad),
            "w": int(bw + 2 * pad), "h": int(bh + 2 * pad),
            "blobs": len(keep), "ink": ink, "char_px": char_px}


def candidates(gray, pack):
    """Print regions on the pack, strongest first. `pack` is (x, y, w, h) px."""
    H, W = gray.shape
    cands = []
    for panel in bright_panels(gray, pack):
        got = print_in(gray, panel)
        if got:
            got["via"] = "label"
            cands.append(got)
    if not cands:
        got = print_in(gray, pack)
        if got:
            got["via"] = "pack"
            cands.append(got)
    for c in cands:
        area = max(c["w"] * c["h"], 1)
        c["score"] = c["ink"] * min(c["blobs"], 60) * min(area / (0.02 * H * W), 1.0)
    cands.sort(key=lambda c: -c["score"])
    return cands


# A read that already carries a lot number is decisive; anything weaker keeps
# the other orientations in play, because upside-down print reads as plausible
# fragments ("10:20" came back as "02:01") and would otherwise stop the search.
DECISIVE_SCORE = 3.0
MAX_READS = 5


def read_pack(frame, pack, rotations=(0, 90, 180, 270), expected="", max_candidates=2):
    """
    Locate and read the code on one pack. Never raises.

    `frame` is BGR or grey; `pack` is the pack's bounding box in pixels
    (x, y, w, h). Candidates are read in the given rotation order; the first
    read that scores as a code wins, so put the rotation that worked last time
    first and it is usually one Tesseract call. Returns a dict with text,
    lines, fields, conf, rotate, box (normalised in the frame), ms, match.
    """
    t0 = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    H, W = gray.shape
    x, y, w, h = pack
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(x + w)), min(H, int(y + h))
    out = {"text": "", "lines": [], "fields": {}, "conf": 0.0, "rotate": 0,
           "box": None, "ms": 0, "via": "", "code_score": 0.0, "error": "", "reads": 0}
    if x1 - x0 < 16 or y1 - y0 < 16:
        out["error"] = "pack box too small"
        out["ms"] = int((time.time() - t0) * 1000)
        return out

    best = None
    decided = False
    try:
        cands = candidates(gray, (x0, y0, x1 - x0, y1 - y0))
        for c in cands[:max_candidates]:
            if decided or out["reads"] >= MAX_READS or time.time() - t0 > READ_BUDGET_S:
                break
            cx0 = max(0, c["x"]); cy0 = max(0, c["y"])
            cx1 = min(W, c["x"] + c["w"]); cy1 = min(H, c["y"] + c["h"])
            win = gray[cy0:cy1, cx0:cx1]
            if win.size < 64:
                continue
            # Scale the glyphs to ~30 px: Tesseract falls apart under ~20 px and
            # the real batch panel came off the 720p camera at ~10 px.
            up = max(1.0, min(4.0, TARGET_CAP_PX / max(c.get("char_px", 8.0), 4.0)))
            # Once one orientation has read anything code-shaped, the second
            # candidate is only tried that way round.
            rots = rotations if best is None or best["code_score"] <= 0 else [best["rotate"]]
            for rot in rots:
                if out["reads"] >= MAX_READS or time.time() - t0 > READ_BUDGET_S:
                    break
                r = read(win, rotate=rot, upscale=up)
                out["reads"] += 1
                cs, fields = code_score(r["text"])
                cand = {"text": r["text"], "lines": r["lines"], "fields": fields,
                        "conf": r["conf"], "rotate": rot, "code_score": cs,
                        "via": c["via"], "error": r["error"],
                        "char_px": round(c.get("char_px", 0.0), 1), "upscale": round(up, 2),
                        "box": {"x": round(cx0 / W, 4), "y": round(cy0 / H, 4),
                                "w": round((cx1 - cx0) / W, 4), "h": round((cy1 - cy0) / H, 4)}}
                if best is None or (cs, r["conf"]) > (best["code_score"], best["conf"]):
                    best = cand
                if expected and compare(r["text"], expected)["ok"]:
                    decided = True        # it is the batch we expect: nothing beats that
                    break
                if cs >= DECISIVE_SCORE:
                    decided = True        # a lot number read: this orientation is right
                    break
        if best is None:
            out["error"] = "no print found on the pack"
    except Exception as exc:
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
    if best is not None:
        out.update(best)
    out["match"] = compare(out["text"], expected)
    out["ms"] = int((time.time() - t0) * 1000)
    return out


# ---------------------------------------------------------------- keywords
_CONFUSE = str.maketrans({"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6"})


def _norm_kw(s):
    return squash(s).replace("\n", "").translate(_CONFUSE)


def keyword_hit(keyword, texts):
    """
    Is this keyword present in any of the read texts, allowing the confusions
    a dot-matrix read produces (O/0, D/0, I/1, S/5, B/8, A/R...)?  Exact after
    confusion-normalising, or a same-length window of the text at >= 0.8
    similarity for keywords of five characters or more.
    """
    k = _norm_kw(keyword)
    if not k:
        return False
    for t in texts:
        n = _norm_kw(t)
        # A short keyword must be (most of) the line it is found in: "INDIA"
        # as a product name is a line of its own, "India" inside an address
        # line is not evidence of anything.
        slack = 12 if any(ch.isdigit() for ch in k) else 6      # "NETWEIGHT:150G" carries its label
        if k in n and len(n) <= max(2 * len(k), len(k) + slack):
            return True
        if len(k) >= 8 and len(n) >= len(k):          # fuzzy only for long ids
            for i in range(0, len(n) - len(k) + 1):
                if difflib.SequenceMatcher(None, k, n[i:i + len(k)]).ratio() >= 0.8:
                    return True
    return False


# ------------------------------------------------------------------ gpu path
def _similar(a, b):
    return difflib.SequenceMatcher(None, normalise(a), normalise(b)).ratio()


def select_lines(lines, expected=""):
    """
    The lines that are the batch code, out of everything the detector found.

    The real white patch also carries the price line; the recipe sample does
    not. Keep lines that read as code fields (lot, date, time) or that resemble
    a line of the expected sample. If nothing qualifies, keep the most
    confident short alphanumeric lines so the operator still sees a read.
    """
    exp_lines = [l for l in (expected or "").split("\n") if l.strip()]
    ef = find_codes(expected) if expected else None
    keep = []
    for l in lines:
        cs, f = code_score(l["text"])
        sim = max((_similar(l["text"], e) for e in exp_lines), default=0.0)
        if ef:
            carries = ((ef["lot"] and f["lot"] == ef["lot"]) or (ef["time"] and f["time"] == ef["time"])
                       or (ef.get("bn") and ef["bn"] in f.get("bns", []))
                       or any(d in ef["dates"] for d in f["dates"]))
            if carries or sim >= 0.5:
                keep.append(l)
        elif cs > 0:
            keep.append(l)
    if keep:
        return keep, "code"
    return [], "none"


def read_pack_gpu(engine, frame, pack, expected="", gain=1.0, rotate=0, min_side=900):
    """
    Locate and read the code on one pack with the TensorRT PP-OCR pipeline.
    Same result shape as read_pack(); rotation is handled by the detector and
    the orientation classifier, so nothing is retried.
    """
    t0 = time.time()
    H, W = frame.shape[:2]
    x, y, w, h = pack
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(x + w)), min(H, int(y + h))
    out = {"text": "", "lines": [], "fields": {}, "conf": 0.0, "rotate": 0,
           "box": None, "ms": 0, "via": "gpu", "code_score": 0.0, "error": "", "reads": 1,
           "all_lines": []}
    if x1 - x0 < 16 or y1 - y0 < 16:
        out["error"] = "pack box too small"
    else:
        crop = frame[y0:y1, x0:x1]
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        if gain and abs(gain - 1.0) > 0.01:
            # Dot-matrix print on dark board: the detector skips a row of faint
            # dots but finds a stroke, so the dots are first blurred into strokes
            # (sigma ~1.6 px at full resolution), then lifted by the gain.
            # Measured on the real Amul carton: this turned nothing into a
            # complete read of batch id and both dates.
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            g = cv2.GaussianBlur(g, (0, 0), 1.6)
            # Auto-level: lift the print to ~230 but never past it, so a change
            # of exposure does not turn the glyphs into saturated blobs. The
            # configured gain is the ceiling.
            p99 = float(np.percentile(g, 99.5))
            lift = min(float(gain), 230.0 / max(p99, 1.0))
            g = np.clip(g.astype(np.float32) * lift, 0, 255).astype(np.uint8)
            crop = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        # A small code window is upscaled so the detector sees glyphs, not
        # specks; a rotated crop turns vertical print into lines it can find.
        scale = 1.0
        if max(crop.shape[:2]) < min_side:
            scale = min(3.0, float(min_side) / max(crop.shape[:2]))
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if rotate:
            crop = cv2.rotate(crop, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                                     270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotate])
        out["rotate"] = rotate
        try:
            lines = engine.read(crop)
        except Exception as exc:
            lines = []
            out["error"] = "gpu ocr failed: %s" % exc
        out["all_lines"] = [{"text": l["text"], "conf": l["conf"]} for l in lines]
        out["timing"] = dict(getattr(engine, "last", {}))
        chosen, how = select_lines(lines, expected)
        if chosen:
            # top to bottom, as printed
            chosen.sort(key=lambda l: min(pt[1] for pt in l["box"]))
            out["lines"] = [l["text"] for l in chosen]
            out["text"] = "\n".join(out["lines"])
            out["conf"] = round(sum(l["conf"] for l in chosen) / len(chosen), 1)
            out["via"] = "gpu" if how == "code" else "gpu-text"
            xs = [pt[0] / scale for l in chosen for pt in l["box"]]
            ys = [pt[1] / scale for l in chosen for pt in l["box"]]
            if rotate:
                xs, ys = [0, x1 - x0], [0, y1 - y0]
            bx0, bx1 = max(0, min(xs)) + x0, min(x1 - x0, max(xs)) + x0
            by0, by1 = max(0, min(ys)) + y0, min(y1 - y0, max(ys)) + y0
            out["box"] = {"x": round(bx0 / W, 4), "y": round(by0 / H, 4),
                          "w": round(max(bx1 - bx0, 1) / W, 4), "h": round(max(by1 - by0, 1) / H, 4)}
        elif not out["error"]:
            seen = ", ".join(l["text"] for l in lines[:3] if l["text"].strip())
            out["error"] = ("no batch code visible - is the code side facing the camera? (saw: %s)" % seen
                            if seen else "no print found on the pack")
        out["code_score"], out["fields"] = code_score(out["text"])
    out["match"] = compare(out["text"], expected)
    out["ms"] = int((time.time() - t0) * 1000)
    return out


if __name__ == "__main__":
    import json
    import sys
    im = cv2.imread(sys.argv[1])
    h, w = im.shape[:2]
    r = read_pack(im, (0, 0, w, h), expected=sys.argv[2] if len(sys.argv) > 2 else "")
    print(json.dumps(r, indent=1))
