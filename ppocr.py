#!/usr/bin/env python3
"""
PP-OCRv4 on the GPU: detect text lines, fix their orientation, read them.

Three TensorRT engines built from the ONNX models that ship inside the
rapidocr_onnxruntime wheel (models/build.sh):

    det  DBNet     image -> probability map of text pixels -> line boxes
    cls  180° test line crop -> upright or upside down
    rec  SVTR/CTC  line crop (48 px tall) -> characters + confidence

Pre- and post-processing follow PaddleOCR's inference code (BGR in, the same
normalisation, DB thresholds, CTC decode); the recogniser's character table is
read out of the ONNX file's own metadata, so the two cannot drift apart.

    ocr = PPOCR()                      # loads the three engines once
    lines = ocr.read(image_bgr)        # [{"text", "conf", "box"}] top to bottom
"""

import pathlib
import time

import cv2
import numpy as np

from gpu import TRTEngine

HERE = pathlib.Path(__file__).resolve().parent
MODELS = HERE / "models"

DET_MAX_SIDE = 960
DET_THRESH = 0.3          # pixel probability -> text
DET_BOX_THRESH = 0.5      # mean probability inside a box to keep it
DET_UNCLIP = 1.6          # DB shrinks boxes at training time; grow them back
DET_MIN_SIZE = 4
CLS_THRESH = 0.9
REC_H = 48
REC_MAX_W = 1600          # the engine profile's maximum width
CLS_SHAPE = (48, 192)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ------------------------------------------------------------- character table
def _varint(b, i):
    s = r = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7f) << s
        s += 7
        if not c & 0x80:
            return r, i


def _walk(b, i, end):
    while i < end:
        k, i = _varint(b, i)
        f, wt = k >> 3, k & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 2:
            n, i = _varint(b, i)
            v = b[i:i + n]
            i += n
        elif wt == 1:
            v = b[i:i + 8]
            i += 8
        elif wt == 5:
            v = b[i:i + 4]
            i += 4
        else:
            raise ValueError("bad wire type %d" % wt)
        yield f, wt, v


def load_charset(onnx_path):
    """The recogniser's characters from the ONNX metadata_props 'character'."""
    b = pathlib.Path(onnx_path).read_bytes()
    for f, wt, v in _walk(b, 0, len(b)):
        if f == 14:                                  # ModelProto.metadata_props
            kv = {}
            for f2, wt2, v2 in _walk(v, 0, len(v)):
                kv[f2] = v2
            if kv.get(1, b"").decode(errors="replace") == "character":
                chars = kv[2].decode("utf-8").split("\n")
                # PaddleOCR's CTC head: blank first, space last.
                return ["blank"] + chars + [" "]
    raise ValueError("no character table in %s" % onnx_path)


# ------------------------------------------------------------------ geometry
def order_points(pts):
    """Four points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, np.float32)
    s = pts.sum(1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def rotate_crop(img, box):
    """Perspective-crop a quad to an upright rectangle; tall crops turn sideways."""
    pts = order_points(box)
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    w, h = max(w, 1), max(h, 1)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    crop = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if h / float(w) >= 1.5:
        crop = np.rot90(crop)
    return crop


# ------------------------------------------------------------------ pipeline
class PPOCR:
    def __init__(self, models=MODELS):
        models = pathlib.Path(models)
        self.det = TRTEngine(models / "ppocr_det.engine")
        self.rec = TRTEngine(models / "ppocr_rec.engine")
        # The 180-degree classifier is optional: its ONNX export carries fixed
        # dimensions that TensorRT builds as 48x48, which is useless. Without
        # it each line is read both ways up and the confident one wins - one
        # extra 5 ms recogniser call, and no third model to look after.
        self.cls = None
        try:
            cls = TRTEngine(models / "ppocr_cls.engine")
            if tuple(cls.inputs[0]["max_shape"][2:]) == CLS_SHAPE:
                self.cls = cls
        except Exception:
            self.cls = None
        self.chars = load_charset(models / "ppocr_rec.onnx")
        self.det_max = self.det.inputs[0]["max_shape"]     # (1,3,H,W)
        self.rec_max_w = self.rec.inputs[0]["max_shape"][3]
        self.last = {}

    # -- detection ---------------------------------------------------------
    def detect(self, img):
        h, w = img.shape[:2]
        limit = min(DET_MAX_SIDE, self.det_max[2], self.det_max[3])
        ratio = min(1.0, limit / float(max(h, w)))
        nh = max(32, int(round(h * ratio / 32)) * 32)
        nw = max(32, int(round(w * ratio / 32)) * 32)
        nh, nw = min(nh, self.det_max[2]), min(nw, self.det_max[3])
        rs = cv2.resize(img, (nw, nh))
        x = (rs.astype(np.float32) / 255.0 - MEAN) / STD
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        prob = self.det.infer(x)[0, 0]                     # (nh, nw)
        boxes = self._boxes(prob, w / float(nw), h / float(nh))
        return boxes

    def _boxes(self, prob, sx, sy):
        binary = (prob > DET_THRESH).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for cnt in contours[:200]:
            if len(cnt) < 3:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), ang = rect
            if min(rw, rh) < DET_MIN_SIZE:
                continue
            # mean probability inside the contour, the way DB scores a box
            x, y, bw, bh = cv2.boundingRect(cnt)
            mask = np.zeros((bh, bw), np.uint8)
            cv2.fillPoly(mask, [cnt - [x, y]], 1)
            score = float(cv2.mean(prob[y:y + bh, x:x + bw], mask)[0])
            if score < DET_BOX_THRESH:
                continue
            # unclip: grow the rectangle by area*ratio/perimeter (pyclipper-free)
            area, per = rw * rh, 2 * (rw + rh)
            d = area * DET_UNCLIP / max(per, 1e-3)
            pts = cv2.boxPoints(((cx, cy), (rw + 2 * d, rh + 2 * d), ang))
            pts[:, 0] *= sx
            pts[:, 1] *= sy
            out.append({"box": order_points(pts), "score": score})
        # top to bottom, then left to right, with a tolerance of half a line
        out.sort(key=lambda b: (b["box"][:, 1].min(), b["box"][:, 0].min()))
        merged = []
        for b in out:
            merged.append(b)
        return merged

    # -- orientation -------------------------------------------------------
    def classify(self, crops):
        flips = []
        for crop in crops:
            x = self._norm(crop, CLS_SHAPE[0], CLS_SHAPE[1], pad_to=CLS_SHAPE[1])
            p = self.cls.infer(x)[0]
            flips.append(bool(p[1] > CLS_THRESH))
        return flips

    # -- recognition -------------------------------------------------------
    def recognize(self, crop):
        h, w = crop.shape[:2]
        rw = int(min(self.rec_max_w, max(32, round(w * REC_H / float(h)))))
        x = self._norm(crop, REC_H, rw)
        probs = self.rec.infer(x)[0]                       # (T, classes)
        idx = probs.argmax(1)
        conf = probs.max(1)
        text, confs, prev = [], [], -1
        for i, c in enumerate(idx):
            if c != 0 and c != prev:
                text.append(self.chars[c] if c < len(self.chars) else "")
                confs.append(float(conf[i]))
            prev = c
        return "".join(text), (float(np.mean(confs)) if confs else 0.0)

    @staticmethod
    def _norm(crop, h, w, pad_to=None):
        rs = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)
        if pad_to and w < pad_to:
            rs = cv2.copyMakeBorder(rs, 0, 0, 0, pad_to - w, cv2.BORDER_CONSTANT, value=0)
        x = (rs.astype(np.float32) / 255.0 - 0.5) / 0.5
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

    # -- everything --------------------------------------------------------
    def read(self, img, use_cls=True, max_lines=12):
        """Text lines in the image, top to bottom: [{"text","conf","box","flip"}]."""
        t0 = time.time()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        boxes = self.detect(img)[:max_lines]
        t1 = time.time()
        crops = [rotate_crop(img, b["box"]) for b in boxes]
        flips = self.classify(crops) if (use_cls and self.cls is not None and crops) else [None] * len(crops)
        t2 = time.time()
        lines = []
        for b, crop, flip in zip(boxes, crops, flips):
            if flip is None:
                # no classifier: read both ways up, keep the confident one
                text, conf = self.recognize(crop)
                text2, conf2 = self.recognize(cv2.rotate(crop, cv2.ROTATE_180))
                if conf2 > conf:
                    text, conf, flip = text2, conf2, True
                else:
                    flip = False
            else:
                if flip:
                    crop = cv2.rotate(crop, cv2.ROTATE_180)
                text, conf = self.recognize(crop)
            if text.strip():
                lines.append({"text": text, "conf": round(conf * 100.0, 1),
                              "box": b["box"].round(1).tolist(), "det": round(b["score"], 3), "flip": flip})
        t3 = time.time()
        self.last = {"det_ms": int((t1 - t0) * 1000), "cls_ms": int((t2 - t1) * 1000),
                     "rec_ms": int((t3 - t2) * 1000), "boxes": len(boxes)}
        return lines


def available():
    return all((MODELS / n).exists() for n in ("ppocr_det.engine", "ppocr_rec.engine"))


if __name__ == "__main__":
    import json
    import sys
    o = PPOCR()
    for path in sys.argv[1:]:
        im = cv2.imread(path)
        t = time.time()
        lines = o.read(im)
        print(path, "%d ms" % int((time.time() - t) * 1000), o.last)
        for l in lines:
            print("   %5.1f  %s" % (l["conf"], l["text"]))
