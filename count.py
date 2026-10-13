#!/usr/bin/env python3
"""
Count pieces in a blister strip: how many pockets there are, how many hold a
gum, how many are empty.

The strip is found by its own colour (the dominant saturated hue inside the
pack box - blue on the Nicogum strip); the pockets are the non-strip holes
inside its outline; a pocket is FILLED when it looks like a gum (light, low
saturation) and EMPTY when it is dark (torn foil) or shows nothing but a flat
patch of belt through the plastic.

    r = count_pieces(frame_bgr, (x, y, w, h), expected=8)
    r["pockets"], r["count"], r["missing"], r["blobs"]
"""

import cv2
import numpy as np

# Measured on the real Nicogum strip (4 ms exposure): strip ribs sat ~154,
# gums seen through the blue plastic sat 80-90 / val 113-122, an empty torn
# pocket sat 139 / val ~55, the belt sat 15. Saturation separates strip from
# gum; brightness separates gum from empty.
DEFAULTS = {
    "strip_sat_min": 130,      # the strip's own colour; gums tinted by the plastic stay below
    "strip_val_min": 40,
    "hue_tol": 14,             # +- around the dominant strip hue
    "pocket_min_frac": 0.006,  # of the strip area; smaller holes are print/glare
    "pocket_max_frac": 0.20,
    "pocket_solidity_min": 0.55,
    "gum_sat_max": 115,        # filled: tinted but not strip-coloured ...
    "gum_val_min": 95,         # ... and not dark (torn foil is ~55)
    "flat_std_max": 3.0,       # a perfectly flat patch is belt seen through an empty pocket
    "foil_std_min": 24.0,      # crinkled torn foil glitters: high contrast inside the pocket
}


def _strip_mask(hsv, c):
    sat, val, hue = hsv[:, :, 1], hsv[:, :, 2], hsv[:, :, 0]
    saturated = (sat >= c["strip_sat_min"]) & (val >= c["strip_val_min"])
    if saturated.sum() < 200:
        return None, None
    hist = np.bincount(hue[saturated].ravel(), minlength=180)
    dom = int(np.argmax(hist))
    diff = np.abs(hue.astype(np.int16) - dom)
    diff = np.minimum(diff, 180 - diff)
    mask = (saturated & (diff <= c["hue_tol"])).astype(np.uint8)
    return mask, dom


def count_pieces(frame, pack, expected, cfg=None):
    c = dict(DEFAULTS)
    c.update(cfg or {})
    H, W = frame.shape[:2]
    x, y, w, h = pack
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(x + w)), min(H, int(y + h))
    out = {"pockets": 0, "count": 0, "missing": 0, "extra": 0, "blobs": [], "box": None,
           "error": "", "strip": None, "hue": None}
    if x1 - x0 < 16 or y1 - y0 < 16:
        out["error"] = "pack box too small"
        return out
    crop = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask, dom = _strip_mask(hsv, c)
    if mask is None:
        out["error"] = "no strip in view"
        return out
    k = max(3, (min(crop.shape[:2]) // 60) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        out["error"] = "no strip in view"
        return out
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    strip = (labels == big).astype(np.uint8)
    contours, _ = cv2.findContours(strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    inside = np.zeros_like(strip)
    cv2.fillPoly(inside, [hull], 1)
    strip_area = float(inside.sum())
    if strip_area < 0.01 * crop.shape[0] * crop.shape[1]:
        out["error"] = "strip too small in view"
        return out
    # pockets: inside the outline but not strip colour, away from the border
    border = max(2, k // 2)
    inner = cv2.erode(inside, np.ones((border * 2 + 1, border * 2 + 1), np.uint8))
    holes = (inner.astype(bool) & ~strip.astype(bool)).astype(np.uint8)
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    hn, hl, hs, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    pockets = []
    for i in range(1, hn):
        bx, by, bw, bh, a = [int(v) for v in hs[i]]
        if a < c["pocket_min_frac"] * strip_area or a > c["pocket_max_frac"] * strip_area:
            continue
        if a / float(max(1, bw * bh)) < c["pocket_solidity_min"]:
            continue
        m = (hl == i)
        ms, mv = float(sat[m].mean()), float(val[m].mean())
        std = float(val[m].std())
        filled = (ms <= c["gum_sat_max"] and mv >= c["gum_val_min"]
                  and c["flat_std_max"] < std < c["foil_std_min"])
        pockets.append({"x": round((bx + x0) / W, 4), "y": round((by + y0) / H, 4),
                        "w": round(bw / W, 4), "h": round(bh / H, 4),
                        "filled": bool(filled), "sat": round(ms, 1), "val": round(mv, 1), "std": round(std, 1),
                        "pieces": 1 if filled else 0})
    if not pockets:
        out["error"] = "no pockets found"
        return out
    # Pockets on one strip are all the same size: drop holes that are much
    # smaller (a fragment, print) or larger (a shadow along the edge) than the
    # typical pocket.
    if len(pockets) >= 3:
        med = float(np.median([p["w"] * p["h"] for p in pockets]))
        pockets = [p for p in pockets if 0.4 * med <= p["w"] * p["h"] <= 1.8 * med]
    # a strip pocket count that disagrees wildly with the expected total means
    # the strip was not segmented properly; say so rather than counting wrong
    pockets.sort(key=lambda p: (round(p["y"], 2), p["x"]))
    filled = sum(1 for p in pockets if p["filled"])
    sx, sy, sw, sh = cv2.boundingRect(hull)
    out.update({
        "pockets": len(pockets), "count": filled,
        # Missing is judged against the strip's known size: a pocket the
        # segmentation could not outline is still a pocket without a gum.
        "missing": (max(0, int(expected) - filled) if expected else len(pockets) - filled),
        "extra": max(0, filled - int(expected)) if expected else 0,
        "blobs": pockets, "hue": dom,
        "strip": {"x": round((sx + x0) / W, 4), "y": round((sy + y0) / H, 4),
                  "w": round(sw / W, 4), "h": round(sh / H, 4)},
        "box": {"x": round(x0 / W, 4), "y": round(y0 / H, 4),
                "w": round((x1 - x0) / W, 4), "h": round((y1 - y0) / H, 4)},
    })
    if expected and len(pockets) != int(expected):
        out["note"] = "%d pockets seen, %d expected" % (len(pockets), int(expected))
    return out


def draw(frame, result):
    """The pack crop with every pocket boxed: green filled, red empty."""
    H, W = frame.shape[:2]
    b = result.get("box")
    if not b:
        return None
    x0, y0 = int(b["x"] * W), int(b["y"] * H)
    x1, y1 = int((b["x"] + b["w"]) * W), int((b["y"] + b["h"]) * H)
    vis = frame[y0:y1, x0:x1].copy()
    for p in result.get("blobs", []):
        bx, by = int(p["x"] * W) - x0, int(p["y"] * H) - y0
        col = (0, 180, 0) if p.get("filled") else (0, 0, 220)
        cv2.rectangle(vis, (bx, by), (bx + int(p["w"] * W), by + int(p["h"] * H)), col, 3)
        if not p.get("filled"):
            cv2.putText(vis, "EMPTY", (bx + 4, by + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    return vis


if __name__ == "__main__":
    import json
    import sys
    im = cv2.imread(sys.argv[1])
    h, w = im.shape[:2]
    r = count_pieces(im, (0, 0, w, h), int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    print(json.dumps({k: v for k, v in r.items() if k != "blobs"}, indent=1))
    for p in r["blobs"]:
        print("  pocket filled=%s sat=%s val=%s std=%s" % (p["filled"], p["sat"], p["val"], p["std"]))
    vis = draw(im, r)
    if vis is not None:
        cv2.imwrite("count_debug.jpg", vis)
