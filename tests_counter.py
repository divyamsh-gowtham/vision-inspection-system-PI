"""
Validate the product counter on a simulated belt, before a single real pack.

What must hold:
  1. exactly one count per pack, counted on ARRIVAL (within a few frames of the
     pack entering the gate);
  2. a pack that stops in the gate and moves on again is still one pack;
  3. a global brightness change (auto-exposure, a shadow across the belt) is
     not a pack;
  4. two packs with a proper gap are two packs; belt noise alone is nothing;
  5. a pack larger than the gate is still one pack.

Run:  python3 tests_counter.py
"""
import sys
import cv2
import numpy as np

sys.path.insert(0, "/home/jark/proinspect")
from counter_service import ProductCounter, LEARN_FRAMES  # noqa: E402

W, H = 320, 180                     # the work size the service analyses at
GATE = {"x": 0.30, "y": 0.25, "w": 0.40, "h": 0.50}
rng = np.random.default_rng(3)


def belt(level=196):
    # Signed noise added in float: casting a negative normal straight to uint8
    # wraps it to ~250 and paints half the belt white.
    return np.clip(level + rng.normal(0, 3, (H, W)), 0, 255).astype(np.uint8)


def pack_at(cx, level=196, size=(64, 44), tone=80):
    img = belt(level)
    w, h = size
    cv2.rectangle(img, (cx - w // 2, H // 2 - h // 2), (cx + w // 2, H // 2 + h // 2), tone, -1)
    cv2.putText(img, "27SEP26", (cx - 28, H // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, 230, 1, cv2.LINE_AA)
    return img


def make_counter():
    c = ProductCounter(GATE)
    c.start_learning()
    for _ in range(LEARN_FRAMES):
        c.feed(belt())
    assert not c.learning() and c.ref is not None, "reference not learned"
    return c


def transit(frames, n=24, **kw):
    """One pack crossing left to right in n frames."""
    for i in range(n):
        cx = -40 + int(i * (W + 80) / (n - 1))
        frames.append(pack_at(cx, **kw))


results = []


def check(label, ok, detail=""):
    results.append(ok)
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", label, ("  (" + detail + ")") if detail else ""))


# 1. twelve packs, gaps between them ------------------------------------------
frames = []
for _ in range(12):
    frames += [belt() for _ in range(10)]
    transit(frames)
frames += [belt() for _ in range(10)]
c = make_counter()
hits = []
for i, f in enumerate(frames):
    if c.feed(f):
        hits.append(i)
check("12 packs -> 12 counts", len(hits) == 12, "got %d" % len(hits))

# counted on arrival: the count lands within a few frames of the pack's leading
# edge crossing into the gate, not after it has left. Transit frames start at
# index 10; the pack is 64 px wide and moves (W+80)/23 px per frame from -40.
gate_left = int(GATE["x"] * W)
enter_frame = 10 + next(i for i in range(24) if (-40 + int(i * (W + 80) / 23)) + 32 >= gate_left)
leave_frame = 10 + next(i for i in range(24) if (-40 + int(i * (W + 80) / 23)) - 32 >= int((GATE["x"] + GATE["w"]) * W))
check("counted on arrival, not on exit",
      bool(hits) and enter_frame <= hits[0] <= enter_frame + 4 and hits[0] < leave_frame,
      "count at frame %d; pack enters gate at %d, leaves at %d" % (hits[0] if hits else -1, enter_frame, leave_frame))

# 2. pack stops in the gate, waits, moves on ----------------------------------
frames = [belt() for _ in range(10)]
for i in range(8):
    frames.append(pack_at(-40 + int(i * (W // 2 + 40) / 7)))
frames += [pack_at(W // 2) for _ in range(40)]          # belt stopped
for i in range(8):
    frames.append(pack_at(W // 2 + int(i * (W // 2 + 40) / 7)))
frames += [belt() for _ in range(10)]
c = make_counter()
n = sum(1 for f in frames if c.feed(f))
check("pack that stops in the gate is one pack", n == 1, "got %d" % n)

# 3. global brightness drift is not a pack -------------------------------------
c = make_counter()
n = 0
for lvl in list(range(196, 226)) + list(range(226, 170, -1)) + list(range(170, 197)):
    n += 1 if c.feed(belt(lvl)) else 0
check("brightness swing +30/-26 levels -> no count", n == 0, "got %d" % n)
peak = 0.0
c = make_counter()
for lvl in range(196, 226):
    c.feed(belt(lvl)); peak = max(peak, c.presence)
check("presence stays under enter during drift", peak < c.enter, "peak %.4f vs enter %.3f" % (peak, c.enter))

# 4. belt noise alone --------------------------------------------------------
c = make_counter()
n = sum(1 for _ in range(200) if c.feed(belt()))
check("200 empty frames -> no count", n == 0, "got %d" % n)

# 5. a pack bigger than the gate -----------------------------------------------
frames = [belt() for _ in range(10)]
transit(frames, n=30, size=(200, 150))
frames += [belt() for _ in range(10)]
c = make_counter()
n = sum(1 for f in frames if c.feed(f))
check("pack larger than the gate is one pack", n == 1, "got %d" % n)

# 6. a pack that drifts through with the drift happening at the same time ----
frames = [belt() for _ in range(10)]
for i in range(24):
    cx = -40 + int(i * (W + 80) / 23)
    frames.append(pack_at(cx, level=196 + i))
frames += [belt(220) for _ in range(10)]
c = make_counter()
n = sum(1 for f in frames if c.feed(f))
check("pack during brightness drift is one pack", n == 1, "got %d" % n)

# 7. track box follows the pack ----------------------------------------------
c = make_counter()
c.feed(belt()); c.feed(pack_at(60)); c.feed(pack_at(60))
t = c.track
check("track box lands on the pack",
      t is not None and abs((t["x"] + t["w"] / 2) - 60 / W) < 0.05,
      "track %s" % (t,))

print("\n%d/%d checks passed" % (sum(results), len(results)))
sys.exit(0 if all(results) else 1)
