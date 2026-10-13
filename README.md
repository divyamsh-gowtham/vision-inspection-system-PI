# ProInspect

Inline code inspection appliance. Runs on a Jetson Orin Nano with an Arducam
AR0234 global-shutter camera, reads date/lot codes off packaging, verifies
print quality, and shows the result on an attached panel.

Current state: **kiosk shell + live product counting + batch-code reading**.
The HMI boots, the operator picks a product and recipe, and the run screen
counts every product that passes the belt camera, reads the batch code printed
on it, and shows the read against the recipe's sample. Print-quality grading
(OCV) is not wired in yet.

## Layout

```
proinspect/
├── kiosk.py                            fullscreen WebKit host for the HMI
├── counter_service.py                  belt camera -> count, read, MJPEG, :8080
├── ocr.py                              locate the code on the pack and read it
├── ppocr.py                            PP-OCRv4 on TensorRT (det + cls + rec)
├── gpu.py                              TensorRT engine wrapper, ctypes CUDA
├── models/                             ONNX models + built engines (engines not committed)
├── tests_counter.py                    synthetic-belt tests for the counter
├── tests_ocr.py                        synthetic-pack tests for the reader
├── ui/
│   ├── index.html                      home grid
│   ├── products.html                   step 1: product on the line
│   ├── recipes.html                    step 2: recipe -> "Load recipe & start"
│   ├── run.html                        live view, gate, products-passed counter
│   ├── app.js / app.css                shared shell: clock, power menu, host bridge
│   └── data/products.json              catalogue
├── config/                             site data, not committed: gate, belt, run
├── branding/                           boot splash theme + generated logo assets
├── scripts/xinitrc                     X session (no window manager, no desktop)
├── systemd/proinspect.service          the HMI
├── systemd/proinspect-counter.service  the counter service
├── systemd/proinspect-clocks.service   pins MAXN_SUPER + jetson_clocks
└── VERSION
```

## Counter service

`counter_service.py` opens the first UVC camera that emits MJPEG (currently the
USB "HZ USB Camera" on `/dev/video0`) at its full **2592x1944, 30 fps**, because
the reader needs the pixels: at 1280x720 the batch code on the real pouch was
~10 px tall and unreadable, at full resolution it is ~27 px. Every frame is
analysed at 1/8 size to count; the panel is sent a 1/4-size re-encode at 10 fps
(~14 KB a frame); only the one frame chosen per pack is decoded in full, for
the reader. Install once:

```bash
sudo cp systemd/proinspect-counter.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now proinspect-counter
journalctl -u proinspect-counter -f
```

```
http://127.0.0.1:8080/               debug page: live view, gate, raw status
http://127.0.0.1:8080/status         camera, counter, run and gate state, one JSON
http://127.0.0.1:8080/stream.mjpg    live MJPEG (15 fps served, 30 analysed)
http://127.0.0.1:8080/run/start?product=&recipe=   count from zero (same job: keeps count)
http://127.0.0.1:8080/run/stop  /run/reset
http://127.0.0.1:8080/gate/set?cx=&cy=&w=&h=       move/resize the gate (0..1)
http://127.0.0.1:8080/belt/learn                   re-learn the empty belt (0.5 s)
http://127.0.0.1:8080/config?enter=&leave=         presence thresholds
http://127.0.0.1:8080/camera/focus                 sweep focus on whatever is under the camera
http://127.0.0.1:8080/camera/set?focus_absolute=   any allow-listed V4L2 control, persisted
http://127.0.0.1:8080/ocr/last.jpg?n=              the patch that was read for pack n
http://127.0.0.1:8080/ocr/frame.jpg                last read's full frame, boxes drawn
```

**How a product is counted.** The service learns the empty belt (median of 15
frames, saved to `config/belt.png`). Presence is how much the *gate* - the
dashed box on the live view - differs from that reference, after removing any
whole-frame brightness drift (auto-exposure, a shadow). Presence above `enter`
for 2 frames counts a product **on arrival**; the gate must fall below `leave`
for 3 frames before the next one can count, so a pack that stops in the gate
and moves on is one pack. Only a clear gate updates the reference.

On the run screen: tap the picture to put the gate where packs pass, use +/-
for its size, and "Learn empty belt" after moving the camera or changing the
lighting. The presence meter shows the score against the `enter` mark, so a
pack that fails to count is diagnosed by eye: if the bar barely moves, the gate
is in the wrong place or the pack looks like the belt.

The run (product, recipe, count) is persisted in `config/run.json`, so a
service restart does not lose the tally; reopening the run screen for the same
job attaches to it rather than resetting. `python3 tests_counter.py` checks
the counting rules on a simulated belt.

**How the code is read.** When a pack is counted the capture thread keeps the
sharpest frame in which the pack is wholly in view, and hands it to a reader
thread as soon as it has two such frames - so the read lands while the pack is
still under the camera - with a second, sharper frame at the end of the transit
if one turned up. `ocr.py` finds the light patch on the pack that carries dense
small print (the batch panel), scales the glyphs to ~30 px, and runs Tesseract
(`/usr/bin/tesseract`, already on the image; there is no pip here). Orientation
is not guessed: the patch is read in the orientations allowed until one yields
a lot number or matches the recipe's sample, and that rotation is tried first
on the next pack. Low-confidence gibberish (foil texture, artwork) is reported
as "nothing legible", not as a code. The run screen shows the read under the
counter with a thumbnail of the patch, MATCH / MISMATCH against the recipe's
`sample`, and the last few reads. Lot, date and time fields must match
**exactly** - a one-digit-off lot is a mismatch, deliberately.

**Everything that can run off the CPU does.** Capture goes through one
GStreamer pipeline: the UVC MJPEG frames are tee'd raw for the reader, decoded
once by the NVJPG engine (`nvv4l2decoder mjpeg=1`), scaled by the VIC into the
grey work image the counter analyses and into the panel frame, which the
hardware JPEG encoder compresses for the stream. The CPU never decodes a
frame. If that pipeline will not start the service falls back to the plain
`v4l2-ctl` path (`PROINSPECT_CAPTURE=v4l2ctl` forces it). Reading runs
PP-OCRv4 (detector, orientation classifier, recogniser) on the GPU through
TensorRT - `gpu.py` is the engine wrapper (ctypes against libcudart, no pip
needed), `ppocr.py` the pipeline. Engines are built once on the box:

```bash
models/build.sh          # ~20 min on the Orin Nano; writes models/*.engine
```

The ONNX models come from the `rapidocr_onnxruntime` wheel on PyPI (the
recogniser's character table is embedded in the ONNX metadata). Until the
engines exist the service reads with Tesseract; it switches to the GPU by
itself when they appear, no restart needed. `/status` reports `ocr_engine`
and each read carries `engine: gpu|tesseract`.

**Focus is fixed, not automatic.** Continuous autofocus parked the lens at its
far end over the featureless belt and every pack came through blurred. The
service sets `focus_automatic_continuous=0` and `focus_absolute` from
`config/camera.json` (default 140, near the belt plane). To calibrate: hold a
pack still under the camera and press **Focus** on the run screen; it sweeps
the range, picks the sharpest, and saves it. `python3 tests_ocr.py` checks
locating, reading in all four orientations, matching, and the reader thread on
a synthetic pack.

While tuning, the reader writes the last 6 packs to `debug/packs/` (full frame
with boxes plus the patch). Set `PROINSPECT_DEBUG=0` in the unit to stop that.

## How the appliance boots

1. UEFI → extlinux, `DEFAULT JetsonIO` (3 s menu timeout)
2. Kernel boots `quiet loglevel=0`, with **no console on the display** — no OS
   text ever reaches the panel
3. Plymouth shows the ProInspect splash once the rootfs mounts
4. `proinspect.service` starts on `graphical.target`, takes tty1, and runs
   `xinit → scripts/xinitrc → kiosk.py`
5. The kiosk opens fullscreen and undecorated. There is no window manager,
   no desktop, and no display manager.

GDM is disabled (`display-manager.service` removed). **xrdp is untouched** and
still works for remote desktop — it spawns its own Xorg independent of GDM.

## Operating it

```bash
sudo systemctl status proinspect       # is the HMI up
sudo systemctl restart proinspect      # reload after a UI change
journalctl -u proinspect -f            # logs, incl. browser console output
```

Develop the UI without disturbing the running panel:

```bash
DISPLAY=:0 python3 kiosk.py --windowed            # windowed, on the panel's X
DISPLAY=:0 python3 kiosk.py --url http://…        # point at a served page
DISPLAY=:0 python3 kiosk.py --hide-cursor         # blank the pointer
```

**The mouse pointer is visible by default.** Pass `--hide-cursor` only once the
station is driven by a touchscreen — blanking it on a mouse-driven unit makes
the machine look broken. The flag blanks the GDK cursor *and* sets
`html.nocursor` so the CSS follows; both are needed, since hiding only one
leaves the pointer visible over part of the window.

`Ctrl+Shift+Q` is the deliberate service exit — systemd restarts it within 3 s.

## Recovery

If the panel stays black after a change, SSH still works. The boot menu offers:

- **JetsonIO** — the appliance entry (default)
- **stock** — stock kernel, verbose console, `/boot/Image.backup`

To get the normal Ubuntu desktop back:

```bash
sudo systemctl disable --now proinspect
sudo systemctl enable --now gdm
```

## Design notes

**WebKit, not Chromium.** `libwebkit2gtk-4.0` and PyGObject already ship on the
JetPack image, so the HMI needs no browser install and avoids Chromium's memory
footprint — which matters on a board with 7.4 GB shared between CPU and GPU.
The UI being HTML means the same pages can later serve the LAN dashboard.

**The initrd is deliberately not regenerated.** `/boot/initrd` is NVIDIA-built
and *both* boot entries reference it, so a bad regeneration would leave nothing
bootable. The splash theme applies from rootfs mount onward instead.

**Counting on arrival, not on exit.** The old software trigger fired as a pack
*left* the gate so it could hand the sharpest frame to the grader. A counter is
different: the operator watches the number, and a number that moves a second
after the pack has gone reads as broken. Count on entry; keep the best frame
for the reader separately when that lands.

**The screen never blanks.** `xinitrc` disables DPMS and the screensaver; a
blank panel on a factory floor is indistinguishable from a dead machine. For the
same reason `kiosk.py` reloads the page if the web process dies, and systemd
restarts the unit if the process exits.

