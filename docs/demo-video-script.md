# Demo video script: DepthWizard (SIH26175)

**Target length:** about 3 minutes, roughly 400 words of voice-over at a calm 140 words per
minute. This is the video behind the QR code on deck slide 6. It is recorded against the live
site, https://project5.zaidansari.tech.

**Who records what:**
- **[C]**: Claude records a scripted browser clip at 1080p. Motion and timing are exact, and
  there is no cursor.
- **[Y]**: you record the screen in OBS, because the shot needs a real cursor and real clicks.

Timings are targets. Speak first, then trim the clips to fit your voice.

Every number below is on screen in the viewer or on deck slide 7. Do not add a number
that is not.

---

## 1 · Hook (0:00–0:15)

**Shot [C]:** Hilly — Sikkim. A slow orbit over the valley at true scale.

> One satellite image. No stereo pair, no laser survey, no new flight. DepthWizard turns it
> into a measured 3D surface — this is Sikkim, rebuilt from a single picture.

## 2 · The problem (0:15–0:35)

**Shot [C]:** Urban — Omaha, top-down. Tilt slowly into 3D.

> Height is what disaster teams and planners need, and it normally takes stereo imagery or
> LiDAR — expensive, and impossible for the past. India has decades of single-view
> satellite archive with no height at all. We estimate it from one image, in metres.

## 3 · Upload your own image (0:35–1:05)

**Shot [Y]:** Click "Upload a satellite image". The file window opens; click
`sikkim_town.tif` (the details line appears), then Upload. Show the progress bar for 3–4
seconds, keeping the "… s of about … s" readout in frame, then cut the wait. The scene opens.
Scroll to the download buttons and hover over them.

> Pick a GeoTIFF — ours, or your own. The model runs on an ordinary CPU server, with no GPU. Because this file
> carries coordinates, we anchor it to open Copernicus terrain and return an absolute DSM —
> exactly what ISRO evaluates — plus height above ground and a per-pixel uncertainty map,
> all as GeoTIFFs you can open in QGIS. A plain PNG or JPG works too; you get relative heights.

## 4 · Explore and measure (1:05–1:35)

**Shot [C]:** Fly-through of Mixed — Jacksonville.
**Shot [Y]:** Measure: click a rooftop, then the ground beside it. Hold on the readout.

> Fly through it like a game. Click any two points to measure: distance on the ground, the
> height difference, and how sure the model is about that height. We open at true scale,
> not stretched, so what you see is what the model predicts.

## 5 · Honest about its own accuracy (1:35–2:10)

**Shot [C]:** Mixed — Jacksonville, then switch "Show" to "How sure we are".
**Shot [C]:** Compare with real LiDAR: the slider sweeps left to right across Omaha.
**Shot [C]:** Switch "Show" to "Where we're wrong" on Omaha.

> Every height comes with a confidence, and that confidence is checked — calibration error
> 0.063. Drag to compare against real airborne LiDAR. On unseen regions we're 3.5 metres per
> building; the published figure over Asia is 5.9. And we show where we're wrong: blue is
> too low. Our known weakness is very tall towers, and the viewer says so instead of hiding
> it.

## 6 · Disaster use case (2:10–2:40)

**Shot [C]:** Hilly — Sikkim. "Flood the valley"; the water level rises slowly to 1,143 m. Hold
on the readout.

> Built for disaster management. Raise the water, and it counts the ground under water and
> the buildings flooded — at 1,143 metres, a fifth of this valley and 44 buildings. It also
> flags buildings too close to call from their own uncertainty.

## 7 · Works offline (2:40–2:52)

**Shot [Y]:** Double-click `viewer_standalone.html` in File Explorer. It opens in the browser
with Wi-Fi off; show the network icon.

> No internet? The whole viewer is one HTML file. Double-click it, and it runs.

## 8 · Close (2:52–3:00)

**Shot [C]:** A title card: DepthWizard · Team Dev Up · SIH26175, with the live, docs and
code links.

> DepthWizard — height from one satellite image, with the evidence to trust it.

---

## Before recording

1. **Viewer deploy is live.** Checked on 25 Sep: the live main.js matches the repo, and the
   render check and the 26 upload checks pass.
2. **Upload images are ready** in `D:\sih2026\demo_uploads\`. There are two GeoTIFFs and
   one plain PNG; the README there describes each. Upload one once before the take so the
   server is warm. The riverside file processed live in 59 s on 25 Sep.
3. **Set up OBS.**
   - Capture a browser window at 1920 × 1080, 60 fps.
   - Hide the bookmarks bar and zoom the browser to 100 %.
   - Close other tabs, and turn off notifications (Focus assist on Windows).
4. **Credit Maxar.** Put "Imagery © Maxar Technologies, CC BY-NC 4.0" in the end card and the
   video description.
5. **Record the voice-over separately.** Record it in a quiet room with the phone about
   20 cm away. Then lay the clips under it; this is easier than talking while clicking.
