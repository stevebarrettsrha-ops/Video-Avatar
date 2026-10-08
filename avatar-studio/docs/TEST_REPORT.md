# Test report — LongCat Avatar Studio

**Result: 237 of 237 checks pass** (`python tests/run.py`, 97 s).

| Suite | Checks | What it proves |
|---|---:|---|
| gate | 5 | Every module compiles; the page's script parses; every id the script uses exists; every button, chip and slider has a listener; `run.sh` parses. |
| units | 48 | Frame and window maths read from kijai's workflow file. The page's copy of the maths gives the same answers. The weight set and folders; the preflight verdicts; the RAM estimate; how ComfyUI is launched. |
| graph | 58 | The graphs `comfy.py` builds, against the real node schema: window wiring, seams, audio routing, every setting, and the fallbacks for missing optional parts. |
| api | 50 | The server end to end: uploads, validation, two-window renders with per-window progress, seeds, cancel, delete, the localhost guard, the preflight, the dependency list, a full set download, and stale or old engines. |
| stress | 20 | 400 fuzzed requests, 12 concurrent renders, parallel uploads and deletes, hostile paths, and damaged gallery/config files. |
| ui | 25 | The page in Chromium: picture and speech, trimming, the plan line, the RAM warning, settings, generate, the lightbox, reuse, **recording from a (fake) microphone**, a draft surviving a reload, and every page. |
| real | 31 | **A real ComfyUI 0.39** with the three node packs. Details below. |

## Against a real ComfyUI

A real ComfyUI with WanVideoWrapper (main), KJNodes and MelBandRoFormer, run
on CPU (this machine has no GPU):

- **Its own validator accepted all 41 graphs**: 5 sizes × 6 lengths
  (0.5 s → 120 s, up to 24 windows), plus 10 settings variations.
- **The stitching and the output actually ran on the real nodes.** Sampling
  needs the 28 GB model, so each window's decode was swapped for 93 copies
  of the resized picture. Everything after that is the app's own graph:
  the overlap cut, the joins, the trim to the speech, `CreateVideo` with
  the audio, and `SaveVideo`. ffprobe of the results:

  | Speech | Windows | Frames | fps | Size | Audio | Picture vs sound |
  |---|---|---|---|---|---|---|
  | 5.8 s | 1 | 93 | 16 | 832×480 | 5.800 s | end together |
  | 12.5 s | 3 | 200 | 16 | 832×480 | 12.500 s | end together |
  | 31.25 s | 7 | 500 | 16 | 832×480 | 31.250 s | end together |

- **The app driving that engine:** it reports ready, sees all three packs
  loaded, uploads round-trip, and a render reaches the engine. The engine
  then stops on the placeholder weights with a plain-words message.

## A real one-click setup

Driven through the setup screen in Chromium:

1. It cloned ComfyUI and the four node packs from GitHub.
2. It built `comfy-venv`, installed PyTorch and every pack's requirements
   (≈ 8½ minutes).
3. It downloaded the six weight files. These came from a stand-in
   HuggingFace, since huggingface.co is blocked from the test machine.
4. It launched ComfyUI.

The app then reported `ready: True`. Running setup a second time over that
install updated it in place in 13 seconds.

## Bugs found and fixed by this testing

1. **Windows were never joined.** `ImageBatchExtendWithOverlap` returns
   `source_images, start_images, extended_images`, and the graph linked
   output 0, a passthrough. Every clip longer than 5.8 s would have
   been cut to its first window. ComfyUI accepts that link (all IMAGE), so
   only running the real nodes showed it. Outputs are now resolved by name,
   and the graph test fails on the old code.
2. **A memory mode of the user's own killed the engine.** ComfyUI refuses
   `--cpu`/`--highvram`/`--novram` together with `--lowvram`, and setup
   then waited the full 15 minutes. A mode passed in `AVATAR_COMFY_ARGS`
   now replaces `--lowvram`. Setup stops within seconds of the engine
   dying, and shows its last log lines.
3. **Damaged weights gave a raw error** (`Error while deserializing
   header`). It now names the node and says to delete the file and
   download it again.
4. **A file uploaded just before Generate could be refused**: the cached
   node schema did not list it yet. Uploads now refresh it.
5. **An old WanVideoWrapper (from before LongCat) was reported as "IMPORT
   FAILED"**. It now says to press Update.
6. **The empty-feed message was split across grid columns**, and the Seed
   row read "Random / Random". Both are fixed.

## A limit the testing measured: RAM for long clips

The decoded frames are float32. While windows are joined, about four copies
are alive at once. On the real engine, 500 frames at 832×480 peaked at
9.3 GB above idle (the app's estimate: 9.6 GB).

- **Without `--cache-none`,** a 31 s clip held about 14 GB and was killed.
  The app always launches with it.
- **With 32 GB of RAM and the fp8 model resident,** about 12 GB is left for
  frames. That is roughly **42 s at 480p, or 18 s at 720p**, per clip. The
  Create page shows the estimate for every clip, and warns with the part
  length to use when it is over.

## What could not be tested here

The model itself. This machine has no GPU, and huggingface.co is blocked,
so LongCat-Avatar never sampled a frame. Lip-sync quality, render time on an
RTX 4060, and VRAM use during sampling still need a first real run. The
screenshots of a finished clip show the real ComfyUI output of the stitching
rehearsal above: a still picture with the real speech track, not model
output.

## Screenshots (`docs/screenshots/`)

| # | Screen |
|---|---|
| 01–03 | Setup sheet; setup running; setup finished (real run) |
| 04–05 | Engine and Models pages after the real setup |
| 06–07 | Create: empty, then with a picture and speech, and the plan line |
| 08 | The settings popover |
| 09–10 | Recording from the microphone, and the recording kept as speech |
| 11 | Rendering: "Window 2 of 3 · step 4 of 12" |
| 12–14 | The finished clip in the feed, in the lightbox, in the Library |
| 15–16 | The RAM warning for 2 min at 720p, and a 40 s part that fits |
| 17 | Phone width |
| 18–19 | An out-of-memory render explained; an engine missing the LongCat nodes |
