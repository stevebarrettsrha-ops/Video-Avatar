# Test report — LongCat Avatar Studio

**Result: 307 of 307 checks pass, including 33 against a real ComfyUI.
An out-of-the-box run from the shipped zip passes, and the error from the
first real PC is reproduced and fixed on the real node.**

Earlier result lines, kept for history: 298 of 298; 290 of 290; 277 of 277; 270 of 270; 267 of 267 checks pass (`python tests/run.py`, 334 s with the real engine).

| Suite | Checks | What it proves |
|---|---:|---|
| gate | 5 | Every module compiles; the page's script parses; every id the script uses exists; every button, chip and slider has a listener; `run.sh` parses. |
| units | 77 | Frame and window maths read from kijai's workflow file. The page's copy of the maths gives the same answers. The weight set and folders; the preflight verdicts; the RAM estimate; how ComfyUI is launched. |
| graph | 73 | The graphs `comfy.py` builds, against the real node schema: window wiring, seams, audio routing, every setting, and the fallbacks for missing optional parts. |
| api | 70 | The server end to end: uploads, validation, two-window renders with per-window progress, seeds, cancel, delete, the localhost guard, the preflight, the dependency list, a full set download, and stale or old engines. |
| stress | 20 | 400 fuzzed requests, 12 concurrent renders, parallel uploads and deletes, hostile paths, and damaged gallery/config files. |
| ui | 29 | The page in Chromium: picture and speech, trimming, the plan line, the RAM warning, settings, generate, the lightbox, reuse, **recording from a (fake) microphone**, a draft surviving a reload, and every page. |
| real | 33 | **A real ComfyUI 0.39** with the three node packs. Details below. |

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
6. **Long clips needed more RAM than 32 GB** (above). They are now rendered
   in parts and joined, with no length limit.
7. **The empty-feed message was split across grid columns**, and the Seed
   row read "Random / Random". Both are fixed.

## The first real PC (RTX 4060, 8 GB, 32 GB RAM, Windows)

A 2.8 s recording at 720p hit two problems.

1. **Reading the prompt took 5½ minutes.** The console showed umT5 running
   its 24 layers on the CPU in bf16, at about 14 s a layer. The text encoder
   now goes to the GPU in fp8 by default, about 6.7 GB, while the DiT is
   still in RAM. That takes seconds, and the result is cached on disk per
   prompt as before. If the card is too full (CUDA OOM in that node), the
   render retries once on the CPU by itself. The api suite drives that
   retry, and the graph, units and ui suites check the defaults. The plan
   line now says 720p takes well over twice as long as 480p.
2. **Then every render failed:** `MultiTalkWav2VecEmbeds: 'NoneType' object
   is not subscriptable`. The cause is transformers 5. The wrapper's
   wav2vec2 subclass asks the encoder for `output_hidden_states`, and from
   5.0 the encoder ignores it. Reproduced on the real ComfyUI from the
   out-of-the-box run, running the real node on a randomly initialised
   wav2vec2 file:

   | transformers | MultiTalkWav2VecEmbeds |
   |---|---|
   | 5.19.0 (what ComfyUI's `>=4.50.3` installs) | **error: 'NoneType' object is not subscriptable** |
   | 4.57.6 | success (13 hidden states) |

   The app now keeps transformers below 5 in ComfyUI's Python. It asks for
   diffusers in the same install, because the newest diffusers needs
   huggingface-hub 1.32 or later and transformers 4 needs below 1.0. pip
   settles on diffusers 0.39, and `pip check` is clean. The fix runs after
   setup, after node and PyTorch installs, and before every engine start.
   Verified from a clean 5.19: the app's own `ComfyProcess.start` installed
   4.57.6, ComfyUI loaded every pack, and the node succeeded.

Code review then found two gaps, and both are closed:

- **Restart on an engine started elsewhere** went through ComfyUI-Manager's
  in-place reboot, which restarts the same packages, so transformers stayed
  5.x. When the configured Python has 5.x, Restart now stops the process
  instead and starts a managed engine, which installs 4.x first, while
  nothing holds the files. (On Windows a running ComfyUI holds tokenizers'
  `.pyd`, so pip cannot replace it in place.) The api suite drives this
  against a mock engine with a Manager reboot endpoint. The reboot is never
  called, 4.x is in place before the new engine starts, and then it is
  ready.
- **A failed install only logged a line and started ComfyUI anyway.** It now
  starts nothing, and the start answers with the reason and the exact pip
  command. `/api/status` stays not ready while ComfyUI's Python has 5.x, so
  an engine started some other way cannot pass as ready. Tested with a
  stand-in Python whose pip fails as a full disk would.

### Then the downgrade itself failed on the real PC

On the Windows PC, pip could not install transformers 4.x, and the app
correctly refused to start the engine. A retry from the Engine page later
went through, but a fix that depends on pip succeeding is fragile. So the
app now carries the fix itself: `compat/avatar_studio_compat`, a ComfyUI
custom node that puts the 4.x hidden states back on transformers 5. It
records the input to each encoder layer, then the output, which is exactly
the tuple 4.x returned.

| Check | Result |
|---|---|
| wrapper's wav2vec2, transformers 5.19.0, no node | `hidden_states` None |
| same, with the node | 13 states |
| 13 states vs transformers 4.57.6, same weights and input | max difference **0.0** |
| node on 4.57.6 | changes nothing (difference 0.0) |
| real ComfyUI on 5.19.0, started by the app's `ComfyProcess.start` | node installed, **no pip call**, `AvatarStudioCompat` listed, `MultiTalkWav2VecEmbeds` **success** |

The downgrade is now only the fallback, for when the node cannot be written
into `custom_nodes`. The api suite covers four cases:

1. pip failing, the node writable: the engine starts with no pip call and
   is ready.
2. Neither possible: the start is refused and nothing says ready.
3. An engine started elsewhere, with ComfyUI-Manager: the node goes on disk
   and then the Manager reboots it.
4. The same, with the node not writable: the Manager reboot is skipped, 4.x
   is installed and the engine ends up ready.

## Out of the box: the shipped zip, start to finish

`tests/out_of_the_box.py` does what a new user does, on a clean folder:

1. Unzip `LongCat-Avatar-Studio.zip`.
2. Run its own `run.sh`. It creates `.venv` and installs the requirements;
   the app answered in 8 s.
3. Click **Install a fresh ComfyUI** on the setup sheet. Every step went
   green in 3 min 6 s: ComfyUI, the four node packs, `comfy-venv` with
   PyTorch and all requirements, the six weight files, and ComfyUI started.
   The engine pill read **Engine ready**.
4. Through the page, render a 12.5 s clip and a 40 s clip. The files, as
   ffprobe sees them:

| Clip | Parts | Frames | fps | Size | Audio |
|---|---|---|---|---|---|
| 12.5 s | 2 | 200 (200 expected) | 16 | 832×480 | 12.500 s |
| 40 s | 4 | 640 (640 expected) | 16 | 832×480 | 40.000 s |

There were no script errors on the page, and the result was **OUT-OF-THE-BOX
RUN PASSED** (log: `docs/out-of-the-box-run.log`, screenshots:
`docs/screenshots/out-of-the-box/`).

Four stand-ins were used because this sandbox has no internet route to
them and no GPU:

- a stand-in HuggingFace (`tests/mock_hf.py`);
- PyTorch from PyPI instead of download.pytorch.org;
- ComfyUI with `--cpu`;
- stand-in frames in place of the model (`AVATAR_REHEARSAL=1`).

On a real machine all four are dropped.

The first attempt found one more bug. **Pressing Generate while a new audio
file was still uploading rendered the previous audio**, without saying so.
Generate is now held ("Uploading…") until every upload has finished, and
the browser test covers it. A progress line that showed a raw node name
("GetVideoComponents") now reads "Picking up from the part before". A unit
check now makes sure every node the app queues has a stage in words.

## No length limit: long clips in parts

The first round of testing measured how much memory the frames need. They
are float32, and about four copies are alive while windows are joined:
500 frames at 832×480 peaked at 9.3 GB. In one graph, two minutes would need
about 36 GB, and a 31 s single graph was in fact killed for lack of memory.

So a long clip is now rendered in **parts** (two windows at 480p, one at
720p, about 3–4 GB of frames each). Each part is its own ComfyUI job that
continues from the last 13 frames of the part before. `assemble.py` then
joins the parts one frame at a time and lays the original soundtrack under
them. Verified on the real engine, with the real app (`AVATAR_REHEARSAL=1`
swaps only the diffusion for stand-in frames):

| Clip | Parts | Joined file | Audio | ComfyUI peak |
|---|---|---|---|---|
| 30 s | 3 | exactly 480 frames, 16 fps | 30.000 s | 4.76 GB |
| 5 min | 30 | exactly 4,800 frames, 16 fps | 300.000 s | 5.60 GB |

Ten times the length, the same memory. The app itself stayed at about
250 MB while joining. Clips can now be up to an hour long, and length costs
time, not memory. (Both peaks include about 4 GB still cached from the
test's earlier sections.)

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
| 15 | Ten minutes of speech planned: 120 windows in 60 parts, the same 3.1 GB as any length |
| 16, 16b, 16c | A two-minute render at "part 3 of 12"; joining the 12 parts; the finished clip ("Rendered in 12 parts") |
| 17 | Phone width |
| 18–19 | An out-of-memory render explained; an engine missing the LongCat nodes |
