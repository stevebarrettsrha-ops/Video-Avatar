# LongCat Avatar Studio — invariants

## Hard rules

1. **`web/index.html` stays one file, no build step.** Same shell as the
   sibling studios: rail, composer over a masonry feed, popover settings.
2. **Never hard-code a ComfyUI workflow.** `comfy.py` builds from
   `/object_info` and matches inputs through candidate-name lists.
   `tests/object_info.json` is the real schema (ComfyUI 0.39 + the three
   packs); refresh it from a real engine, never by hand-typing.
3. **The frame maths is the workflow's.** 16 fps, 93-frame windows, 13
   frames of overlap, so each later window adds 80;
   `windows = 1 + ceil((frames − 93) / 80)`. The wav2vec2 embeds count in
   audio frames, 32 a second, so their `num_frames` is twice the video's.
   The page's `frameCount()`/`windowCount()` mirror Python's and
   `test_units` holds them together, against numbers read from the
   workflow file in `assets/`.
4. **Window wiring, exactly as kijai's example.** Window 0: ExtendEmbeds
   with overlap 1, frames_processed 0, no ref_latent, prev_latents = the
   encoded picture. Window n: prev_latents = the previous *sampler's* raw
   latents, ref_latent = the encoded picture, frames_processed = frames
   joined so far (93, 173, …), then the last 13 decoded frames re-encoded
   into the seam with `ReplaceVideoLatentFrames` and joined with
   `ImageBatchExtendWithOverlap` (cut, new_images side). Seeds are
   `seed + window`.
5. **LongCat only runs at bf16 base precision.** Memory savings come from
   `quantization` (fp8_e4m3fn by default) and block swap, never from
   base_precision. Attention defaults to sdpa: sageattention 1.0.6 breaks it.
6. **The clip carries the original audio.** The lips listen to the
   MelBandRoFormer vocals; `CreateVideo` gets the trimmed original.
   Without the separator the clip still renders, with a note saying so.
7. **The distill LoRA is the alpha64 one.** The older rank-128 file is not
   meant for this model and harms extension (kijai's note).
8. **Uploads invalidate the cached schema.** LoadImage/LoadAudio list
   ComfyUI/input; a cached schema refuses a file uploaded after it.
9. **Recordings become WAV in the browser** (decodeAudioData + a 16-bit PCM
   encoder) before upload. No server-side ffmpeg.
10. **The preflight tells the truth.** 8 GB VRAM / 32 GB RAM is "tight", and
    the note says it is untimed on real hardware until someone times it.
    Python detection by execution, downloads resumable, model deletes
    path-checked — as in the sibling apps.
11. **Link outputs by name.** `ImageBatchExtendWithOverlap`'s output 0 is
    a passthrough; joining from it dropped every window but the first, and
    ComfyUI accepted it (all IMAGE). `_out()` resolves by output name.
12. **Long clips are parts; ComfyUI's cache stays on.** One graph holds
    all its frames (~4 float32 copies while joining: 18.6 MB a frame at
    480p, measured), so `comfy.plan()` splits a clip into parts of
    `windows_per_part()` windows (4 GB budget: 2 at 480p, 1 at 720p). Part
    k loads part k−1's video (LoadVideo), re-encodes its last 13 frames as
    prev_latents and the seam, reads its own audio slice starting 13 frames
    before its first new frame (frames_processed 13 → audio index 0), and
    outputs only its new frames. Seeds and the progress map use the window's
    place in the whole clip. Parts carry no audio; `assemble.py` joins them
    frame by frame (PyAV) and lays the original soundtrack under them in one
    piece. `--cache-none` is NOT used: the cache keeps the model loaded
    between parts. `--lowvram` unless the person set a memory mode
    (`AVATAR_COMFY_ARGS`) — ComfyUI refuses two. Verified on a real engine:
    5 min (30 parts) peaks within 1 GB of 30 s, both joined to the frame.
13. **`AVATAR_REHEARSAL=1` is a test seam only.** `comfy.rehearse()` swaps
    the diffusion for stand-in frames so the real app runs end to end on an
    engine without the model.
14. **transformers < 5 in ComfyUI's Python.** From 5.0 the wrapper's
    wav2vec2 gets `hidden_states=None` and MultiTalkWav2VecEmbeds fails
    ("'NoneType' object is not subscriptable"), as on the first real PC.
    `bootstrap.pin_transformers` (with diffusers, so huggingface-hub agrees)
    runs after setup, node/PyTorch installs, and before every engine start.
15. **umT5 reads the prompt on the GPU in fp8 by default.** On the CPU (bf16)
    it took 5½ minutes on an RTX 4060 PC. A CUDA OOM in
    WanVideoTextEncodeCached is retried once with `t5_cpu` (run_job).
    Saved drafts before v2 do not carry the old CPU default forward.

## Graceful degradation

No MelBandRoFormer node or model → speech goes straight to wav2vec2, with a
note on the clip. No `Wav2VecModelLoader` (older wrapper) →
`DownloadAndLoadWav2VecModel`. No `ImageResizeKJv2` → core `ImageScale`
with centre crop. An unknown scheduler → euler. Block swap 0 → no swap
node. Required nodes missing → a message naming each node and its pack.

## SaveVideo's nested dynamic combos

Recent ComfyUI serves `SaveVideo.format` as a V3 dynamic combo whose option
carries another, `format.codec`. `_fill_dynamic` recurses; the mock
validates nested ones too. A flat implementation once looked fine against a
hand-written schema.

## Saved locations are verified, never trusted

The config keeps absolute paths, which go stale the moment the app folder is
moved, renamed or re-extracted — and then everything reads "missing" though
it is all on disk. `bootstrap.verify_locations()` runs at every start (in the
boot thread, before the engine is launched) and on every Recheck
(`/api/deps`): `heal_paths()` rebases a stale path onto the app's current
folder, and if ComfyUI is still nowhere, `find_comfy_installs()` walks the
drives breadth-first under a time budget, and `pick_comfy()` prefers the
install holding the LongCat weights. While it walks, `/api/deps` reports
`searching` and the page re-polls. `AVATAR_STUDIO_NO_SEARCH=1` (set by every
test that starts the server) turns it off: test configs name made-up folders
on purpose.

## Validation gate

```bash
python tests/run.py            # gate units graph api stress ui (+ real)
python tests/run.py gate       # compile, inline script parse, ids, wiring
```
