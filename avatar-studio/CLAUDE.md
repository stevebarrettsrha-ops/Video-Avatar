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
   Block prefetch and pinned (non-blocking) transfers are Settings, off by
   default: the 8 GB audit gives that VRAM and host RAM to activations.
   Block swap's default follows the card: `bootstrap.blocks_to_swap_default`
   (48 under 7 GB, 40 on 8, 30 on 12, 24 on 16, 16 on 24, none from 32; 40
   for a card not read), which `/api/status` reports from the engine's own
   VRAM figure, else the last preflight's (`cfg["vram"]`). The page's slider
   starts there and Back to defaults returns there; a moved slider keeps its
   value (draft v3 `swap_auto`), and drafts before v3 do not carry the old
   fixed 40 forward. `comfy.build()` alone still falls back to 40.
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
14. **The lip sync must work on transformers 5.** From 5.0 the wrapper's
    wav2vec2 gets `hidden_states=None` and MultiTalkWav2VecEmbeds fails
    ("'NoneType' object is not subscriptable"), as on the first real PC.
    The fix is `compat/avatar_studio_compat`, a custom node that restores
    the 4.x hidden states (bit-identical to 4.57.6) and registers the marker
    node `AvatarStudioCompat`. `bootstrap.install_compat` copies it into
    custom_nodes before every start, after setup and installs; nothing in
    the person's Python changes. Only if it cannot be copied does
    `pin_transformers` downgrade (a downgrade failed on a real Windows PC).
    If neither works, `ComfyProcess.start` raises `TransformersBlocked` and
    launches nothing. `/api/status` is not ready while a running engine has
    5.x without the marker node (`transformers_bad`). Every start goes
    through `server._start_engine`. Restart copies the node in before a
    ComfyUI-Manager reboot; if it could not, the reboot is skipped.
15. **umT5 reads the prompt on the GPU in fp8 by default.** On the CPU (bf16)
    it took 5½ minutes on an RTX 4060 PC. A CUDA OOM in
    WanVideoTextEncodeCached is retried once with `t5_cpu` (run_job).
    Saved drafts before v2 do not carry the old CPU default forward.
16. **matplotlib stays off Tk.** The wrapper's sampler plots its sigmas on
    every run; on Windows with Tk installed that made Tk objects on the
    render thread which were collected on the server thread, and ComfyUI
    died mid-render ("Tcl_AsyncDelete … wrong thread", fatal exception), as
    on the first real PC. ComfyUI is launched with `MPLBACKEND=Agg`, and the
    compatibility node forces Agg for engines started some other way.
    Reproduced with Tk under Xvfb; both fixes stop it.
17. **A managed engine that dies fails the job at once** (`engine_died_message`
    quotes its fatal line), instead of waiting five minutes unreachable. The
    sampler's "Loading transformer parameters" progress (1896 tensors) is
    labelled as loading, not as steps: the stage counts steps only when the
    progress max equals the scheduler's steps. The wrapper zeroes that bar
    once the weights are on the card (`load_weights` ends with
    `update_absolute(0)`), so `ws_listener` keeps each bar's high-water
    mark and a 0 after counting up reads "model loaded, on step 1 of 12",
    never "0 of 1896". The load is worth `LOAD_SHARE` (a quarter) of that
    window's slice of the bar and its steps fill the rest; the bar never
    walks backwards, so the pre-sampling creep stops at what the first
    window's first step is worth. The mock engine replays the zeroing.
18. **A busy engine is not a broken one.** Moving the model, the engine
    answers its ping (`comfy_online`) but can miss `/object_info` and
    `/history` replies: a real PC's status read "Restart the engine" and
    "Set up" mid-load, and its card froze on "1551 of 1896". A failed
    schema refresh serves the last schema (`ComfyClient.schema_stale`,
    asked again in a minute; a forced read or none to fall back on still
    raises), `stale_models` needs a read `dits`, and with no schema at all
    the status says `engine_busy`/`schema_error` and the page "Engine
    busy". A job whose engine answers nothing keeps its clock and says so
    on the card; it fails after `engine_wait_limit`: five minutes for an
    engine this app cannot see, half an hour for a live managed process,
    whose death still fails it at once. The mock's `POST /busy` is that
    engine.

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
it is all on disk. `bootstrap.verify_locations()` checks the exact saved
paths at start. Only a missing location triggers relocation: `heal_paths()`
tries the moved app folder, then `find_comfy_installs()` searches under a
time budget. A failed attempt is saved and does not repeat on polls or
restarts. Engine Recheck (`/api/deps?recheck=1`) explicitly permits another
attempt. `pick_comfy()` prefers the install holding the LongCat weights.
While it walks, `/api/deps` reports `searching` and the page re-polls.
Model discovery likewise saves exact verified paths (including nested
folders and `extra_model_paths.yaml`) and searches again only when a saved
file disappears or Recheck is pressed. A small custom node registers
Settings' shared models folder with the local engine without rewriting the
person's YAML or copying weights. Incomplete safetensors and LFS pointers
do not count as installed; this is not a model identity/checksum check.
A missing separate models root is retained, not replaced by an empty stock
folder. Only a relocated engine's own models folder follows that engine;
a separately rebased folder must hold valid required weights. A previously
verified missing root cannot silently become a new download destination.
`AVATAR_STUDIO_NO_SEARCH=1` (set by every
test that starts the server) turns it off: test configs name made-up folders
on purpose.

A saved path can also be *there* and still wrong: an app folder on a drive
shared by Windows and Linux carries the Windows `comfy-venv/Scripts/python.exe`
(or a portable build's `python_embeded/python.exe`), and every run of it is
`[Errno 8] Exec format error`. `foreign_binary()` reads the first bytes (MZ,
ELF, Mach-O); `_location_state()` counts a foreign saved interpreter as
missing so `heal_paths()` clears it, and `comfy_python()` /
`portable_python()` pass over one. The foreign venv is never rebuilt in
place — that would break it for the other system — so `venv_dir()` gives
this one `comfy-venv-linux` (or `-windows`, `-darwin`) beside it, and
`run.sh` does the same for a Windows `.venv` (`.venv-linux`). Setup never
puts the interpreter that *builds* the venv (often the app's own, torch-less)
into `cfg["python"]`: a setup that stops early must not leave the engine on
it. Git's `sudo apt-get` goes through `manager._elevate()` (`sudo -n`, else
`pkexec` on a desktop), because a task has no terminal to type a password in.

## Validation gate

```bash
python tests/run.py            # gate units graph api stress ui (+ real)
python tests/run.py gate       # compile, inline script parse, ids, wiring
```
