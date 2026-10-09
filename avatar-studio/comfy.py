"""
comfy.py - talks to ComfyUI and builds the LongCat-Avatar graphs.

Built from /object_info rather than a stored workflow, so a renamed input shows
up as a clear message instead of a silently wrong value.

Generation mirrors kijai's LongCatAvatar_audio_image_to_video_example_01.json
(ComfyUI-WanVideoWrapper, example_workflows/), with one difference that
matters: the workflow hard-wires three windows, this builds as many as the
audio needs.

  LoadImage ─ ImageResizeKJv2 ─ WanVideoEncode ─────────────┐ (ref latent)
  LoadAudio ─ TrimAudioDuration ─┬─ MelBandRoFormerSampler ─ MultiTalkWav2VecEmbeds
                                 │        (vocals only)            │
                                 │                                 ▼
  WanVideoModelLoader ◄─ BlockSwap, LoraSelect(distill)   window 0: ExtendEmbeds(overlap 1)
  WanVideoSchedulerv2(longcat_distill_euler) ─────────────►  WanVideoSamplerv2 ─ WanVideoDecode
  WanVideoTextEncodeCached ──────────────────────────────►        │
                                                          window n: ExtendEmbeds(overlap 13,
                                                            prev latents, ref latent)
                                                            ─ WanVideoSamplerv2
                                                            ─ ReplaceVideoLatentFrames(last 13
                                                              decoded frames, re-encoded)
                                                            ─ WanVideoDecode
                                                            ─ ImageBatchExtendWithOverlap(cut)
                                 └──────────── original audio (music kept) ─ CreateVideo ─ SaveVideo

The numbers the front end computes rather than asking nodes to do:

  fps     = 16 — the model's audio stride is 2 over 32 audio frames a second
  frames  = ceil(seconds * 16), the clip's real length
  windows = 1 + ceil((frames - 93) / 80): each window is 93 frames, the
            next one re-uses the last 13 of the one before
"""

from __future__ import annotations

import json
import math
import random
import threading
import time
import uuid

import requests

EXTEND = "WanVideoLongCatAvatarExtendEmbeds"
SAMPLER = "WanVideoSamplerv2"
SCHEDULER = "WanVideoSchedulerv2"
EMBEDS = "MultiTalkWav2VecEmbeds"
MELBAND = "MelBandRoFormerSampler"
MELBAND_LOADER = "MelBandRoFormerModelLoader"

FPS = 16                 # video frames a second (audio features run at 32)
AUDIO_FPS = 32
WINDOW = 93              # frames per window, as in the workflow
OVERLAP = 13             # frames each window re-uses from the one before
STEP = WINDOW - OVERLAP  # new frames each extra window adds
# Long speech is rendered in parts (see plan()), so length costs time, not
# memory: an hour is a sanity bound, not a resource one.
MAX_SECONDS = 3600.0

# Output sizes. Every one divides by 16, which ImageResizeKJv2 is asked for
# and the Wan VAE needs. 480p is what the workflow ships with and what an
# 8 GB card can carry; 720p wants a 24 GB card.
SIZES = {
    "832x480": (832, 480),
    "480x832": (480, 832),
    "640x640": (640, 640),
    "1280x720": (1280, 720),
    "720x1280": (720, 1280),
}
DEFAULT_SIZE = "832x480"

DEFAULT_NEGATIVE = (
    "Close-up, bright tones, overexposed, static, blurred details, subtitles, "
    "style, works, paintings, images, static, overall gray, worst quality, low "
    "quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly "
    "drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
    "fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards")


class ComfyError(RuntimeError):
    pass


def output_size(key) -> tuple[int, int]:
    """The exact pixel size of a finished clip; unknown keys get the default."""
    return SIZES.get(str(key or ""), SIZES[DEFAULT_SIZE])


def frame_count(seconds: float) -> int:
    """Video frames that cover `seconds` of speech at 16 fps."""
    return max(1, int(math.ceil(round(float(seconds) * FPS, 6))))


def window_count(frames: int) -> int:
    """Windows needed for `frames`: the first gives 93, every later one 80."""
    return 1 + max(0, int(math.ceil((frames - WINDOW) / STEP)))


# Decoded frames are float32 RGB, and while the windows are joined about four
# copies of the batch are alive at once. Measured on a real ComfyUI running
# the app's own joins: 500 frames at 832x480 peaked 9.3 GB above idle,
# 18.6 MB a frame, 3.9 frame-sized copies. One graph for a whole clip would
# need that for every frame of it — two minutes at 480p is ~36 GB — so a
# long clip is rendered in parts, each holding PART_BUDGET of frames at most.
FRAME_COPIES = 4
PART_BUDGET = 4 * 1024 ** 3


def frame_ram(frames: int, width: int, height: int) -> int:
    """Bytes of RAM the decoded frames need at their peak."""
    return frames * width * height * 3 * 4 * FRAME_COPIES


def frames_fitting(width: int, height: int, budget: int = PART_BUDGET) -> int:
    """How many frames at this size fit the budget."""
    return budget // (width * height * 3 * 4 * FRAME_COPIES)


def windows_per_part(width: int, height: int, budget: int = PART_BUDGET) -> int:
    """Windows one part may hold: 2 at 480p, 1 at 720p. Never fewer than 1."""
    return max(1, 1 + (frames_fitting(width, height, budget) - WINDOW) // STEP)


def plan(seconds: float, size=None, per_part: int | None = None) -> dict:
    """How a clip is rendered: its windows, grouped into parts.

    Every part is its own ComfyUI prompt. Part 0 starts from the picture;
    part k continues from the last 13 frames of part k-1, re-encoded, exactly
    as a window continues from the one before inside a part. Each part
    outputs only its new frames (part 0: 93 + 80 per extra window; later
    parts: 80 per window), so the parts laid end to end are the clip.
    """
    frames = frame_count(min(max(float(seconds), 0.01), MAX_SECONDS))
    windows = window_count(frames)
    width, height = output_size(size)
    per = max(1, int(per_part or windows_per_part(width, height)))
    parts, made = [], 0
    for first in range(0, windows, per):
        count = min(per, windows - first)
        out = (WINDOW + (count - 1) * STEP) if first == 0 else count * STEP
        keep = min(out, frames - made)
        parts.append({"index": len(parts), "first": first, "windows": count,
                      "start_frame": made, "frames": keep,
                      "trim": keep < out})
        made += keep
    return {"frames": frames, "windows": windows, "parts": parts,
            "per_part": per, "width": width, "height": height,
            "rendered": rendered_frames(windows),
            "part_ram": frame_ram(WINDOW + (per - 1) * STEP, width, height)}


def rendered_frames(windows: int) -> int:
    """Frames the windows actually produce, before trimming to the audio."""
    return WINDOW + (windows - 1) * STEP


def rehearse(graph: dict, part: int = 0) -> dict:
    """The graph with the diffusion swapped for stand-in frames.

    For testing the whole app on an engine without the 28 GB model (and
    without a GPU): every WanVideoDecode becomes 93 copies of the resized
    picture — a part's first window also gets the previous part's real last
    13 frames in front, as its decode would — and the nodes that only feed
    the model are dropped. Everything else runs as built: the picture, the
    audio trim, loading the previous part, the overlap cuts, the joins, the
    trim to the speech, CreateVideo and SaveVideo. Switched on by
    AVATAR_REHEARSAL=1; never in normal use.
    """
    g = json.loads(json.dumps(graph))
    resized = next(k for k, n in g.items()
                   if n["class_type"] in ("ImageResizeKJv2", "ImageScale"))
    for nid, node in list(g.items()):
        if node["class_type"] != "WanVideoDecode":
            continue
        g[nid] = {"class_type": "RepeatImageBatch",
                  "inputs": {"image": [resized, 0], "amount": WINDOW}}
        if part > 0 and nid == "105":          # a part's first window
            g[nid + "r"] = {"class_type": "RepeatImageBatch",
                            "inputs": {"image": [resized, 0], "amount": STEP}}
            g[nid] = {"class_type": "ImageBatch",
                      "inputs": {"image1": ["32", 0], "image2": [nid + "r", 0]}}
    keep = {"LoadImage", "ImageResizeKJv2", "ImageScale", "LoadAudio",
            "TrimAudioDuration", "RepeatImageBatch", "ImageBatch",
            "GetImageRangeFromBatch", "ImageBatchExtendWithOverlap",
            "LoadVideo", "GetVideoComponents", "CreateVideo", "SaveVideo"}
    g = {k: n for k, n in g.items() if n["class_type"] in keep}
    while True:                          # drop what nothing uses any more
        used = {str(v[0]) for n in g.values() for v in n["inputs"].values()
                if isinstance(v, list) and len(v) == 2 and isinstance(v[1], int)}
        dead = [k for k, n in g.items() if k not in used
                and n["class_type"] not in ("SaveVideo",)]
        if not dead:
            return g
        for k in dead:
            del g[k]


class ComfyClient:
    def __init__(self, url: str = "http://127.0.0.1:8188") -> None:
        self.url = url.rstrip("/")
        import os
        self.rehearsal = os.environ.get("AVATAR_REHEARSAL") == "1"
        self.client_id = str(uuid.uuid4())
        self._schema: dict | None = None
        self._schema_at = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # schema
    # ------------------------------------------------------------------ #
    def schema(self, force: bool = False) -> dict:
        with self._lock:
            if force or self._schema is None or time.time() - self._schema_at > 120:
                r = requests.get(f"{self.url}/object_info", timeout=30)
                r.raise_for_status()
                self._schema = r.json()
                self._schema_at = time.time()
            return self._schema

    def has(self, class_type: str) -> bool:
        return class_type in self.schema()

    def node_inputs(self, class_type: str) -> dict:
        info = self.schema().get(class_type)
        if not info:
            raise ComfyError(
                f"This ComfyUI has no '{class_type}' node. Install or update "
                "the custom nodes from the Engine page, then restart ComfyUI.")
        spec = info.get("input", {})
        merged = {}
        merged.update(spec.get("required", {}) or {})
        merged.update(spec.get("optional", {}) or {})
        return merged

    # Which pack each node comes from, so "missing" can say what to install.
    REQUIRED_NODES = {
        "WanVideoModelLoader": "ComfyUI-WanVideoWrapper",
        "WanVideoVAELoader": "ComfyUI-WanVideoWrapper",
        "WanVideoTextEncodeCached": "ComfyUI-WanVideoWrapper",
        "WanVideoEncode": "ComfyUI-WanVideoWrapper",
        "WanVideoDecode": "ComfyUI-WanVideoWrapper",
        SAMPLER: "ComfyUI-WanVideoWrapper",
        SCHEDULER: "ComfyUI-WanVideoWrapper",
        EXTEND: "ComfyUI-WanVideoWrapper (with LongCat Avatar)",
        EMBEDS: "ComfyUI-WanVideoWrapper",
        "GetImageRangeFromBatch": "ComfyUI-KJNodes",
        "ImageBatchExtendWithOverlap": "ComfyUI-KJNodes",
        "ReplaceVideoLatentFrames": "ComfyUI (core, 0.5 or newer)",
        "LoadImage": "ComfyUI (core)",
        "LoadAudio": "ComfyUI (core)",
        "CreateVideo": "ComfyUI (core)",
        "SaveVideo": "ComfyUI (core)",
    }

    def missing_nodes(self) -> list[str]:
        return [n for n in self.REQUIRED_NODES if not self.has(n)]

    def ensure_supported(self) -> None:
        missing = self.missing_nodes()
        if not (self.has("Wav2VecModelLoader")
                or self.has("DownloadAndLoadWav2VecModel")):
            missing.append("Wav2VecModelLoader")
        if missing:
            packs = sorted({self.REQUIRED_NODES.get(n, "ComfyUI-WanVideoWrapper")
                            for n in missing})
            raise ComfyError("This ComfyUI cannot run LongCat Avatar — it is "
                             "missing " + ", ".join(missing) + " (from "
                             + "; ".join(packs) + "). Install them from the "
                             "Engine page, then restart ComfyUI.")

    @staticmethod
    def _combo_options(spec) -> list:
        """The options of a combo input, whichever schema wrote it: classic
        [[options], {...}] or the newer ["COMBO", {"options": [...]}]."""
        if not isinstance(spec, (list, tuple)) or not spec:
            return []
        kind = spec[0]
        if isinstance(kind, list):
            return kind
        opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        if isinstance(kind, str) and kind.upper().startswith(
                ("COMBO", "COMFY_DYNAMICCOMBO")):
            options = opts.get("options") or []
            if options and isinstance(options[0], dict):
                return [o.get("key") for o in options]
            return list(options)
        return []

    def _enum(self, class_type: str, name: str) -> list[str]:
        try:
            spec = self.node_inputs(class_type).get(name)
        except ComfyError:
            return []
        return [str(v) for v in self._combo_options(spec)]

    # the file lists ComfyUI scanned at startup
    def dits(self) -> list[str]:
        return self._enum("WanVideoModelLoader", "model")

    def text_encoders(self) -> list[str]:
        return self._enum("WanVideoTextEncodeCached", "model_name")

    def vaes(self) -> list[str]:
        return self._enum("WanVideoVAELoader", "model_name")

    def loras(self) -> list[str]:
        return self._enum("WanVideoLoraSelect", "lora")

    def wav2vecs(self) -> list[str]:
        return self._enum("Wav2VecModelLoader", "model")

    def melbands(self) -> list[str]:
        return [m for m in self._enum(MELBAND_LOADER, "model_name")
                if "melband" in m.lower() or "roformer" in m.lower()]

    def schedulers(self) -> list[str]:
        return self._enum(SCHEDULER, "scheduler")

    def attention_modes(self) -> list[str]:
        return self._enum("WanVideoModelLoader", "attention_mode")

    def quantizations(self) -> list[str]:
        return self._enum("WanVideoModelLoader", "quantization")

    def images(self) -> list[str]:
        return self._enum("LoadImage", "image")

    def capabilities(self) -> dict:
        return {"longcat": self.has(EXTEND),
                "vocals": self.has(MELBAND) and bool(self.melbands()),
                "trim": self.has("TrimAudioDuration"),
                "kjnodes": self.has("ImageBatchExtendWithOverlap"),
                "blockswap": self.has("WanVideoBlockSwap")}

    # ------------------------------------------------------------------ #
    # picking files
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pick(names: list[str], wanted: str, contains: list[str],
              avoid: list[str] | None = None) -> str:
        if wanted and wanted in names:
            return wanted
        for n in names:
            low = n.lower().replace("\\", "/")
            if all(c in low for c in contains) and \
                    not any(a in low for a in (avoid or [])):
                return n
        return ""

    def resolve_models(self, p: dict) -> dict:
        dit = self._pick(self.dits(), p.get("dit", ""), ["longcat", "avatar"])
        if not dit:
            raise ComfyError(
                "ComfyUI's model list has no LongCat-Avatar model. If "
                "LongCat-Avatar_comfy_bf16.safetensors is already on disk, "
                "restart ComfyUI from the Engine page — it scans its model "
                "folders once, at startup. Otherwise download it on the "
                "Models page.")
        t5 = self._pick(self.text_encoders(), p.get("text_encoder", ""),
                        ["umt5"]) or self._pick(self.text_encoders(), "", ["t5"])
        if not t5:
            raise ComfyError("No umT5 text encoder found. Download "
                             "umt5-xxl-enc-bf16.safetensors on the Models page.")
        vae = self._pick(self.vaes(), p.get("vae", ""), ["wan2_1_vae"]) or \
            self._pick(self.vaes(), "", ["wan_2.1_vae"]) or \
            self._pick(self.vaes(), "", ["vae"], avoid=["2_2", "2.2"])
        if not vae:
            raise ComfyError("No Wan 2.1 VAE found. Download "
                             "Wan2_1_VAE_bf16.safetensors on the Models page.")
        lora = ""
        if p.get("lora") is not False and self.has("WanVideoLoraSelect"):
            loras = self.loras()
            lora = self._pick(loras, p.get("lora") or "",
                              ["longcat", "distill", "alpha64"]) or \
                self._pick(loras, "", ["longcat", "distill"])
        wav2vec = self._pick(self.wav2vecs(), p.get("wav2vec", ""),
                             ["wav2vec"]) if self.has("Wav2VecModelLoader") else ""
        melband = ""
        if p.get("isolate_voice", True) and self.has(MELBAND):
            melband = self._pick(self.melbands(), p.get("melband", ""),
                                 ["melband"])
        return {"dit": dit, "text_encoder": t5, "vae": vae, "lora": lora,
                "wav2vec": wav2vec, "melband": melband}

    # ------------------------------------------------------------------ #
    # graph building
    # ------------------------------------------------------------------ #
    @staticmethod
    def _match(available: dict, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in available:
                return c
        low = {k.lower(): k for k in available}
        for c in candidates:
            if c.lower() in low:
                return low[c.lower()]
        return None

    def _out(self, class_type: str, name: str, fallback: int = 0) -> int:
        """The index of a node's output, found by its name.

        ImageBatchExtendWithOverlap returns source_images, start_images,
        extended_images — all IMAGE, so a link to the wrong one is valid,
        queues, runs, and quietly drops every window but the first. Asking
        by name is what the workflow's links do.
        """
        names = (self.schema().get(class_type) or {}).get("output_name") or []
        return names.index(name) if name in names else fallback

    def _default(self, definition):
        """(True, value) for an input ComfyUI would want filled, else (False, None)."""
        if not isinstance(definition, (list, tuple)) or not definition:
            return False, None
        kind = definition[0]
        opts = definition[1] if len(definition) > 1 else {}
        if not isinstance(opts, dict):
            opts = {}
        combo = self._combo_options(definition)
        if combo:
            return True, opts.get("default", combo[0])
        if kind in ("INT", "FLOAT", "STRING", "BOOLEAN"):
            if "default" in opts:
                return True, opts["default"]
            if kind == "STRING":
                return True, ""
        return False, None

    @staticmethod
    def _dynamic_options(definition) -> dict:
        """A V3 dynamic combo's options: {key: (required, optional)}.

        Newer nodes hang settings off a dropdown — SaveVideo's format, and
        under it format.codec. The prompt carries them as "<combo>.<sub>",
        and only for the option chosen; a missing required one is a
        rejected prompt.
        """
        if not isinstance(definition, (list, tuple)) or len(definition) < 2:
            return {}
        kind, opts = definition[0], definition[1]
        if not (isinstance(kind, str) and kind.upper().startswith(
                "COMFY_DYNAMICCOMBO") and isinstance(opts, dict)):
            return {}
        out = {}
        for option in opts.get("options") or []:
            if not isinstance(option, dict) or "key" not in option:
                continue
            ins = option.get("inputs") or {}
            if "required" in ins or "optional" in ins:
                out[option["key"]] = (dict(ins.get("required") or {}),
                                      dict(ins.get("optional") or {}))
            else:
                out[option["key"]] = (dict(ins), {})
        return out

    def _fill_dynamic(self, inputs: dict, name: str, definition) -> None:
        """Choose the combo's option (what was asked, else its default) and
        fill that option's required sub-inputs, recursing into nested
        combos; sub-inputs of options not chosen are dropped."""
        options = self._dynamic_options(definition)
        if not options:
            return
        opts = definition[1] if isinstance(definition[1], dict) else {}
        chosen = inputs.get(name)
        if chosen not in options:
            chosen = opts.get("default", next(iter(options)))
            if chosen not in options:
                chosen = next(iter(options))
            inputs[name] = chosen
        required, optional = options[chosen]
        allowed = {f"{name}.{k}" for k in list(required) + list(optional)}
        for key in [k for k in inputs if k.startswith(name + ".")
                    and k.count(".") == name.count(".") + 1]:
            if key not in allowed:
                del inputs[key]
        for sub, d in required.items():
            full = f"{name}.{sub}"
            if self._dynamic_options(d):
                self._fill_dynamic(inputs, full, d)
            elif full not in inputs:
                has, value = self._default(d)
                if has:
                    inputs[full] = value
        # an optional sub-combo that was asked for is filled out too
        # (format.codec.encoding = re-encode needs its .crf)
        for sub, d in optional.items():
            full = f"{name}.{sub}"
            if full in inputs and self._dynamic_options(d):
                self._fill_dynamic(inputs, full, d)
        # anything asked for deeper down that the choices made did not
        # open (a codec setting under a format that has no such codec)
        for key in [k for k in inputs if k.startswith(name + ".")]:
            parent = key.rsplit(".", 1)[0]
            if parent != name and parent not in inputs:
                del inputs[key]

    def _node(self, class_type: str, wanted: dict) -> dict:
        """One prompt node: the wanted values matched to the node's real input
        names, and every other widget the node has filled with its default
        (an unfilled required widget is a rejected prompt)."""
        spec = self.node_inputs(class_type)
        required = (self.schema()[class_type].get("input", {})
                    .get("required", {}) or {})
        dynamic = {n: d for n, d in spec.items() if self._dynamic_options(d)}
        # sub-inputs of dynamic combos are matchable by their prompt names,
        # "<combo>.<sub>", nested ones too ("format.codec.encoding.crf")
        available = dict(spec)
        stack = list(dynamic.items())
        while stack:
            name, d = stack.pop()
            for req, opt in self._dynamic_options(d).values():
                for sub, sd in list(req.items()) + list(opt.items()):
                    full = f"{name}.{sub}"
                    if full not in available:
                        available[full] = sd
                        if self._dynamic_options(sd):
                            stack.append((full, sd))
        inputs: dict = {}
        for key, want in wanted.items():
            name = self._match(available, want["names"])
            if name is None:
                if want.get("required"):
                    raise ComfyError(
                        f"{class_type} has no input for '{key}'. This ComfyUI "
                        "does not match the Avatar Studio graph — update the "
                        "custom nodes from the Engine page.")
                continue
            value = want["value"]
            options = self._combo_options(available[name])
            if options and not isinstance(value, list) and value not in options:
                if want.get("required"):
                    raise ComfyError(f"{class_type}: '{value}' is not one of "
                                     f"its {name} options.")
                continue                 # leave it to the default below
            inputs[name] = value
        for name, definition in spec.items():
            if name in inputs or name == "control_after_generate" \
                    or name in dynamic:
                continue
            # optional widgets are left alone: an absent one is the node's
            # own default, and some are links that merely look like widgets
            if name not in required:
                continue
            has, value = self._default(definition)
            if has:
                inputs[name] = value
        for name, definition in dynamic.items():
            if name in required or name in inputs:
                self._fill_dynamic(inputs, name, definition)
        return {"class_type": class_type, "inputs": inputs}

    def build(self, p: dict, part: int = 0, prev_video: str = "") -> dict:
        """The prompt for one part of the clip (plan()); a short clip is one
        part. prev_video is the previous part's video, already in
        ComfyUI/input — every part after the first continues from its last
        13 frames.

        p: image (in ComfyUI/input), audio (in ComfyUI/input),
        audio_seconds (the clip length after the trim), audio_start,
        prompt, negative, size, steps, shift, cfg, audio_cfg, audio_scale,
        seed, lora_strength, quantization, attention, blocks_to_swap,
        tiled_vae, t5_cpu, isolate_voice, ref_frame_index,
        ref_mask_frame_range."""
        self.ensure_supported()
        if not p.get("image"):
            raise ComfyError("Add a picture of the person first.")
        if not p.get("audio"):
            raise ComfyError("Add the speech — a recording or an audio file.")
        files = self.resolve_models(p)
        seed = int(p.get("seed") if p.get("seed") not in (None, "") else
                   random.randint(0, 2**40))
        seconds = min(max(float(p.get("audio_seconds") or 0), 0.1), MAX_SECONDS)
        start = max(float(p.get("audio_start") or 0), 0.0)
        layout = plan(seconds, p.get("size"), p.get("windows_per_part"))
        frames, windows = layout["frames"], layout["windows"]
        total = layout["rendered"]
        width, height = layout["width"], layout["height"]
        if not 0 <= part < len(layout["parts"]):
            raise ComfyError(f"This clip has {len(layout['parts'])} parts; "
                             f"there is no part {part + 1}.")
        this = layout["parts"][part]
        single = len(layout["parts"]) == 1
        if part > 0 and not prev_video:
            raise ComfyError("A later part needs the part before it.")
        # Frames this part's audio slice starts at, and the frames_processed
        # the first window sees: part 0 from the top; part k from 13 frames
        # before its first new frame, so ExtendEmbeds reads its audio from
        # the slice's own start (audio index = (processed - overlap) * 2).
        joined_before = 0 if part == 0 else WINDOW + (this["first"] - 1) * STEP
        slice_from = 0 if part == 0 else joined_before - OVERLAP
        span = (WINDOW + (this["windows"] - 1) * STEP if part == 0
                else OVERLAP + this["windows"] * STEP)
        slice_frames = max(1, min(span, frames - slice_from))
        slice_start = start + slice_from / FPS
        slice_seconds = slice_frames / FPS if not single else seconds
        tiled = bool(p.get("tiled_vae", True))
        g: dict = {}

        # ---------------- model ----------------
        loader_wanted = {
            "model": {"names": ["model"], "value": files["dit"], "required": True},
            # LongCat only runs at bf16 base precision (kijai's own note)
            "precision": {"names": ["base_precision"], "value": "bf16"},
            "quant": {"names": ["quantization"],
                      "value": p.get("quantization") or "fp8_e4m3fn"},
            "device": {"names": ["load_device"], "value": "offload_device"},
            # sdpa: always there. sageattention 1.0.6 does not work with
            # LongCat, so the faster modes are an explicit choice
            "attention": {"names": ["attention_mode"],
                          "value": p.get("attention") or "sdpa"},
        }
        if self.has("WanVideoBlockSwap") and int(p.get("blocks_to_swap", 40)) > 0:
            g["1"] = self._node("WanVideoBlockSwap", {
                "blocks": {"names": ["blocks_to_swap"],
                           "value": int(p.get("blocks_to_swap", 40))},
                "img": {"names": ["offload_img_emb"], "value": False},
                "txt": {"names": ["offload_txt_emb"], "value": False},
                "nb": {"names": ["use_non_blocking"],
                       "value": bool(p.get("non_blocking", False))},
                "vace": {"names": ["vace_blocks_to_swap"], "value": 0},
                "prefetch": {"names": ["prefetch_blocks"], "value": 1},
                "debug": {"names": ["block_swap_debug"], "value": False}})
            loader_wanted["swap"] = {"names": ["block_swap_args"],
                                     "value": ["1", 0]}
        if files["lora"]:
            g["2"] = self._node("WanVideoLoraSelect", {
                "lora": {"names": ["lora"], "value": files["lora"],
                         "required": True},
                "strength": {"names": ["strength"],
                             "value": float(p.get("lora_strength", 1.0))},
                "low_mem": {"names": ["low_mem_load"], "value": False},
                "merge": {"names": ["merge_loras"], "value": False}})
            loader_wanted["lora"] = {"names": ["lora"], "value": ["2", 0]}
        g["3"] = self._node("WanVideoModelLoader", loader_wanted)
        g["4"] = self._node("WanVideoVAELoader", {
            "vae": {"names": ["model_name"], "value": files["vae"],
                    "required": True},
            "precision": {"names": ["precision"], "value": "bf16"}})
        schedulers = self.schedulers()
        sched = p.get("scheduler") or "longcat_distill_euler"
        if schedulers and sched not in schedulers:
            sched = "euler" if "euler" in schedulers else schedulers[0]
        g["5"] = self._node(SCHEDULER, {
            "scheduler": {"names": ["scheduler"], "value": sched},
            "steps": {"names": ["steps"], "value": int(p.get("steps") or 12)},
            "shift": {"names": ["shift"], "value": float(p.get("shift") or 12)},
            "start": {"names": ["start_step"], "value": 0},
            "end": {"names": ["end_step"], "value": -1}})
        t5_on_cpu = bool(p.get("t5_cpu", False))
        g["6"] = self._node("WanVideoTextEncodeCached", {
            "model": {"names": ["model_name"], "value": files["text_encoder"],
                      "required": True},
            "precision": {"names": ["precision"], "value": "bf16"},
            "positive": {"names": ["positive_prompt"],
                         "value": (p.get("prompt") or "").strip()
                         or "A person is talking to the camera.",
                         "required": True},
            "negative": {"names": ["negative_prompt"],
                         "value": p.get("negative") or DEFAULT_NEGATIVE},
            # On the GPU the weights go over as fp8 (~6.7 GB with the bf16
            # token embedding), which fits an 8 GB card: the DiT is still on
            # the CPU at this point, and the encoder is dropped right after.
            # It takes seconds. On the CPU it stays bf16 and took 5½ minutes
            # on an RTX 4060 PC (24 layers at ~14 s). Either way the result
            # is cached on disk per prompt, and a CUDA out-of-memory here
            # makes the server retry on the CPU (server.run_job).
            "quant": {"names": ["quantization"],
                      "value": "disabled" if t5_on_cpu else "fp8_e4m3fn"},
            "cache": {"names": ["use_disk_cache"], "value": True},
            "device": {"names": ["device"],
                       "value": "cpu" if t5_on_cpu else "gpu"}})

        # ---------------- the picture ----------------
        g["10"] = self._node("LoadImage", {
            "image": {"names": ["image"], "value": p["image"], "required": True}})
        if self.has("ImageResizeKJv2"):
            g["11"] = self._node("ImageResizeKJv2", {
                "image": {"names": ["image"], "value": ["10", 0], "required": True},
                "width": {"names": ["width"], "value": width},
                "height": {"names": ["height"], "value": height},
                "method": {"names": ["upscale_method"], "value": "lanczos"},
                "keep": {"names": ["keep_proportion"], "value": "crop"},
                "crop": {"names": ["crop_position"], "value": "center"},
                "div": {"names": ["divisible_by"], "value": 16},
                "device": {"names": ["device"], "value": "cpu"}})
        else:
            g["11"] = self._node("ImageScale", {
                "image": {"names": ["image"], "value": ["10", 0], "required": True},
                "method": {"names": ["upscale_method"], "value": "lanczos"},
                "width": {"names": ["width"], "value": width},
                "height": {"names": ["height"], "value": height},
                "crop": {"names": ["crop"], "value": "center"}})
        g["12"] = self._encode(["4", 0], ["11", 0], tiled)

        # ---------------- the speech ----------------
        g["20"] = self._node("LoadAudio", {
            "audio": {"names": ["audio", "file"], "value": p["audio"],
                      "required": True}})
        audio_ref: list = ["20", 0]
        if self.has("TrimAudioDuration"):
            # cut to exactly what is rendered: the speech and the soundtrack
            # the clip is muxed with must be the same stretch of audio
            g["21"] = self._node("TrimAudioDuration", {
                "audio": {"names": ["audio"], "value": audio_ref,
                          "required": True},
                "start": {"names": ["start_index", "start"],
                          "value": round(slice_start, 4)},
                "duration": {"names": ["duration"],
                             "value": round(slice_seconds, 4)}})
            audio_ref = ["21", 0]
        elif slice_start > 0:
            raise ComfyError("Starting part-way into the audio needs ComfyUI's "
                             "TrimAudioDuration node. Update ComfyUI.")
        speech_ref = audio_ref
        note = ""
        if files["melband"]:
            # the model listens to the voice alone; music and room noise in
            # the recording would otherwise move the lips
            g["22"] = self._node(MELBAND_LOADER, {
                "model": {"names": ["model_name"], "value": files["melband"],
                          "required": True}})
            g["23"] = self._node(MELBAND, {
                "model": {"names": ["model"], "value": ["22", 0], "required": True},
                "audio": {"names": ["audio"], "value": audio_ref,
                          "required": True}})
            speech_ref = ["23", self._out(MELBAND, "vocals")]
        elif p.get("isolate_voice", True):
            note = ("Voice isolation was skipped (MelBandRoFormer is not "
                    "installed or has no model), so background sound in the "
                    "recording also drives the lips.")
        if files["wav2vec"]:
            g["24"] = self._node("Wav2VecModelLoader", {
                "model": {"names": ["model"], "value": files["wav2vec"],
                          "required": True},
                "precision": {"names": ["base_precision"], "value": "fp16"},
                "device": {"names": ["load_device"], "value": "main_device"}})
        else:
            if not self.has("DownloadAndLoadWav2VecModel"):
                raise ComfyError("No wav2vec2 model. Download "
                                 "wav2vec2-chinese-base_fp16.safetensors on "
                                 "the Models page.")
            g["24"] = self._node("DownloadAndLoadWav2VecModel", {
                "model": {"names": ["model"],
                          "value": "TencentGameMate/chinese-wav2vec2-base"},
                "precision": {"names": ["base_precision"], "value": "fp16"},
                "device": {"names": ["load_device"], "value": "main_device"}})
        g["25"] = self._node(EMBEDS, {
            "model": {"names": ["wav2vec_model"], "value": ["24", 0],
                      "required": True},
            "audio": {"names": ["audio_1"], "value": speech_ref, "required": True},
            "norm": {"names": ["normalize_loudness"], "value": True},
            # counted in audio frames (32 a second), not video frames
            "frames": {"names": ["num_frames"],
                       "value": int(min(span * 2, 10000))},
            "fps": {"names": ["fps"], "value": float(AUDIO_FPS)},
            "scale": {"names": ["audio_scale"],
                      "value": float(p.get("audio_scale", 1.0))},
            "acfg": {"names": ["audio_cfg_scale"],
                     "value": float(p.get("audio_cfg", 2.0))},
            "multi": {"names": ["multi_audio_type"], "value": "para"},
            "floor": {"names": ["add_noise_floor"], "value": True},
            "smooth": {"names": ["smooth_transients"], "value": True}})

        # ---------------- the windows ----------------
        cfg = float(p.get("cfg") or 1.0)
        samplers: dict[str, int] = {}
        window_of: dict[str, int] = {}
        samples_out = self._out(SAMPLER, "samples")
        joined_out = self._out("ImageBatchExtendWithOverlap", "extended_images", 2)
        images_ref: list | None = None
        prev_samples: list | None = None
        done = 0 if part == 0 else OVERLAP        # frames_processed, slice-local
        if part > 0:
            # the previous part's last 13 frames, re-encoded: where this part
            # carries on from (and the seam its first window is pinned to)
            g["30"] = self._node("LoadVideo", {
                "file": {"names": ["file", "video"], "value": prev_video,
                         "required": True}})
            g["31"] = self._node("GetVideoComponents", {
                "video": {"names": ["video"], "value": ["30", 0],
                          "required": True}})
            g["32"] = self._node("GetImageRangeFromBatch", {
                "images": {"names": ["images"],
                           "value": ["31", self._out("GetVideoComponents",
                                                     "images")],
                           "required": True},
                "start": {"names": ["start_index"], "value": -1},
                "count": {"names": ["num_frames"], "value": OVERLAP}})
            g["33"] = self._encode(["4", 0], ["32", 0], tiled)
            prev_samples = ["33", 0]
        for local in range(this["windows"]):
            w = this["first"] + local             # the window's place in the clip
            base = 100 + local * 10
            ext, smp, rng, enc, rep, dec, cat = (str(base + k) for k in range(7))
            ext_wanted = {
                "prev": {"names": ["prev_latents"],
                         "value": ["12", 0] if w == 0 else prev_samples,
                         "required": True},
                "audio": {"names": ["audio_embeds"], "value": ["25", 0],
                          "required": True},
                "frames": {"names": ["num_frames"], "value": WINDOW},
                # the first window starts from the picture alone (overlap 1);
                # later ones continue from the last 13 frames
                "overlap": {"names": ["overlap"], "value": 1 if w == 0 else OVERLAP},
                "processed": {"names": ["frames_processed"], "value": done},
                "pad": {"names": ["if_not_enough_audio"], "value": "pad_with_start"},
                "ref_index": {"names": ["ref_frame_index"],
                              "value": int(p.get("ref_frame_index", 10))},
                "ref_range": {"names": ["ref_mask_frame_range"],
                              "value": int(p.get("ref_mask_frame_range", 3))},
            }
            if w > 0:
                # the picture stays the identity anchor for every window
                ext_wanted["ref"] = {"names": ["ref_latent"], "value": ["12", 0]}
            g[ext] = self._node(EXTEND, ext_wanted)
            g[smp] = self._node(SAMPLER, {
                "model": {"names": ["model"], "value": ["3", 0], "required": True},
                "embeds": {"names": ["image_embeds"], "value": [ext, 0],
                           "required": True},
                "cfg": {"names": ["cfg"], "value": cfg},
                "seed": {"names": ["seed"], "value": (seed + w) % 2**64},
                "offload": {"names": ["force_offload"], "value": True},
                "scheduler": {"names": ["scheduler"], "value": ["5", 0],
                              "required": True},
                "text": {"names": ["text_embeds"], "value": ["6", 0],
                         "required": True},
                "noise": {"names": ["add_noise_to_samples"], "value": False}})
            samplers[smp] = w
            if w == 0:
                g[dec] = self._decode(["4", 0], [smp, samples_out], tiled)
                images_ref = [dec, 0]
                done = WINDOW
            elif local == 0:
                # a part's first window: pin its overlap to the previous
                # part's frames, then keep only the 80 frames that are new
                g[rep] = self._node("ReplaceVideoLatentFrames", {
                    "dest": {"names": ["destination"],
                             "value": [smp, samples_out], "required": True},
                    "src": {"names": ["source"], "value": ["33", 0],
                            "required": True},
                    "index": {"names": ["index"], "value": 0}})
                g[dec] = self._decode(["4", 0], [rep, 0], tiled)
                g[cat] = self._node("GetImageRangeFromBatch", {
                    "images": {"names": ["images"], "value": [dec, 0],
                               "required": True},
                    "start": {"names": ["start_index"], "value": OVERLAP},
                    "count": {"names": ["num_frames"], "value": STEP}})
                images_ref = [cat, 0]
                done += STEP
            else:
                # the overlap frames are swapped for the decoded frames they
                # continue from, re-encoded, so the seam does not drift
                g[rng] = self._node("GetImageRangeFromBatch", {
                    "images": {"names": ["images"], "value": images_ref,
                               "required": True},
                    "start": {"names": ["start_index"], "value": -1},
                    "count": {"names": ["num_frames"], "value": OVERLAP}})
                g[enc] = self._encode(["4", 0], [rng, 0], tiled)
                g[rep] = self._node("ReplaceVideoLatentFrames", {
                    "dest": {"names": ["destination"],
                             "value": [smp, samples_out],
                             "required": True},
                    "src": {"names": ["source"], "value": [enc, 0],
                            "required": True},
                    "index": {"names": ["index"], "value": 0}})
                g[dec] = self._decode(["4", 0], [rep, 0], tiled)
                g[cat] = self._node("ImageBatchExtendWithOverlap", {
                    "source": {"names": ["source_images"], "value": images_ref,
                               "required": True},
                    "new": {"names": ["new_images"], "value": [dec, 0],
                            "required": True},
                    "overlap": {"names": ["overlap"], "value": OVERLAP},
                    "side": {"names": ["overlap_side"], "value": "new_images"},
                    "mode": {"names": ["overlap_mode"], "value": "cut"}})
                images_ref = [cat, joined_out]
                done += STEP
            prev_samples = [smp, samples_out]
            window_of.update({k: w for k in (ext, smp, rng, enc, rep, dec, cat)
                              if k in g})

        # ---------------- the clip ----------------
        frames_ref = images_ref
        if this["trim"]:
            # the last window is padded with audio that is not there; cut the
            # picture back to the length of the speech
            g["90"] = self._node("GetImageRangeFromBatch", {
                "images": {"names": ["images"], "value": images_ref,
                           "required": True},
                "start": {"names": ["start_index"], "value": 0},
                "count": {"names": ["num_frames"], "value": this["frames"]}})
            frames_ref = ["90", 0]
        video_wanted = {
            "images": {"names": ["images"], "value": frames_ref, "required": True},
            "fps": {"names": ["fps"], "value": float(FPS)}}
        if single:
            # the original audio, music and all — only the lips listen to
            # the isolated voice. A part has none: the app lays the whole
            # soundtrack under the joined parts in one piece
            video_wanted["audio"] = {"names": ["audio"], "value": audio_ref}
        g["91"] = self._node("CreateVideo", video_wanted)
        save_wanted = {
            "video": {"names": ["video"], "value": ["91", 0], "required": True},
            "prefix": {"names": ["filename_prefix"],
                       "value": "video/LongCatAvatar" if single
                       else "video/LongCatAvatar_part"}}
        if not single:
            # a part is read back for the next part's seam and re-encoded
            # into the clip: keep it near-lossless where SaveVideo allows
            save_wanted.update({
                "format": {"names": ["format"], "value": "mp4"},
                "codec": {"names": ["format.codec"], "value": "h264"},
                "encoding": {"names": ["format.codec.encoding"],
                             "value": "re-encode"},
                "crf": {"names": ["format.codec.encoding.crf"], "value": 10.0}})
        g["92"] = self._node("SaveVideo", save_wanted)
        if self.rehearsal:
            g = rehearse(g, part)

        return {"prompt": g, "seed": seed, "files": files, "frames": frames,
                "windows": windows, "rendered": total, "samplers": samplers,
                "window_of": window_of, "part": part,
                "steps": int(p.get("steps") or 12),
                "parts": layout["parts"], "this_part": this,
                "width": width, "height": height, "fps": FPS,
                "size": f"{width}x{height}",
                "seconds": round(frames / FPS, 2), "note": note,
                "part_ram": layout["part_ram"]}

    def _encode(self, vae: list, image: list, tiled: bool) -> dict:
        return self._node("WanVideoEncode", {
            "vae": {"names": ["vae"], "value": vae, "required": True},
            "image": {"names": ["image"], "value": image, "required": True},
            "tiling": {"names": ["enable_vae_tiling"], "value": tiled},
            "noise": {"names": ["noise_aug_strength"], "value": 0.0},
            "strength": {"names": ["latent_strength"], "value": 1.0}})

    def _decode(self, vae: list, samples: list, tiled: bool) -> dict:
        return self._node("WanVideoDecode", {
            "vae": {"names": ["vae"], "value": vae, "required": True},
            "samples": {"names": ["samples"], "value": samples, "required": True},
            "tiling": {"names": ["enable_vae_tiling"], "value": tiled}})

    # ------------------------------------------------------------------ #
    # queue / results
    # ------------------------------------------------------------------ #
    def queue(self, prompt: dict) -> str:
        body = {"prompt": prompt, "client_id": self.client_id}
        r = requests.post(f"{self.url}/prompt", json=body, timeout=60)
        if r.status_code >= 400:
            try:
                raise ComfyError(_readable(r.json()))
            except ValueError:
                raise ComfyError(r.text[:400])
        return r.json()["prompt_id"]

    def interrupt(self) -> None:
        try:
            requests.post(f"{self.url}/interrupt", timeout=10)
        except Exception:
            pass

    def cancel(self, prompt_id: str) -> bool:
        """Stop this prompt and only this one; True once that is done.

        A bare /interrupt stops whatever is running, which may be another
        job; a prompt still waiting is taken off the queue instead. False
        when ComfyUI could not be asked, so the caller tries again.
        """
        try:
            r = requests.get(f"{self.url}/queue", timeout=10)
            r.raise_for_status()
            q = r.json()
            running = {e[1] for e in q.get("queue_running") or []
                       if isinstance(e, list) and len(e) > 1}
            if prompt_id in running:
                requests.post(f"{self.url}/interrupt", timeout=10) \
                    .raise_for_status()
            else:
                requests.post(f"{self.url}/queue",
                              json={"delete": [prompt_id]}, timeout=10) \
                    .raise_for_status()
            return True
        except Exception:
            return False

    def history(self, prompt_id: str) -> dict:
        r = requests.get(f"{self.url}/history/{prompt_id}", timeout=20)
        r.raise_for_status()
        return r.json().get(prompt_id) or {}

    VIDEO_SUFFIX = (".mp4", ".webm", ".mkv", ".mov", ".gif", ".avi")

    def outputs(self, prompt_id: str) -> list[dict]:
        hist = self.history(prompt_id)
        found = []
        for node_out in (hist.get("outputs") or {}).values():
            for key in ("videos", "video", "gifs", "images", "files"):
                for item in node_out.get(key, []) or []:
                    if not isinstance(item, dict) or not item.get("filename"):
                        continue
                    if item.get("type") == "temp":
                        continue
                    if key in ("videos", "video", "gifs", "files") or \
                            item["filename"].lower().endswith(self.VIDEO_SUFFIX):
                        found.append(item)
        return found

    def failed(self, prompt_id: str) -> str | None:
        status = (self.history(prompt_id).get("status") or {})
        if status.get("status_str") == "error":
            for kind, data in status.get("messages", []):
                if kind == "execution_error":
                    msg = str(data.get("exception_message", ""))
                    low = msg.lower()
                    if "deserializing header" in low or \
                            "incomplete metadata" in low or \
                            "safetensor" in low and "invalid" in low:
                        return (f"{data.get('node_type')}: a model file it "
                                "loads is damaged or unfinished ({}). Delete "
                                "it on the Models page and download it "
                                "again.".format(msg.strip()[:80]))
                    if "nonetype" in low and "subscriptable" in low and \
                            "wav2vec" in str(data.get("node_type", "")).lower():
                        return ("MultiTalkWav2VecEmbeds: ComfyUI has "
                                "transformers 5, which returns nothing from "
                                "wav2vec2. Press Restart ComfyUI on the Engine "
                                "page: the app adds its compatibility node "
                                "first. Or install 4.x: python -m pip install "
                                "\"transformers<5\"")
                    if "out of memory" in low:
                        return (f"{data.get('node_type')}: out of memory. Use "
                                "480p, raise Block swap in Settings, keep fp8 "
                                "weights and tiled VAE on.")
                    return f"{data.get('node_type')}: {msg}"
            return "ComfyUI reported an error while generating."
        return None

    def view(self, item: dict):
        params = {"filename": item.get("filename", ""),
                  "subfolder": item.get("subfolder", ""),
                  "type": item.get("type", "output")}
        return requests.get(f"{self.url}/view", params=params, stream=True,
                            timeout=600)

    def upload(self, file_storage) -> str:
        """Into ComfyUI/input. /upload/image takes any file, audio included —
        LoadAudio lists the same folder."""
        files = {"image": (file_storage.filename, file_storage.stream,
                           file_storage.mimetype or "application/octet-stream")}
        r = requests.post(f"{self.url}/upload/image", files=files,
                          data={"type": "input", "overwrite": "false"}, timeout=600)
        r.raise_for_status()
        # LoadImage and LoadAudio list ComfyUI/input: the cached schema does
        # not have this file yet, and a build against it would refuse it
        with self._lock:
            self._schema = None
        data = r.json()
        name = data.get("name") or file_storage.filename
        sub = data.get("subfolder") or ""
        return f"{sub}/{name}" if sub else name


def _readable(err: dict) -> str:
    for node_id, info in (err.get("node_errors") or {}).items():
        for e in info.get("errors", []):
            return (f"{info.get('class_type', 'node ' + str(node_id))}: "
                    f"{e.get('message')} {e.get('details', '')}".strip())
    top = err.get("error") or {}
    if top:
        return f"{top.get('message', 'Rejected by ComfyUI')} " \
               f"{top.get('details', '')}".strip()
    return json.dumps(err)[:300]
