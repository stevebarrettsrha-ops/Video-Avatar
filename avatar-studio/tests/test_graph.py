"""comfy.py — the graphs handed to ComfyUI.

The mock's schema is the real /object_info of ComfyUI with the three node
packs loaded, and it validates every prompt the way ComfyUI does — unknown
inputs, values outside a combo's options, missing required inputs (nested
dynamic-combo ones too) and dangling links are all rejected. So "accepted"
here means the real server takes it too; a 3-window graph built this way
was also posted to a real CPU-only ComfyUI and passed its validation.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import comfy                                       # noqa: E402
from comfy import ComfyClient, ComfyError          # noqa: E402
from harness import Suite, comfy as mock_comfy     # noqa: E402


def nodes_of(graph: dict, cls: str) -> list[tuple[str, dict]]:
    return sorted([(nid, n) for nid, n in graph.items()
                   if n["class_type"] == cls], key=lambda t: int(t[0]))


def upload(url: str, name: str, body: bytes = b"x") -> None:
    requests.post(f"{url}/upload/image",
                  files={"image": (name, io.BytesIO(body))},
                  data={"type": "input"}, timeout=10).raise_for_status()


BASE = {"image": "face.png", "audio": "speech.wav", "prompt": "a woman talks",
        "seed": 7}


def run(slow: bool = False) -> Suite:
    s = Suite("graph")
    with mock_comfy() as mock:
        upload(mock.url, "face.png")
        upload(mock.url, "speech.wav")
        client = ComfyClient(mock.url)

        # -- schema reads ---------------------------------------------------
        s.check("the LongCat Avatar extender is seen", client.has(comfy.EXTEND))
        s.equal("the DiT list is the WanVideoModelLoader combo",
                client._pick(client.dits(), "", ["longcat", "avatar"]),
                "LongCat-Avatar_comfy_bf16.safetensors")
        s.check("the distill scheduler is offered",
                "longcat_distill_euler" in client.schedulers())
        s.check("sdpa is an attention mode", "sdpa" in client.attention_modes())
        s.check("MelBandRoFormer files only, not the DiT that shares its folder",
                client.melbands() == ["MelBandRoformer_fp32.safetensors"])
        s.check("capabilities report everything on",
                all(client.capabilities().values()), str(client.capabilities()))
        s.equal("nothing required is missing", client.missing_nodes(), [])

        # -- a short clip: one window ---------------------------------------
        built = client.build({**BASE, "audio_seconds": 4.0})
        g = built["prompt"]
        client.queue(g)
        s.check("ComfyUI accepts a one-window clip", True)
        s.equal("4 s is 64 frames in one window",
                (built["frames"], built["windows"]), (64, 1))
        s.equal("one sampler", len(nodes_of(g, comfy.SAMPLER)), 1)
        ext = nodes_of(g, comfy.EXTEND)[0][1]["inputs"]
        s.check("the first window starts from the picture (overlap 1, none "
                "processed, no ref latent) — as the workflow",
                ext["overlap"] == 1 and ext["frames_processed"] == 0
                and "ref_latent" not in ext and ext["num_frames"] == 93
                and ext["prev_latents"] == [nodes_of(g, "WanVideoEncode")[0][0], 0])
        trim = nodes_of(g, "GetImageRangeFromBatch")
        s.check("the 93 frames are cut back to the 64 the speech covers",
                len(trim) == 1 and trim[0][1]["inputs"]["start_index"] == 0
                and trim[0][1]["inputs"]["num_frames"] == 64)
        s.check("no stitching for a single window",
                not nodes_of(g, "ImageBatchExtendWithOverlap")
                and not nodes_of(g, "ReplaceVideoLatentFrames"))

        # -- the workflow's own shape: three windows ------------------------
        built = client.build({**BASE, "audio_seconds": 12.5})
        g = built["prompt"]
        client.queue(g)
        s.check("ComfyUI accepts a three-window clip", True)
        s.equal("12.5 s: 200 frames, 3 windows, 253 rendered",
                (built["frames"], built["windows"], built["rendered"]),
                (200, 3, 253))
        exts = [n["inputs"] for _, n in nodes_of(g, comfy.EXTEND)]
        smps = nodes_of(g, comfy.SAMPLER)
        s.equal("frames_processed walks 0, 93, 173 — the workflow's counts",
                [e["frames_processed"] for e in exts], [0, 93, 173])
        s.equal("overlap 1 for the first window, 13 after",
                [e["overlap"] for e in exts], [1, 13, 13])
        s.check("each later window continues from the previous sampler's "
                "raw latents",
                exts[1]["prev_latents"] == [smps[0][0], 0]
                and exts[2]["prev_latents"] == [smps[1][0], 0])
        enc0 = nodes_of(g, "WanVideoEncode")[0][0]
        s.check("the picture's latent anchors every later window",
                all(e.get("ref_latent") == [enc0, 0] for e in exts[1:]))
        s.equal("each window samples with its own seed",
                [n["inputs"]["seed"] for _, n in smps], [7, 8, 9])
        s.equal("the progress map names each sampler's window",
                built["samplers"], {smps[0][0]: 0, smps[1][0]: 1, smps[2][0]: 2})
        reps = nodes_of(g, "ReplaceVideoLatentFrames")
        s.check("the seam latents are replaced by the re-encoded last 13 "
                "decoded frames",
                len(reps) == 2 and all(r[1]["inputs"]["index"] == 0 for r in reps))
        ranges = [n["inputs"] for _, n in nodes_of(g, "GetImageRangeFromBatch")
                  if n["inputs"]["start_index"] == -1]
        s.check("the overlap frames come from the end of the batch so far",
                len(ranges) == 2 and all(r["num_frames"] == 13 for r in ranges))
        cats = nodes_of(g, "ImageBatchExtendWithOverlap")
        s.check("windows are joined with cut, on the new images' side",
                len(cats) == 2 and all(
                    c[1]["inputs"]["overlap_mode"] == "cut"
                    and c[1]["inputs"]["overlap_side"] == "new_images"
                    and c[1]["inputs"]["overlap"] == 13 for c in cats))
        s.check("the second join extends the first join's batch",
                cats[1][1]["inputs"]["source_images"] == [cats[0][0], 0])

        # -- the audio half -------------------------------------------------
        trim_audio = nodes_of(g, "TrimAudioDuration")[0]
        s.equal("the audio is cut to the stretch rendered",
                (trim_audio[1]["inputs"]["start_index"],
                 trim_audio[1]["inputs"]["duration"]), (0.0, 12.5))
        mel = nodes_of(g, comfy.MELBAND)[0]
        emb = nodes_of(g, comfy.EMBEDS)[0][1]["inputs"]
        s.check("the lips listen to the isolated voice",
                emb["audio_1"] == [mel[0], 0])
        s.check("the embeds run at 32 a second over every rendered frame",
                emb["fps"] == 32.0 and emb["num_frames"] == 2 * 253)
        s.check("the workflow's audio shaping is on",
                emb["add_noise_floor"] and emb["smooth_transients"]
                and emb["normalize_loudness"])
        cv = nodes_of(g, "CreateVideo")[0][1]["inputs"]
        s.check("the clip carries the original audio, music and all, at 16 fps",
                cv["audio"] == [trim_audio[0], 0] and cv["fps"] == 16.0)
        sv = nodes_of(g, "SaveVideo")[0][1]["inputs"]
        s.check("SaveVideo's nested dynamic combos are filled",
                sv.get("format") == "auto" and sv.get("format.codec") == "auto")

        # -- the model half -------------------------------------------------
        loader = nodes_of(g, "WanVideoModelLoader")[0][1]["inputs"]
        s.check("bf16 base precision, fp8 storage, sdpa, off the card",
                loader["base_precision"] == "bf16"
                and loader["quantization"] == "fp8_e4m3fn"
                and loader["attention_mode"] == "sdpa"
                and loader["load_device"] == "offload_device")
        swap = nodes_of(g, "WanVideoBlockSwap")[0][1]["inputs"]
        s.equal("40 of 48 blocks swapped by default", swap["blocks_to_swap"], 40)
        lora = nodes_of(g, "WanVideoLoraSelect")[0][1]["inputs"]
        s.check("the alpha64 distill LoRA at 1.0, not merged",
                lora["lora"] == "LongCat_distill_lora_alpha64_bf16.safetensors"
                and lora["strength"] == 1.0 and lora["merge_loras"] is False)
        sch = nodes_of(g, comfy.SCHEDULER)[0][1]["inputs"]
        s.equal("longcat_distill_euler, 12 steps, shift 12",
                (sch["scheduler"], sch["steps"], sch["shift"]),
                ("longcat_distill_euler", 12, 12.0))
        t5 = nodes_of(g, "WanVideoTextEncodeCached")[0][1]["inputs"]
        s.check("umT5 encodes on the CPU, cached, with the workflow's negative",
                t5["device"] == "cpu" and t5["use_disk_cache"]
                and t5["negative_prompt"] == comfy.DEFAULT_NEGATIVE)
        rs = nodes_of(g, "ImageResizeKJv2")[0][1]["inputs"]
        s.check("the picture is centre-cropped to 832×480, a multiple of 16",
                (rs["width"], rs["height"], rs["keep_proportion"],
                 rs["crop_position"], rs["divisible_by"])
                == (832, 480, "crop", "center", 16))

        # -- choices travel -------------------------------------------------
        built = client.build({**BASE, "audio_seconds": 3, "audio_start": 5,
                              "size": "480x832", "quantization": "disabled",
                              "blocks_to_swap": 0, "t5_cpu": False,
                              "tiled_vae": False, "audio_cfg": 4,
                              "steps": 8, "isolate_voice": False,
                              "attention": "sageattn"})
        g = built["prompt"]
        client.queue(g)
        s.check("ComfyUI accepts a graph with every choice flipped", True)
        s.equal("portrait is 480×832", (built["width"], built["height"]),
                (480, 832))
        s.equal("the trim starts where asked",
                nodes_of(g, "TrimAudioDuration")[0][1]["inputs"]["start_index"], 5.0)
        s.check("no block swap node when it is 0",
                not nodes_of(g, "WanVideoBlockSwap")
                and "block_swap_args" not in
                nodes_of(g, "WanVideoModelLoader")[0][1]["inputs"])
        s.check("isolation off feeds the trimmed audio straight in",
                not nodes_of(g, comfy.MELBAND)
                and nodes_of(g, comfy.EMBEDS)[0][1]["inputs"]["audio_1"]
                == [nodes_of(g, "TrimAudioDuration")[0][0], 0])
        s.equal("lip-sync strength lands on audio_cfg_scale",
                nodes_of(g, comfy.EMBEDS)[0][1]["inputs"]["audio_cfg_scale"], 4.0)
        s.check("tiling off on encode and decode",
                all(not n["inputs"]["enable_vae_tiling"]
                    for _, n in nodes_of(g, "WanVideoEncode")
                    + nodes_of(g, "WanVideoDecode")))
        s.equal("bf16 storage asked for, bf16 storage given",
                nodes_of(g, "WanVideoModelLoader")[0][1]["inputs"]["quantization"],
                "disabled")
        s.equal("the T5 on the GPU when asked",
                nodes_of(g, "WanVideoTextEncodeCached")[0][1]["inputs"]["device"],
                "gpu")
        s.equal("an attention mode the loader offers is kept",
                nodes_of(g, "WanVideoModelLoader")[0][1]["inputs"]["attention_mode"],
                "sageattn")
        built = client.build({**BASE, "audio_seconds": 2,
                              "scheduler": "not-a-scheduler"})
        s.equal("an unknown scheduler falls back to euler",
                nodes_of(built["prompt"], comfy.SCHEDULER)[0][1]["inputs"]
                ["scheduler"], "euler")

        # -- what is refused, and how it says so ----------------------------
        s.fails_with("no picture is a clear message",
                     lambda: client.build({**BASE, "image": "",
                                           "audio_seconds": 2}),
                     ComfyError, "picture")
        s.fails_with("no audio is a clear message",
                     lambda: client.build({**BASE, "audio": "",
                                           "audio_seconds": 2}),
                     ComfyError, "speech")
        s.fails_with("a file ComfyUI does not have is named before queueing",
                     lambda: client.build({**BASE, "audio": "never-uploaded.wav",
                                           "audio_seconds": 2}),
                     ComfyError, "never-uploaded.wav")

        class Up:
            filename = "late.wav"
            stream = io.BytesIO(b"RIFF")
            mimetype = "audio/wav"
        client.schema()                  # cached before the upload
        name = client.upload(Up())
        built = client.build({**BASE, "audio": name, "audio_seconds": 2})
        client.queue(built["prompt"])
        s.check("a file uploaded after the schema was cached is still usable",
                True)

    with mock_comfy(MOCK_OMIT="WanVideoLongCatAvatarExtendEmbeds") as mock:
        client = ComfyClient(mock.url)
        s.fails_with("a wrapper without LongCat names what is missing and "
                     "where it comes from",
                     lambda: client.build({**BASE, "audio_seconds": 2}),
                     ComfyError, "WanVideoLongCatAvatarExtendEmbeds")
        s.equal("and says so in the status list", client.missing_nodes(),
                ["WanVideoLongCatAvatarExtendEmbeds"])

    with mock_comfy(MOCK_NO_MELBAND_MODEL="1") as mock:
        upload(mock.url, "face.png")
        upload(mock.url, "speech.wav")
        client = ComfyClient(mock.url)
        built = client.build({**BASE, "audio_seconds": 2})
        client.queue(built["prompt"])
        s.check("no separator model: the clip still renders, with a note",
                not nodes_of(built["prompt"], comfy.MELBAND)
                and "isolation" in built["note"])

    with mock_comfy(MOCK_OMIT="Wav2VecModelLoader") as mock:
        upload(mock.url, "face.png")
        upload(mock.url, "speech.wav")
        client = ComfyClient(mock.url)
        built = client.build({**BASE, "audio_seconds": 2})
        client.queue(built["prompt"])
        s.check("an older wrapper falls back to (Down)load Wav2Vec Model",
                nodes_of(built["prompt"], "DownloadAndLoadWav2VecModel")
                [0][1]["inputs"]["model"] == "TencentGameMate/chinese-wav2vec2-base")

    with mock_comfy(MOCK_BLANK_UNETS="99") as mock:
        client = ComfyClient(mock.url)
        s.fails_with("weights the engine has not scanned say to restart it",
                     lambda: client.build({**BASE, "audio_seconds": 2}),
                     ComfyError, "restart ComfyUI")
    return s
