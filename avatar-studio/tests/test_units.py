"""The arithmetic and the tables, against kijai's workflow's own numbers.

The workflow (assets/LongCatAvatar_audio_image_to_video_example_01.json)
renders 93-frame windows at 16 fps, each later window re-using the last 13
frames, with the wav2vec2 embeds at 32 frames a second. Those numbers are read
from the workflow file here rather than retyped, so the app cannot drift from
it unnoticed — and the page's copy of the maths is run through node and held
to Python's.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bootstrap                                   # noqa: E402
import comfy                                       # noqa: E402
from harness import Suite                          # noqa: E402


def workflow() -> dict:
    return json.loads((ROOT / "assets" /
                       "LongCatAvatar_audio_image_to_video_example_01.json")
                      .read_text())


def widgets(wf: dict, cls: str) -> list:
    return [n.get("widgets_values") for n in wf["nodes"] if n["type"] == cls]


def run(slow: bool = False) -> Suite:
    s = Suite("units")
    wf = workflow()

    # -- the numbers come from the workflow --------------------------------
    consts = {n.get("title"): n["widgets_values"][0] for n in wf["nodes"]
              if n["type"] == "INTConstant"}
    s.equal("93 frames a window, as the workflow's frames_per_window",
            comfy.WINDOW, consts.get("frames_per_window"))
    s.equal("13 frames of overlap, as the workflow's Overlap", comfy.OVERLAP,
            consts.get("Overlap"))
    combine = widgets(wf, "VHS_VideoCombine")[0]
    s.equal("16 fps, as the workflow's VideoCombine", comfy.FPS,
            combine["frame_rate"])
    embeds = widgets(wf, "MultiTalkWav2VecEmbeds")[0]
    s.equal("wav2vec2 embeds at 32 a second (audio stride 2)", comfy.AUDIO_FPS,
            embeds[2])
    sched = widgets(wf, "WanVideoSchedulerv2")[0]
    s.equal("the distill scheduler the workflow ships with",
            sched[0], "longcat_distill_euler")
    s.check("windows divide by 4 plus 1, as the Wan VAE needs",
            (comfy.WINDOW - 1) % 4 == 0 and (comfy.STEP) % 4 == 0)

    # -- frames and windows -------------------------------------------------
    s.equal("1 s of speech is 16 frames", comfy.frame_count(1), 16)
    s.equal("a sliver still makes a frame", comfy.frame_count(0.01), 1)
    s.equal("12.5 s is 200 frames", comfy.frame_count(12.5), 200)
    s.equal("float noise does not add a frame (0.1 × 16 × 3)",
            comfy.frame_count(4.8), 77)
    s.equal("up to 93 frames is one window", comfy.window_count(93), 1)
    s.equal("94 frames needs a second", comfy.window_count(94), 2)
    s.equal("173 frames is exactly two windows", comfy.window_count(173), 2)
    s.equal("174 frames is three", comfy.window_count(174), 3)
    s.equal("three windows render 253 frames", comfy.rendered_frames(3), 253)
    s.equal("the two-minute cap is 24 windows",
            comfy.window_count(comfy.frame_count(comfy.MAX_SECONDS)), 24)
    for sec in (0.5, 3, 5.8, 5.9, 10.81, 11, 30, 60, 120):
        f = comfy.frame_count(sec)
        w = comfy.window_count(f)
        if not (comfy.rendered_frames(w) >= f
                and (w == 1 or comfy.rendered_frames(w - 1) < f)):
            s.check(f"{sec} s: the fewest windows that cover it", False,
                    f"{f} frames, {w} windows")
            break
    else:
        s.check("every length gets the fewest windows that cover it", True)
    s.check("every size divides by 16",
            all(w % 16 == 0 and h % 16 == 0 for w, h in comfy.SIZES.values()))
    s.equal("the default size is the workflow's 832×480",
            comfy.output_size(None), (832, 480))
    s.equal("an unknown size falls back to it", comfy.output_size("9x9"),
            (832, 480))

    # -- the page computes the same ----------------------------------------
    page = (ROOT / "web" / "index.html").read_text()
    if shutil.which("node"):
        import re
        script = re.findall(r"<script>(.*?)</script>", page, re.S)[0]
        consts_js = re.search(r"var FPS = .*?;\n", script).group(0)
        fns = "".join(re.search(rf"function {n}\(.*?\n}}\n", script, re.S).group(0)
                      for n in ("frameCount", "windowCount"))
        cases = [0.01, 0.5, 1, 4.8, 5.8, 5.81, 10.8, 10.81, 12.5, 59.99, 120]
        probe = consts_js + fns + "console.log(JSON.stringify(" + json.dumps(
            cases) + ".map(function(s){var f=frameCount(s);return [f,windowCount(f)];})))"
        out = subprocess.run(["node", "-e", probe], capture_output=True,
                             text=True, timeout=30)
        got = json.loads(out.stdout or "null")
        want = [[comfy.frame_count(c), comfy.window_count(comfy.frame_count(c))]
                for c in cases]
        s.equal("the page's frame and window counts match Python's", got, want)
        sizes = json.loads(re.search(r"var SIZES = (\{.*?\});", script, re.S)
                           .group(1).replace("\n", " ").replace("\"", "\"")
                           .replace("'", "\""))
        s.equal("the page offers exactly the server's sizes",
                {k: tuple(v) for k, v in sizes.items()}, comfy.SIZES)
        neg = re.search(r'var DEFAULT_NEGATIVE = (.*?);\n', script, re.S).group(1)
        neg_text = subprocess.run(["node", "-e", f"console.log({neg})"],
                                  capture_output=True, text=True).stdout.strip()
        s.equal("the page's default negative prompt is the server's",
                neg_text, comfy.DEFAULT_NEGATIVE)
    else:
        print("  --   node is not installed, so the page's maths was not run")

    s.equal("the server's negative prompt is the workflow's",
            comfy.DEFAULT_NEGATIVE,
            widgets(wf, "WanVideoTextEncodeCached")[0][3])

    # -- the weight set -----------------------------------------------------
    items = bootstrap.model_set({})
    names = {m["name"] for m in items}
    s.check("the set is the five files the workflow loads",
            names == {"LongCat-Avatar_comfy_bf16.safetensors",
                      "LongCat_distill_lora_alpha64_bf16.safetensors",
                      "umt5-xxl-enc-bf16.safetensors",
                      "Wan2_1_VAE_bf16.safetensors",
                      "wav2vec2-chinese-base_fp16.safetensors"}, str(names))
    s.check("the DiT comes from the Avatar folder of Kijai/LongCat-Video_comfy",
            any(m["repo"] == "Kijai/LongCat-Video_comfy"
                and m["path"] == "Avatar/LongCat-Avatar_comfy_bf16.safetensors"
                and m["folder"] == "diffusion_models" for m in items))
    s.check("the distill LoRA is the alpha64 one the workflow's note asks for",
            any(m["name"].startswith("LongCat_distill_lora_alpha64")
                and m["folder"] == "loras" for m in items))
    s.check("wav2vec2 lands where Wav2VecModelLoader looks",
            any(m["folder"] == "wav2vec2" for m in items))
    extras = bootstrap.extra_models({"want_melband": True})
    s.check("MelBandRoFormer lands in diffusion_models, where its loader looks",
            [m["folder"] for m in extras] == ["diffusion_models"])
    s.equal("no MelBandRoFormer file without its node",
            bootstrap.extra_models({"want_melband": False}), [])
    s.check("every set file names its repo and path",
            all(m["repo"] and m["path"] for m in items + extras))
    s.check("the wrapper and KJNodes are not optional",
            {n["id"] for n in bootstrap.CUSTOM_NODES if not n["optional"]}
            == {"wrapper", "kjnodes"})
    s.equal("every node pack has a marker the engine proves it with",
            set(bootstrap.NODE_MARKERS),
            {n["id"] for n in bootstrap.CUSTOM_NODES if n["id"] != "manager"})
    s.equal("fp8 and bf16 are the two ways the model can sit in memory",
            {k: v["quantization"] for k, v in bootstrap.PRECISIONS.items()},
            {"fp8": "fp8_e4m3fn", "bf16": "disabled"})

    # -- the preflight verdicts --------------------------------------------
    gib = 1024 ** 3
    v, notes = bootstrap.assess(8 * gib, 32 * gib, 500e9, 42e9, 16e9)
    s.equal("an RTX 4060 with 32 GB of RAM is tight, not hard", v, "tight")
    s.check("and the note says it is untimed", any("Untimed" in n for n in notes))
    v, _ = bootstrap.assess(6 * gib, 32 * gib, 500e9, 42e9, 16e9)
    s.equal("6 GB of VRAM is hard", v, "hard")
    v, _ = bootstrap.assess(24 * gib, 64 * gib, 500e9, 42e9, 16e9)
    s.equal("a 24 GB card with 64 GB of RAM is ok", v, "ok")
    v, _ = bootstrap.assess(24 * gib, 8 * gib, 500e9, 42e9, 16e9)
    s.equal("8 GB of RAM against a 16 GB peak is hard", v, "hard")
    v, _ = bootstrap.assess(24 * gib, 64 * gib, 30e9, 42e9, 16e9)
    s.equal("no disk for the download is hard", v, "hard")
    v, notes = bootstrap.assess(8 * gib, 32 * gib, 500e9, 42e9, 30e9,
                                precision="bf16")
    s.check("bf16 on a small card says to use fp8",
            any("fp8 is the setting" in n for n in notes))
    return s
