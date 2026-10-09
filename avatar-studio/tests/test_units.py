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
    s.equal("two minutes is 24 windows",
            comfy.window_count(comfy.frame_count(120)), 24)
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
        fns_ram = consts_js + re.search(r"var FRAME_COPIES.*?\n", script).group(0) \
            + re.search(r"function frameRam\(.*?\n", script).group(0) \
            + re.search(r"function framesFitting\(.*?\n}\n", script, re.S).group(0) \
            + re.search(r"function windowsPerPart\(.*?\n}\n", script, re.S).group(0)
        probe = fns_ram + "console.log(JSON.stringify([frameRam(500,832,480)," \
            "framesFitting(832,480),framesFitting(1280,720)," \
            "windowsPerPart(832,480),windowsPerPart(1280,720)," \
            "windowsPerPart(640,640),windowsPerPart(480,832)]))"
        got = json.loads(subprocess.run(["node", "-e", probe], capture_output=True,
                                        text=True).stdout or "null")
        s.equal("the page's RAM estimate matches Python's", got,
                [comfy.frame_ram(500, 832, 480), comfy.frames_fitting(832, 480),
                 comfy.frames_fitting(1280, 720),
                 comfy.windows_per_part(832, 480),
                 comfy.windows_per_part(1280, 720),
                 comfy.windows_per_part(640, 640),
                 comfy.windows_per_part(480, 832)])
        neg = re.search(r'var DEFAULT_NEGATIVE = (.*?);\n', script, re.S).group(1)
        neg_text = subprocess.run(["node", "-e", f"console.log({neg})"],
                                  capture_output=True, text=True).stdout.strip()
        s.equal("the page's default negative prompt is the server's",
                neg_text, comfy.DEFAULT_NEGATIVE)
    else:
        print("  --   node is not installed, so the page's maths was not run")

    measured = 9.32e9           # the real run: 500 frames, 832x480, above idle
    s.check("the RAM estimate is within 10% of the real measurement",
            abs(comfy.frame_ram(500, 832, 480) - measured) / measured < 0.1,
            f"{comfy.frame_ram(500, 832, 480)/1e9:.2f} GB vs 9.32 GB")
    s.equal("a part holds 2 windows at 480p and 1 at 720p",
            (comfy.windows_per_part(832, 480), comfy.windows_per_part(1280, 720)),
            (2, 1))
    for size in comfy.SIZES:
        w, h = comfy.SIZES[size]
        per = comfy.windows_per_part(w, h)
        if comfy.frame_ram(comfy.WINDOW + (per - 1) * comfy.STEP, w, h) \
                > comfy.PART_BUDGET:
            s.check(f"{size}: a part fits its 4 GB budget", False)
            break
    else:
        s.check("at every size a part's frames fit the 4 GB budget", True)
    for sec in (0.5, 5.8, 10.8, 10.9, 30, 61.3, 120, 600, 3600):
        for size in ("832x480", "1280x720"):
            lay = comfy.plan(sec, size)
            ps = lay["parts"]
            ok = (sum(p["frames"] for p in ps) == lay["frames"]
                  and sum(p["windows"] for p in ps) == lay["windows"]
                  and all(p["windows"] <= lay["per_part"] for p in ps)
                  and all(not p["trim"] for p in ps[:-1])
                  and [p["first"] for p in ps]
                  == list(range(0, lay["windows"], lay["per_part"]))
                  and all(ps[i]["start_frame"] == sum(q["frames"] for q in ps[:i])
                          for i in range(len(ps))))
            if not ok:
                s.check(f"{sec} s at {size}: parts add up to the clip", False,
                        str(ps)[:300])
                break
        else:
            continue
        break
    else:
        s.check("for every length up to an hour, the parts' frames add up "
                "to exactly the clip, and only the last is trimmed", True)
    s.equal("an hour at 480p is 720 windows in 360 parts",
            (comfy.plan(3600)["windows"], len(comfy.plan(3600)["parts"])),
            (720, 360))
    s.equal("a short clip is one part — no joining at all",
            len(comfy.plan(10.8)["parts"]), 1)
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

    # -- every node the graphs use has a stage in words ----------------------
    import server
    import json as _json
    used = set(_json.loads((ROOT / "tests" / "object_info.json").read_text()))
    raw = sorted(c for c in used if server.stage_for(c) == c)
    s.equal("every node the app queues shows as a stage in words, never its "
            "class name", raw, [])

    # -- launching the engine ----------------------------------------------
    import os
    import tempfile
    import time as _time
    tmp = Path(tempfile.mkdtemp(prefix="avatar-launch-"))
    (tmp / "main.py").write_text(
        "import sys, pathlib\n"
        "pathlib.Path(__file__).with_name('argv.txt').write_text(' '.join(sys.argv[1:]))\n"
        "print('main.py: error: pretend ComfyUI refused its flags')\n"
        "sys.exit(2)\n")
    bootstrap.DATA_DIR = tmp
    from harness import free_port

    def launch(extra: str) -> tuple[str, float, bool]:
        os.environ["AVATAR_COMFY_ARGS"] = extra
        proc, prog = bootstrap.ComfyProcess(), bootstrap.Progress()
        proc.start(sys.executable, tmp, free_port(), prog, lowvram=True)
        began = _time.time()
        up = bootstrap.wait_for_comfy(f"http://127.0.0.1:{free_port()}",
                                      timeout=60, alive=proc.alive)
        return (tmp / "argv.txt").read_text(), _time.time() - began, up
    try:
        argv, took, up = launch("")
        s.check("the engine starts in low-VRAM mode with previews, and its "
                "cache on (it keeps the model loaded from part to part)",
                "--lowvram" in argv and "--cache-none" not in argv
                and "--preview-method auto" in argv, argv)
        argv, took, up = launch("--cpu")
        s.check("a memory mode of the person's own replaces --lowvram "
                "(ComfyUI refuses both)",
                "--cpu" in argv and "--lowvram" not in argv, argv)
        argv, _, _ = launch("--use-sage-attention")
        s.check("other flags ride along beside low-VRAM mode",
                "--use-sage-attention" in argv and "--lowvram" in argv, argv)
        s.check("an engine that dies at start ends the wait in seconds, not "
                "15 minutes", not up and took < 10, f"{took:.1f} s")
    finally:
        os.environ.pop("AVATAR_COMFY_ARGS", None)

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

    # -- transformers 5 breaks the lip sync: kept under 5 --------------------
    s.check("transformers 4.x and none at all pass, 5.x does not",
            bootstrap.transformers_ok("4.57.6") and bootstrap.transformers_ok("")
            and not bootstrap.transformers_ok("5.19.0")
            and not bootstrap.transformers_ok("5.0.0"))
    calls: list = []
    real_v, real_pip = bootstrap.transformers_version, bootstrap.pip_install
    try:
        bootstrap.pip_install = lambda py, args, log, *a, **k: calls.append(args)
        bootstrap.transformers_version = lambda py: "5.19.0"
        changed = bootstrap.pin_transformers("py", lambda m: None)
        s.check("5.19 is replaced with 4.x", changed and calls == [
            bootstrap.PIN_ARGS] and "<5" in bootstrap.TRANSFORMERS_PIN
            and any(a.startswith("diffusers") for a in bootstrap.PIN_ARGS),
            str(calls))
        calls.clear()
        bootstrap.transformers_version = lambda py: "4.57.6"
        s.check("4.57 is left alone",
                not bootstrap.pin_transformers("py", lambda m: None)
                and not calls)
    finally:
        bootstrap.transformers_version, bootstrap.pip_install = real_v, real_pip

    cl = comfy.ComfyClient("http://127.0.0.1:9")
    cl.history = lambda pid: {"status": {"status_str": "error", "messages": [
        ["execution_error", {"node_type": "MultiTalkWav2VecEmbeds",
                             "exception_message":
                             "'NoneType' object is not subscriptable"}]]}}
    msg = cl.failed("x") or ""
    s.check("the error a real PC hit says what it is and how to fix it",
            "transformers 5" in msg and "transformers<5" in msg, msg)

    # -- a moved app folder: stale saved paths are found again ---------------
    import tempfile
    real_app = bootstrap.APP_DIR
    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp) / "Text-to-Video-Model-main" / real_app.name
        (app / "ComfyUI" / "models").mkdir(parents=True)
        (app / "ComfyUI" / "main.py").write_text("")
        bootstrap.APP_DIR = app
        try:
            old = "C:\\AI\\Text-to-Video-Model\\" + real_app.name + "\\ComfyUI"
            moved = dict(bootstrap.DEFAULT_CONFIG, comfy_dir=old,
                         models_dir=old + "\\models",
                         python="C:\\gone\\python.exe")
            notes = bootstrap.heal_paths(moved)
            s.equal("a stale ComfyUI path is rebased onto the moved app",
                    moved["comfy_dir"], str(app / "ComfyUI"))
            s.equal("a stale models path follows it",
                    moved["models_dir"], str(app / "ComfyUI" / "models"))
            s.equal("a vanished Python path is cleared, not kept",
                    moved["python"], "")
            s.check("each repair is reported", len(notes) == 3)
            blank = dict(bootstrap.DEFAULT_CONFIG)
            bootstrap.heal_paths(blank)
            s.equal("an empty config adopts the ComfyUI inside the app",
                    blank["comfy_dir"], str(app / "ComfyUI"))
            s.check("a valid config is left alone",
                    bootstrap.heal_paths(blank) == [])
        finally:
            bootstrap.APP_DIR = real_app

    # -- the start-up search: finds ComfyUI anywhere, prefers the weights ----
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        def make(rel: str) -> Path:
            c = root / rel
            (c / "models").mkdir(parents=True)
            (c / "main.py").write_text("")
            (c / "folder_paths.py").write_text("")
            return c
        bare = make("a/ComfyUI")
        rich = make("x/y/z/w/ComfyUI")
        make("a/ComfyUI/custom_nodes/inner/ComfyUI")   # never walked into
        (root / "Windows" / "ComfyUI").mkdir(parents=True)
        (root / "Windows" / "ComfyUI" / "main.py").write_text("")
        found = bootstrap.find_comfy_installs([root], max_depth=6, budget=10)
        s.equal("the search finds every ComfyUI, shallowest first",
                found, [bare, rich])
        s.equal("the search stops at the depth limit",
                bootstrap.find_comfy_installs([root], max_depth=3, budget=10),
                [bare])
        wcfg = dict(bootstrap.DEFAULT_CONFIG)
        for m in bootstrap.model_set(wcfg):
            p = bootstrap.model_path(rich / "models", m)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"")
        s.equal("the install holding the LongCat weights is the one chosen",
                bootstrap.pick_comfy(found, wcfg), rich)
        real_find = bootstrap.find_comfy_installs
        real_detect = bootstrap.detect_comfy_dirs
        bootstrap.find_comfy_installs = lambda: [bare, rich]
        bootstrap.detect_comfy_dirs = lambda: []   # this machine's own ComfyUI
        try:
            lost = dict(bootstrap.DEFAULT_CONFIG, comfy_dir=str(root / "gone"))
            bootstrap.verify_locations(lost)
            s.equal("a lost ComfyUI is found by the search",
                    lost["comfy_dir"], str(rich))
            s.equal("and its models folder with it",
                    lost["models_dir"], str(rich / "models"))
            s.check("the report says both check out",
                    all("not found" not in ln
                        for ln in bootstrap.location_report(lost)))
            quiet = dict(bootstrap.DEFAULT_CONFIG, comfy_dir=str(root / "gone"))
            bootstrap.verify_locations(quiet, search=False)
            s.equal("no search when asked not to",
                    quiet["comfy_dir"], str(root / "gone"))
        finally:
            bootstrap.find_comfy_installs = real_find
            bootstrap.detect_comfy_dirs = real_detect
    return s
