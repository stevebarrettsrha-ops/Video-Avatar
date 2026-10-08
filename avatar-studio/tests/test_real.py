"""Against a REAL ComfyUI — opt in with AVATAR_REAL_COMFY=http://host:port.

The engine must have ComfyUI-WanVideoWrapper, ComfyUI-KJNodes and
ComfyUI-MelBandRoFormer loaded, and the six weight files present in its
model folders (placeholders are enough: nothing here loads a model).
AVATAR_REAL_MODELS points at that models folder. Three things are proved:

1. ComfyUI's own validator accepts every graph comfy.py builds — every size,
   lengths from half a second to the two-minute cap (25 windows of graph),
   every switch flipped.
2. The stitching and the output really execute. The sampling and decoding
   need 28 GB of weights, so they are swapped for RepeatImageBatch stand-ins
   making 93 frames per window; everything after them — the overlap cut, the
   joins, the trim to the speech, CreateVideo with the trimmed audio,
   SaveVideo — is the app's own graph running on the real nodes. ffprobe then
   checks the file: the frame count of the speech, 16 fps, the size, and an
   audio track as long as the speech.
3. The app itself, pointed at the engine: status, the dependency list,
   uploads, and a render that gets as far as loading the (placeholder) model
   and comes back with ComfyUI's own error, readable.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
import wave
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import comfy                                       # noqa: E402
from comfy import ComfyClient                      # noqa: E402
from harness import Suite, Workspace, studio, wait_for  # noqa: E402

URL = os.environ.get("AVATAR_REAL_COMFY", "")
MODELS = os.environ.get("AVATAR_REAL_MODELS", "")


def available() -> str:
    if not URL:
        return "set AVATAR_REAL_COMFY to a real ComfyUI to run this"
    try:
        requests.get(f"{URL}/system_stats", timeout=5).raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return f"no ComfyUI at {URL} ({exc})"
    if not shutil.which("ffprobe"):
        return "ffprobe is not installed"
    return ""


def speech_wav(seconds: float, rate: int = 22050) -> bytes:
    """A tone that rises and falls like a voice: enough to have a length."""
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        frames = bytearray()
        for i in range(int(rate * seconds)):
            t = i / rate
            amp = 0.5 + 0.5 * math.sin(2 * math.pi * 3 * t)
            frames += struct.pack("<h", int(9000 * amp * math.sin(
                2 * math.pi * (180 + 40 * math.sin(t)) * t)))
        w.writeframes(bytes(frames))
    return buf.getvalue()


def portrait_png(path: Path) -> None:
    """A 600×800 face-ish picture, made with ffmpeg so it is a real PNG."""
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=0x2a3d4f:s=600x800,drawbox=x=170:y=170:w=260:h=330"
                    ":color=0xd9a37e:t=fill,drawbox=x=230:y=290:w=40:h=24"
                    ":color=0x222222:t=fill,drawbox=x=330:y=290:w=40:h=24"
                    ":color=0x222222:t=fill,drawbox=x=255:y=410:w=90:h=22"
                    ":color=0x8a3b3b:t=fill", "-frames:v", "1", str(path)],
                   check=True)


def upload(name: str, body: bytes) -> str:
    r = requests.post(f"{URL}/upload/image", files={"image": (name, body)},
                      data={"type": "input", "overwrite": "true"}, timeout=60)
    r.raise_for_status()
    return r.json()["name"]


def post(graph: dict) -> tuple[int, dict]:
    """Queue, and take it straight back off: validation is the point."""
    r = requests.post(f"{URL}/prompt", json={"prompt": graph}, timeout=60)
    body = r.json()
    if r.ok:
        requests.post(f"{URL}/queue", json={"delete": [body["prompt_id"]]},
                      timeout=10)
    return r.status_code, body


def idle(timeout: float = 120) -> None:
    """Wait for the engine's queue to drain (an accepted prompt may have
    started before it could be taken back; it fails at the model loader)."""
    def empty():
        q = requests.get(f"{URL}/queue", timeout=10).json()
        if q.get("queue_running"):
            requests.post(f"{URL}/interrupt", timeout=10)
        return not q.get("queue_running") and not q.get("queue_pending")
    wait_for(empty, timeout, 0.5)


def rehearsal(built: dict) -> dict:
    """The app's graph with sampling and decoding swapped for stand-ins.

    Every WanVideoDecode becomes RepeatImageBatch(the resized picture, 93):
    a window's worth of frames, as a decode would produce. The nodes that
    only feed the model (loaders, embeds, encodes, samplers) are dropped;
    everything downstream of the decodes is kept exactly as built.
    """
    g = json.loads(json.dumps(built["prompt"]))
    resized = next(k for k, n in g.items()
                   if n["class_type"] in ("ImageResizeKJv2", "ImageScale"))
    for nid, node in g.items():
        if node["class_type"] == "WanVideoDecode":
            g[nid] = {"class_type": "RepeatImageBatch",
                      "inputs": {"image": [resized, 0],
                                 "amount": comfy.WINDOW}}
    keep = {"LoadImage", "ImageResizeKJv2", "ImageScale", "LoadAudio",
            "TrimAudioDuration", "RepeatImageBatch", "GetImageRangeFromBatch",
            "ImageBatchExtendWithOverlap", "CreateVideo", "SaveVideo"}
    g = {k: n for k, n in g.items() if n["class_type"] in keep}
    # the overlap ranges fed the re-encode; with no encode they are unused
    used = {v[0] for n in g.values() for v in n["inputs"].values()
            if isinstance(v, list) and len(v) == 2 and isinstance(v[1], int)}
    g = {k: n for k, n in g.items()
         if n["class_type"] != "GetImageRangeFromBatch"
         or n["inputs"]["start_index"] != -1 or k in used}
    g[next(k for k, n in g.items() if n["class_type"] == "SaveVideo")][
        "inputs"]["filename_prefix"] = "video/AvatarRehearsal"
    return g


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames",
                          "-show_entries",
                          "stream=codec_type,width,height,r_frame_rate,"
                          "nb_read_frames,duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True)
    streams = json.loads(out.stdout)["streams"]
    v = next(s for s in streams if s["codec_type"] == "video")
    a = [s for s in streams if s["codec_type"] == "audio"]
    num, den = v["r_frame_rate"].split("/")
    return {"frames": int(v["nb_read_frames"]), "fps": float(num) / float(den),
            "size": (v["width"], v["height"]),
            "audio": float(a[0]["duration"]) if a else 0.0}


def run(slow: bool = False) -> Suite:
    s = Suite("real")
    client = ComfyClient(URL)
    work = Path(os.environ.get("AVATAR_REAL_OUT") or
                Path(__file__).resolve().parent / ".real-out")
    work.mkdir(parents=True, exist_ok=True)

    stats = requests.get(f"{URL}/system_stats", timeout=10).json()["system"]
    print(f"  ..   ComfyUI {stats.get('comfyui_version')} at {URL}")
    s.equal("the engine has every node the app needs", client.missing_nodes(), [])

    portrait_png(work / "portrait.png")
    img = upload("avatar_test_portrait.png", (work / "portrait.png").read_bytes())
    lengths = {0.5: None, 5.8: None, 5.9: None, 12.5: None, 31.25: None,
               120.0: None}
    for sec in lengths:
        lengths[sec] = upload(f"avatar_test_{sec}s.wav", speech_wav(sec + 0.4))
    base = {"image": img, "prompt": "A person talks to the camera.", "seed": 3}
    client.schema(force=True)        # uploaded behind the client's back

    # -- 1. the real validator ---------------------------------------------
    idle()
    rejected = []
    count = 0
    for size in comfy.SIZES:
        for sec, aud in lengths.items():
            built = client.build({**base, "audio": aud, "audio_seconds": sec,
                                  "size": size})
            code, body = post(built["prompt"])
            count += 1
            if code != 200:
                rejected.append((size, sec, body.get("node_errors") or body))
        idle()
    s.check(f"the real validator accepts all {count} size × length graphs",
            not rejected, str(rejected)[:600])
    built = client.build({**base, "audio": lengths[120.0],
                          "audio_seconds": 120.0})
    s.equal("two minutes is 24 windows, 1920 frames",
            (built["windows"], built["frames"]), (24, 1920))
    s.check("the 24-window graph is accepted too",
            post(built["prompt"])[0] == 200)
    idle()
    flips = [{"quantization": "disabled"}, {"blocks_to_swap": 0},
             {"t5_cpu": False}, {"tiled_vae": False},
             {"isolate_voice": False}, {"audio_start": 3.0},
             {"attention": "sageattn"}, {"steps": 30, "cfg": 4.0,
                                         "audio_cfg": 5.0, "shift": 7},
             {"lora": False}, {"scheduler": "unipc"}]
    bad = []
    for flip in flips:
        built = client.build({**base, "audio": lengths[12.5],
                              "audio_seconds": 6.0, **flip})
        code, body = post(built["prompt"])
        if code != 200:
            bad.append((flip, body.get("node_errors") or body))
        idle()
    s.check(f"and every one of {len(flips)} switches flipped", not bad,
            str(bad)[:600])

    # -- 2. the stitching and the output, executed --------------------------
    for sec in (5.8, 12.5, 31.25):
        built = client.build({**base, "audio": lengths[sec],
                              "audio_seconds": sec})
        graph = rehearsal(built)
        r = requests.post(f"{URL}/prompt", json={"prompt": graph}, timeout=60)
        if not s.check(f"{sec} s: the rehearsal graph is accepted", r.ok,
                       r.text[:400]):
            continue
        pid = r.json()["prompt_id"]
        done = wait_for(lambda: requests.get(f"{URL}/history/{pid}",
                                             timeout=10).json().get(pid), 600, 1)
        hist = requests.get(f"{URL}/history/{pid}", timeout=10).json()[pid]
        status = hist.get("status", {}).get("status_str")
        if not s.check(f"{sec} s: it runs to the end on the real nodes",
                       done and status == "success",
                       json.dumps(hist.get("status"))[:500]):
            continue
        items = client.outputs(pid)
        dest = work / f"rehearsal_{sec}s.mp4"
        with client.view(items[0]) as resp:
            dest.write_bytes(resp.content)
        got = probe(dest)
        s.equal(f"{sec} s: {built['windows']} windows joined and cut to "
                f"{built['frames']} frames", got["frames"], built["frames"])
        s.equal(f"{sec} s: at 16 fps", got["fps"], 16.0)
        s.equal(f"{sec} s: at 832×480", got["size"], (832, 480))
        s.check(f"{sec} s: the audio track is the speech, {sec} s long",
                abs(got["audio"] - sec) < 0.1, f"{got['audio']:.3f} s")
        s.check(f"{sec} s: picture and sound end together",
                abs(got["frames"] / 16 - got["audio"]) < 1 / 16 + 0.05,
                f"{got['frames'] / 16:.3f} s of picture, {got['audio']:.3f} s of sound")

    # -- 3. the app, driving the real engine --------------------------------
    if not MODELS:
        print("  --   AVATAR_REAL_MODELS not set, so the app was not run "
              "against the engine")
        return s
    idle()
    with Workspace() as ws, studio(URL, ws / "data", Path(MODELS)) as app:
        st = requests.get(f"{app.url}/api/status", timeout=60).json()
        s.check("the app sees a ready engine",
                st["ready"] and st["nodes_ready"] and not st["missing_models"],
                str({k: st.get(k) for k in ("ready", "missing_nodes",
                                            "missing_models", "schema_error")}))
        deps = {d["id"]: d for d in requests.get(
            f"{app.url}/api/deps", timeout=180).json()["items"]}
        s.check("the three node packs are reported loaded by the real engine",
                all(deps[k]["state"] == "ok" for k in
                    ("node:wrapper", "node:kjnodes", "node:melband")),
                str({k: deps[k]["state"] for k in deps}))
        up = requests.post(f"{app.url}/api/upload", files={
            "file": ("app_portrait.png", (work / "portrait.png").read_bytes())},
            timeout=60).json()["name"]
        back = requests.get(f"{app.url}/api/input", params={"name": up},
                            timeout=30)
        s.check("an upload through the app comes back from the real engine",
                back.ok and back.content[:4] == b"\x89PNG")
        aud = requests.post(f"{app.url}/api/upload", files={
            "file": ("app_speech.wav", speech_wav(3))}, timeout=60).json()["name"]
        r = requests.post(f"{app.url}/api/generate", json={
            "image": up, "audio": aud, "audio_seconds": 3,
            "prompt": "A person talks."}, timeout=60)
        s.equal("the app queues a render on the real engine", r.status_code, 200)
        job_id = r.json()["jobs"][0]
        wait_for(lambda: next(j for j in requests.get(
            f"{app.url}/api/jobs", timeout=10).json()
            if j["id"] == job_id)["status"] != "running", 600, 1)
        job = next(j for j in requests.get(f"{app.url}/api/jobs",
                                           timeout=10).json()
                   if j["id"] == job_id)
        print(f"  ..   the render ended: {job['status']} — "
              f"{(job.get('error') or '')[:160]}")
        s.check("a placeholder weight file stops it with a plain-words "
                "message: which node, and to download the file again",
                job["status"] == "error" and "damaged or unfinished"
                in job["error"] and "download it again" in job["error"],
                job.get("error", ""))
    return s
