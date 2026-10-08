"""The HTTP surface, end to end: the real server.py against the mock ComfyUI
(the real schema, the real validation) and a mock HuggingFace.

Every test gets its own ports and data folder; nothing touches a real
install's library or config.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import (Suite, Workspace, comfy, fake_weights,  # noqa: E402
                     finish_jobs, hub, studio, wait_for)

WAV = (b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
       b"\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
       b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
       b"\x01\x01\x01\x00\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82")


def upload(app: str, name: str, body: bytes) -> requests.Response:
    return requests.post(f"{app}/api/upload", files={"file": (name, body)},
                         timeout=20)


def run(slow: bool = False) -> Suite:
    s = Suite("api")
    with comfy(delay=0.6) as mock, Workspace() as ws:
        fake_weights(ws / "models")
        with studio(mock.url, ws / "data", ws / "models") as app:
            a = app.url
            st = requests.get(f"{a}/api/status", timeout=20).json()
            s.check("ready with the engine, the nodes and the weights",
                    st["ready"] and st["nodes_ready"] and not st["missing_models"],
                    str({k: st.get(k) for k in ("ready", "missing_nodes",
                                                "missing_models")}))
            s.check("the status offers the sizes and attention modes",
                    "832x480" in st["sizes"] and "sdpa" in st["attentions"])
            s.check("not stale: the engine lists the LongCat model",
                    st["stale_models"] is False)

            # -- uploads, and the picture comes back -----------------------
            img = upload(a, "face.png", PNG).json()["name"]
            aud = upload(a, "speech.wav", WAV).json()["name"]
            s.equal("an upload is named as ComfyUI stored it", (img, aud),
                    ("face.png", "speech.wav"))
            r = requests.get(f"{a}/api/input", params={"name": img}, timeout=20)
            s.check("the picture comes back from ComfyUI/input as an image",
                    r.status_code == 200 and r.content == PNG
                    and r.headers["Content-Type"].startswith("image/"))
            r = requests.get(f"{a}/api/input", params={"name": "../x.png"},
                             timeout=20)
            s.equal("a path out of the input folder is refused", r.status_code, 400)
            r = requests.get(f"{a}/api/input", params={"name": "gone.png"},
                             timeout=20)
            s.equal("a file that is not there is a 404", r.status_code, 404)

            # -- what generate refuses -------------------------------------
            def gen(**kw):
                body = {"image": img, "audio": aud, "audio_seconds": 7.0,
                        "prompt": "a woman talks to the camera", "seed": 11}
                body.update(kw)
                return requests.post(f"{a}/api/generate", json=body, timeout=20)

            for what, kw, word in (
                    ("no picture", {"image": ""}, "picture"),
                    ("no audio", {"audio": ""}, "speech"),
                    ("no length", {"audio_seconds": 0}, "audio_seconds"),
                    ("over two minutes", {"audio_seconds": 121}, "audio_seconds"),
                    ("an unknown size", {"size": "999x999"}, "Size"),
                    ("steps that are text", {"steps": "many"}, "steps"),
                    ("block swap past 48", {"blocks_to_swap": 49}, "blocks_to_swap"),
                    ("a prompt that is not text", {"prompt": 5}, "prompt"),
                    ("five runs", {"runs": 5}, "runs"),
                    ("a negative seed", {"seed": -1}, "seed")):
                r = gen(**kw)
                s.check(f"{what} is a 400 that says why",
                        r.status_code == 400 and word.lower() in
                        r.json().get("error", "").lower(),
                        f"{r.status_code} {r.text[:90]}")
            r = requests.post(f"{a}/api/generate", data="[1,2]",
                              headers={"Content-Type": "application/json"},
                              timeout=20)
            s.equal("a body that is not an object is a 400", r.status_code, 400)

            # -- a two-window render, end to end ---------------------------
            # slow enough that the one-second poll sees each window's steps
            requests.post(f"{mock.url}/delay", json={"seconds": 3}, timeout=5)
            r = gen()
            s.equal("a render starts", r.status_code, 200)
            job_id = r.json()["jobs"][0]
            seen = set()

            def watch():
                jobs = requests.get(f"{a}/api/jobs", timeout=10).json()
                job = next(j for j in jobs if j["id"] == job_id)
                seen.add(job.get("stage", "").split(" · ")[0])
                return job["status"] != "running"
            wait_for(watch, 60, 0.3)
            jobs = finish_jobs(a)
            requests.post(f"{mock.url}/delay", json={"seconds": 0.6}, timeout=5)
            job = next(j for j in jobs if j["id"] == job_id)
            s.equal("it finishes", job["status"], "done")
            s.check("progress names the windows as it goes",
                    any(x.startswith("Window 2 of 2") for x in seen), str(seen))
            clip = job["images"][0]
            s.check("the clip records what it was made from",
                    clip["image"] == img and clip["audio"] == aud
                    and clip["windows"] == 2 and clip["frames"] == 112
                    and (clip["width"], clip["height"]) == (832, 480)
                    and clip["seed"] == 11, str(clip)[:300])
            s.equal("fp8 storage was the config's default",
                    clip["quantization"], "fp8_e4m3fn")
            graph = list(requests.get(f"{mock.url}/prompts", timeout=10)
                         .json().values())[-1]
            s.equal("the queued graph has two samplers",
                    sum(n["class_type"] == "WanVideoSamplerv2"
                        for n in graph.values()), 2)
            r = requests.get(f"{a}/api/clip/{clip['id']}", timeout=20)
            s.check("the clip is served as video",
                    r.status_code == 200
                    and r.headers["Content-Type"].startswith("video/"))
            s.check("and listed in the gallery",
                    any(c["id"] == clip["id"] for c in
                        requests.get(f"{a}/api/clips", timeout=10).json()))

            # -- two runs, two seeds ---------------------------------------
            r = gen(runs=2, audio_seconds=2)
            ids = r.json()["jobs"]
            jobs = finish_jobs(a)
            seeds = sorted(j["images"][0]["seed"] for j in jobs if j["id"] in ids)
            s.equal("a fixed seed gives each run its own", seeds, [11, 1011])

            # -- cancel ----------------------------------------------------
            requests.post(f"{mock.url}/delay", json={"seconds": 4}, timeout=5)
            job_id = gen(audio_seconds=20).json()["jobs"][0]
            wait_for(lambda: any(j["id"] == job_id and j.get("prompt_id")
                                 for j in requests.get(f"{a}/api/jobs",
                                                       timeout=10).json()), 20)
            r = requests.post(f"{a}/api/jobs/{job_id}/cancel", timeout=10)
            s.equal("a running render can be stopped", r.status_code, 200)
            jobs = finish_jobs(a, 60)
            s.equal("and it says so", next(j for j in jobs if j["id"] == job_id)
                    ["status"], "cancelled")
            requests.post(f"{mock.url}/delay", json={"seconds": 0.6}, timeout=5)

            # -- delete ----------------------------------------------------
            r = requests.delete(f"{a}/api/clip/{clip['id']}", timeout=10)
            s.check("a clip can be deleted, file and all",
                    r.status_code == 200
                    and not any(c["id"] == clip["id"] for c in
                                requests.get(f"{a}/api/clips", timeout=10).json())
                    and not list((ws / "data" / "clips").glob(clip["id"] + "*")))

            # -- the guard -------------------------------------------------
            r = requests.get(f"{a}/api/status",
                             headers={"Host": "evil.example:80"}, timeout=10)
            s.equal("another host name is refused", r.status_code, 403)
            r = requests.post(f"{a}/api/generate", json={},
                              headers={"Origin": "https://evil.example"},
                              timeout=10)
            s.equal("a cross-site post is refused", r.status_code, 403)

            # -- the engine and the machine --------------------------------
            pf = requests.get(f"{a}/api/preflight", timeout=120).json()
            s.check("the preflight gives a verdict and a peak",
                    pf["verdict"] in ("ok", "tight", "hard") and pf["peak"] > 0)
            deps = requests.get(f"{a}/api/deps", timeout=120).json()["items"]
            by_id = {d["id"]: d for d in deps}
            s.check("the dependency list names the three node packs",
                    {"node:wrapper", "node:kjnodes", "node:melband"} <= set(by_id))
            s.check("loaded packs count as ok, wherever they live",
                    all(by_id[k]["state"] == "ok" for k in
                        ("node:wrapper", "node:kjnodes", "node:melband")))
            s.equal("the weights row is ok", by_id["models"]["state"], "ok")

    # -- a render that runs out of memory says what to change --------------
    with comfy(delay=0.2, MOCK_FAIL_AFTER="1") as mock, Workspace() as ws:
        fake_weights(ws / "models")
        with studio(mock.url, ws / "data", ws / "models") as app:
            upload(app.url, "face.png", PNG)
            upload(app.url, "speech.wav", WAV)
            requests.post(f"{app.url}/api/generate", json={
                "image": "face.png", "audio": "speech.wav",
                "audio_seconds": 3}, timeout=20)
            job = finish_jobs(app.url)[0]
            s.check("out of memory names the settings that help",
                    job["status"] == "error" and "Block swap" in job["error"],
                    job.get("error", ""))

    # -- missing weights, and the set download -----------------------------
    with comfy() as mock, hub() as hf, Workspace() as ws:
        (ws / "models").mkdir()
        with studio(mock.url, ws / "data", ws / "models",
                    hf_endpoint=hf.url) as app:
            st = requests.get(f"{app.url}/api/status", timeout=20).json()
            s.check("missing weights block ready and are named",
                    not st["ready"] and
                    "LongCat-Avatar_comfy_bf16.safetensors" in st["missing_models"])
            hs = requests.get(f"{app.url}/api/hf/settings", timeout=20).json()
            s.check("the Models page lists the six files with their roles",
                    len(hs["curated"]["files"]) == 6 and
                    {f["role"] for f in hs["curated"]["files"]}
                    == {"required", "optional"})
            r = requests.post(f"{app.url}/api/hf/download", json={"set": True},
                              timeout=20).json()
            s.equal("one press queues every missing file", len(r["tasks"]), 6)

            def downloads_done():
                ts = [t for t in requests.get(f"{app.url}/api/tasks",
                                              timeout=10).json()
                      if t["kind"] == "download"]
                return ts and all(t["state"] != "running" for t in ts)
            wait_for(downloads_done, 60)
            tasks = [t for t in requests.get(f"{app.url}/api/tasks",
                                             timeout=10).json()
                     if t["kind"] == "download"]
            s.check("they all land", all(t["state"] == "done" for t in tasks),
                    str([(t["title"], t["state"], t["detail"]) for t in tasks]))
            s.check("each in the folder its loader reads",
                    (ws / "models/diffusion_models/LongCat-Avatar_comfy_bf16.safetensors").exists()
                    and (ws / "models/wav2vec2/wav2vec2-chinese-base_fp16.safetensors").exists()
                    and (ws / "models/loras/LongCat_distill_lora_alpha64_bf16.safetensors").exists()
                    and (ws / "models/diffusion_models/MelBandRoformer_fp32.safetensors").exists())
            st = requests.get(f"{app.url}/api/status", timeout=20).json()
            s.equal("nothing is missing after", st["missing_models"], [])
            br = requests.get(f"{app.url}/api/hf/browse",
                              params={"repo": "Kijai/WanVideo_comfy"},
                              timeout=20).json()
            folders = {f["name"]: f["folder"] for f in br["files"]}
            s.equal("browsing guesses umT5's folder", folders.get(
                "umt5-xxl-enc-bf16.safetensors"), "text_encoders")
            s.check("and marks what is installed", all(
                f["installed"] for f in br["files"] if f["inset"]))
            r = requests.post(f"{app.url}/api/config",
                              json={"precision": "bf16"}, timeout=10)
            st = requests.get(f"{app.url}/api/status", timeout=20).json()
            s.equal("the memory precision is a setting",
                    st["config"]["precision"], "bf16")

    # -- an engine without the LongCat nodes -------------------------------
    with comfy(MOCK_OMIT="WanVideoLongCatAvatarExtendEmbeds") as mock, \
            Workspace() as ws:
        fake_weights(ws / "models")
        with studio(mock.url, ws / "data", ws / "models") as app:
            st = requests.get(f"{app.url}/api/status", timeout=20).json()
            s.check("an old wrapper is not ready, and the node is named",
                    not st["ready"] and st["missing_nodes"]
                    == ["WanVideoLongCatAvatarExtendEmbeds"])
            deps = {d["id"]: d for d in requests.get(
                f"{app.url}/api/deps", timeout=120).json()["items"]}
            s.equal("the wrapper row says to update it", deps["node:wrapper"]
                    ["state"], "missing")
        (ws / "ComfyUI" / "custom_nodes" / "ComfyUI-WanVideoWrapper").mkdir(
            parents=True)
        (ws / "ComfyUI" / "main.py").write_text("")
        with studio(mock.url, ws / "data2", ws / "models",
                    comfy_dir=str(ws / "ComfyUI")) as app:
            deps = {d["id"]: d for d in requests.get(
                f"{app.url}/api/deps", timeout=120).json()["items"]}
            row = deps["node:wrapper"]
            s.check("an installed wrapper from before LongCat says Update, "
                    "not IMPORT FAILED",
                    row["state"] == "missing" and "Update" in row["detail"]
                    and row["action"] == "update", str(row))

    # -- an engine that scanned before the weights landed ------------------
    with comfy(MOCK_BLANK_UNETS="99") as mock, Workspace() as ws:
        fake_weights(ws / "models")
        with studio(mock.url, ws / "data", ws / "models") as app:
            time.sleep(0.5)
            st = requests.get(f"{app.url}/api/status", timeout=20).json()
            s.check("weights on disk the engine cannot see read as stale",
                    st["stale_models"] is True)
    return s
