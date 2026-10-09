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

from harness import (Suite, Workspace, comfy, fake_install,  # noqa: E402
                     fake_python, fake_weights, finish_jobs, free_port, hub,
                     studio, wait_for)

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
                    ("over an hour", {"audio_seconds": 3601}, "audio_seconds"),
                    ("parts of 30 windows", {"windows_per_part": 30},
                     "windows_per_part"),
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

    # -- a long clip: parts, handed on, joined -----------------------------
    import io, math, struct, wave
    import av
    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4") as out:
        st = out.add_stream("libx264", rate=16)
        st.width, st.height, st.pix_fmt = 832, 480, "yuv420p"
        for i in range(180):
            frame = av.VideoFrame(832, 480, "yuv420p")
            for plane in frame.planes:
                plane.update(bytes([(i * 7) % 256]) * plane.buffer_size)
            frame.pts = i
            for pkt in st.encode(frame):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)
    part_video = buf.getvalue()
    tone = io.BytesIO()
    with wave.open(tone, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", int(5000 * math.sin(i / 9)))
                               for i in range(16000 * 31)))
    with comfy(delay=0.2) as mock, Workspace() as ws:
        fake_weights(ws / "models")
        requests.post(f"{mock.url}/testvideo", data=part_video, timeout=20)
        with studio(mock.url, ws / "data", ws / "models") as app:
            upload(app.url, "face.png", PNG)
            upload(app.url, "speech31.wav", tone.getvalue())
            r = requests.post(f"{app.url}/api/generate", json={
                "image": "face.png", "audio": "speech31.wav", "seed": 4,
                "audio_seconds": 30}, timeout=20)
            job_id = r.json()["jobs"][0]
            stages = set()

            def watch():
                job = next(j for j in requests.get(f"{app.url}/api/jobs",
                                                   timeout=10).json()
                           if j["id"] == job_id)
                stages.add(job.get("stage", "").split(" · ")[0])
                stages.add(" · ".join(job.get("stage", "").split(" · ")[1:2]))
                return job["status"] != "running"
            wait_for(watch, 120, 0.2)
            job = next(j for j in finish_jobs(app.url) if j["id"] == job_id)
            s.equal("a 30 s clip renders as 3 parts and finishes",
                    (job["status"], job.get("parts")), ("done", 3))
            queued = list(requests.get(f"{mock.url}/prompts", timeout=10)
                          .json().values())[-3:]
            loads = [[n["inputs"]["file"] for n in g.values()
                      if n["class_type"] == "LoadVideo"] for g in queued]
            s.check("each part after the first loads the part before it, "
                    "uploaded under the job's name",
                    loads[0] == [] and len(loads[1]) == 1 and len(loads[2]) == 1
                    and loads[1][0].endswith("part000.mp4")
                    and loads[2][0].endswith("part001.mp4"), str(loads))
            s.check("progress says which part is rendering",
                    any("part 2 of 3" in x.lower() for x in stages), str(stages))
            s.check("the joining step is shown",
                    any(x.startswith("Joining 3 parts") for x in stages))
            clip = job["images"][0]
            path = ws / "data" / "clips" / clip["file"]
            with av.open(str(path)) as got:
                frames = sum(1 for _ in got.decode(video=0))
            with av.open(str(path)) as got:
                a = got.streams.audio[0]
                audio_s = float(a.duration * a.time_base) if a.duration else 0
                rate = got.streams.video[0].average_rate
            s.equal("the joined clip is exactly the speech's 480 frames at 16 fps",
                    (frames, float(rate)), (480, 16.0))
            s.check("with the speech underneath, 30 s of it",
                    abs(audio_s - 30) < 0.05, f"{audio_s:.3f} s")
            s.check("and the parts are cleaned away",
                    not list((ws / "data" / "parts").glob("*/*.mp4")))
            s.equal("the seed is the clip's, one for all its parts",
                    clip["seed"], 4)

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

    # -- a card too full for umT5 in fp8: the prompt is read on the CPU -----
    with comfy(delay=0.2, MOCK_T5_OOM="1") as mock, Workspace() as ws:
        fake_weights(ws / "models")
        with studio(mock.url, ws / "data", ws / "models") as app:
            upload(app.url, "face.png", PNG)
            upload(app.url, "speech.wav", WAV)
            requests.post(f"{app.url}/api/generate", json={
                "image": "face.png", "audio": "speech.wav",
                "audio_seconds": 3}, timeout=20)
            job = finish_jobs(app.url)[0]
            sent = list(requests.get(f"{mock.url}/prompts", timeout=10)
                        .json().values())
            devices = [n["inputs"]["device"] for g in sent for n in g.values()
                       if n["class_type"] == "WanVideoTextEncodeCached"]
            s.check("a text-encoder OOM on the GPU is retried on the CPU, "
                    "and the clip still arrives",
                    job["status"] == "done" and devices == ["gpu", "cpu"],
                    f"{job.get('status')} {job.get('error', '')} {devices}")

    # -- the engine dies mid-render: said at once, not after five minutes --
    with Workspace() as ws:
        install = fake_install(ws / "app")
        url = f"http://127.0.0.1:{free_port()}"
        with studio(url, ws / "data", install / "models",
                    comfy_dir=str(install), python=sys.executable,
                    auto_start_comfy=False) as app:
            requests.post(f"{app.url}/api/comfy/start", timeout=60)
            wait_for(lambda: requests.get(f"{app.url}/api/status", timeout=20)
                     .json()["ready"], timeout=60)
            requests.post(f"{url}/delay", json={"seconds": 30}, timeout=5)
            upload(app.url, "face.png", PNG)
            upload(app.url, "speech.wav", WAV)
            job_id = requests.post(f"{app.url}/api/generate", json={
                "image": "face.png", "audio": "speech.wav",
                "audio_seconds": 3}, timeout=20).json()["jobs"][0]

            def job():
                return next(j for j in requests.get(
                    f"{app.url}/api/jobs", timeout=10).json()
                    if j["id"] == job_id)
            wait_for(lambda: "Queued" not in job().get("stage", "Queued"),
                     30, 0.3)
            time.sleep(1)
            s.check("(the render is under way in the engine)",
                    job()["status"] == "running", str(job()))
            requests.post(f"{url}/crash", timeout=5)
            began = time.time()
            wait_for(lambda: job()["status"] != "running", 60, 0.3)
            took, j = time.time() - began, job()
            s.check("a managed engine that dies mid-render fails the job within "
                    "seconds, with its fatal line",
                    j["status"] == "error" and took < 15
                    and "stopped in the middle" in j.get("error", "")
                    and "fatal" in j.get("error", "").lower(),
                    f"{took:.1f}s {j.get('status')} {j.get('error', '')}")

    # -- the sampler also counts tensors onto the card: not steps ----------
    with comfy(delay=3, MOCK_LOAD_TENSORS="1896") as mock, Workspace() as ws:
        fake_weights(ws / "models")
        with studio(mock.url, ws / "data", ws / "models") as app:
            upload(app.url, "face.png", PNG)
            upload(app.url, "speech.wav", WAV)
            job_id = requests.post(f"{app.url}/api/generate", json={
                "image": "face.png", "audio": "speech.wav",
                "audio_seconds": 3}, timeout=20).json()["jobs"][0]
            seen: list = []

            def watch():
                job = next(j for j in requests.get(
                    f"{app.url}/api/jobs", timeout=10).json()
                    if j["id"] == job_id)
                seen.append((job.get("stage", ""), job.get("pct", 0)))
                return job["status"] != "running"
            wait_for(watch, 60, 0.2)
            loading = [x for x in seen if "loading the model onto the GPU" in x[0]]
            s.check("loading the model's 1896 tensors is named as such, never "
                    "\"step 424 of 1896\" (what a real PC showed)",
                    loading and not any("of 1896" in x[0] and "step" in x[0]
                                        for x in seen), str(seen[:6]))
            s.check("and it does not move the bar as if steps were done",
                    all(p <= 12 for _, p in loading), str(loading[:3]))
            s.check("the real steps are still counted",
                    any("step 12 of 12" in x[0] or "step 11 of 12" in x[0]
                        for x in seen) or any("step" in x[0] and "of 12" in x[0]
                                              for x in seen), str(seen[-4:]))

    # -- transformers 5 in ComfyUI's Python ---------------------------------
    # the real PC's case: pip cannot downgrade it. The compatibility node is
    # copied in instead, pip is never run, and the engine comes up ready.
    with Workspace() as ws:
        install = fake_install(ws / "app")
        py = fake_python(ws / "py", "5.19.0", pip_ok=False)
        url = f"http://127.0.0.1:{free_port()}"
        with studio(url, ws / "data", install / "models",
                    comfy_dir=str(install), python=str(py),
                    auto_start_comfy=False) as app:
            r = requests.post(f"{app.url}/api/comfy/start", timeout=60)
            s.check("transformers 5 with a pip that fails: the engine starts "
                    "anyway, with the compatibility node and no pip at all",
                    r.status_code == 200
                    and (install / "custom_nodes" / "avatar_studio_compat"
                         / "__init__.py").exists()
                    and not (ws / "py" / "pip.log").exists(), r.text[:300])
            s.check("and it is ready, because the engine has loaded the node",
                    wait_for(lambda: requests.get(f"{app.url}/api/status",
                                                  timeout=20).json()["ready"],
                             timeout=60))
            deps = requests.get(f"{app.url}/api/deps", timeout=60).json()
            tf = next((i for i in deps["items"] if i["id"] == "transformers"), {})
            s.check("the Engine page lists transformers 5.19 as fine, with the "
                    "node", tf.get("state") == "ok"
                    and "compatibility node" in tf.get("detail", ""), str(tf))

    # neither fix possible (custom_nodes not writable, pip failing): the
    # engine is not started, and nothing says ready
    with Workspace() as ws:
        install = fake_install(ws / "app")
        (install / "custom_nodes").write_text("not a folder")
        py = fake_python(ws / "py", "5.19.0", pip_ok=False)
        url = f"http://127.0.0.1:{free_port()}"
        with studio(url, ws / "data", install / "models",
                    comfy_dir=str(install), python=str(py),
                    auto_start_comfy=False) as app:
            r = requests.post(f"{app.url}/api/comfy/start", timeout=60)
            err = r.json().get("error", "")
            s.check("with no way to fix the lip sync, the engine is not "
                    "started, and the answer says why and how",
                    r.status_code == 409 and "transformers 5.19.0" in err
                    and "transformers>=4.50.3,<5" in err, err)
            time.sleep(3)
            st = requests.get(f"{app.url}/api/status", timeout=20).json()
            s.check("so nothing comes up, and nothing says ready",
                    not st["comfy_online"] and not st["ready"]
                    and st["transformers_bad"] == "5.19.0", str(st)[:200])

    # an engine already running with ComfyUI-Manager: the node goes on disk
    # first, so the Manager's in-place reboot loads it; no pip
    with Workspace() as ws:
        install = fake_install(ws / "app")
        py = fake_python(ws / "py", "5.19.0", pip_ok=False)
        reboots = ws / "manager.log"
        with comfy(delay=0.2, MOCK_MANAGER_LOG=str(reboots)) as outside:
            with studio(outside.url, ws / "data", install / "models",
                        comfy_dir=str(install), python=str(py),
                        auto_start_comfy=False) as app:
                st = requests.get(f"{app.url}/api/status", timeout=20).json()
                s.check("an engine with transformers 5 and without the node "
                        "is not ready, and the status says which version",
                        st["comfy_online"] and not st["ready"]
                        and st["transformers_bad"] == "5.19.0", str(st)[:200])
                r = requests.post(f"{app.url}/api/comfy/restart",
                                  timeout=120).json()
                s.check("Restart puts the node on disk, then lets "
                        "ComfyUI-Manager reboot the engine to load it",
                        r.get("how") == "manager-reboot" and reboots.exists()
                        and (install / "custom_nodes" / "avatar_studio_compat"
                             / "__init__.py").exists()
                        and not (ws / "py" / "pip.log").exists(), str(r))

    # the same, but the node cannot be written: the Manager reboot could not
    # fix anything, so it is skipped; the engine is stopped, 4.x installed
    # while nothing holds the files, and a managed engine started
    with Workspace() as ws:
        install = fake_install(ws / "app")
        (install / "custom_nodes").write_text("not a folder")
        py = fake_python(ws / "py", "5.19.0")
        reboots = ws / "manager.log"
        with comfy(delay=0.2, MOCK_MANAGER_LOG=str(reboots)) as outside:
            with studio(outside.url, ws / "data", install / "models",
                        comfy_dir=str(install), python=str(py),
                        auto_start_comfy=False) as app:
                r = requests.post(f"{app.url}/api/comfy/restart",
                                  timeout=120).json()
                s.check("without the node, Restart skips the Manager reboot "
                        "and takes the engine over",
                        r.get("how") == "takeover" and not reboots.exists(),
                        str(r))
                s.equal("4.x was installed before the new engine started",
                        (ws / "py" / "transformers.version").read_text(),
                        "4.57.6")
                s.check("and then the engine is ready",
                        wait_for(lambda: requests.get(
                            f"{app.url}/api/status", timeout=20)
                            .json()["ready"], timeout=60))

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
