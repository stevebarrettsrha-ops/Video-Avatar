"""Abuse: fuzzed bodies, concurrency, damaged files on disk, hostile paths.

Nothing a page (or a person with curl) sends may crash the server into a 500,
concurrent renders must not lose each other's clips, and a damaged config or
gallery must be set aside, not take the app down.
"""

from __future__ import annotations

import json
import random
import sys
import threading
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import (Suite, Workspace, comfy, fake_weights,  # noqa: E402
                     finish_jobs, studio)

WAV = (b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
       b"\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

ODD = [None, "", " ", "0", "-1", "1e309", "NaN", "inf", "-inf", 0, -1, 1e309,
       float("nan") if False else 1e-9, 2**70, -2**70, True, False, [], {}, [1],
       {"a": 1}, "x" * 5000, "\u0000", "../../etc/passwd", "<script>", 3.5,
       12, 120, 121, 48, 49]
KEYS = ["image", "audio", "audio_seconds", "audio_start", "prompt", "negative",
        "size", "steps", "shift", "cfg", "audio_cfg", "audio_scale",
        "lora_strength", "blocks_to_swap", "seed", "runs", "quantization",
        "attention", "isolate_voice", "tiled_vae", "t5_cpu", "audio_label",
        "ref_frame_index", "ref_mask_frame_range", "title"]


def run(slow: bool = False) -> Suite:
    s = Suite("stress")
    rng = random.Random(1234)
    with comfy(delay=0.15) as mock, Workspace() as ws:
        fake_weights(ws / "models")
        with studio(mock.url, ws / "data", ws / "models") as app:
            a = app.url
            for name, body in (("face.png", b"\x89PNG\r\n\x1a\n"),
                               ("speech.wav", WAV)):
                requests.post(f"{a}/api/upload", files={"file": (name, body)},
                              timeout=20).raise_for_status()

            # -- fuzz /api/generate --------------------------------------
            codes: dict[int, int] = {}
            fails = []
            for i in range(400):
                body = {"image": "face.png", "audio": "speech.wav",
                        "audio_seconds": 2}
                for key in rng.sample(KEYS, rng.randint(1, 6)):
                    body[key] = rng.choice(ODD)
                try:
                    payload = json.dumps(body, allow_nan=True)
                except ValueError:
                    continue
                r = requests.post(f"{a}/api/generate", data=payload,
                                  headers={"Content-Type": "application/json"},
                                  timeout=30)
                codes[r.status_code] = codes.get(r.status_code, 0) + 1
                if r.status_code >= 500:
                    fails.append((body, r.text[:200]))
                elif r.status_code == 400 and not r.json().get("error"):
                    fails.append((body, "a 400 that says nothing"))
            print(f"  ..   400 fuzzed bodies → status counts {codes}")
            s.check("no fuzzed body is a 500, every 400 says why", not fails,
                    str(fails[:3]))
            jobs = finish_jobs(a, 300)
            bad = [j for j in jobs if j["status"] == "error"
                   and "Traceback" in (j.get("error") or "")]
            s.check("bodies that passed validation render or fail in words",
                    not bad, str(bad[:2]))
            for raw in (b"", b"not json", b"[]", b"null", b"\"str\"", b"{",
                        b"\xff\xfe"):
                r = requests.post(f"{a}/api/generate", data=raw,
                                  headers={"Content-Type": "application/json"},
                                  timeout=10)
                if r.status_code >= 500:
                    fails.append(raw)
            s.check("broken JSON is never a 500", not fails, str(fails))

            # -- concurrency ---------------------------------------------
            before = len(requests.get(f"{a}/api/clips", timeout=10).json())
            started: list[str] = []
            lock = threading.Lock()

            def fire():
                r = requests.post(f"{a}/api/generate", json={
                    "image": "face.png", "audio": "speech.wav",
                    "audio_seconds": 1.5, "runs": 2}, timeout=30)
                with lock:
                    started.extend(r.json().get("jobs", []))
            threads = [threading.Thread(target=fire) for _ in range(6)]
            [t.start() for t in threads]
            [t.join() for t in threads]
            jobs = finish_jobs(a, 300)
            done = [j for j in jobs if j["id"] in started and j["status"] == "done"]
            s.equal("12 renders fired from 6 threads all finish",
                    len(done), 12)
            clips = requests.get(f"{a}/api/clips", timeout=10).json()
            s.equal("and the gallery holds every one of them — none lost to "
                    "a write race", len(clips) - before, 12)
            ids = [c["id"] for c in clips]
            s.equal("no clip id is repeated", len(ids), len(set(ids)))

            # concurrent uploads of the same name
            def up(i):
                requests.post(f"{a}/api/upload", files={
                    "file": (f"many_{i % 3}.wav", WAV)}, timeout=20)
            threads = [threading.Thread(target=up, args=(i,)) for i in range(15)]
            [t.start() for t in threads]
            [t.join() for t in threads]
            s.check("15 parallel uploads leave the server answering",
                    requests.get(f"{a}/api/status", timeout=20).ok)

            # concurrent deletes of the same clip
            victim = clips[0]["id"]
            codes_del = []
            threads = [threading.Thread(target=lambda: codes_del.append(
                requests.delete(f"{a}/api/clip/{victim}", timeout=10).status_code))
                for _ in range(5)]
            [t.start() for t in threads]
            [t.join() for t in threads]
            left = requests.get(f"{a}/api/clips", timeout=10).json()
            s.check("five deletes of one clip at once: gone once, nothing else "
                    "touched", all(c < 500 for c in codes_del)
                    and len(left) == len(clips) - 1)

            # -- hostile paths -------------------------------------------
            for name in ("../../../etc/passwd", "/etc/passwd", "a/../../b.png",
                         "..\\..\\x.png", "%2e%2e/x"):
                r = requests.get(f"{a}/api/input", params={"name": name},
                                 timeout=10)
                if r.status_code == 200:
                    fails.append(name)
            s.check("no input path escapes ComfyUI/input", not fails, str(fails))
            for folder, name in (("../", "x"), ("loras", "../../config.json"),
                                 ("loras", "..\\x"), ("nope", "a.safetensors")):
                r = requests.delete(f"{a}/api/hf/local", json={
                    "folder": folder, "name": name}, timeout=10)
                if r.status_code != 400:
                    fails.append((folder, name, r.status_code))
            s.check("model deletes outside the models folder are refused",
                    not fails, str(fails))
            r = requests.get(f"{a}/api/clip/..%2F..%2Fserver.py", timeout=10)
            s.check("a clip id cannot reach a file", r.status_code == 404)
            r = requests.get(f"{a}/api/jobs/nope/preview", timeout=10)
            s.equal("an unknown job's preview is a 404", r.status_code, 404)
            r = requests.post(f"{a}/api/jobs/nope/cancel", timeout=10)
            s.equal("an unknown job's cancel is a 404", r.status_code, 404)

        # -- damaged files on disk ---------------------------------------
        data = ws / "data"
        (data / "gallery.json").write_text("{not json")
        with studio(mock.url, data, ws / "models") as app:
            r = requests.get(f"{app.url}/api/clips", timeout=10)
            s.check("a damaged gallery reads as empty, not a 500",
                    r.ok and r.json() == [])
            s.check("and is kept aside for recovery",
                    any(p.name.startswith("gallery.json.bad-")
                        for p in data.iterdir()))
            r = requests.post(f"{app.url}/api/generate", json={
                "image": "face.png", "audio": "speech.wav",
                "audio_seconds": 1}, timeout=20)
            finish_jobs(app.url, 60)
            s.equal("and the next clip starts a fresh gallery",
                    len(requests.get(f"{app.url}/api/clips", timeout=10).json()), 1)
        (data / "gallery.json").write_text(json.dumps(
            [{"id": "x"}, "junk", 5, {"file": "y"}, {"id": "ok", "file": "ok.mp4"}]))
        with studio(mock.url, data, ws / "models") as app:
            clips = requests.get(f"{app.url}/api/clips", timeout=10).json()
            s.equal("entries without an id and a file are dropped",
                    [c["id"] for c in clips], ["ok"])
            r = requests.get(f"{app.url}/api/clip/ok", timeout=10)
            s.equal("a listed clip whose file is gone is a 404", r.status_code, 404)
        (data / "config.json").write_text("[1, 2")
        from harness import Server, free_port
        import os
        import sys as _sys
        port = free_port()
        with Server([_sys.executable, "server.py"], port, "/api/status",
                    env={"AVATAR_STUDIO_PORT": str(port),
                         "AVATAR_STUDIO_NO_BROWSER": "1",
                         "AVATAR_STUDIO_NO_SEARCH": "1",
                         "AVATAR_STUDIO_DATA": str(data)}) as app:
            st = requests.get(f"{app.url}/api/status", timeout=20).json()
            s.check("a damaged config starts the app on defaults, with setup "
                    "offered", st["setup_complete"] is False
                    and st["config"]["precision"] == "fp8")
            s.check("and the damaged config is kept aside",
                    any(p.name.startswith("config.json.bad-")
                        for p in data.iterdir()))
    return s
