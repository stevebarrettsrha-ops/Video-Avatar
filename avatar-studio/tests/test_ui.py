"""The interface itself, in a real Chromium.

Wants Playwright and a Chromium; says so and steps aside when either is
missing (AVATAR_CHROMIUM points at a browser executable if Playwright's own
download is not installed). Chromium's fake media devices stand in for a
microphone, so the Record button is exercised for real: MediaRecorder,
decodeAudioData, the WAV encoder and the upload.
"""

from __future__ import annotations

import io
import math
import os
import shutil
import struct
import sys
import tempfile
import time
import wave
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import (Suite, Workspace, comfy, fake_weights,  # noqa: E402
                     studio, wait_for)


def chromium_path() -> str:
    if os.environ.get("AVATAR_CHROMIUM"):
        return os.environ["AVATAR_CHROMIUM"]
    for cand in ("/opt/pw-browsers/chromium",
                 shutil.which("chromium"), shutil.which("chromium-browser"),
                 shutil.which("google-chrome")):
        if cand and Path(cand).exists():
            return cand
    return ""       # let Playwright try its own download


def available() -> str:
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        return "Playwright is not installed (pip install playwright)"
    return ""


def tone(seconds: float) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"".join(
            struct.pack("<h", int(6000 * math.sin(2 * math.pi * 220 * i / 16000)))
            for i in range(int(16000 * seconds))))
    return buf.getvalue()


def png() -> bytes:
    import zlib

    def chunk(kind, body):
        return (struct.pack(">I", len(body)) + kind + body
                + struct.pack(">I", zlib.crc32(kind + body) & 0xffffffff))
    rows = b"".join(b"\x00" + bytes([200, 150, 120]) * 32 for _ in range(40))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 40, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def run(slow: bool = False) -> Suite:
    from playwright.sync_api import sync_playwright

    s = Suite("ui")
    tmp = Path(tempfile.mkdtemp(prefix="avatar-ui-"))
    (tmp / "face.png").write_bytes(png())
    (tmp / "speech.wav").write_bytes(tone(7.5))
    with comfy(delay=0.4) as mock, Workspace() as ws:
        fake_weights(ws / "models")
        with studio(mock.url, ws / "data", ws / "models") as app, \
                sync_playwright() as pw:
            exe = chromium_path()
            browser = pw.chromium.launch(
                executable_path=exe or None,
                args=["--use-fake-device-for-media-stream",
                      "--use-fake-ui-for-media-stream",
                      "--autoplay-policy=no-user-gesture-required"])
            ctx = browser.new_context(viewport={"width": 1400, "height": 900})
            ctx.grant_permissions(["microphone"], origin=app.url)
            page = ctx.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text)
                    if m.type == "error" and "favicon" not in m.text else None)
            page.goto(app.url)
            page.wait_for_function(
                "document.querySelector('#enginePill span').textContent"
                " === 'Engine ready'", timeout=20000)
            s.check("the page boots to a ready engine", True)
            s.check("the setup sheet stays shut on a set-up machine",
                    page.locator("#veil-setup").is_hidden())

            # -- picture and speech ---------------------------------------
            page.locator("#imgFile").set_input_files(str(tmp / "face.png"))
            page.wait_for_selector("#imgThumb:not([hidden])", timeout=10000)
            s.check("the picture shows on its card",
                    page.locator("#imgClear").is_visible())
            page.locator("#audioFile").set_input_files(str(tmp / "speech.wav"))
            page.wait_for_function(
                "document.getElementById('audioInfo').textContent.indexOf('7.5') > 0",
                timeout=10000)
            plan = page.locator("#planNote").inner_text()
            s.check("the plan says 7.5 s is 120 frames in two windows",
                    "120 frames" in plan and "2 windows" in plan, plan)
            page.fill("#trimLen", "3")
            plan = page.locator("#planNote").inner_text()
            s.check("a length of 3 s is one window",
                    "3.0 s" in plan and "1 window" in plan, plan)
            page.fill("#trimLen", "")
            page.fill("#trimStart", "9")
            s.check("a start past the end says so",
                    "past the end" in page.locator("#planNote").inner_text())
            page.fill("#trimStart", "0")
            s.check("a short clip is one part — nothing to join",
                    page.locator("#partsNote").count() == 0)
            page.evaluate("S.audio.duration = 600; syncPlan()")
            plan = page.locator("#planNote").inner_text()
            s.check("ten minutes at 480p: 120 windows in 60 parts, no warning, "
                    "and the same memory as any other length",
                    "120 windows" in plan and "60 parts" in plan
                    and "the same for any length" in plan
                    and "3.1 GB" in plan and "bad-text" not in
                    page.inner_html("#planNote"), plan)
            page.evaluate("S.audio.duration = 3600; syncPlan()")
            s.check("an hour is allowed too",
                    "720 windows" in page.locator("#planNote").inner_text())
            page.evaluate("S.audio.duration = 7.5; syncPlan()")
            page.click('#segSize button[data-v="1280x720"]')
            s.check("720p warns that it is well over twice as slow",
                    "start at 480p" in page.locator("#planNote").inner_text())
            s.equal("the prompt is read on the GPU by default (5½ min on a real "
                    "PC's CPU)", page.evaluate("segOn('segT5')"), "gpu")
            page.click('#segSize button[data-v="480x832"]')
            s.check("portrait is chosen",
                    "480×832" in page.locator("#planNote").inner_text())

            # -- settings popover -----------------------------------------
            page.click("#btnSettings")
            s.check("settings open", page.locator("#settingsPop").is_visible())
            page.locator("#steps-sl").fill("8")
            s.equal("the steps readout follows its slider",
                    page.locator("#stepsVal").inner_text(), "8")
            page.click('#segT5 button[data-v="gpu"]')
            page.click("#btnCloseSettings")

            # -- an upload in flight holds Generate back --------------------
            page.route("**/api/upload", lambda route: (time.sleep(1.5),
                                                       route.continue_()))
            page.locator("#audioFile").set_input_files(str(tmp / "speech.wav"))
            page.wait_for_function("S.uploading > 0", timeout=5000)
            s.check("while the speech uploads, Generate is held and says so",
                    page.locator("#btnGenerate").is_disabled()
                    and "Uploading" in page.locator("#genLabel").inner_text())
            page.wait_for_function("S.uploading === 0", timeout=20000)
            s.check("and comes back when it is done",
                    page.locator("#btnGenerate").is_enabled()
                    and page.locator("#genLabel").inner_text() == "Generate")
            page.unroute("**/api/upload")

            # -- generate --------------------------------------------------
            page.fill("#description", "A woman talks to the camera.")
            page.click("#btnGenerate")
            page.wait_for_selector(".tile video", timeout=40000)
            s.check("the clip lands in the feed", True)
            graph = list(requests.get(f"{mock.url}/prompts", timeout=10)
                         .json().values())[-1]
            by = {}
            for n in graph.values():
                by.setdefault(n["class_type"], []).append(n["inputs"])
            s.check("the page's choices reached the graph",
                    by["WanVideoSchedulerv2"][0]["steps"] == 8
                    and by["WanVideoTextEncodeCached"][0]["device"] == "gpu"
                    and by["ImageResizeKJv2"][0]["width"] == 480
                    and len(by["WanVideoSamplerv2"]) == 2,
                    str({k: len(v) for k, v in by.items()}))

            # -- lightbox and reuse ---------------------------------------
            page.click("#imgClear")            # reuse must bring it back
            s.check("the picture can be removed",
                    page.locator("#imgThumb").is_hidden())
            page.locator(".tile .over").first.click(force=True)
            page.wait_for_selector("#lightbox:not([hidden])")
            meta = page.locator("#lbMeta").inner_text()
            s.check("the lightbox shows the windows and the speech",
                    "2 windows" in meta and "speech.wav" in meta, meta)
            page.click("#lbReuse")
            s.check("reuse brings the picture back, from ComfyUI/input",
                    page.wait_for_function(
                        "document.getElementById('imgThumb').naturalWidth > 0",
                        timeout=10000) is not None)

            # -- the microphone -------------------------------------------
            page.click("#btnRecord")
            page.wait_for_function(
                "document.getElementById('recLabel').textContent.indexOf('Stop') === 0")
            page.wait_for_timeout(1600)
            page.click("#btnRecord")
            page.wait_for_function(
                "document.getElementById('audioInfo').textContent"
                ".indexOf('Recording') === 0", timeout=15000)
            s.check("a recording becomes the speech", True)
            name = page.evaluate("S.audio.name")
            s.check("uploaded as a WAV the engine can read",
                    name.startswith("recording-") and name.endswith(".wav"), name)
            r = requests.get(f"{app.url}/api/input", params={"name": name},
                             timeout=10)
            s.check("with a real RIFF header", r.content[:4] == b"RIFF"
                    and r.content[8:12] == b"WAVE")

            # -- draft survives a reload ----------------------------------
            page.reload()
            page.wait_for_function("S.audio && S.image", timeout=10000)
            s.equal("the prompt survives a reload",
                    page.input_value("#description"),
                    "A woman talks to the camera.")

            # -- the other pages ------------------------------------------
            page.click('.nav[data-view="models"]')
            page.wait_for_selector("#setsList .setrow")
            s.check("the Models page lists the set as ready",
                    "ready" in page.locator("#setsList").inner_text())
            page.click('.nav[data-view="engine"]')
            page.wait_for_selector("#dep-list .fitem", timeout=60000)
            s.check("the Engine page lists the node packs",
                    "ComfyUI-WanVideoWrapper" in page.locator("#dep-list").inner_text())
            page.click('.nav[data-view="library"]')
            s.check("the library shows the clip",
                    page.locator("#libGrid .tile").count() >= 1)
            s.equal("no script errors along the way", errors, [])
            browser.close()
    shutil.rmtree(tmp, ignore_errors=True)
    return s
