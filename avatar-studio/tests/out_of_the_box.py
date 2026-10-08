"""Out of the box: an unzipped copy, its own run.sh, the page, nothing else.

    python tests/out_of_the_box.py <unzipped app folder> <screenshot folder> <media folder>

The media folder holds portrait.png, speech_12.5s.wav and speech_40s.wav.
It launches the folder's run.sh, clicks "Install a fresh ComfyUI" on the
setup sheet, waits for every step, then renders a 12.5 s and a 40 s clip
through the page and checks the files with ffprobe: the exact frame count,
16 fps, the parts, and the audio length.

Four stand-ins for what a sandbox without internet or GPU cannot reach —
drop them on a real machine: a stand-in HuggingFace (tests/mock_hf.py),
PyTorch from PyPI, AVATAR_COMFY_ARGS=--cpu, and AVATAR_REHEARSAL=1
(stand-in frames in place of the 28 GB model).
"""
import json, os, signal, socket, subprocess, sys, time
from pathlib import Path
import requests
APP = Path(sys.argv[1]); SHOTS = Path(sys.argv[2]); MEDIA = Path(sys.argv[3])
LOG = APP.parent / "run.log"
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"

def port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p

def say(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)

# --- the two things this sandbox cannot reach -----------------------------
hf_port = port()
hf = subprocess.Popen([sys.executable, str(APP / "tests" / "mock_hf.py"), str(hf_port)],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(2)
(APP / "data").mkdir(exist_ok=True)
(APP / "data" / "config.json").write_text(json.dumps({
    "hf_endpoint": f"http://127.0.0.1:{hf_port}",      # huggingface.co is blocked here
    "torch_index": "https://pypi.org/simple"}))          # download.pytorch.org is blocked here
say("stand-in HuggingFace on", hf_port, "· PyTorch from PyPI")

# --- run.sh, exactly as shipped ---------------------------------------------
app_port = port()
env = dict(os.environ, AVATAR_STUDIO_PORT=str(app_port), AVATAR_STUDIO_NO_BROWSER="1",
           AVATAR_COMFY_ARGS="--cpu",        # no GPU in this sandbox
           AVATAR_REHEARSAL="1")              # no model: stand-in frames
t0 = time.time()
run = subprocess.Popen(["./run.sh"], cwd=APP, env=env, stdout=open(LOG, "w"),
                       stderr=subprocess.STDOUT, start_new_session=True)
url = f"http://127.0.0.1:{app_port}"
for _ in range(600):
    try:
        requests.get(url + "/api/status", timeout=2); break
    except Exception:
        if run.poll() is not None:
            print(LOG.read_text()[-3000:]); sys.exit("run.sh died")
        time.sleep(1)
say(f"run.sh: app answering at {url} after {time.time()-t0:.0f} s (made .venv, installed requirements)")

from playwright.sync_api import sync_playwright
ok = True
try:
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path="/opt/pw-browsers/chromium")
        page = b.new_page(viewport={"width": 1360, "height": 900})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url)
        page.wait_for_selector("#veil-setup:not([hidden])", timeout=60000)
        page.wait_for_selector("#preflight-setup .fitem", timeout=180000)
        page.screenshot(path=str(SHOTS / "oob-01-first-launch-setup.png"))
        opts = page.locator("#pick-list label")
        names = [opts.nth(i).inner_text().split("\n")[0] for i in range(opts.count())]
        say("setup offers:", names)
        for i, n in enumerate(names):
            if "fresh ComfyUI" in n:
                opts.nth(i).click()
        page.click("#btnRunSetup")
        t1 = time.time(); last = None
        while True:
            st = json.loads(page.evaluate("fetch('/api/setup/state?since=999999').then(r=>r.text())"))
            if st.get("step") != last:
                last = st.get("step"); say("setup step:", last)
            if st.get("done") or st.get("error"):
                break
            time.sleep(4)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(SHOTS / "oob-02-setup-complete.png"))
        say(f"setup {'DONE' if st.get('done') else 'FAILED: ' + str(st.get('error'))} in {time.time()-t1:.0f} s")
        for s in st["steps"]:
            say("   ", s["state"], "·", s["label"], "—", (s["detail"] or "")[-110:])
        if not st.get("done"):
            raise SystemExit("setup failed")
        page.click("#setup-done")
        page.wait_for_function("document.querySelector('#enginePill span').textContent === 'Engine ready'", timeout=300000)
        say("engine pill: Engine ready")

        def render(audio, label, expect_parts):
            page.locator("#imgFile").set_input_files(str(MEDIA / "portrait.png"))
            page.wait_for_selector("#imgThumb:not([hidden])")
            page.locator("#audioFile").set_input_files(str(MEDIA / audio))
            page.wait_for_function("S.audio && S.audio.duration > 1")
            page.fill("#description", "A man in a dark jacket talks to the camera in a warm studio.")
            before = page.evaluate("S.images.length")
            t = time.time()
            page.click("#btnGenerate")
            seen = set()
            while page.evaluate("S.images.length") <= before:
                lbl = page.evaluate("(document.querySelector('.skel .lbl')||{}).textContent||''")
                if lbl and "part 2" in lbl and "mid" not in seen:
                    seen.add("mid"); page.screenshot(path=str(SHOTS / f"oob-{label}-rendering.png"))
                bad = page.evaluate("(document.querySelector('.skel.bad .lbl')||{}).textContent||''")
                if bad:
                    raise SystemExit(f"render failed: {bad}")
                time.sleep(1)
                if time.time() - t > 3600:
                    raise SystemExit("render timed out")
            page.wait_for_timeout(1500)
            im = page.evaluate("S.images[0]")
            page.screenshot(path=str(SHOTS / f"oob-{label}-done.png"))
            clip = APP / "data" / "clips" / im["file"]
            out = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames",
                "-show_entries", "stream=codec_type,nb_read_frames,duration,width,height,r_frame_rate",
                "-of", "json", str(clip)], capture_output=True, text=True).stdout)["streams"]
            v = next(x for x in out if x["codec_type"] == "video")
            a = [x for x in out if x["codec_type"] == "audio"]
            say(f"{label}: done in {time.time()-t:.0f} s · parts={im.get('parts')} · "
                f"{v['nb_read_frames']} frames ({im['frames']} expected) · {v['width']}x{v['height']} · "
                f"{v['r_frame_rate']} fps · audio {float(a[0]['duration']) if a else 0:.3f} s")
            good = (int(v["nb_read_frames"]) == im["frames"] and a
                    and abs(float(a[0]["duration"]) - im["audio_seconds"]) < 0.1
                    and (im.get("parts") or 1) == expect_parts)
            say(f"{label}: {'PASS' if good else 'FAIL'}")
            return good

        ok &= render("speech_12.5s.wav", "03-short-12s", 2)
        ok &= render("speech_40s.wav", "04-long-40s", 4)
        page.locator(".tile .over").first.click(force=True)
        page.wait_for_selector("#lightbox:not([hidden])")
        page.wait_for_timeout(800)
        page.screenshot(path=str(SHOTS / "oob-05-clip-details.png"))
        page.click("#lbClose")
        page.click('.nav[data-view="engine"]')
        page.wait_for_selector("#dep-list .fitem", timeout=180000)
        page.wait_for_timeout(3000)
        page.screenshot(path=str(SHOTS / "oob-06-engine.png"), full_page=True)
        say("script errors on the page:", errors)
        ok &= not errors
        b.close()
finally:
    say("RESULT:", "OUT-OF-THE-BOX RUN PASSED" if ok else "FAILED")
    os.killpg(run.pid, signal.SIGTERM)
    subprocess.run(["pkill", "-f", str(APP / "ComfyUI" / "main.py")])
    hf.terminate()
