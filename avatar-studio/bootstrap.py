"""
bootstrap.py - first-launch setup for LongCat Avatar Studio.

LongCat-Avatar is a 13.6-billion-parameter video model shipped in bf16, so the
weights are the floor: the DiT alone is about 28 GB on disk and the umT5 text
encoder another 11 GB. preflight() measures the machine and says plainly what
that means, before forty-odd GB is downloaded.

Steps: Python -> ComfyUI -> custom nodes -> dependencies -> weights -> launch.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

# The engine is always local. Without this, requests asks the system proxy
# settings about 127.0.0.1 on every call — on Windows that is a registry read
# and a reverse-DNS lookup, seconds per status poll, and a proxy that does not
# exempt loopback turns a running engine into an "offline" one.
_loopback = ["localhost", "127.0.0.1", "::1"]
for _key in ("NO_PROXY", "no_proxy"):
    _have = [h.strip() for h in os.environ.get(_key, "").split(",") if h.strip()]
    os.environ[_key] = ",".join(_have + [h for h in _loopback if h not in _have])

APP_DIR = Path(__file__).resolve().parent
# Config, gallery and finished clips. AVATAR_STUDIO_DATA moves the lot, which
# is what lets the tests run against a throwaway folder — as in the sibling apps.
DATA_DIR = Path(os.environ.get("AVATAR_STUDIO_DATA") or (APP_DIR / "data"))
CONFIG_PATH = DATA_DIR / "config.json"

COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
HF_BASE = "https://huggingface.co"
LONGCAT_REPO = "Kijai/LongCat-Video_comfy"
WAN_REPO = "Kijai/WanVideo_comfy"
MELBAND_REPO = "Kijai/MelBandRoFormer_comfy"
WAV2VEC_REPO = "Kijai/wav2vec2_safetensors"
MODEL_REPO = LONGCAT_REPO

# The LongCat Avatar nodes were built on the wrapper's longcat_avatar branch
# and have since been merged into main, which also carries kijai's official
# example workflow (copied to assets/). main is what gets installed.
CUSTOM_NODES = [
    {"id": "wrapper", "dir": "ComfyUI-WanVideoWrapper",
     "label": "ComfyUI-WanVideoWrapper",
     "repo": "https://github.com/kijai/ComfyUI-WanVideoWrapper.git",
     "fallback": "",
     "why": "The LongCat Avatar nodes: model loader, sampler, the window "
            "extender and the wav2vec2 audio embeds.",
     "optional": False},
    {"id": "kjnodes", "dir": "ComfyUI-KJNodes", "label": "ComfyUI-KJNodes",
     "repo": "https://github.com/kijai/ComfyUI-KJNodes.git", "fallback": "",
     "why": "Resizes the picture and stitches the windows together "
            "(ImageBatchExtendWithOverlap).",
     "optional": False},
    {"id": "melband", "dir": "ComfyUI-MelBandRoFormer",
     "label": "ComfyUI-MelBandRoFormer",
     "repo": "https://github.com/kijai/ComfyUI-MelBandRoFormer.git",
     "fallback": "",
     "why": "Separates the voice from music and noise, so only the speech "
            "moves the lips.",
     "optional": True},
    {"id": "manager", "dir": "ComfyUI-Manager", "label": "ComfyUI-Manager",
     "repo": "https://github.com/Comfy-Org/ComfyUI-Manager.git",
     "fallback": "https://github.com/ltdrdata/ComfyUI-Manager.git",
     "why": "Installs and updates other nodes from inside ComfyUI.",
     "optional": True},
]

# What each pack registers that proves ComfyUI really loaded it.
NODE_MARKERS = {"wrapper": "WanVideoLongCatAvatarExtendEmbeds",
                "kjnodes": "ImageBatchExtendWithOverlap",
                "melband": "MelBandRoFormerSampler"}

# The weight set. Sizes marked approx are estimates until HuggingFace is
# asked: setup reads the real ones from the repo before it downloads, and
# the Models page shows the real ones once a repo has been browsed.
DIT = {"name": "LongCat-Avatar_comfy_bf16.safetensors", "repo": LONGCAT_REPO,
       "path": "Avatar/LongCat-Avatar_comfy_bf16.safetensors",
       "folder": "diffusion_models", "size": 28_000_000_000, "approx": True,
       "why": "LongCat-Avatar: a picture and speech in, a talking video out."}
DISTILL_LORA = {"name": "LongCat_distill_lora_alpha64_bf16.safetensors",
                "repo": LONGCAT_REPO,
                "path": "LongCat_distill_lora_alpha64_bf16.safetensors",
                "folder": "loras", "size": 1_400_000_000, "approx": True,
                "why": "The distill LoRA — 12 steps at cfg 1 instead of 50 "
                       "with guidance. Use this one: the older rank-128 "
                       "LoRA hurts the window extension."}
TEXT_ENCODER = {"name": "umt5-xxl-enc-bf16.safetensors", "repo": WAN_REPO,
                "path": "umt5-xxl-enc-bf16.safetensors",
                "folder": "text_encoders", "size": 11_360_000_000,
                "approx": True,
                "why": "umT5-XXL. Reads the prompt; cached on disk per prompt."}
VAE = {"name": "Wan2_1_VAE_bf16.safetensors", "repo": WAN_REPO,
       "path": "Wan2_1_VAE_bf16.safetensors", "folder": "vae",
       "size": 254_000_000, "approx": True,
       "why": "The Wan 2.1 VAE — encodes the picture, decodes the frames."}
WAV2VEC = {"name": "wav2vec2-chinese-base_fp16.safetensors",
           "repo": WAV2VEC_REPO,
           "path": "wav2vec2-chinese-base_fp16.safetensors",
           "folder": "wav2vec2", "size": 190_000_000, "approx": True,
           "why": "wav2vec2: turns the speech into what drives the lips."}
MELBAND = {"name": "MelBandRoformer_fp32.safetensors", "repo": MELBAND_REPO,
           "path": "MelBandRoformer_fp32.safetensors",
           "folder": "diffusion_models", "size": 913_000_000, "approx": True,
           "why": "MelBandRoFormer: lifts the voice out of music and noise."}

# How the DiT sits in memory once loaded. The file is bf16 either way; fp8
# is applied by the wrapper's loader as it reads it.
PRECISIONS = {
    "fp8": {"label": "fp8 in memory — for 8–16 GB cards",
            "note": "The bf16 file, stored as fp8_e4m3fn as it loads: about "
                    "14 GB resident instead of 28. What makes 32 GB of RAM "
                    "enough.",
            "quantization": "fp8_e4m3fn"},
    "bf16": {"label": "bf16 — full precision, 24 GB+ cards",
             "note": "About 28 GB resident. Wants 64 GB of system RAM unless "
                     "the card holds most of it.",
             "quantization": "disabled"},
}

DEFAULT_CONFIG = {
    "comfy_url": "http://127.0.0.1:8188",
    "comfy_dir": "", "models_dir": "", "python": "",
    "managed": True, "auto_start_comfy": True, "torch_index": "",
    "hf_token": "", "hf_endpoint": HF_BASE, "hf_repo": MODEL_REPO,
    "precision": "fp8",
    "want_melband": True, "want_manager": True,
    "lowvram": True, "setup_complete": False,
}


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
_write_lock = threading.Lock()


def atomic_write(path: Path, text: str) -> None:
    """Write through a temporary file and swap it in, so a crash mid-write
    never leaves half a JSON document. On Windows the swap is retried: an
    antivirus scan or OneDrive sync briefly holding the file is routine."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.1)


def quarantine(path: Path) -> Path | None:
    """Move a file that will not parse aside, so the next save cannot bury
    what was in it. Returns where it went."""
    dest = path.with_name(f"{path.name}.bad-{time.strftime('%Y%m%d-%H%M%S')}")
    try:
        os.replace(path, dest)
        print(f"[avatar-studio] {path.name} could not be read; kept as "
              f"{dest.name}", flush=True)
        return dest
    except OSError:
        return None


def load_config() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if not isinstance(saved, dict):
                raise ValueError("not an object")
            cfg.update(saved)
        except (OSError, ValueError):
            # keep the damaged file for recovery; defaults are not silent
            quarantine(CONFIG_PATH)
    if cfg.get("precision") not in PRECISIONS:
        cfg["precision"] = "fp8"
    return cfg


def save_config(cfg: dict) -> None:
    # several threads save (requests, setup, tasks): one writer at a time
    with _write_lock:
        atomic_write(CONFIG_PATH, json.dumps(cfg, indent=2))


# --------------------------------------------------------------------------- #
# model set
# --------------------------------------------------------------------------- #
def _item(spec: dict, role: str) -> dict:
    return {"folder": spec["folder"], "repo": spec["repo"],
            "path": spec["path"], "name": spec["name"], "size": spec["size"],
            "approx": spec.get("approx", False), "role": role,
            "why": spec["why"]}


def model_set(cfg: dict) -> list[dict]:
    """The files a render cannot start without. The precision choice does
    not change the files — LongCat ships bf16 only — only how they load."""
    return [_item(DIT, "required"), _item(DISTILL_LORA, "required"),
            _item(TEXT_ENCODER, "required"), _item(VAE, "required"),
            _item(WAV2VEC, "required")]


def model_path(models_dir: Path, item: dict) -> Path:
    return models_dir / item["folder"] / item["name"]


def missing_models(models_dir: Path, cfg: dict) -> list[dict]:
    return [m for m in model_set(cfg) if not model_path(models_dir, m).exists()]


def extra_models(cfg: dict) -> list[dict]:
    """Optional files: the vocal separator, only wanted with its node."""
    out = []
    if cfg.get("want_melband", True):
        out.append(_item(MELBAND, "optional"))
    return out


def missing_extras(models_dir: Path, cfg: dict) -> list[dict]:
    return [m for m in extra_models(cfg)
            if not model_path(models_dir, m).exists()]


def node_installed(comfy_dir: Path, node: dict) -> bool:
    return (comfy_dir / "custom_nodes" / node["dir"]).is_dir()


def wanted_nodes(cfg: dict) -> list[dict]:
    keys = {"manager": "want_manager", "melband": "want_melband"}
    return [n for n in CUSTOM_NODES
            if not n.get("optional") or cfg.get(keys.get(n["id"], ""), True)]


# --------------------------------------------------------------------------- #
# human-readable numbers
# --------------------------------------------------------------------------- #
def fmt_size(n: float) -> str:
    n = float(n or 0)
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.1f} MB" if n < 1e8 else f"{n / 1e6:.0f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.0f} kB"
    return f"{int(n)} B"


def fmt_eta(seconds: float) -> str:
    s = int(max(seconds or 0, 0))
    if s >= 3600:
        return f"{s // 3600}h {(s % 3600) // 60}m left"
    if s >= 60:
        return f"{s // 60}m {s % 60}s left"
    return f"{s}s left"


def fmt_transfer(got: float, total: float, speed: float, eta: float) -> str:
    """One line of download state: how much, how fast, how much longer."""
    bits = [f"{fmt_size(got)} of {fmt_size(total)}" if total
            else f"{fmt_size(got)} so far"]
    if speed > 0:
        bits.append(f"{speed / 1e6:.1f} MB/s")
    if total and speed > 0:
        bits.append(fmt_eta(eta))
    return " · ".join(bits)


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #
def _ram_bytes() -> int:
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        pass
    if platform.system() == "Windows":
        try:
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            stat = MS()
            stat.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullTotalPhys)
        except Exception:
            return 0
    return 0


def _vram_bytes(python: str) -> tuple[int, str]:
    if not python or not Path(python).exists():
        return 0, ""
    code = ("import torch,json;d=torch.cuda.is_available();"
            "print(json.dumps({'v':(torch.cuda.get_device_properties(0)"
            ".total_memory if d else 0),"
            "'n':(torch.cuda.get_device_name(0) if d else '')}))")
    try:
        out = subprocess.run([python, "-c", code], capture_output=True,
                             text=True, timeout=90)
        if out.returncode != 0:
            return 0, ""
        d = json.loads(out.stdout.strip().splitlines()[-1])
        return int(d["v"]), d["n"]
    except Exception:
        return 0, ""


# what people call "8 GB" or "32 GB" is GiB: an RTX 4060 reports 8.59e9
# bytes, which divided by 1e9 read as "9 GB"
GIB = 1024 ** 3


def assess(vram: int, ram: int, free_disk: int, download: int,
           peak: int, lowvram: bool = True,
           precision: str = "fp8") -> tuple[str, list[str]]:
    """The verdict from the measurements.

    Nothing here has been timed on a real 8 GB card yet, and it says so.
    What is known: the DiT is 13.6B parameters, bf16 on disk; the wrapper's
    block swap keeps all but a few of its 48 blocks in system RAM, and fp8
    storage halves what that RAM has to hold. So 8 GB of VRAM with 32 GB of
    RAM is "tight", not "hard" — "hard" is kept for what genuinely blocks:
    under ~7 GB of VRAM, RAM far below the peak, no disk for the download.
    """
    notes, verdict = [], "ok"

    def worse(level: str) -> None:
        nonlocal verdict
        order = ("ok", "tight", "hard")
        if order.index(level) > order.index(verdict):
            verdict = level

    if vram and vram < 7e9:
        worse("hard")
        notes.append(f"{vram/GIB:.0f} GB of VRAM is under the 8 GB this app "
                     "is tuned for. It will install and queue, but every "
                     "step swaps weights and a clip can take hours.")
    elif vram and vram < 12e9:
        worse("tight")
        notes.append(f"{vram/GIB:.0f} GB of VRAM — tight. Keep fp8 weights, "
                     "Block swap at 40 or more, tiled VAE and 480p. The "
                     "blocks stream from system RAM on every step, so expect "
                     "several minutes per 5.8-second window. Untimed on a "
                     "real 8 GB card so far; the first render gives the true "
                     "number.")
    elif vram and vram < 20e9:
        worse("tight")
        notes.append(f"{vram/GIB:.0f} GB of VRAM — workable with block swap "
                     "at 20–30; 480p is the comfortable size.")
    if vram and vram < 20e9 and not lowvram:
        worse("tight")
        notes.append("Low-VRAM mode is off. Block swap still keeps the DiT "
                     "in RAM, but ComfyUI will hold more of everything else "
                     "on the card — turn it back on if a render stops out "
                     "of memory.")
    if vram and vram < 20e9 and precision == "bf16":
        worse("tight")
        notes.append("bf16 in memory doubles what system RAM has to hold "
                     "(about 28 GB for the DiT). fp8 is the setting for this "
                     "card.")
    if ram and peak and ram < peak * 0.7:
        worse("hard")
        notes.append(f"{ram/GIB:.0f} GB of system RAM against a "
                     f"{peak/1e9:.0f} GB peak — the DiT as it sits in "
                     "memory, plus the audio models and the VAE. That is not "
                     "enough to page through; expect out-of-memory stops.")
    elif ram and peak and ram < peak * 1.15:
        worse("tight")
        notes.append(f"{ram/GIB:.0f} GB of system RAM against a "
                     f"{peak/1e9:.0f} GB peak — close the browser tabs and "
                     "apps you can while a clip renders; a fast SSD for the "
                     "page file matters.")
    if free_disk and download and free_disk < download * 1.1:
        worse("hard")
        notes.append(f"{free_disk/1e9:.0f} GB free where the models go, and "
                     f"the set needs {download/1e9:.0f} GB.")
    if not vram:
        notes.append("Could not read the GPU — install PyTorch, then recheck.")
    if verdict == "hard":
        notes.append("It will still install and queue; it may simply be too "
                     "slow to use. Pointing Settings at a ComfyUI on a "
                     "rented 24 GB box is the way round it.")
    return verdict, notes


def preflight(cfg: dict) -> dict:
    """Measure the machine and say what that means for this model."""
    vram, gpu = _vram_bytes(comfy_python(cfg))
    ram = _ram_bytes()
    try:
        free_disk = shutil.disk_usage(cfg.get("models_dir") or str(APP_DIR)).free
    except Exception:
        free_disk = 0

    items = model_set(cfg) + extra_models(cfg)
    download = sum(i["size"] for i in items)
    precision = cfg.get("precision") or "fp8"
    dit = DIT["size"] // 2 if precision == "fp8" else DIT["size"]
    # the text encoder loads, encodes and unloads before the DiT is needed;
    # what is resident together is the DiT and the small models
    small = VAE["size"] + WAV2VEC["size"] + MELBAND["size"] + DISTILL_LORA["size"]
    peak = max(dit, TEXT_ENCODER["size"]) + small

    verdict, notes = assess(vram, ram, free_disk, download, peak,
                            cfg.get("lowvram", True), precision)
    return {"vram": vram, "gpu": gpu, "ram": ram, "free_disk": free_disk,
            "download": download, "peak": peak, "verdict": verdict,
            "notes": notes, "precision": precision}


# --------------------------------------------------------------------------- #
# progress
# --------------------------------------------------------------------------- #
class Progress:
    STEPS = [("python", "Check Python"), ("comfyui", "Install ComfyUI"),
             ("nodes", "Install the custom nodes"),
             ("deps", "Install dependencies"),
             ("models", "Download the LongCat Avatar weights"),
             ("launch", "Start ComfyUI")]

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.lines: list[str] = []
        self.running = False
        self.done = False
        self.error: str | None = None
        self.step = ""
        self.steps = {k: {"key": k, "label": v, "state": "pending",
                          "detail": "", "pct": None} for k, v in self.STEPS}

    def log(self, msg: str) -> None:
        with self._lock:
            self.lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            if len(self.lines) > 4000:
                del self.lines[:2000]
        print(f"[setup] {msg}", flush=True)

    def begin(self, key: str, detail: str = "") -> None:
        with self._lock:
            self.step = key
            self.steps[key]["state"] = "running"
            self.steps[key]["detail"] = detail
            self.steps[key]["pct"] = None

    def detail(self, key: str, detail: str) -> None:
        with self._lock:
            self.steps[key]["detail"] = detail

    def track(self, key: str, pct: float | None, detail: str = "") -> None:
        """Move a step's bar. `pct` None means running with no number yet —
        the front end shows an indeterminate bar rather than a fake 0%."""
        with self._lock:
            self.steps[key]["pct"] = (None if pct is None
                                      else round(max(0.0, min(100.0, pct)), 1))
            if detail:
                self.steps[key]["detail"] = detail

    def finish(self, key: str, detail: str = "") -> None:
        with self._lock:
            self.steps[key]["state"] = "done"
            self.steps[key]["pct"] = None
            if detail:
                self.steps[key]["detail"] = detail

    def fail(self, key: str, detail: str) -> None:
        with self._lock:
            self.steps[key]["state"] = "error"
            self.steps[key]["pct"] = None
            self.steps[key]["detail"] = detail

    def snapshot(self, since: int = 0) -> dict:
        with self._lock:
            # A list, in the order the steps actually happen: jsonify sorts
            # dict keys, which listed Check Python last on the one screen
            # where order is the whole point.
            steps = [dict(self.steps[k]) for k, _ in self.STEPS]
            return {"running": self.running, "done": self.done,
                    "error": self.error, "step": self.step, "steps": steps,
                    "cursor": len(self.lines), "lines": self.lines[since:]}


# --------------------------------------------------------------------------- #
# interpreters
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _stream(cmd: list[str], on_line, cwd: str | None = None,
            env: dict | None = None, should_cancel=None) -> int:
    r"""Run `cmd` and hand every line of its output to `on_line` as it appears.

    Splits on carriage returns as well as newlines: git writes its progress by
    rewriting one line with \r, so a plain line iterator would hold all of it
    back until the clone finished — which is exactly the silence this is meant
    to fill. Reads with read1() so a partial block is delivered straight away.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, cwd=cwd, env=env)
    assert proc.stdout
    buf = b""
    while True:
        block = proc.stdout.read1(8192)
        if not block:
            break
        if should_cancel and should_cancel():
            proc.terminate()
            try:
                proc.wait(15)
            except subprocess.TimeoutExpired:
                proc.kill()
            return proc.wait()
        buf += block
        parts = re.split(rb"[\r\n]", buf)
        buf = parts.pop()
        for raw in parts:
            text = raw.decode("utf-8", "replace").strip()
            if text:
                on_line(text)
    if buf.strip():
        on_line(buf.decode("utf-8", "replace").strip())
    return proc.wait()


# git reports each phase as its own 0-100%; "Receiving objects" is the download.
GIT_PHASE = re.compile(r"(Counting objects|Compressing objects|Receiving objects"
                       r"|Resolving deltas|Updating files):\s+(\d+)%")


def git_run(cmd: list[str], log, on_pct=None) -> tuple[int, str]:
    """A git command with its progress forwarded. Returns (code, last output)."""
    tail: list[str] = []

    def line(text: str) -> None:
        m = GIT_PHASE.search(text)
        if m:
            if on_pct:
                on_pct(m.group(1), float(m.group(2)))
            return
        tail.append(text)
        log(text[:200])

    # C locale: the phase names above are what git prints in English, and a
    # translated git would otherwise report no progress at all.
    code = _stream(cmd, line, env=dict(os.environ, LC_ALL="C"))
    return code, "\n".join(tail[-8:])


def git_clone(url: str, target: Path, log, on_pct=None,
              depth: int = 1) -> tuple[int, str]:
    return git_run(["git", "clone", "--depth", str(depth), "--progress",
                    url, str(target)], log, on_pct)


def find_python(prog: Progress | None = None) -> str:
    candidates: list[list[str]] = [[sys.executable]]
    if platform.system() == "Windows":
        candidates += [["py", "-3.12"], ["py", "-3.11"], ["py", "-3.10"],
                       ["py", "-3"], ["python"]]
    else:
        candidates += [["python3.12"], ["python3.11"], ["python3.10"],
                       ["python3"], ["python"]]
    for cand in candidates:
        try:
            out = _run(cand + ["-c", "import sys;print(sys.executable);"
                                     "print('%d.%d' % sys.version_info[:2])"],
                       timeout=25)
        except Exception:
            continue
        if out.returncode != 0:
            continue
        parts = [p.strip() for p in out.stdout.strip().splitlines() if p.strip()]
        if len(parts) < 2 or not parts[0]:
            continue
        try:
            major, minor = (int(x) for x in parts[1].split("."))
        except ValueError:
            continue
        if (major, minor) >= (3, 10):
            if prog:
                prog.log(f"Using Python {parts[1]} at {parts[0]}")
            return parts[0]
    raise RuntimeError("No Python 3.10 or newer found. Install it from "
                       "python.org, tick 'Add to PATH', and run setup again.")


def portable_python(comfy_dir: Path) -> Path | None:
    for base in (comfy_dir.parent, comfy_dir):
        cand = base / "python_embeded" / "python.exe"
        if cand.exists():
            return cand
    return None


def venv_python(comfy_dir: Path) -> Path:
    venv = comfy_dir.parent / "comfy-venv"
    return venv / ("Scripts/python.exe" if platform.system() == "Windows"
                   else "bin/python")


def comfy_python(cfg: dict) -> str:
    comfy_dir = Path(cfg["comfy_dir"]) if cfg.get("comfy_dir") else None
    if comfy_dir:
        p = portable_python(comfy_dir)
        if p:
            return str(p)
        v = venv_python(comfy_dir)
        if v.exists():
            return str(v)
    return cfg.get("python") or ""


def have_git() -> bool:
    return shutil.which("git") is not None


def detect_comfy_dirs() -> list[str]:
    home = Path.home()
    # beside the app first: setup puts ComfyUI in here, and a portable build
    # unpacked next to the repo is the other common layout
    near = [APP_DIR, APP_DIR.parent, APP_DIR.parent.parent]
    cands = [b / "ComfyUI" for b in near] + \
            [b / "ComfyUI_windows_portable" / "ComfyUI" for b in near] + \
            [home / "ComfyUI",
             home / "Documents" / "ComfyUI", home / "Desktop" / "ComfyUI",
             Path("C:/ComfyUI"), Path("C:/ComfyUI_windows_portable/ComfyUI"),
             Path("D:/ComfyUI"), Path("D:/ComfyUI_windows_portable/ComfyUI")]
    appdata, local = os.environ.get("APPDATA"), os.environ.get("LOCALAPPDATA")
    if appdata:
        cands.append(Path(appdata) / "ComfyUI")
    if local:
        cands.append(Path(local) / "Programs" / "@comfyorgcomfyui-electron"
                     / "resources" / "ComfyUI")
    out, seen = [], set()
    for c in cands:
        try:
            if ((c / "main.py").exists() or (c / "models").is_dir()) \
                    and str(c) not in seen:
                seen.add(str(c))
                out.append(str(c))
        except OSError:
            continue
    return out


def rebase_path(old: str) -> Path | None:
    """Where a path saved under an earlier location of this app lives now.

    The config keeps absolute paths. Move, rename or re-extract the folder
    (Text-to-Video-Model -> Text-to-Video-Model-main, C: -> D:) and every one
    of them points nowhere, though ComfyUI and the weights moved with it.
    The tail after this app's own folder name is the same; graft it on here.
    """
    parts = [p for p in re.split(r"[\\/]+", old or "") if p]
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == APP_DIR.name.lower():
            cand = APP_DIR.joinpath(*parts[i + 1:])
            try:
                if cand.exists():
                    return cand
            except OSError:
                return None
    return None


def heal_paths(cfg: dict) -> list[str]:
    """Repair saved paths that no longer exist. Returns what changed."""
    notes: list[str] = []
    old_comfy = cfg.get("comfy_dir") or ""
    comfy = Path(old_comfy) if old_comfy else None
    if not (comfy and (comfy / "main.py").exists()):
        cands = [rebase_path(old_comfy)] + [Path(d) for d in detect_comfy_dirs()]
        for c in cands:
            if c and (c / "main.py").exists():
                cfg["comfy_dir"] = str(c)
                comfy = c
                notes.append(f"ComfyUI found at {c}")
                break
    old_models = cfg.get("models_dir") or ""
    if not (old_models and Path(old_models).is_dir()):
        moved = rebase_path(old_models)
        if moved and moved.is_dir():
            cfg["models_dir"] = str(moved)
        elif comfy and (comfy / "models").is_dir():
            cfg["models_dir"] = str(comfy / "models")
        if cfg.get("models_dir") != old_models:
            notes.append(f"Models folder found at {cfg['models_dir']}")
    py = cfg.get("python") or ""
    if py and not Path(py).exists():
        moved = rebase_path(py)
        cfg["python"] = str(moved) if moved else ""
        notes.append(f"Python path {py} is gone"
                     + (f"; using {moved}" if moved else "; cleared"))
    return notes


# folders never worth walking into when hunting for ComfyUI: system trees,
# package caches, and the inside of a ComfyUI (its models alone can hold
# thousands of entries)
_SKIP_DIRS = {"windows", "program files", "program files (x86)", "programdata",
              "$recycle.bin", "system volume information", "recovery",
              "node_modules", ".git", "__pycache__", "site-packages", "lib",
              "libs", "scripts", ".cache", ".venv", "venv", "comfy-venv",
              "python_embeded", "models", "custom_nodes", "output", "input",
              "temp", "proc", "sys", "dev", "snap"}


def is_comfy_dir(path: Path) -> bool:
    try:
        return (path / "main.py").is_file() and \
            (path / "folder_paths.py").is_file()
    except OSError:
        return False


def search_roots() -> list[Path]:
    """Where a full search starts: around the app, home, then every drive."""
    roots = [APP_DIR.parent.parent, Path.home()]
    if platform.system() == "Windows":
        roots += [Path(f"{c}:/") for c in "CDEFGHIJKLMNOPQRSTUVWXYZ"
                  if os.path.exists(f"{c}:/")]
    else:
        roots += [Path("/opt"), Path("/srv"), Path("/mnt"), Path("/media")]
    return roots


def find_comfy_installs(roots: list[Path] | None = None, max_depth: int = 6,
                        budget: float = 45.0) -> list[Path]:
    """Every ComfyUI under `roots`, shallowest first, within a time budget.

    Breadth-first, so the install a person put somewhere sensible is met
    long before the walk wanders into deep trees, and a slow or huge drive
    ends the search on time rather than holding up the engine.
    """
    deadline = time.monotonic() + budget
    found: list[Path] = []
    seen: set[str] = set()
    queue = [(r, 0) for r in (roots if roots is not None else search_roots())]
    while queue and time.monotonic() < deadline:
        path, depth = queue.pop(0)
        try:
            key = os.path.normcase(str(path.resolve()))
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        if is_comfy_dir(path):
            found.append(path)
            continue                # nothing worth finding inside one
        if depth >= max_depth:
            continue
        try:
            with os.scandir(path) as it:
                for e in it:
                    if time.monotonic() >= deadline:
                        break
                    try:
                        if not e.is_dir(follow_symlinks=False):
                            continue
                    except OSError:
                        continue
                    if e.name.lower() in _SKIP_DIRS or e.name.startswith("."):
                        continue
                    queue.append((Path(e.path), depth + 1))
        except OSError:
            continue
    return found


def pick_comfy(installs: list[Path], cfg: dict) -> Path | None:
    """The install to use: the one holding the LongCat weights, then the one
    inside this app, then the first found."""
    def score(c: Path) -> tuple:
        models = c / "models"
        weights = models.is_dir() and not missing_models(models, cfg)
        try:
            inside = c.resolve().is_relative_to(APP_DIR.parent.resolve())
        except (OSError, ValueError):
            inside = False
        return (not weights, not inside)
    return min(installs, key=score) if installs else None


def verify_locations(cfg: dict, search: bool = True,
                     log=None) -> list[str]:
    """Check every saved location at start; repair what moved.

    The quick repair (heal_paths) handles a moved app folder. When ComfyUI is
    still nowhere, and `search` is on, the drives are searched for it.
    """
    say = log or (lambda _m: None)
    notes = heal_paths(cfg)
    comfy = Path(cfg["comfy_dir"]) if cfg.get("comfy_dir") else None
    if comfy and (comfy / "main.py").exists():
        return notes
    if not search:
        return notes
    say("Searching this computer for ComfyUI…")
    hit = pick_comfy(find_comfy_installs(), cfg)
    if not hit:
        say("No ComfyUI found on this computer — install it from the "
            "Engine page, or set its folder in Settings.")
        return notes
    cfg["comfy_dir"] = str(hit)
    notes.append(f"ComfyUI found at {hit}")
    models = cfg.get("models_dir") or ""
    if not (models and Path(models).is_dir()) and (hit / "models").is_dir():
        cfg["models_dir"] = str(hit / "models")
        notes.append(f"Models folder found at {cfg['models_dir']}")
    return notes


def location_report(cfg: dict) -> list[str]:
    """One line per saved location, saying whether it checks out."""
    out = []
    comfy = cfg.get("comfy_dir") or ""
    out.append(f"ComfyUI: {comfy} — ok" if comfy and Path(comfy, "main.py").exists()
               else "ComfyUI: not found")
    models = cfg.get("models_dir") or ""
    if models and Path(models).is_dir():
        gone = missing_models(Path(models), cfg)
        out.append(f"Models: {models} — "
                   + ("all LongCat weights present" if not gone else
                      f"{len(gone)} LongCat file(s) missing"))
    else:
        out.append("Models: not found")
    return out


# --------------------------------------------------------------------------- #
# huggingface
# --------------------------------------------------------------------------- #
def hf_headers(cfg: dict) -> dict:
    token = (cfg.get("hf_token") or "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def hf_endpoint(cfg: dict) -> str:
    return (cfg.get("hf_endpoint") or HF_BASE).rstrip("/")


def hf_tree(cfg: dict, repo: str, revision: str = "main") -> list[dict]:
    base = hf_endpoint(cfg)
    last = ""
    for kind in ("models", "datasets"):
        url = f"{base}/api/{kind}/{repo}/tree/{revision}?recursive=1"
        try:
            r = requests.get(url, headers=hf_headers(cfg), timeout=30)
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
            continue
        if r.status_code == 401:
            raise RuntimeError("This repo needs a HuggingFace token. Add one on "
                               "the Models page, then try again.")
        if r.status_code == 403:
            raise RuntimeError("Your token cannot read this repo. If it is "
                               "gated, accept its terms on the model page "
                               "first.")
        if r.status_code == 404:
            continue
        r.raise_for_status()
        files = []
        for e in r.json():
            if e.get("type") != "file":
                continue
            size = (e.get("lfs") or {}).get("size") or e.get("size") or 0
            files.append({"path": e["path"], "size": size})
        return files
    raise RuntimeError(f"Could not find '{repo}' on {base}. "
                       + (last or "Check the spelling, or add a token."))


def download_file(cfg: dict, repo: str, path: str, dest: Path,
                  on_progress=None, should_cancel=None,
                  revision: str = "main") -> None:
    with _writing_lock:
        if dest in _writing:
            raise RuntimeError(f"{dest.name} is already downloading.")
        _writing.add(dest)
    try:
        _download_file(cfg, repo, path, dest, on_progress, should_cancel,
                       revision)
    finally:
        with _writing_lock:
            _writing.discard(dest)


# one writer per .part — setup and the Models page can otherwise both append
_writing: set[Path] = set()
_writing_lock = threading.Lock()


def _download_file(cfg: dict, repo: str, path: str, dest: Path,
                   on_progress=None, should_cancel=None,
                   revision: str = "main") -> None:
    import urllib.parse
    url = (f"{hf_endpoint(cfg)}/{repo}/resolve/{revision}/"
           + urllib.parse.quote(path))
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = dict(hf_headers(cfg))
    if have:
        headers["Range"] = f"bytes={have}-"
    with requests.get(url, headers=headers, stream=True, timeout=60,
                      allow_redirects=True) as r:
        if r.status_code == 416:
            # the part already covers the file — but only trust that when the
            # server's own size ("bytes */SIZE") agrees; otherwise start over
            remote = r.headers.get("Content-Range", "").rsplit("/", 1)[-1]
            if remote.isdigit() and int(remote) == have:
                part.replace(dest)
                return
            part.unlink(missing_ok=True)
            r.close()
            return _download_file(cfg, repo, path, dest, on_progress,
                                  should_cancel, revision)
        if r.status_code in (401, 403):
            raise RuntimeError("HuggingFace refused the download. If the "
                               "repo is gated, accept its terms on the model "
                               "page, then add a token on the Models page.")
        r.raise_for_status()
        mode = "ab" if (have and r.status_code == 206) else "wb"
        if mode == "wb":
            have = 0
        length = int(r.headers.get("Content-Length", 0))
        total = length + have if length else 0
        got, last, started = have, 0.0, time.time()
        with open(part, mode) as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if should_cancel and should_cancel():
                    return
                if not chunk:
                    continue
                fh.write(chunk)
                got += len(chunk)
                now = time.time()
                if on_progress and now - last > 0.6:
                    last = now
                    speed = (got - have) / max(now - started, .1)
                    eta = (total - got) / speed if speed > 0 and total else 0
                    on_progress(got, total, speed, eta)
    if total and part.stat().st_size != total:
        # a stream that ended early; the part stays so the next try resumes
        raise RuntimeError(f"{dest.name} stopped short at "
                           f"{fmt_size(part.stat().st_size)} of "
                           f"{fmt_size(total)} — try again to resume.")
    part.replace(dest)


# --------------------------------------------------------------------------- #
# ComfyUI process
# --------------------------------------------------------------------------- #
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


VRAM_MODES = ("--gpu-only", "--highvram", "--normalvram", "--lowvram",
              "--novram", "--cpu")


class ComfyProcess:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.lines: list[str] = []
        self._lock = threading.Lock()

    def note(self, msg: str) -> None:
        """An app-side line in the engine console — what the app is doing TO
        the engine belongs next to what the engine itself says."""
        with self._lock:
            self.lines.append(f"[Avatar Studio] {msg}")
            if len(self.lines) > 2000:
                del self.lines[:1000]

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, python: str, comfy_dir: Path, port: int, prog: Progress,
              lowvram: bool = True) -> None:
        if self.alive():
            return
        # main.py by full path: /system_stats reports argv, and a bare
        # "main.py" says nothing about which install is answering
        cmd = [python, str(Path(comfy_dir).resolve() / "main.py"),
               "--listen", "127.0.0.1", "--port", str(port),
               "--disable-auto-launch"]
        extra = os.environ.get("AVATAR_COMFY_ARGS", "").split()
        # ComfyUI takes one memory mode; a mode of the person's own (--cpu on
        # a machine with no GPU, --highvram on a big card) replaces ours —
        # with both, ComfyUI refuses to start at all
        if lowvram and not any(a in VRAM_MODES for a in extra):
            # The wrapper's block swap is what keeps the DiT in system RAM;
            # --lowvram keeps ComfyUI's own models (the audio models, the VAE)
            # off the card between uses. ComfyUI's cache stays ON: a long
            # clip is a chain of prompts (comfy.plan), and the cache is what
            # keeps the model loaded from one part to the next. The parts
            # are what bound the frames in memory, whatever the length.
            cmd += ["--lowvram"]
        # ComfyUI sends no step previews unless asked; the live preview in
        # the app (the wrapper's sampler previews each step) needs them
        cmd += ["--preview-method", "auto"]
        # anything else the person wants ComfyUI started with — --cpu on a
        # machine with no GPU, --use-sage-attention, a different cache mode
        cmd += extra
        # transformers 5 breaks the lip sync: the compatibility node fixes it
        # without touching the Python; failing that, 4.x. If neither, do not
        # start: the engine would come up "ready" and every render would
        # fail in wav2vec2.
        try:
            ensure_lipsync(python, comfy_dir, prog.log)
        except Exception as exc:  # noqa: BLE001
            have = transformers_version(python)
            if not lipsync_ok(python, comfy_dir):
                msg = (f"ComfyUI was not started: it has transformers {have}, "
                       "with which every render fails in the lip sync, and "
                       "neither the compatibility node nor transformers 4.x "
                       f"could be installed ({str(exc)[:200]}). Check that "
                       f"{Path(comfy_dir) / 'custom_nodes'} is writable, or "
                       "run: " + transformers_fix_command(python))
                prog.log(msg)
                self.note(msg)
                raise TransformersBlocked(msg) from exc
            prog.log(f"transformers check: {exc}")
        prog.log("Launching ComfyUI: " + " ".join(cmd))
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
            if platform.system() == "Windows" else 0
        # Output goes to a file, not a pipe. Closing the console on Windows
        # kills this app without a word to the windowless engine; with a pipe
        # its next progress line would hit a dead handle and fail every
        # render. A file outlives us, so an engine the next launch adopts
        # still works.
        log = DATA_DIR / "comfy.log"
        try:
            out = open(log, "wb")
        except OSError:                  # an orphan still holds it on Windows
            log = DATA_DIR / f"comfy-{os.getpid()}.log"
            out = open(log, "wb")
        # MPLBACKEND=Agg: the wrapper's sampler plots its sigmas with
        # matplotlib on every run. On Windows with Tk installed, matplotlib
        # picks the Tk GUI backend, and Tk objects made on the render thread
        # then get collected on the web-server thread, which kills ComfyUI
        # ("Tcl_AsyncDelete: async handler deleted by the wrong thread",
        # Windows fatal exception) mid-render. A real RTX 4060 PC hit it.
        env = {**os.environ, "PYTHONUNBUFFERED": "1",
               "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}
        try:
            self.proc = subprocess.Popen(cmd, cwd=str(comfy_dir), stdout=out,
                                         stderr=subprocess.STDOUT, env=env,
                                         creationflags=flags)
        finally:
            out.close()                  # the child has its own handle
        threading.Thread(target=self._pump, args=(prog, log, self.proc),
                         daemon=True).start()

    def _pump(self, prog: Progress, log: Path, proc) -> None:
        """Follow the engine's log file for as long as the engine runs.
        Bytes are decoded leniently: one odd byte must not stop the reader."""
        try:
            fh = open(log, "rb")
        except OSError:
            return
        buf = b""
        with fh:
            while True:
                chunk = fh.read(65536)
                if not chunk:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.3)
                    continue
                buf += chunk
                *lines, buf = buf.split(b"\n")
                for raw in lines:
                    self._line(prog, raw.decode("utf-8", "replace").rstrip())
        if buf:
            self._line(prog, buf.decode("utf-8", "replace").rstrip())

    def _line(self, prog: Progress, line: str) -> None:
        line = _ANSI.sub("", line)        # ComfyUI colours its log levels
        with self._lock:
            self.lines.append(line)
            if len(self.lines) > 2000:
                del self.lines[:1000]
        if any(k in line for k in ("Error", "Traceback", "error:",
                                   "IMPORT FAILED", "Starting server",
                                   "out of memory")):
            prog.log(f"ComfyUI: {line}")

    def tail(self, n: int = 40) -> list[str]:
        with self._lock:
            return self.lines[-n:]

    def stop(self) -> None:
        if self.alive():
            try:
                self.proc.terminate()
                self.proc.wait(timeout=15)
            except Exception:
                try:
                    self.proc.kill()
                    self.proc.wait(timeout=5)     # reap it: no zombie left
                except Exception:
                    pass


def normal_url(url: str) -> str:
    """The ComfyUI address as stored: trimmed, no trailing slash."""
    return str(url or "").strip().rstrip("/")


def comfy_port(url: str) -> int:
    """The port in a ComfyUI address; ComfyUI's own 8188 when none is given."""
    from urllib.parse import urlsplit
    try:
        return urlsplit(normal_url(url)).port or 8188
    except ValueError:
        return 8188


def comfy_online(url: str) -> bool:
    try:
        return requests.get(f"{url}/system_stats", timeout=3).status_code == 200
    except requests.exceptions.ReadTimeout:
        # It took the connection but is slow to answer: an engine loading a
        # 28 GB model through --lowvram, not an engine that is off. Calling
        # it offline would offer a second Start onto a port already taken.
        return True
    except Exception:
        return False


def _pids_from_proc_net(port: int) -> list[int]:
    """Linux, no tools needed: the socket inode from /proc/net/tcp*, then the
    process whose fd table holds it."""
    inodes = set()
    for name in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(name).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10:
                continue
            local, state, inode = parts[1], parts[3], parts[9]
            if state == "0A" and local.rsplit(":", 1)[-1] == f"{port:04X}":
                inodes.add(inode)
    pids = set()
    if not inodes:
        return []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            for fd in (proc / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if any(f"socket:[{i}]" == target for i in inodes):
                    pids.add(int(proc.name))
                    break
        except OSError:
            continue
    return sorted(pids)


def port_pids(port: int) -> list[int]:
    """Whoever is listening on the port."""
    if platform.system() == "Windows":
        pids = set()
        try:
            out = _run(["netstat", "-ano", "-p", "TCP"], timeout=25).stdout
        except Exception:
            return []
        for line in out.splitlines():
            parts = line.split()
            # the state column is translated on localised Windows, so a
            # listener is recognised by its empty foreign address instead
            if len(parts) >= 5 and parts[0] == "TCP" \
                    and parts[2] in ("0.0.0.0:0", "[::]:0") \
                    and parts[1].rsplit(":", 1)[-1] == str(port):
                try:
                    pids.add(int(parts[4]))
                except ValueError:
                    pass
        return sorted(pids)
    found = _pids_from_proc_net(port)
    if found:
        return found
    if shutil.which("lsof"):
        try:
            out = _run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                       timeout=25).stdout
            return sorted({int(t) for t in out.split() if t.strip().isdigit()})
        except Exception:
            pass
    return []


def pid_cmdline(pid: int) -> str:
    try:
        if platform.system() == "Windows":
            # wmic is gone from current Windows 11; CIM through PowerShell
            # is the supported way to read another process's command line
            out = _run(["powershell", "-NoProfile", "-NonInteractive",
                        "-Command",
                        f"(Get-CimInstance Win32_Process -Filter "
                        f"'ProcessId={int(pid)}').CommandLine"],
                       timeout=25).stdout
            lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
            return lines[0] if lines else ""
        cmd = Path(f"/proc/{pid}/cmdline")
        if cmd.exists():
            return cmd.read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace").strip()
        return _run(["ps", "-p", str(pid), "-o", "command="],
                    timeout=25).stdout.strip()
    except Exception:
        return ""


def _pid_gone(pid: int) -> bool:
    """True once the pid is no longer a running process.

    A zombie still answers kill(pid, 0): it keeps its pid until its parent
    reaps it, while holding no sockets and running no code. Counting one as
    alive costs five seconds of polling and then reports a SIGKILL that
    stopped nothing — the opposite of what kill_pid is for.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    try:
        # "12 (a name with spaces) Z 1 ..." — split after the last ')'.
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[-1].split()
        return bool(state) and state[0] == "Z"
    except OSError:
        return False


def kill_pid(pid: int) -> str:
    """Stop a process: politely first, firmly if it lingers. Returns what the
    system said about it, so a refusal (access denied, already gone) can be
    shown instead of guessed at."""
    if platform.system() == "Windows":
        try:
            out = _run(["taskkill", "/PID", str(pid), "/T", "/F"], timeout=30)
            return (out.stdout or out.stderr or "").strip()
        except Exception as exc:  # noqa: BLE001
            return str(exc)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already gone"
    except PermissionError:
        return "access denied"
    for _ in range(25):
        time.sleep(0.2)
        if _pid_gone(pid):
            return "stopped"
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return "stopped"
    except PermissionError:
        return "access denied"
    return "sent SIGKILL"


def comfy_stats(url: str) -> dict | None:
    """What is actually answering on the address — argv says which install."""
    try:
        r = requests.get(f"{url}/system_stats", timeout=3)
        if r.status_code == 200:
            return r.json().get("system") or {}
    except Exception:
        pass
    return None


def engine_foreign(stats: dict | None, comfy_dir: str | Path | None) -> bool:
    """Whether the engine answering is provably a different install.

    Only an absolute main.py path proves anything. A relative one — this
    app's own launches before it passed the full path, the portable build's
    `ComfyUI\\main.py`, a hand-typed `python main.py` — says nothing about
    where it runs, and accusing it would flag the right engine as foreign.
    """
    argv = (stats or {}).get("argv") or []
    if not argv or not comfy_dir:
        return False
    ran = str(argv[0]).replace("\\", "/").lower()
    if not (ran.startswith("/") or re.match(r"^[a-z]:/", ran)):
        return False
    wants = {str(d).replace("\\", "/").lower().rstrip("/") + "/"
             for d in (comfy_dir, Path(comfy_dir).resolve())}
    return not any(ran.startswith(w) for w in wants)


def engine_lowvram(stats: dict | None) -> bool | None:
    """Whether the engine answering was launched in low-VRAM mode.

    None when its command line is not known. On an 8 GB card an engine
    started without --lowvram (a launcher script, ComfyUI Desktop, a manual
    `python main.py`) is the difference between a render and an OOM stop.
    """
    argv = [str(a) for a in (stats or {}).get("argv") or []]
    if not argv:
        return None
    return any(a in ("--lowvram", "--novram") for a in argv)


def wait_for_comfy(url: str, timeout: int = 900, on_wait=None,
                   alive=None) -> bool:
    """Poll until ComfyUI answers. `on_wait(elapsed, timeout)` runs each pass —
    there is no percentage to give here, only how long it has been waiting.
    `alive()`, when given, ends the wait the moment the process has died:
    an engine that exited on a bad flag is not going to answer in 15 minutes."""
    started = time.time()
    deadline = started + timeout
    while time.time() < deadline:
        if comfy_online(url):
            return True
        if alive is not None and not alive():
            time.sleep(1)                # the log needs a moment to land
            return comfy_online(url)
        if on_wait:
            on_wait(time.time() - started, timeout)
        time.sleep(2)
    return False


# --------------------------------------------------------------------------- #
# pip / nodes
# --------------------------------------------------------------------------- #
# pip's own progress bar is a terminal animation and vanishes when its output
# is a pipe, which is why a 2.4 GB torch wheel looks like a hang. `--progress-bar
# raw` makes it print "Progress <done> of <total>" lines instead, which survive
# the pipe. Older pips do not have it, so ask before using it.
PIP_RAW = re.compile(r"^Progress (\d+) of (\d+)$")
PIP_GET = re.compile(r"^\s*(?:Downloading|Using cached)\s+(\S+)")
_PIP_RAW_OK: dict[str, bool] = {}


def pip_has_raw_progress(python: str) -> bool:
    if python not in _PIP_RAW_OK:
        ok = False
        try:
            out = _run([python, "-m", "pip", "install", "--help"], timeout=60)
            at = out.stdout.find("--progress-bar")
            ok = at >= 0 and "raw" in out.stdout[at:at + 300]
        except Exception:  # noqa: BLE001
            ok = False
        _PIP_RAW_OK[python] = ok
    return _PIP_RAW_OK[python]


def pip_install(python: str, args: list[str], log, on_pct=None,
                should_cancel=None) -> None:
    """Install with pip. `on_pct(pct|None, detail)` is called as it downloads."""
    import urllib.parse
    cmd = [python, "-m", "pip", "install"]
    if on_pct and pip_has_raw_progress(python):
        cmd += ["--progress-bar", "raw"]
    cmd += args
    log("$ " + " ".join(cmd[:8]) + (" …" if len(cmd) > 8 else ""))

    # Per-wheel transfer state: pip restarts the counter for every file.
    cur = {"name": "", "base": 0, "started": 0.0, "last": 0.0}

    def line(text: str) -> None:
        m = PIP_RAW.match(text)
        if m:
            if not on_pct:
                return
            got, total = int(m.group(1)), int(m.group(2))
            now = time.time()
            if got < cur["base"] or not cur["started"]:
                cur["base"], cur["started"] = got, now       # a new file
            if now - cur["last"] < 0.4 and not (total and got >= total):
                return
            cur["last"] = now
            speed = (got - cur["base"]) / max(now - cur["started"], .1)
            eta = (total - got) / speed if speed > 0 and total else 0
            label = cur["name"] or "package"
            on_pct((got / total * 100) if total else None,
                   f"{label} — {fmt_transfer(got, total, speed, eta)}")
            return
        m = PIP_GET.match(text)
        if m:
            cur.update(name=urllib.parse.unquote(
                m.group(1).rsplit("/", 1)[-1])[:60],
                base=0, started=0.0, last=0.0)
        if text.startswith(("Collecting", "Downloading", "Installing",
                            "Successfully", "ERROR", "Building", "WARNING: ")):
            log(text[:200])
            if on_pct and text.startswith(("Installing", "Building")):
                # Unpacking and byte-compiling: no byte count to report, and
                # torch takes minutes over it, so say what is happening.
                on_pct(None, text[:120])

    if _stream(cmd, line, should_cancel=should_cancel) != 0:
        if should_cancel and should_cancel():
            raise RuntimeError("Cancelled.")
        raise RuntimeError("pip install failed — see the log.")
    if "pip" in args:
        # pip just upgraded itself, so whether it can report progress may have
        # changed. The very first install in a new venv is that upgrade.
        _PIP_RAW_OK.pop(python, None)


# transformers 5 broke the lip sync. WanVideoWrapper's wav2vec2 (a subclass of
# transformers' Wav2Vec2Model) calls the encoder with output_hidden_states=True
# and reads .hidden_states; from 5.0 the encoder ignores that argument and
# returns None, so MultiTalkWav2VecEmbeds fails with "'NoneType' object is not
# subscriptable" on every render. Measured: 13 hidden states on 4.57.6, None
# on 5.19.0. ComfyUI asks only for >=4.50.3, so a fresh install gets 5.x.
TRANSFORMERS_PIN = "transformers>=4.50.3,<5"
# transformers 4.x wants huggingface-hub <1.0, and the newest diffusers wants
# >=1.32; naming diffusers in the same install lets pip step it back to one
# that agrees (0.39 on the day). `pip check` clean afterwards, measured.
PIN_ARGS = [TRANSFORMERS_PIN, "diffusers>=0.33.0"]


_TF_SEEN: dict[str, str] = {}


class TransformersBlocked(RuntimeError):
    """transformers 5 is installed, and neither the compatibility node nor
    4.x could be put in place: every render would fail in wav2vec2, so the
    engine is not started."""


def transformers_version(python: str, cached: bool = False) -> str:
    """The transformers in ComfyUI's Python, without importing it ('' if none).
    cached=True answers from the last probe (the status poll runs often);
    every uncached probe refreshes it."""
    if cached and python in _TF_SEEN:
        return _TF_SEEN[python]
    try:
        r = _run([python, "-c", "import importlib.metadata as m;"
                  "print(m.version('transformers'))"], timeout=60)
        v = r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        v = ""
    _TF_SEEN[python] = v
    return v


def forget_transformers() -> None:
    _TF_SEEN.clear()


def transformers_ok(version: str) -> bool:
    try:
        return not version or int(version.split(".")[0]) < 5
    except ValueError:
        return True


def transformers_fix_command(python: str) -> str:
    return f'"{python}" -m pip install ' + " ".join(f'"{a}"' for a in PIN_ARGS)


def pin_transformers(python: str, log, on_pct=None, should_cancel=None) -> bool:
    """Bring transformers back under 5 if it is not. True if it changed.
    Raises if it is still 5.x afterwards. Run it with the engine stopped: on
    Windows a running ComfyUI holds tokenizers' .pyd, and pip cannot replace
    it."""
    have = transformers_version(python)
    if transformers_ok(have):
        return False
    log(f"transformers {have} breaks the lip sync (wav2vec2 returns no hidden "
        "states); installing 4.x")
    try:
        pip_install(python, PIN_ARGS, log, on_pct,
                    should_cancel=should_cancel)
    finally:
        forget_transformers()
    now = transformers_version(python)
    if not transformers_ok(now):
        raise RuntimeError(f"transformers is still {now} after the install")
    return True


# The first fix: Avatar Studio's own custom node puts the 4.x hidden states
# back on transformers 5 (compat/avatar_studio_compat, identical output to
# 4.57.6, measured). Nothing in the person's Python changes, so nothing can
# fail the way a downgrade did on a real Windows PC. The downgrade above
# stays only for when the shim cannot be copied into custom_nodes.
COMPAT_NAME = "avatar_studio_compat"
COMPAT_SRC = APP_DIR / "compat" / COMPAT_NAME
COMPAT_NODE = "AvatarStudioCompat"


def compat_installed(comfy_dir) -> bool:
    if not comfy_dir:
        return False
    dest = Path(comfy_dir) / "custom_nodes" / COMPAT_NAME / "__init__.py"
    try:
        return dest.read_bytes() == (COMPAT_SRC / "__init__.py").read_bytes()
    except OSError:
        return False


def fix_t5_cpu_loading(comfy_dir, log=None) -> bool:
    """Repair the known wrapper call that loads T5 on CUDA in CPU mode.

    Only replace that call's literal third argument, located with the AST.
    Keep the original alongside it; never rewrite unfamiliar upstream code.
    This must run before ComfyUI imports the wrapper (restart an old engine).
    """
    import ast
    import hashlib
    path = Path(comfy_dir) / "custom_nodes" / "ComfyUI-WanVideoWrapper" / "nodes.py"
    if not path.is_file():
        return False
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for cls in tree.body:
            if not isinstance(cls, ast.ClassDef) or cls.name != "WanVideoTextEncodeCached":
                continue
            for call in ast.walk(cls):
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "loadmodel" and len(call.args) >= 3
                        and isinstance(call.func.value, ast.Call)
                        and isinstance(call.func.value.func, ast.Name)
                        and call.func.value.func.id == "LoadWanVideoT5TextEncoder"):
                    continue
                arg = call.args[2]
                replacement = '"offload_device" if device == "cpu" else "main_device"'
                if ast.dump(arg) == ast.dump(ast.parse(replacement, mode="eval").body):
                    return True
                if not isinstance(arg, ast.Constant) or arg.value != "main_device":
                    continue
                # AST columns are UTF-8 byte offsets, not character offsets.
                rows = source.encode("utf-8").splitlines(keepends=True)
                start = sum(map(len, rows[:arg.lineno - 1])) + arg.col_offset
                end = sum(map(len, rows[:arg.end_lineno - 1])) + arg.end_col_offset
                raw = source.encode("utf-8")
                fixed = raw[:start] + replacement.encode() + raw[end:]
                compile(fixed, str(path), "exec")
                digest = hashlib.sha256(raw).hexdigest()[:12]
                backup = path.with_suffix(f".py.avatar-studio-original-{digest}")
                if not backup.exists():
                    backup.write_bytes(raw)
                temp = path.with_suffix(".py.avatar-studio-tmp")
                temp.write_bytes(fixed)
                temp.replace(path)
                if log:
                    log("Fixed CPU text encoding: T5 now loads off the GPU. "
                        "The original wrapper file is saved beside nodes.py.")
                return True
        if log:
            log("T5 CPU-load fix: wrapper code differs; left it unchanged. "
                "CPU fallback has not been verified for this version.")
    except (OSError, SyntaxError, ValueError) as exc:
        if log:
            log(f"Could not apply the T5 CPU-load fix: {exc}")
    return False


def install_compat(comfy_dir, log=None) -> bool:
    """Copy the shim into ComfyUI/custom_nodes (if it is not there already).
    True when it is in place."""
    if not comfy_dir or not (Path(comfy_dir) / "main.py").exists():
        return False
    fix_t5_cpu_loading(comfy_dir, log)
    if compat_installed(comfy_dir):
        return True
    dest = Path(comfy_dir) / "custom_nodes" / COMPAT_NAME
    try:
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(COMPAT_SRC / "__init__.py", dest / "__init__.py")
    except OSError as exc:
        if log:
            log(f"Could not install the wav2vec2 compatibility node: {exc}")
        return False
    if log:
        log("Installed Avatar Studio's wav2vec2 compatibility node "
            "(lip sync on transformers 5).")
    return True


def lipsync_ok(python: str, comfy_dir, cached: bool = False) -> bool:
    """Will wav2vec2 give the lip sync its hidden states in this install?"""
    return (transformers_ok(transformers_version(python, cached=cached))
            or compat_installed(comfy_dir))


def ensure_lipsync(python: str, comfy_dir, log, on_pct=None,
                   should_cancel=None) -> None:
    """The shim if it can be copied; else transformers 4.x. Raises if
    neither, with transformers 5 installed."""
    if install_compat(comfy_dir, log):
        return
    pin_transformers(python, log, on_pct, should_cancel=should_cancel)


def torch_index(cfg: dict) -> str:
    if cfg.get("torch_index"):
        return cfg["torch_index"]
    if platform.system() == "Darwin":
        return ""
    if shutil.which("nvidia-smi"):
        try:
            if _run(["nvidia-smi"], timeout=20).returncode == 0:
                return "https://download.pytorch.org/whl/cu128"
        except Exception:
            pass
    return "https://download.pytorch.org/whl/cpu"


def clone_node(node: dict, comfy_dir: Path, log, on_pct=None) -> Path:
    target = comfy_dir / "custom_nodes" / node["dir"]
    if target.exists():
        log(f"Updating {node['label']}")
        git_run(["git", "-C", str(target), "pull", "--ff-only", "--progress"],
                log, on_pct)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    urls = [node["repo"]] + ([node["fallback"]] if node.get("fallback") else [])
    errors = []
    for url in urls:
        log(f"git clone {url}")
        code, out = git_clone(url, target, log, on_pct)
        if code == 0:
            return target
        errors.append(out[-300:])
        shutil.rmtree(target, ignore_errors=True)
    raise RuntimeError(f"Could not download {node['label']}: " + " | ".join(errors))


# --------------------------------------------------------------------------- #
# progress adapters
# --------------------------------------------------------------------------- #
# Each git phase is its own 0-100%, so stack them into one bar that only ever
# moves forwards. Receiving objects is the transfer and takes nearly all of it.
GIT_WEIGHT = {"Counting objects": (0.00, 0.02), "Compressing objects": (0.02, 0.03),
              "Receiving objects": (0.05, 0.90), "Resolving deltas": (0.95, 0.04),
              "Updating files": (0.95, 0.05)}


def _git_pct(prog: Progress, key: str, head: str = "",
             base: float = 0.0, span: float = 100.0):
    """A git on_pct callback that drives `key`'s bar between base and base+span."""
    def on_pct(phase: str, pct: float) -> None:
        start, width = GIT_WEIGHT.get(phase, (0.0, 0.0))
        line = f"{phase} — {pct:.0f}%"
        prog.track(key, base + (start + width * pct / 100) * span,
                   f"{head} · {line}" if head else line)
    return on_pct


def _pip_pct(prog: Progress, head: str, key: str = "deps"):
    """A pip on_pct callback: pct is None while pip is not transferring bytes."""
    def on_pct(pct: float | None, detail: str) -> None:
        prog.track(key, pct, f"{head} · {detail}" if head else detail)
    return on_pct


# --------------------------------------------------------------------------- #
# setup run
# --------------------------------------------------------------------------- #
def run_setup(cfg: dict, prog: Progress, comfy: ComfyProcess,
              chosen_dir: str = "", mode: str = "auto") -> None:
    prog.running = True
    prog.done = False
    prog.error = None
    try:
        prog.begin("python")
        if mode == "external":
            prog.finish("python", "Not needed — you run ComfyUI yourself")
            py = cfg.get("python") or sys.executable
        else:
            py = find_python(prog)
            cfg["python"] = py
            prog.finish("python", py)

        prog.begin("comfyui")
        comfy_dir = None
        if mode == "external":
            if not comfy_online(cfg["comfy_url"]):
                raise RuntimeError(f"Nothing is answering at {cfg['comfy_url']}.")
            if not cfg.get("models_dir"):
                raise RuntimeError("Set the ComfyUI models folder in Settings.")
            cfg["managed"] = False
            if cfg.get("comfy_dir"):
                comfy_dir = Path(cfg["comfy_dir"])
            prog.finish("comfyui", cfg["comfy_url"])
        else:
            if chosen_dir:
                comfy_dir = Path(chosen_dir)
                cfg["managed"] = False
                prog.log(f"Using existing ComfyUI at {comfy_dir}")
            else:
                comfy_dir = APP_DIR / "ComfyUI"
                cfg["managed"] = True
                if not (comfy_dir / "main.py").exists():
                    if not have_git():
                        raise RuntimeError("Git is not installed. Install it "
                                           "from the Engine page first.")
                    prog.track("comfyui", None, "Downloading ComfyUI…")
                    code, out = git_clone(COMFY_REPO, comfy_dir, prog.log,
                                          _git_pct(prog, "comfyui"))
                    if code != 0:
                        raise RuntimeError("git clone failed: " + out[-600:])
                else:
                    prog.track("comfyui", None, "Updating ComfyUI…")
                    git_run(["git", "-C", str(comfy_dir), "pull", "--ff-only",
                             "--progress"], prog.log,
                            _git_pct(prog, "comfyui"))
            if not (comfy_dir / "main.py").exists():
                raise RuntimeError(f"No main.py in {comfy_dir}.")
            cfg["comfy_dir"] = str(comfy_dir)
            cfg["models_dir"] = str(comfy_dir / "models")
            prog.finish("comfyui", str(comfy_dir))

        models_dir = Path(cfg["models_dir"])

        prog.begin("nodes")
        node_paths: list[Path] = []
        if comfy_dir is None:
            prog.finish("nodes", "Install the nodes in your own ComfyUI")
        else:
            wanted = wanted_nodes(cfg)
            if wanted and not have_git():
                raise RuntimeError("Git is needed to install the custom nodes.")
            for index, node in enumerate(wanted):
                # one bar across all the nodes: each clone gets its slice
                prog.track("nodes", index / len(wanted) * 100,
                           f"Installing {node['label']}…")
                try:
                    node_paths.append(clone_node(
                        node, comfy_dir, prog.log,
                        _git_pct(prog, "nodes", node["label"],
                                 base=index / len(wanted) * 100,
                                 span=100 / len(wanted))))
                except Exception as exc:  # noqa: BLE001
                    if node.get("optional"):
                        prog.log(f"Skipped {node['label']}: {exc}")
                    else:
                        raise
            prog.finish("nodes", ", ".join(n["label"] for n in wanted) or "none")

        prog.begin("deps")
        if mode == "external":
            prog.finish("deps", "Handled by your own ComfyUI install")
        else:
            target = portable_python(Path(cfg["comfy_dir"]))
            if target:
                prog.log(f"Portable ComfyUI detected — installing into {target}")
            else:
                vpy = venv_python(Path(cfg["comfy_dir"]))
                if not vpy.exists():
                    prog.detail("deps", "Creating the Python environment…")
                    res = _run([py, "-m", "venv",
                                str(Path(cfg["comfy_dir"]).parent / "comfy-venv")])
                    if res.returncode != 0:
                        raise RuntimeError("venv creation failed: " +
                                           (res.stderr or res.stdout)[-600:])
                target = vpy
                prog.track("deps", None, "Upgrading pip…")
                pip_install(str(target), ["--upgrade", "pip", "wheel"],
                            prog.log, _pip_pct(prog, "pip"))
                args = ["torch", "torchvision", "torchaudio"]
                idx = torch_index(cfg)
                if idx:
                    args += ["--index-url", idx]
                prog.track("deps", None, "Installing PyTorch — the long one…")
                pip_install(str(target), args, prog.log,
                            _pip_pct(prog, "PyTorch"))
                prog.track("deps", None, "Installing ComfyUI requirements…")
                pip_install(str(target),
                            ["-r", str(Path(cfg["comfy_dir"]) / "requirements.txt")],
                            prog.log, _pip_pct(prog, "ComfyUI requirements"))
            cfg["python"] = str(target)
            for path in node_paths:
                reqs = path / "requirements.txt"
                if reqs.exists():
                    prog.track("deps", None, f"Requirements for {path.name}…")
                    try:
                        pip_install(str(target), ["-r", str(reqs)], prog.log,
                                    _pip_pct(prog, path.name))
                    except Exception as exc:  # noqa: BLE001
                        prog.log(f"Skipped {path.name} requirements: {exc}")
            # after every requirements file: any of them can pull transformers 5
            prog.track("deps", None, "Checking transformers…")
            ensure_lipsync(cfg["python"], cfg["comfy_dir"], prog.log,
                           _pip_pct(prog, "transformers"))
            prog.finish("deps", f"Installed into {Path(cfg['python']).name}")

        prog.begin("models")
        todo = missing_models(models_dir, cfg) + missing_extras(models_dir, cfg)
        if not todo:
            prog.finish("models", "Everything is already downloaded")
        else:
            for note in preflight(cfg)["notes"]:
                prog.log("Preflight: " + note)
            # Ask each repo for the real sizes so one bar can cover the whole
            # set. The hard-coded sizes are rough and some are zero, and a bar
            # that only knows the current file jumps back to 0% five times.
            sizes: dict[str, int] = {}
            prog.track("models", None, "Checking what is on the repo…")
            for repo in {m["repo"] for m in todo}:
                try:
                    for f in hf_tree(cfg, repo):
                        sizes[f"{repo}/{f['path']}"] = f["size"]
                except Exception as exc:  # noqa: BLE001
                    prog.log(f"Could not read {repo}'s file list ({exc}). The "
                             "bar will follow one file at a time instead.")
            plan = [(item, int(sizes.get(f"{item['repo']}/{item['path']}")
                               or item.get("size") or 0)) for item in todo]
            grand = sum(s for _, s in plan)
            prog.log(f"{len(plan)} file(s) to download "
                     f"({cfg.get('precision', 'fp8')} in memory"
                     + (f", {fmt_size(grand)}" if grand else "") + ")")
            done_bytes = 0
            for i, (item, size) in enumerate(plan, 1):
                dest = model_path(models_dir, item)
                head = f"{item['name']} ({i} of {len(plan)})"

                def on_prog(got, total, speed, eta, _head=head):
                    whole = ((done_bytes + got) / grand * 100) if grand else (
                        (got / total * 100) if total else None)
                    prog.track("models", whole,
                               f"{_head} — " + fmt_transfer(got, total,
                                                            speed, eta))

                prog.track("models",
                           (done_bytes / grand * 100) if grand else None,
                           f"{head} — starting…")
                try:
                    download_file(cfg, item["repo"], item["path"], dest,
                                  on_prog)
                except Exception as exc:  # noqa: BLE001
                    if item.get("role") != "optional":
                        raise
                    # optional files are niceties: never fail setup on them
                    prog.log(f"Skipped {item['name']} ({exc}) — clips still "
                             "render without it.")
                    done_bytes += size
                    continue
                done_bytes += size or (dest.stat().st_size
                                       if dest.exists() else 0)
                prog.log(f"Downloaded {item['name']}")
            prog.finish("models", f"{len(plan)} file(s) ready"
                        + (f" · {fmt_size(grand)}" if grand else ""))

        prog.begin("launch")
        url = cfg["comfy_url"]
        if mode == "external" or not cfg.get("auto_start_comfy", True):
            if not comfy_online(url):
                raise RuntimeError(f"ComfyUI is not answering at {url}.")
        elif comfy_online(url):
            prog.log("ComfyUI is already running — restart it so it picks up the "
                     "new nodes and weights.")
        else:
            port = comfy_port(url)
            comfy.start(cfg["python"], Path(cfg["comfy_dir"]), port, prog,
                        cfg.get("lowvram", True))

            def waiting(elapsed: float, limit: int) -> None:
                s = int(elapsed)
                been = f"{s // 60}m {s % 60}s" if s >= 60 else f"{s}s"
                prog.track("launch", None, "Waiting for ComfyUI — "
                           f"{been} so far. The first start is slow.")

            prog.track("launch", None, "Waiting for ComfyUI…")
            if not wait_for_comfy(url, timeout=900, on_wait=waiting,
                                  alive=comfy.alive):
                if not comfy.alive():
                    raise RuntimeError("ComfyUI stopped while starting. Its "
                                       "last words:\n"
                                       + "\n".join(comfy.tail(12)))
                raise RuntimeError("ComfyUI did not start within 15 minutes.\n"
                                   + "\n".join(comfy.tail(25)))
        prog.finish("launch", url)

        cfg["setup_complete"] = True
        save_config(cfg)
        prog.done = True
        prog.log("Setup complete.")
    except Exception as exc:  # noqa: BLE001
        prog.error = str(exc)
        if prog.step:
            prog.fail(prog.step, str(exc))
        prog.log(f"FAILED: {exc}")
    finally:
        prog.running = False
