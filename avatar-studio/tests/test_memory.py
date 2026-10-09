"""Reproduce CPU mode allocating T5 on CUDA, then execute the repaired call."""
import subprocess
import tempfile
from unittest.mock import patch
from pathlib import Path

import bootstrap
from harness import Suite


# Same load call as the upstream cached encoder; no model downloads needed.
SOURCE = '''class WanVideoTextEncodeCached:
    def process(self, model_name, precision, device="gpu", quantization="disabled"):
        t5, = LoadWanVideoT5TextEncoder().loadmodel(model_name, precision, "main_device", quantization)
        return t5
'''


def run(slow=False):
    s = Suite("memory")
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        (root / "main.py").write_text("# stand-in engine\n")
        path = root / "custom_nodes/ComfyUI-WanVideoWrapper/nodes.py"
        path.parent.mkdir(parents=True)
        path.write_text(SOURCE)
        calls = []

        class Loader:
            def loadmodel(self, name, precision, load_device, quantization):
                calls.append(load_device)
                if load_device == "main_device":
                    raise RuntimeError("CUDA out of memory")
                return ("CPU weights",)

        ns = {"LoadWanVideoT5TextEncoder": Loader}
        exec(SOURCE, ns)
        try:
            ns["WanVideoTextEncodeCached"]().process("t5", "bf16", device="cpu")
        except RuntimeError:
            s.equal("original CPU fallback tries to allocate on CUDA", calls, ["main_device"])
        else:
            s.check("original CPU fallback fails as reproduced", False)
        s.check("known upstream call repaired", bootstrap.fix_t5_cpu_loading(root))
        exec(path.read_text(), ns)
        result = ns["WanVideoTextEncodeCached"]().process("t5", "bf16", device="cpu")
        s.equal("CPU fallback now loads directly on the offload device", result, "CPU weights")
        try:
            ns["WanVideoTextEncodeCached"]().process("t5", "bf16", device="gpu")
        except RuntimeError:
            s.equal("GPU mode still loads on the GPU", calls[-1], "main_device")
        else:
            s.check("GPU mode is unchanged", False)
        fixed = path.read_bytes()
        s.check("second application succeeds", bootstrap.fix_t5_cpu_loading(root))
        s.equal("second application changes no bytes", path.read_bytes(), fixed)
        s.equal("original wrapper preserved", next(path.parent.glob("*.avatar-studio-original-*")).read_text(), SOURCE)
        # Existing wav2vec shims must not skip the new CPU-loading repair.
        path.write_text(SOURCE)
        bootstrap.install_compat(root)
        s.equal("normal compatibility install also repairs T5", path.read_bytes(), fixed)
        path.write_text("# an unfamiliar future wrapper\n")
        s.check("unknown source is refused", not bootstrap.fix_t5_cpu_loading(root))
        s.equal("unknown source stays untouched", path.read_text(), "# an unfamiliar future wrapper\n")
        # Future upstream methods must not be patched with an undefined name.
        unfamiliar = SOURCE.replace('device="gpu"', 'target="gpu"')
        path.write_text(unfamiliar)
        s.check("a loader without the device argument is refused",
                not bootstrap.fix_t5_cpu_loading(root))
        s.equal("a changed method signature stays untouched", path.read_text(), unfamiliar)
        crlf = SOURCE.replace("\n", "\r\n").encode()
        path.write_bytes(crlf)
        bootstrap.fix_t5_cpu_loading(root)
        s.check("repair preserves Windows newlines", b"\r\n" in path.read_bytes()
                and path.read_bytes().count(b"\r\n") == crlf.count(b"\r\n"))
        s.check("repair can restore the exact Windows source",
                bootstrap.restore_t5_cpu_loading(root))
        s.equal("restoration preserves every original byte", path.read_bytes(), crlf)

    # The repaired file belongs to an upstream Git checkout. Reproduce the
    # dirty-pull failure with real Git, then exercise the app's update path.
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        origin = root / "origin"
        origin.mkdir()
        def git(*args, cwd=origin, check=True):
            return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                                  text=True, check=check)
        git("init")
        (origin / "nodes.py").write_text(SOURCE)
        git("add", "nodes.py")
        def commit(message):
            git("-c", "user.name=Memory test", "-c", "user.email=test@example.invalid",
                "commit", "-m", message)
        commit("original wrapper")
        engine = root / "engine"
        target = engine / "custom_nodes" / "ComfyUI-WanVideoWrapper"
        target.parent.mkdir(parents=True)
        git("clone", str(origin), str(target))
        path = target / "nodes.py"
        bootstrap.fix_t5_cpu_loading(engine)
        updated = SOURCE + "\n# a later upstream revision\n"
        (origin / "nodes.py").write_text(updated)
        git("add", "nodes.py")
        commit("upstream update")
        failed = git("pull", "--ff-only", cwd=target, check=False)
        s.check("a real upstream update is blocked by the first repair",
                failed.returncode != 0 and "local changes" in failed.stderr.lower(),
                failed.stderr[-250:])
        node = {"dir": target.name, "label": target.name}
        bootstrap.clone_node(node, engine, lambda _: None)
        expected = bootstrap._t5_cpu_source(updated.encode())
        s.equal("wrapper update succeeds and repairs the new source", path.read_bytes(), expected)
        s.equal("upstream checkout advanced to the new revision",
                git("rev-parse", "HEAD", cwd=target).stdout,
                git("rev-parse", "HEAD").stdout)
        observed = []
        def fail_pull(*args, **kwargs):
            observed.append(path.read_bytes())
            return 1, "network unavailable"
        with patch.object(bootstrap, "git_run", side_effect=fail_pull):
            s.fails_with("a failed wrapper update is reported",
                         lambda: bootstrap.clone_node(node, engine, lambda _: None),
                         RuntimeError, "network unavailable")
        s.equal("our repair is removed before a failing pull", observed, [updated.encode()])
        s.equal("the CPU repair is restored even when the pull fails", path.read_bytes(), expected)
        custom = expected + b"# user's local adjustment\n"
        path.write_bytes(custom)
        observed.clear()
        with patch.object(bootstrap, "git_run", side_effect=fail_pull):
            s.fails_with("an update with other local edits reports its failure",
                         lambda: bootstrap.clone_node(node, engine, lambda _: None),
                         RuntimeError)
        s.equal("user edits are not discarded before an update", observed, [custom])
        s.equal("user edits survive a failed update", path.read_bytes(), custom)
        # A damaged backup must not be used even if it would reproduce the patch.
        path.write_bytes(expected)
        for backup in target.glob("nodes.py.avatar-studio-original-*"):
            backup.unlink()
        (target / "nodes.py.avatar-studio-original-000000000000").write_text(updated)
        s.check("a backup with a mismatched hash is refused",
                not bootstrap.restore_t5_cpu_loading(engine))
        s.equal("a mismatched backup cannot change the wrapper", path.read_bytes(), expected)
    # A later, changed prompt can load CPU T5 while the video model remains
    # cached. Preflight must budget their coexistence, not just the larger one.
    with patch.object(bootstrap, "_vram_bytes", return_value=(8 * bootstrap.GIB, "test GPU")), \
            patch.object(bootstrap, "_ram_bytes", return_value=32 * bootstrap.GIB), \
            patch.object(bootstrap, "comfy_python", return_value="unused"):
        measured = bootstrap.preflight({"precision": "fp8"})
    s.check("preflight budgets cached video weights and CPU text weights together",
            measured["peak"] > bootstrap.DIT["size"] // 2 + bootstrap.TEXT_ENCODER["size"])
    s.check("preflight distinguishes weight estimates from a measured RAM peak",
            any("not a measured RAM peak" in note for note in measured["notes"]))
    return s
