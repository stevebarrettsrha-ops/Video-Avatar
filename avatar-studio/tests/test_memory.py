"""Reproduce CPU mode allocating T5 on CUDA, then execute the repaired call."""
import tempfile
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
    return s
