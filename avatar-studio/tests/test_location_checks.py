"""Location discovery is event-driven; polling never repeats directory walks."""
import importlib.util
import os
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bootstrap
from harness import ROOT, Suite


def run(slow=False):
    s = Suite("location_checks")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        engine = root / "chosen" / "ComfyUI"
        (engine / "models").mkdir(parents=True)
        (engine / "main.py").write_text("")
        valid = dict(bootstrap.DEFAULT_CONFIG, comfy_dir=str(engine),
                     models_dir=str(engine / "models"))
        with patch.object(bootstrap, "detect_comfy_dirs", return_value=[]) as near, \
                patch.object(bootstrap, "find_comfy_installs", return_value=[]) as walk:
            for _ in range(4):
                bootstrap.verify_locations(valid)
            s.equal("verified choices are preserved without nearby scans", near.call_count, 0)
            s.equal("verified choices need no drive searches", walk.call_count, 0)
            s.equal("verified ComfyUI stays selected", valid["comfy_dir"], str(engine))

            cfg = dict(bootstrap.DEFAULT_CONFIG, comfy_dir=str(root / "missing"))
            bootstrap.verify_locations(cfg, search=False)
            bootstrap.verify_locations(cfg, search=False)
            s.equal("failed quick relocation runs once", near.call_count, 1)
            s.check("quick relocation leaves one full search available",
                    bootstrap.locations_need_search(cfg))
            for _ in range(4):
                bootstrap.verify_locations(cfg)
            s.equal("a full search reuses the previous quick check", near.call_count, 1)
            s.equal("failed full search is not repeated by polling", walk.call_count, 1)
            s.check("failed full search is recorded", not bootstrap.locations_need_search(cfg))

            with patch.object(bootstrap, "CONFIG_PATH", root / "config.json"), \
                    patch.object(bootstrap, "DATA_DIR", root):
                bootstrap.save_config(cfg)
                cfg = bootstrap.load_config()
                bootstrap.verify_locations(cfg)
            s.equal("saved failed relocation survives a restart", walk.call_count, 1)
            cfg["_model_locations"] = {"files": {"missing": None}}
            bootstrap.verify_locations(cfg, force=True)
            s.equal("manual Recheck permits another quick relocation", near.call_count, 2)
            s.equal("manual Recheck permits one more full search", walk.call_count, 2)
            s.check("manual Recheck clears model discovery misses",
                    "_model_locations" not in cfg)

            cfg["comfy_dir"] = str(root / "different")
            bootstrap.verify_locations(cfg)
            s.equal("changing a saved location permits a new search", walk.call_count, 3)
            with patch.object(bootstrap, "APP_DIR", root / "moved-app"):
                bootstrap.verify_locations(cfg)
                bootstrap.verify_locations(cfg)
            s.equal("moving the app permits exactly one new search", walk.call_count, 4)

        # A location can become valid, then disappear, without its text changing.
        cfg = dict(bootstrap.DEFAULT_CONFIG, comfy_dir=str(engine),
                   models_dir=str(engine / "models"))
        main = engine / "main.py"
        with patch.object(bootstrap, "detect_comfy_dirs", return_value=[]), \
                patch.object(bootstrap, "find_comfy_installs", return_value=[]) as walk:
            main.unlink()
            bootstrap.verify_locations(cfg)
            main.write_text("")
            bootstrap.verify_locations(cfg)
            s.check("a recovered location forgets its old failure", "_location_check" not in cfg)
            main.unlink()
            bootstrap.verify_locations(cfg)
            s.equal("a newly invalid verified location gets a fresh search", walk.call_count, 2)
            main.write_text("")

        remote = dict(bootstrap.DEFAULT_CONFIG, managed=False, comfy_dir="", models_dir="")
        with patch.object(bootstrap, "detect_comfy_dirs", return_value=[str(engine)]) as near, \
                patch.object(bootstrap, "find_comfy_installs", return_value=[engine]) as walk:
            bootstrap.verify_locations(remote)
            bootstrap.verify_locations(remote, force=True)
            s.check("an external engine never adopts a local install",
                    not remote["comfy_dir"] and not near.called and not walk.called)
            s.check("an external engine does not request a drive search",
                    not bootstrap.locations_need_search(remote))

        # Execute the Flask routes and their background work without sockets or
        # real filesystem walks. Importing this isolated module starts no app.
        spec = importlib.util.spec_from_file_location("avatar_location_test_server", ROOT / "server.py")
        server = importlib.util.module_from_spec(spec)
        cfg = dict(bootstrap.DEFAULT_CONFIG, comfy_dir=str(root / "lost"))
        with patch.dict(os.environ, {"AVATAR_STUDIO_NO_SEARCH": "1"}), \
                patch.object(bootstrap, "load_config", return_value=cfg):
            spec.loader.exec_module(server)

        class ImmediateThread:
            def __init__(self, target, args=(), **_kwargs):
                self.target, self.args = target, args

            def start(self):
                self.target(*self.args)

        with patch.dict(os.environ, {"AVATAR_STUDIO_NO_SEARCH": "0"}), \
                patch.object(bootstrap, "detect_comfy_dirs", return_value=[]) as near, \
                patch.object(bootstrap, "find_comfy_installs", return_value=[]) as walk, \
                patch.object(bootstrap, "CONFIG_PATH", root / "api-config.json"), \
                patch.object(server.threading, "Thread", ImmediateThread), \
                patch.object(server, "comfy_online", return_value=False), \
                patch.object(server, "_transformers_bad", return_value=False), \
                patch.object(server.manager, "dependencies", return_value=[]):
            browser = server.app.test_client()
            for _ in range(4):
                s.equal("routine dependency request succeeds", browser.get("/api/deps").status_code, 200)
                browser.get("/api/status")
            s.equal("routine dependency and status polls perform one nearby scan", near.call_count, 1)
            s.equal("routine dependency and status polls perform one full search", walk.call_count, 1)
            s.check("a failed API search is saved without path repairs",
                    (root / "api-config.json").is_file())
            browser.get("/api/deps?recheck=1")
            browser.get("/api/deps")
            s.equal("only an explicit API Recheck repeats the full search", walk.call_count, 2)

    return s
