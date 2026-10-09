"""Completed weights must be reused from the same roots the engine sees."""
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bootstrap
import manager
from harness import Suite, tiny_safetensors


def put(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tiny_safetensors())
    return path


def run(slow=False):
    s = Suite("reuse")
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        engine = root / "ComfyUI"
        engine.mkdir()
        (engine / "main.py").write_text("# engine fixture")
        models = engine / "models"
        models.mkdir()
        cfg = dict(bootstrap.DEFAULT_CONFIG, comfy_dir=str(engine), models_dir=str(models))
        items = bootstrap.model_set(cfg) + bootstrap.extra_models(cfg)
        for item in items:
            put(models / item["folder"] / "my collection" / item["name"])
        with patch.object(bootstrap.requests, "get", side_effect=AssertionError("network used")):
            s.equal("nested completed model set schedules no download", manager.download_set(cfg), [])
            s.check("curated status recognises every nested file", all(i["installed"] for i in manager.curated(cfg)["files"]))
            item = items[0]
            existing = bootstrap.model_path(models, item, cfg)
            before = existing.read_bytes()
            bootstrap.download_file(cfg, item["repo"], item["path"], existing)
            s.equal("single-file request preserves completed weights without HTTP", existing.read_bytes(), before)
        saved = json.loads(json.dumps(cfg))
        with patch.object(Path, "rglob", side_effect=AssertionError("repeat tree walk")):
            s.equal("persisted exact file locations need no recursive scan", bootstrap.missing_models(models, saved), [])
        duplicate = put(models / item["folder"] / item["name"])
        s.equal("a new default copy does not replace the verified saved location", bootstrap.model_path(models, item, cfg), existing)
        refreshed = json.loads(json.dumps(cfg))
        bootstrap.clear_model_cache(refreshed)
        s.equal("explicit Recheck permits selecting the newly added default copy", bootstrap.model_path(models, item, refreshed), duplicate)
        duplicate.unlink()

        # A vanished individual file gets one relocation attempt; another
        # missing-file poll uses the saved miss until the person Rechecks.
        existing.unlink()
        moved = put(models / item["folder"] / "relocated" / item["name"])
        s.equal("a failed verified file is relocated once", bootstrap.model_path(models, item, cfg), moved)
        moved.unlink()
        bootstrap.model_path(models, item, cfg)
        with patch.object(Path, "rglob", side_effect=AssertionError("repeat missing scan")):
            s.equal("an unsuccessful relocation does not rescan on polling", bootstrap.model_path(models, item, cfg), models / item["folder"] / item["name"])
        put(moved)
        bootstrap.clear_model_cache(cfg)
        s.equal("explicit Recheck discovers the file at its new position", bootstrap.model_path(models, item, cfg), moved)

        # ComfyUI's own extra paths use a relative base, newline lists and
        # legacy category names. The first empty folder must not hide a hit.
        outside = root / "shared weights"
        encoder = next(i for i in items if i["folder"] == "text_encoders")
        t5 = put(outside / "encoders" / "nested" / encoder["name"])
        (models / encoder["folder"] / "my collection" / encoder["name"]).unlink()
        yaml_file = engine / "extra_model_paths.yaml"
        yaml_file.write_text("shared:\n  base_path: ../shared weights\n  clip: |\n    absent\n    encoders\n  is_default: true\n", encoding="utf-8")
        bootstrap.clear_model_cache(cfg)
        s.equal("extra_model_paths relative base and multiline clip alias are reused", bootstrap.model_path(models, encoder, cfg), t5)
        s.check("extra model paths do not require copying weights", not (models / encoder["folder"] / encoder["name"]).exists())

        # Settings' custom root must also reach the engine, not just the
        # app's installed indicator. Execute the actual registration module.
        shared_cfg = dict(cfg, models_dir=str(outside))
        shared_item = items[0]
        shared_file = put(outside / shared_item["folder"] / "nested" / shared_item["name"])
        bootstrap.model_path(outside, shared_item, shared_cfg)
        put(outside / shared_item["folder"] / shared_item["name"])
        original_yaml = yaml_file.read_bytes()
        bootstrap.install_model_paths(shared_cfg)
        module_path = engine / "custom_nodes/avatar_studio_model_paths/__init__.py"
        registrations = {}
        fake_paths = types.ModuleType("folder_paths")
        def register(category, path, is_default=False):
            entries = registrations.setdefault(category, [])
            entries.insert(0 if is_default else len(entries), path)
        fake_paths.add_model_folder_path = register
        spec = importlib.util.spec_from_file_location("avatar_paths_fixture", module_path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"folder_paths": fake_paths}):
            spec.loader.exec_module(module)
        s.check("engine registers the configured shared model root", str(outside / "diffusion_models") in registrations["diffusion_models"])
        runtime_choice = next(Path(d) / shared_item["name"]
                              for d in registrations[shared_item["folder"]]
                              if (Path(d) / shared_item["name"]).is_file())
        s.equal("runtime prioritises the saved hit over a new default copy", runtime_choice, shared_file)
        s.check("engine registers legacy clip directories too", str(outside / "clip") in registrations["text_encoders"])
        s.equal("registration never rewrites the person's extra-model YAML", yaml_file.read_bytes(), original_yaml)
        found_by_runtime = [p for d in registrations[shared_item["folder"]] for p in Path(d).rglob(shared_item["name"])]
        s.equal("runtime and installer resolve the same shared file", bootstrap.model_path(outside, shared_item, shared_cfg), found_by_runtime[0])
        s.equal("no copy is made into the engine default folder", (models / shared_item["folder"] / shared_item["name"]).exists(), False)

        # A download-looking filename alone is not proof the weights exist.
        bad = models / "vae/bad.safetensors"
        bad.parent.mkdir(exist_ok=True)
        for label, body in [("empty", b""), ("LFS pointer", b"version https://git-lfs.github.com/spec/v1\n"),
                            ("truncated payload", tiny_safetensors()[:-1]), ("truncated header", tiny_safetensors()[:50])]:
            bad.write_bytes(body)
            s.check(f"{label} is not treated as installed", not bootstrap.usable_model(bad))
        bad.write_bytes(tiny_safetensors())
        s.check("structurally complete weights are reusable", bootstrap.usable_model(bad))
        partial = bad.with_suffix(".safetensors.part")
        partial.write_bytes(tiny_safetensors())
        s.check("a part file is never an installed model", not bootstrap.usable_model(partial))

        # Setup can be re-run without forgetting the chosen engine or shared
        # model root. Stop at dependencies so no package install is needed.
        setup_cfg = dict(shared_cfg, want_manager=False, want_melband=False)
        class StopSetup(Exception):
            pass
        with patch.object(bootstrap, "find_python", return_value=sys.executable), \
                patch.object(bootstrap, "wanted_nodes", return_value=[]), \
                patch.object(bootstrap, "portable_python", side_effect=StopSetup("test stop")), \
                patch.object(bootstrap, "save_config"), \
                patch.object(bootstrap, "git_clone", side_effect=AssertionError("new engine download")):
            bootstrap.run_setup(setup_cfg, bootstrap.Progress(), bootstrap.ComfyProcess())
        s.equal("setup retains the existing ComfyUI", setup_cfg["comfy_dir"], str(engine))
        s.equal("setup retains the selected shared model folder", setup_cfg["models_dir"], str(outside))

        stock_engine = root / "stock-engine"
        stock_engine.mkdir()
        (stock_engine / "main.py").write_text("# engine fixture")
        (stock_engine / "custom_nodes").write_text("not writable as a directory")
        stock_cfg = dict(bootstrap.DEFAULT_CONFIG, comfy_dir=str(stock_engine),
                         models_dir=str(stock_engine / "models"))
        for item in items:
            put(stock_engine / "models" / item["folder"] / item["name"])
        bootstrap.install_model_paths(stock_cfg)
        s.check("stock model folders need no writable registration node", (stock_engine / "custom_nodes").is_file())
    return s
