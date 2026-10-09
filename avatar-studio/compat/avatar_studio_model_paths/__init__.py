"""Expose Avatar Studio's selected shared model folders to ComfyUI."""
import json
from pathlib import Path

import folder_paths

for category, directories in json.loads(
        Path(__file__).with_name("paths.json").read_text(encoding="utf-8")).items():
    for directory in reversed(directories):
        folder_paths.add_model_folder_path(category, directory, is_default=True)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
