# Video-Avatar

Talking avatars, run locally. Give it a picture of a person and some speech,
recorded or from a file. Get back a video of them saying it, lips in sync.
It runs on **LongCat-Avatar** through kijai's ComfyUI-WanVideoWrapper.

| What | What it is |
|---|---|
| [`avatar-studio/`](avatar-studio/) | The local app: setup, model downloads, the engine, and the generator UI. **Start here.** |
| [`REFERENCE_LINKS.md`](REFERENCE_LINKS.md) | Every model and custom node with its download link and target folder. |
| `LongCat-Avatar-Studio.zip` | The `avatar-studio` folder as one download. |
| `ComfyUI-WanVideoWrapper-longcat_avatar.zip` | The wrapper's `longcat_avatar` branch as uploaded. Its nodes are now in the wrapper's `main`, which the app installs. |

To run it on Windows, double-click `avatar-studio/run.bat`. On Linux or
macOS, run `avatar-studio/run.sh`. Either one opens <http://127.0.0.1:7808>
and walks you through setup.

The [app README](avatar-studio/README.md) has the hardware notes. On an
RTX 4060 (8 GB) with 32 GB of RAM it runs with fp8 weights, block swap and
the text encoder on the CPU. Clips can be any length up to an hour. Long
speech is rendered in parts and joined, so memory stays the same whatever
the length. The download is about 42 GB, and a render takes minutes per
5.8-second window; that timing hasn't been measured on real hardware yet.
See [the test report](avatar-studio/docs/TEST_REPORT.md) and
[screenshots](avatar-studio/docs/screenshots/).
