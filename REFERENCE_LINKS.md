# Reference links — LongCat Avatar in ComfyUI

Everything LongCat Avatar Studio installs or downloads, with where it goes.
Avatar Studio fetches all of it for you from its setup sheet and Models
page. This list is for doing it by hand, or for checking what it did.

## AI models

Put each file in the `ComfyUI/models/<folder>` shown.

| What | File | Folder | Download | Repo page |
|---|---|---|---|---|
| **Diffusion model** (LongCat-Avatar) | `LongCat-Avatar_comfy_bf16.safetensors` | `diffusion_models` | [download](https://huggingface.co/Kijai/LongCat-Video_comfy/resolve/main/Avatar/LongCat-Avatar_comfy_bf16.safetensors) | [Kijai/LongCat-Video_comfy › Avatar](https://huggingface.co/Kijai/LongCat-Video_comfy/blob/main/Avatar/LongCat-Avatar_comfy_bf16.safetensors) |
| **Distill LoRA** (12 steps, cfg 1) | `LongCat_distill_lora_alpha64_bf16.safetensors` | `loras` | [download](https://huggingface.co/Kijai/LongCat-Video_comfy/resolve/main/LongCat_distill_lora_alpha64_bf16.safetensors) | [Kijai/LongCat-Video_comfy](https://huggingface.co/Kijai/LongCat-Video_comfy/blob/main/LongCat_distill_lora_alpha64_bf16.safetensors) |
| **Text encoder** (umT5-XXL) | `umt5-xxl-enc-bf16.safetensors` | `text_encoders` | [download](https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors) | [Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy/tree/main) |
| **VAE** (Wan 2.1) | `Wan2_1_VAE_bf16.safetensors` | `vae` | [download](https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors) | [Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy/tree/main) |
| **Audio encoder** (wav2vec2) | `wav2vec2-chinese-base_fp16.safetensors` | `wav2vec2` | [download](https://huggingface.co/Kijai/wav2vec2_safetensors/resolve/main/wav2vec2-chinese-base_fp16.safetensors) | [Kijai/wav2vec2_safetensors](https://huggingface.co/Kijai/wav2vec2_safetensors/tree/main) |
| **Vocal separator** (optional) | `MelBandRoformer_fp32.safetensors` | `diffusion_models` | [download](https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/MelBandRoformer_fp32.safetensors) | [Kijai/MelBandRoFormer_comfy](https://huggingface.co/Kijai/MelBandRoFormer_comfy/tree/main) |

Notes from kijai's workflow:

- Use the **alpha64** distill LoRA. The older
  `LongCat_distill_lora_rank128_bf16.safetensors` "is not meant for this
  model and may have negative effects especially to extension capability."
- "LongCat models only run with bf16 base precision." (fp8 *storage* at
  load time is fine; that's the quantization setting.)
- "sageattention 1.0.6 does NOT work." Use sdpa, the default.
- Instead of the wav2vec2 safetensors file, the `(Down)load Wav2Vec Model`
  node can fetch
  [TencentGameMate/chinese-wav2vec2-base](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base)
  by itself.

## ComfyUI custom nodes

| Node pack | Why | Link |
|---|---|---|
| **ComfyUI-WanVideoWrapper** (required) | The LongCat Avatar nodes: loader, sampler, `WanVideo LongCat Avatar Extend Embeds`, wav2vec2 embeds | [github.com/kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) |
| ↳ `longcat_avatar` branch | Where the Avatar nodes were first built. It is now merged into `main`, which is what Avatar Studio installs. The zip in this repo is that branch. | [tree/longcat_avatar](https://github.com/kijai/ComfyUI-WanVideoWrapper/tree/longcat_avatar) |
| ↳ official example workflow | What Avatar Studio's graph follows (copied to `avatar-studio/assets/`) | [LongCatAvatar_audio_image_to_video_example_01.json](https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/main/example_workflows/LongCatAvatar_audio_image_to_video_example_01.json) |
| **ComfyUI-KJNodes** (required) | Image resize, window stitching (`ImageBatchExtendWithOverlap`, `GetImageRangeFromBatch`) | [github.com/kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) |
| **ComfyUI-MelBandRoFormer** (optional) | Separates the voice from music and noise | [github.com/kijai/ComfyUI-MelBandRoFormer](https://github.com/kijai/ComfyUI-MelBandRoFormer) |
| ComfyUI-Manager (optional) | Installs and updates nodes from inside ComfyUI | [github.com/Comfy-Org/ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager) |
| ComfyUI-VideoHelperSuite (not needed) | The workflow saves with `VHS_VideoCombine`; Avatar Studio uses ComfyUI's built-in `CreateVideo`/`SaveVideo` instead | [github.com/Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) |

## The engine and the model

- ComfyUI: [github.com/comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)
  (version 0.5 or newer, for `ReplaceVideoLatentFrames` and
  `TrimAudioDuration`)
- LongCat-Video, the original model project by Meituan:
  [github.com/meituan-longcat/LongCat-Video](https://github.com/meituan-longcat/LongCat-Video)

## Where this came from

- The video tutorial these links were taken from:
  [youtube.com/watch?v=eZTdaLbzqL4](https://www.youtube.com/watch?v=eZTdaLbzqL4)
- The sibling app this one is built on:
  [stevebarrettsrha-ops/Text-to-Video-Model](https://github.com/stevebarrettsrha-ops/Text-to-Video-Model)
  (MiniMax Studio)
