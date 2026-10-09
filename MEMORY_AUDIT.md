# 8 GB memory audit — 9 October 2026

## Fixed

- WanVideoWrapper's cached text encoder loaded T5 with `main_device` even
  when CPU encoding was selected. The supposed OOM fallback could therefore
  run out of GPU memory before CPU encoding began. Engine start/restart now
  repairs that exact loader argument using its Python syntax tree, preserves
  a source-hashed backup, and leaves unfamiliar upstream code untouched with
  a console message. The fix was also checked against current upstream source.
- Transformer block prefetch defaults to zero so an extra block does not
  consume space needed for activations on a small card.
- Unsupported fit guarantees have been qualified. CPU T5 still needs host
  RAM; changing its execution device does not eliminate the model weights.

## Validation

Gate, units, new memory reproducer, graph, API and stress suites:
**260 checks passed; 3 process-takeover checks failed in this runner**.
The same three also failed when the API suite was run on unchanged `main`.
All ten new CPU-loading checks and all 74 graph checks passed. The test first
reproduces the old GPU allocation in CPU mode, then executes the repaired
call and verifies CPU/GPU selection, backups and repeat application.

The API tests use a mock engine. No full LongCat checkpoint was run on an
8 GB GPU; neither a real VRAM peak nor production visual quality is verified.

## Apply

Use the updated `avatar-studio/` source, not the historical ZIP in the repo.
Restart ComfyUI from the Engine page after updating. Start with a short 480p
clip, fp8, block swap and tiled VAE. Other AI engines should be closed.
If sampling still exhausts VRAM, try increasing Block swap. For remote
ComfyUI, the loader fix must be applied on that host and it must be restarted.
Save the final error and report the GPU model and system RAM for a GPU check.
