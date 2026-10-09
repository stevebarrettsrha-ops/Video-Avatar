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
- Second pass: preflight now budgets cached video-model weights and CPU T5
  together. The earlier estimate incorrectly assumed one had unloaded before
  the other loaded. The UI labels this as model weights and explains that
  frames, activations, loading copies and the OS need additional RAM.
- Second pass: Avatar-managed wrapper updates temporarily undo only the exact
  source repair, then reapply it after success or failure. A real Git reproducer
  showed that the repair otherwise blocked an upstream `nodes.py` update.
  User edits and backups with mismatched hashes are preserved. Failed pulls
  are reported instead of silently counted as successful updates.
- The repair preserves Windows line endings and refuses a changed upstream
  method without the required `device` parameter.

## Validation

Second-pass gate, units, memory, graph, API and stress suites:
**278 checks passed; 3 process-takeover checks failed in this runner**.
The same three also failed when the API suite was run on unchanged `main`.
On clean GitHub runners, **all 281 checks passed on Python 3.10 and 3.13**,
including the process-takeover checks ([run 37993778113](https://github.com/stevebarrettsrha-ops/Video-Avatar/actions/runs/37993778113)).
The second-pass gate, unit, memory and graph suites passed **186 checks**,
including **28 memory checks** and all 74 graph checks. Tests reproduce the
old GPU allocation in CPU mode, execute the repaired call, and verify CPU/GPU
selection, exact backups, repeat application and the real Git update lifecycle.
The changed helper was rechecked against current upstream `nodes.py`: only
the intended loader argument changes and restoration recovers every byte.

The API tests use a mock engine. No full LongCat checkpoint was run on an
8 GB GPU; neither a real VRAM peak nor production visual quality is verified.

## Apply

Use the updated `avatar-studio/` source, not the historical ZIP in the repo.
Restart ComfyUI from the Engine page after updating. Start with a short 480p
clip, fp8, block swap and tiled VAE. Other AI engines should be closed.
If sampling still exhausts VRAM, try increasing Block swap. For remote
ComfyUI, the loader fix must be applied on that host and it must be restarted.
Save the final error and report the GPU model and system RAM for a GPU check.
