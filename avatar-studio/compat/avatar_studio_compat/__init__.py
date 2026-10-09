"""Avatar Studio's compatibility shim, loaded by ComfyUI as a custom node.

transformers 5 dropped `output_hidden_states` from the wav2vec2 encoder:
the encoder now returns only its last layer, and per-layer states are
collected by a decorator on the top-level model's forward. WanVideoWrapper's
wav2vec2 (multitalk/wav2vec2.py) overrides that forward and calls the encoder
itself, so it gets `hidden_states=None`, and MultiTalkWav2VecEmbeds fails with
"'NoneType' object is not subscriptable" on every render.

This puts the 4.x behaviour back for callers that ask for it: the input to
each layer, then the encoder's output, the same tuple 4.x returned (13 for
the 12-layer model, equal to 4.57.6's to the last bit; docs/TEST_REPORT.md).
It changes nothing on transformers 4.x, or when hidden states are not asked
for. Avatar Studio copies it into ComfyUI/custom_nodes before every start
(bootstrap.install_compat). Its one node does nothing; it is there so the app
can see in /object_info that a running engine has loaded the shim.
"""

MARK = "_avatar_studio_compat"
VERSION = 1


class AvatarStudioCompat:
    """A marker: present in /object_info when this shim is loaded."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "Avatar Studio"
    DESCRIPTION = ("Avatar Studio's wav2vec2 compatibility shim is loaded. "
                   "Nothing to wire up.")

    def noop(self):
        return ()


NODE_CLASS_MAPPINGS = {"AvatarStudioCompat": AvatarStudioCompat}
NODE_DISPLAY_NAME_MAPPINGS = {"AvatarStudioCompat": "Avatar Studio compatibility"}


def _wrap(cls):
    original = cls.forward
    if getattr(original, MARK, False):
        return

    def forward(self, hidden_states, *args, **kwargs):
        want = kwargs.get("output_hidden_states")
        if want is None:
            want = getattr(self.config, "output_hidden_states", False)
        if not want:
            return original(self, hidden_states, *args, **kwargs)
        seen = []

        def before_layer(module, a, kw):
            seen.append(a[0] if a else kw.get("hidden_states"))

        hooks = [layer.register_forward_pre_hook(before_layer, with_kwargs=True)
                 for layer in self.layers]
        try:
            out = original(self, hidden_states, *args, **kwargs)
        finally:
            for h in hooks:
                h.remove()
        last = out[0] if isinstance(out, tuple) else out.last_hidden_state
        states = tuple(seen) + (last,)
        if isinstance(out, tuple):
            return (out[0], states) + tuple(out[1:])
        out.hidden_states = states
        return out

    setattr(forward, MARK, True)
    forward.__wrapped__ = original
    cls.forward = forward


def apply() -> bool:
    """Patch the wav2vec2 encoders if transformers is 5 or later."""
    try:
        import transformers
        if int(transformers.__version__.split(".")[0]) < 5:
            return False
        from transformers.models.wav2vec2 import modeling_wav2vec2 as m
    except Exception as exc:  # noqa: BLE001
        print(f"[Avatar Studio] wav2vec2 compatibility not applied: {exc}")
        return False
    for name in ("Wav2Vec2Encoder", "Wav2Vec2EncoderStableLayerNorm"):
        cls = getattr(m, name, None)
        if cls is not None:
            _wrap(cls)
    print(f"[Avatar Studio] transformers {transformers.__version__}: wav2vec2 "
          "hidden states restored for the lip sync")
    return True


apply()
