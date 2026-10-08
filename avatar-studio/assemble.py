"""
assemble.py - join a long clip's parts into one video, one frame at a time.

A long clip is rendered in parts (comfy.plan) so ComfyUI never holds more
than a few windows of frames. Joining them here keeps that promise: frames
are decoded, encoded and written one at a time, so a ten-minute clip needs
the same few megabytes as a ten-second one. The soundtrack goes underneath
in one piece, straight from the original recording — no seam in the sound
at the joins, music and all.

PyAV (the `av` package, the same library ComfyUI saves its videos with).
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

AUDIO_RATE = 48000


def assemble(parts: list[tuple[Path, int]], out: Path, fps: int,
             audio: Path | None = None, audio_start: float = 0.0,
             audio_seconds: float | None = None, crf: int = 17,
             should_cancel=None) -> dict:
    """parts: (video file, how many of its frames to take), in order.

    Writes H.264 (yuv420p, crf 17) at `fps`, and AAC from `audio` cut to
    audio_start .. audio_start + audio_seconds. Returns what was written.
    """
    import av

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part.mp4")
    written = 0
    samples = 0
    if not parts:
        raise RuntimeError("There were no parts to join.")
    with av.open(str(parts[0][0])) as first:
        size = first.streams.video[0].codec_context
        width, height = size.width, size.height
    with av.open(str(tmp), "w") as dst:
        # every stream is declared before the first packet: mp4 writes its
        # header then, and refuses a stream added after
        vstream = dst.add_stream("libx264", rate=fps)
        vstream.width, vstream.height = width, height
        vstream.pix_fmt = "yuv420p"
        vstream.options = {"crf": str(crf), "preset": "medium"}
        astream = None
        if audio is not None:
            astream = dst.add_stream("aac", rate=AUDIO_RATE)
            astream.layout = "stereo"
        for path, take in parts:
            got = 0
            with av.open(str(path)) as src:
                for frame in src.decode(video=0):
                    if got >= take:
                        break
                    if should_cancel and should_cancel():
                        raise RuntimeError("Cancelled.")
                    img = frame.reformat(width=vstream.width,
                                         height=vstream.height,
                                         format="yuv420p")
                    # the frame keeps its source file's time base unless told
                    # otherwise: a count of frames must be read as 1/fps
                    img.pts = written
                    img.time_base = Fraction(1, fps)
                    for packet in vstream.encode(img):
                        dst.mux(packet)
                    got += 1
                    written += 1
            if got < take:
                raise RuntimeError(f"{path.name} has {got} frames; "
                                   f"{take} were expected.")
        for packet in vstream.encode():
            dst.mux(packet)

        if astream is not None:
            samples = _write_audio(av, dst, astream, audio, audio_start,
                                   audio_seconds if audio_seconds is not None
                                   else written / fps)
    tmp.replace(out)
    return {"frames": written, "seconds": written / fps,
            "audio_seconds": samples / AUDIO_RATE}


def _write_audio(av, dst, astream, src_path: Path, start: float,
                 seconds: float) -> int:
    """The source's audio from `start` for `seconds`, as AAC. Returns samples.

    Everything goes through one FIFO at the output rate: the samples before
    `start` are read out and dropped, then exactly `seconds` worth are
    encoded in the encoder's frame size. No numpy, nothing held but a frame.
    """
    skip = int(round(start * AUDIO_RATE))
    left = int(round(seconds * AUDIO_RATE))
    resampler = av.AudioResampler(format="fltp", layout="stereo",
                                  rate=AUDIO_RATE)
    fifo = av.AudioFifo()
    written = 0
    size = astream.codec_context.frame_size or 1024

    def pump(final: bool = False) -> None:
        nonlocal skip, left, written
        if skip and fifo.samples:
            drop = min(skip, fifo.samples)
            fifo.read(drop)
            skip -= drop
        while left and (fifo.samples >= min(size, left)
                        or (final and fifo.samples)):
            chunk = fifo.read(min(size, left, fifo.samples))
            chunk.pts = written
            chunk.time_base = Fraction(1, AUDIO_RATE)
            written += chunk.samples
            left -= chunk.samples
            for packet in astream.encode(chunk):
                dst.mux(packet)

    with av.open(str(src_path)) as src:
        if not src.streams.audio:
            raise RuntimeError(f"{src_path.name} has no audio.")
        for frame in src.decode(audio=0):
            frame.pts = None
            for res in resampler.resample(frame):
                fifo.write(res)
            pump()
            if not left:
                break
    for res in resampler.resample(None):
        fifo.write(res)
    pump(final=True)
    for packet in astream.encode():
        dst.mux(packet)
    return written
