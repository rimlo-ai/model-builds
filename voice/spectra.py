"""Spectrograms as Rimlo computes them, in numpy (check.py runs where there's no PyTorch)."""
import wave

import numpy as np


def read_wav(path):
    """A 16-bit PCM WAV as float32 mono."""
    with wave.open(str(path)) as f:
        assert f.getsampwidth() == 2
        data = np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float32) / 32768
        return data.reshape(-1, f.getnchannels()).mean(axis=1)


def hann(n):
    """Periodic, as torch.hann_window."""
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)


def vorbis(n):
    s = np.sin(0.5 * np.pi * (np.arange(n) + 0.5) / (n / 2))
    return np.sin(0.5 * np.pi * s * s).astype(np.float32)


def stft(x, n_fft, hop, window):
    """Centred with reflect padding, as torch.stft and librosa.stft: (1, bins, frames, 2) float32."""
    x = np.pad(x, (n_fft // 2, n_fft // 2), mode='reflect')
    frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    spec = np.fft.rfft(x[idx] * window, axis=-1).T
    return np.ascontiguousarray(np.stack([spec.real, spec.imag], -1)[None], np.float32)


def run_steps(run, spec, block, delay, shapes):
    """A whole spectrogram (1, bins, T, 2) through a step function run(block, t0, total, states) -> (out, states),
    as Rimlo steps it: blocks of `block` frames (the last padded with zeros) until the output, `delay` frames
    behind, has caught up."""
    T = spec.shape[2]
    states = [np.zeros(s, np.float32) for s in shapes]
    outs = []
    t0 = 0
    while t0 - delay < T:
        blk = spec[:, :, t0:t0 + block]
        if blk.shape[2] < block:
            blk = np.pad(blk, ((0, 0), (0, 0), (0, block - blk.shape[2]), (0, 0)))
        out, states = run(np.ascontiguousarray(blk, np.float32), t0, T, states)
        outs.append(out)
        t0 += block
    return np.concatenate(outs, 2)[:, :, delay:delay + T]
