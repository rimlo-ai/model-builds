"""Checks the voice models on the CPU this runs on, against reference.py's outputs:

- enhance.onnx, stepped through each clip's spectrogram, must match PyTorch's whole-clip output to within
  1e-3 of its peak (in practice it's about 1e-6).
- denoise.onnx must match the build machine's output to within 1e-3 of its peak.

Needs only numpy and onnxruntime.

    python check.py DIST_DIR           the reference is in DIST_DIR/reference
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from spectra import run_steps, stft, vorbis

TOLERANCE = 1e-3


def denoise(session, config, x):
    """DPDFNet on a whole signal, one frame at a time, as Rimlo runs it; the output lines up with the input."""
    n, hop, lag = config['fft'], config['hop'], config['lag']
    # Zeros after the end, so the last of the output (lag frames behind, and half a frame of centring) is made.
    padded = np.concatenate([x, np.zeros(n + lag * hop, np.float32)])
    spec = stft(padded, n, hop, vorbis(n))
    state = np.zeros(config['stateSize'], np.float32)
    state[:len(config['stateInit'])] = config['stateInit']
    out = np.empty_like(spec)
    for t in range(spec.shape[2]):
        frame = np.ascontiguousarray(spec[:, :, t:t + 1].transpose(0, 2, 1, 3))  # (1, 1, bins, 2)
        y, state = session.run(None, {'spec': frame, 'state_in': state})
        out[:, :, t] = y[0, 0]
    # Inverse STFT: the Vorbis window's squares sum to 1 at 50% overlap, so overlap-add needs no normalising.
    frames = np.fft.irfft(out[0, :, :, 0] + 1j * out[0, :, :, 1], n=n, axis=0).T * vorbis(n)
    y = np.zeros(hop * (frames.shape[0] + 1), np.float32)
    for t, f in enumerate(frames):
        y[t * hop:t * hop + n] += f
    start = n // 2 + lag * hop  # the centring, then the model's lag
    return y[start:start + len(x)]


def enhance(dist, config, spec):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.enable_mem_pattern = False
    session = ort.InferenceSession(str(dist / 'enhance.onnx'), so, providers=['CPUExecutionProvider'])
    names = [s['name'] for s in config['states']]
    shapes = [s['shape'] for s in config['states']]

    def run(block, t0, total, states):
        feeds = {'spec': block, 't0': np.array(t0, np.int64), 'total': np.array(total, np.int64)}
        feeds.update(zip(names, states))
        out, *states = session.run(None, feeds)
        return out, states

    return run_steps(run, spec, config['block'], config['delay'], shapes)


def main():
    dist = Path(sys.argv[1])
    config = json.loads((dist / 'voice.json').read_text())
    session = ort.InferenceSession(str(dist / 'denoise.onnx'), providers=['CPUExecutionProvider'])
    failures = []
    for ref in sorted((dist / 'reference').glob('*.npz')):
        r = np.load(ref)
        for name, got, want in (('enhance', enhance(dist, config['enhance'], r['spec']), r['enhanced']),
                                ('denoise', denoise(session, config['denoise'], r['x']), r['denoised'])):
            err = float(np.abs(got - want).max() / np.abs(want).max())
            print(f'{ref.stem} {name}: max difference {err:.2e} of peak', flush=True)
            if not err <= TOLERANCE:
                failures.append(f'{ref.stem} {name}')
    if failures:
        sys.exit('check.py: failed: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
