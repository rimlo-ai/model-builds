"""Writes what check.py compares builds against, into OUT_DIR:

- For Enhance voice, each test clip's spectrogram and PyTorch's output for the whole clip at once (fp32), so the
  step graph must reproduce a whole-file run.
- For noise removal, DPDFNet's output on each clip from this machine's ONNX Runtime, so the other architecture
  must agree with it.

    python reference.py SRS_DIR DIST_DIR CLIPS_DIR OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

import srs_step
from spectra import hann, read_wav, stft
from check import denoise


def main():
    srs_dir, dist, clips, out = (Path(a) for a in sys.argv[1:5])
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((dist / 'voice.json').read_text())
    model = srs_step.load(srs_dir)
    session = ort.InferenceSession(str(dist / 'denoise.onnx'), providers=['CPUExecutionProvider'])
    torch.set_num_threads(4)
    for clip in sorted(clips.glob('*.wav')):
        # The clips are 16 kHz; taken as 48 kHz they're simply faster speech, which is all a check needs.
        x = read_wav(clip)[:48000 * 8]
        x = x / np.abs(x).max()
        spec = stft(x, srs_step.N_FFT, srs_step.HOP, hann(srs_step.N_FFT))
        with torch.no_grad():
            enhanced = model(torch.from_numpy(spec)).numpy()
        np.savez_compressed(out / f'{clip.stem}.npz', x=x, spec=spec, enhanced=enhanced,
                            denoised=denoise(session, config['denoise'], x))
        print('reference', clip.name, flush=True)


if __name__ == '__main__':
    main()
