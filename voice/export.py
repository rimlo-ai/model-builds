"""Exports Rimlo's voice models into DIST_DIR:

- enhance.onnx: Smule Renaissance Small as a stateful step of BLOCK frames (srs_step.py), fp32.
- denoise.onnx: Ceva's DPDFNet-2 48 kHz ONNX as released (one 10 ms frame per step, with its state).
- voice.json: what running them needs besides the graphs: STFT settings, DPDFNet's initial state (from the
  graph's metadata) and the step model's state names and shapes.

    python export.py SRS_DIR DPDFNET_ONNX DIST_DIR
"""
import json
import shutil
import sys
from pathlib import Path

import onnx
import torch

import srs_step

BLOCK = 16  # frames per step: 0.68 s. Larger blocks are no faster and hold more memory.


def export_enhance(srs_dir, out):
    step = srs_step.Step(srs_step.load(srs_dir)).eval()
    shapes = srs_step.state_shapes(step.m)
    args = (torch.zeros(1, 2049, BLOCK, 2), torch.tensor(0), torch.tensor(1000),
            *[torch.zeros(*s) for _, s in shapes])
    names = [n for n, _ in shapes]
    with torch.no_grad():
        torch.onnx.export(step, args, str(out), input_names=['spec', 't0', 'total'] + names,
                          output_names=['out'] + [n + '_next' for n in names], opset_version=18, dynamo=False)
    return {
        'sampleRate': 48000,
        'fft': srs_step.N_FFT,
        'hop': srs_step.HOP,
        'window': 'hann',
        'block': BLOCK,
        'delay': srs_step.DELAY,
        'highPass': 60,
        'states': [{'name': n, 'shape': list(s)} for n, s in shapes],
    }


def denoise_config(path):
    meta = {p.key: p.value for p in onnx.load(str(path), load_external_data=False).metadata_props}
    floats = lambda key: [float(v) for v in meta[key].split(',')]  # noqa: E731
    assert len(floats('erb_norm_init')) == int(meta['erb_norm_state_size'])
    assert len(floats('spec_norm_init')) == int(meta['spec_norm_state_size'])
    assert meta['window_type'] == 'vorbis' and meta['center'] == '1' and meta['pad_mode'] == 'reflect'
    return {
        'sampleRate': int(meta['sample_rate']),
        'fft': int(meta['n_fft']),
        'hop': int(meta['hop_length']),
        'window': 'vorbis',
        'stateSize': int(meta['state_size']),
        # The state starts as zeros with these running normalisation means at its start.
        'stateInit': floats('erb_norm_init') + floats('spec_norm_init'),
        # The output lags the input by this many frames (DPDFNet's look-back alignment).
        'lag': 4,
    }


def main():
    srs_dir, dpdfnet, dist = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    dist.mkdir(parents=True, exist_ok=True)
    config = {'enhance': export_enhance(srs_dir, dist / 'enhance.onnx')}
    shutil.copyfile(dpdfnet, dist / 'denoise.onnx')
    config['denoise'] = denoise_config(dist / 'denoise.onnx')
    (dist / 'voice.json').write_text(json.dumps(config, indent=1) + '\n')
    print('exported', dist)


if __name__ == '__main__':
    main()
