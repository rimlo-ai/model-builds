"""Checks the build before it's released:

1. The fp32 encoder graph matches the PyTorch model (Transformers) on every test clip.
2. Transcripts from the graphs, decoded as Rimlo decodes them, match PyTorch's own greedy decoding: exactly for fp32,
   and within a character error rate for the int8 and int4 builds.

    python verify.py CHECKPOINT_DIR FP32_DIR DIST_DIR TESTS_DIR
"""
import json
import os
import sys
import unicodedata
import wave

import numpy as np
import onnxruntime as ort
import torch
from transformers import Nemotron3_5AsrForRNNT

from decode import Transcriber
from export import LOOKAHEAD, Encoder
from features import features

MAX_ENCODER_DIFF = 2e-3
# Mean character error rate against PyTorch's transcript, per build.
MAX_CER = {'fp32': 0.01, 'int8': 0.05, 'int4': 0.08}


def read(path):
    with wave.open(path) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(np.float32) / 32768


def norm(text):
    text = unicodedata.normalize('NFKC', text).lower()
    return ''.join(ch for ch in text if not unicodedata.category(ch).startswith(('P', 'S', 'Z')))


def cer(ref, hyp):
    a, b = norm(ref), norm(hyp)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1] / max(len(a), 1)


def main():
    checkpoint, fp32, dist, tests = sys.argv[1:5]
    torch.set_num_threads(os.cpu_count() or 4)
    clips = json.load(open(f'{tests}/clips.json'))['clips']
    audio = {c['file']: read(f'{tests}/clips/{c["file"]}') for c in clips}
    model = Nemotron3_5AsrForRNNT.from_pretrained(checkpoint, dtype=torch.float32).eval()
    tokens = f'{dist}/tokens.json'
    failures = []

    # PyTorch's transcripts, with its own decoding, are the reference.
    vocab = json.load(open(tokens))['vocab']
    reference = {}
    for c in clips:
        x = features(audio[c['file']])
        with torch.inference_mode():
            out = model.generate(input_features=x, attention_mask=torch.ones(x.shape[:2], dtype=torch.long),
                                 prompt_ids=torch.tensor([101]), num_lookahead_tokens=LOOKAHEAD)
        ids = (out.sequences if hasattr(out, 'sequences') else out)[0].tolist()
        reference[c['file']] = [i for i in ids if i < len(vocab)]

    # 1. The encoder graph against PyTorch.
    wrapped = Encoder(model)
    session = ort.InferenceSession(f'{fp32}/encoder.onnx', providers=['CPUExecutionProvider'])
    for c in clips:
        x = features(audio[c['file']])
        with torch.inference_mode():
            expected = wrapped(x, torch.tensor([101])).numpy()
        got = session.run(None, {'features': x.numpy(), 'prompt': np.array([101], dtype=np.int64)})[0]
        diff = float(np.abs(got - expected).max())
        print(f'encoder {c["file"]}: {x.shape[1]} frames, max |diff| {diff:.2e}')
        if diff > MAX_ENCODER_DIFF:
            failures.append(f'encoder differs from PyTorch on {c["file"]} by {diff:.2e}')
    del session, wrapped, model

    # 2. Transcripts from each build, one at a time to keep memory down.
    builds = {
        'fp32': (f'{fp32}/encoder.onnx', f'{fp32}/decoder.onnx', f'{fp32}/joint.onnx'),
        'int8': (f'{dist}/int8/encoder.onnx', f'{dist}/decoder.onnx', f'{dist}/joint.onnx'),
        'int4': (f'{dist}/int4/encoder.onnx', f'{dist}/decoder.onnx', f'{dist}/joint.onnx'),
    }
    for name, paths in builds.items():
        transcribe = Transcriber(*paths, tokens)
        errors = []
        for c in clips:
            text = transcribe(audio[c['file']])
            expected = transcribe.text(reference[c['file']])
            errors.append(cer(expected, text))
            print(f'{name} {c["file"]}: CER vs PyTorch {errors[-1]:.3f}, vs human {cer(c["text"], text):.3f}  {text[:70]}')
        del transcribe
        mean = sum(errors) / len(errors)
        print(f'{name}: mean CER vs PyTorch {mean:.3f} (limit {MAX_CER[name]})')
        if mean > MAX_CER[name]:
            failures.append(f'{name} transcripts drift from PyTorch: mean CER {mean:.3f} > {MAX_CER[name]}')

    if failures:
        print('\n'.join(['FAILED'] + failures))
        sys.exit(1)
    print('verified')


if __name__ == '__main__':
    main()
