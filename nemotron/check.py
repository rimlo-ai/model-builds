"""Checks one architecture's quantized builds on the CPU it runs on: their transcripts of the test clips must stay
within a mean character error rate of PyTorch's (reference.py wrote both). Needs only numpy and onnxruntime.

    python check.py DIST_DIR ARCH          ARCH: arm64 or x64; the reference is in DIST_DIR/reference
"""
import json
import sys
import unicodedata

import numpy as np

# Mean character error rate against PyTorch's transcript, per build.
MAX_CER = {'int8': 0.05, 'int4': 0.08}


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
    from decode import Transcriber

    dist, arch = sys.argv[1:3]
    reference = json.load(open(f'{dist}/reference/transcripts.json'))
    failures = []
    for variant, limit in MAX_CER.items():
        transcribe = Transcriber(f'{dist}/{arch}/{variant}/encoder.onnx', f'{dist}/{arch}/decoder.onnx',
                                 f'{dist}/{arch}/joint.onnx', f'{dist}/tokens.json')
        errors = []
        for clip, expected in reference.items():
            text = transcribe(np.load(f'{dist}/reference/features/{clip}.npy'))
            errors.append(cer(expected, text))
            print(f'{arch} {variant} {clip}: CER vs PyTorch {errors[-1]:.3f}  {text[:70]}')
        mean = sum(errors) / len(errors)
        print(f'{arch} {variant}: mean CER vs PyTorch {mean:.3f} (limit {limit})')
        if mean > limit:
            failures.append(f'{arch} {variant} transcripts drift from PyTorch: mean CER {mean:.3f} > {limit}')
    if failures:
        print('\n'.join(['FAILED'] + failures))
        sys.exit(1)
    print('checked', arch)


if __name__ == '__main__':
    main()
