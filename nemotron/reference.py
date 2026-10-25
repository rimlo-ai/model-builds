"""Checks the fp32 export against the PyTorch model (Transformers), and writes what check.py compares each quantized
build against, so that can run on any CPU without PyTorch or the checkpoint:

1. The fp32 encoder graph must match PyTorch within MAX_ENCODER_DIFF on every test clip.
2. Transcripts from the fp32 graphs, decoded as Rimlo decodes them, must equal PyTorch's own greedy decoding.

Writes OUT_DIR/features/<clip>.npy (the clips' log-mel features) and OUT_DIR/transcripts.json (PyTorch's).

    python reference.py CHECKPOINT_DIR FP32_DIR TOKENS_JSON TESTS_DIR OUT_DIR
"""
import json
import os
import sys
import wave

import numpy as np
import onnxruntime as ort
import torch
from transformers import Nemotron3_5AsrForRNNT

from check import cer
from decode import Transcriber
from export import LOOKAHEAD, Encoder
from features import features

MAX_ENCODER_DIFF = 2e-3


def read(path):
    with wave.open(path) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(np.float32) / 32768


def main():
    checkpoint, fp32, tokens, tests, out = sys.argv[1:6]
    torch.set_num_threads(os.cpu_count() or 4)
    os.makedirs(f'{out}/features', exist_ok=True)
    clips = json.load(open(f'{tests}/clips.json'))['clips']
    vocab = json.load(open(tokens))['vocab']
    model = Nemotron3_5AsrForRNNT.from_pretrained(checkpoint, dtype=torch.float32).eval()
    failures = []

    inputs, transcripts = {}, {}
    for c in clips:
        x = features(read(f'{tests}/clips/{c["file"]}'))
        inputs[c['file']] = x
        np.save(f'{out}/features/{c["file"]}.npy', x.numpy())
        with torch.inference_mode():
            generated = model.generate(input_features=x, attention_mask=torch.ones(x.shape[:2], dtype=torch.long),
                                       prompt_ids=torch.tensor([101]), num_lookahead_tokens=LOOKAHEAD)
        ids = (generated.sequences if hasattr(generated, 'sequences') else generated)[0].tolist()
        transcripts[c['file']] = [i for i in ids if i < len(vocab)]

    wrapped = Encoder(model)
    session = ort.InferenceSession(f'{fp32}/encoder.onnx', providers=['CPUExecutionProvider'])
    for c in clips:
        x = inputs[c['file']]
        with torch.inference_mode():
            expected = wrapped(x, torch.tensor([101])).numpy()
        got = session.run(None, {'features': x.numpy(), 'prompt': np.array([101], dtype=np.int64)})[0]
        diff = float(np.abs(got - expected).max())
        print(f'encoder {c["file"]}: {x.shape[1]} frames, max |diff| {diff:.2e}')
        if diff > MAX_ENCODER_DIFF:
            failures.append(f'the fp32 encoder differs from PyTorch on {c["file"]} by {diff:.2e}')
    del session, wrapped, model

    transcribe = Transcriber(f'{fp32}/encoder.onnx', f'{fp32}/decoder.onnx', f'{fp32}/joint.onnx', tokens)
    texts = {}
    for c in clips:
        texts[c['file']] = transcribe.text(transcripts[c['file']])
        text = transcribe(inputs[c['file']].numpy())
        print(f'fp32 {c["file"]}: vs human {cer(c["text"], text):.3f}  {text[:70]}')
        if text != texts[c['file']]:
            failures.append(f'fp32 transcript of {c["file"]} differs from PyTorch: "{text}" vs "{texts[c["file"]]}"')
    json.dump(texts, open(f'{out}/transcripts.json', 'w'), ensure_ascii=False, indent=1)

    if failures:
        print('\n'.join(['FAILED'] + failures))
        sys.exit(1)
    print('reference written')


if __name__ == '__main__':
    main()
