"""Transcribes with the exported graphs, the way Rimlo does: one encoder pass over the segment's features, then greedy
RNN-T decoding (at most maxSymbolsPerStep tokens per encoder step, the decoder re-run only after a token). Needs only
numpy and onnxruntime."""
import json
import re

import numpy as np
import onnxruntime as ort

SPECIAL = re.compile(r'<(?:[a-z]{2}-[A-Z]{2}|unk)>')


class Transcriber:
    def __init__(self, encoder, decoder, joint, tokens, threads=2):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        session = lambda path: ort.InferenceSession(path, options, providers=['CPUExecutionProvider'])
        self.encoder, self.decoder, self.joint = session(encoder), session(decoder), session(joint)
        self.tokens = json.load(open(tokens))

    def __call__(self, features, language='auto'):
        """The transcript of log-mel `features` [1, frames, 128] (see features.py)."""
        t = self.tokens
        prompt = t['autoPrompt'] if language == 'auto' else t['prompts'][language]
        encoded = self.encoder.run(None, {'features': features, 'prompt': np.array([prompt], dtype=np.int64)})[0]
        layers, hidden = t['decoder']['layers'], t['decoder']['hidden']
        h = np.zeros((layers, 1, hidden), np.float32)
        c = np.zeros((layers, 1, hidden), np.float32)
        last, decoded, out = t['blank'], None, []
        for step in range(encoded.shape[1]):
            frame = encoded[:, step:step + 1, :]
            for _ in range(t['maxSymbolsPerStep']):
                if decoded is None:
                    decoded = self.decoder.run(None, {'token': np.array([[last]], dtype=np.int64), 'h': h, 'c': c})
                best = int(self.joint.run(None, {'encoded': frame, 'decoded': decoded[0]})[0][0].argmax())
                if best == t['blank']:
                    break
                out.append(best)
                last, h, c, decoded = best, decoded[1], decoded[2], None
        return self.text(out)

    def text(self, ids):
        """The transcript, without <unk> or the language tag the model adds after the last word."""
        pieces = (self.tokens['vocab'][i] for i in ids)
        return ' '.join(''.join(p for p in pieces if not SPECIAL.fullmatch(p)).replace('▁', ' ').split())
