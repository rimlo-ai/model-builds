"""Exports Nemotron 3.5 ASR for one offline pass per segment, as fp32 ONNX:

- encoder.onnx: log-mel [1, frames, 128] and a language prompt id [1] -> encoded [1, steps, 640]. The FastConformer
  encoder, the language-prompt fusion and the joint network's encoder projection, with the chunked-limited attention
  the model was trained with (56 steps of left context, 3 of lookahead). The mask is built from the input's length,
  so any length works.
- decoder.onnx: one prediction-network step: token [1, 1], LSTM state h and c [2, 1, 640] -> decoded [1, 1, 640].
- joint.onnx: one encoder step and one decoder output -> logits [1, 13088].

The convolution modules' 1x1 convolutions are written as linear layers, so they export as MatMul and get quantized
with the rest.

    python export.py CHECKPOINT_DIR OUT_DIR
"""
import os
import sys

import torch
from torch import nn
from transformers import Nemotron3_5AsrForRNNT

LOOKAHEAD = 3


def conv_forward(self, hidden_states, all_masked_rows=None, padding_cache=None):
    h = nn.functional.linear(hidden_states, self.pointwise_conv1.weight.squeeze(-1), self.pointwise_conv1.bias)
    h = nn.functional.glu(h, dim=-1).transpose(1, 2)
    h = self.depthwise_conv(h, padding_cache=None).transpose(1, 2)
    h = self.activation(self.norm(h))
    return nn.functional.linear(h, self.pointwise_conv2.weight.squeeze(-1), self.pointwise_conv2.bias)


class Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        encoder = model.encoder
        for layer in encoder.layers:
            layer.conv.forward = conv_forward.__get__(layer.conv)
        self.chunk = LOOKAHEAD + 1
        self.left_chunks = (encoder.config.sliding_window - 1) // self.chunk
        self.prompts = model.config.num_prompts

    def forward(self, features, prompt):
        encoder = self.model.encoder
        hidden = encoder.subsampling(features, None, padding_cache=None) * encoder.input_scale
        chunks = torch.arange(hidden.shape[1], device=hidden.device) // self.chunk
        diff = chunks[:, None] - chunks[None, :]
        mask = ((diff >= 0) & (diff <= self.left_chunks))[None, None]
        positions = encoder.encode_positions(hidden, cached_frames=0)
        for layer in encoder.layers:
            hidden = layer(hidden, attention_mask=mask, all_masked_rows=None, position_embeddings=positions)
        one_hot = nn.functional.one_hot(prompt, num_classes=self.prompts).to(hidden.dtype)[:, None, :]
        fused = self.model.prompt_projector(torch.cat([hidden, one_hot.expand(-1, hidden.shape[1], -1)], dim=-1))
        return self.model.encoder_projector(fused)


class Decoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, token, h, c):
        out, (h2, c2) = self.decoder.lstm(self.decoder.embedding(token), (h, c))
        return self.decoder.decoder_projector(out), h2, c2


class Joint(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.joint = model.joint

    def forward(self, encoded, decoded):
        return self.joint.head(self.joint.activation(encoded + decoded))[:, 0, :]


def main():
    checkpoint, out = sys.argv[1:3]
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)
    model = Nemotron3_5AsrForRNNT.from_pretrained(checkpoint, dtype=torch.float32).eval()
    config = model.config
    hidden, layers = config.decoder_hidden_size, config.num_decoder_layers
    with torch.inference_mode():
        torch.onnx.export(Encoder(model), (torch.randn(1, 400, 128) * 2 - 8, torch.tensor([101])), f'{out}/encoder.onnx',
                          input_names=['features', 'prompt'], output_names=['encoded'],
                          dynamic_axes={'features': {1: 'frames'}, 'encoded': {1: 'steps'}},
                          opset_version=18, dynamo=False, external_data=True)
        torch.onnx.export(Decoder(model), (torch.tensor([[5]]), torch.zeros(layers, 1, hidden), torch.zeros(layers, 1, hidden)),
                          f'{out}/decoder.onnx', input_names=['token', 'h', 'c'], output_names=['decoded', 'h_out', 'c_out'],
                          opset_version=18, dynamo=False)
        torch.onnx.export(Joint(model), (torch.randn(1, 1, hidden), torch.randn(1, 1, hidden)), f'{out}/joint.onnx',
                          input_names=['encoded', 'decoded'], output_names=['logits'], opset_version=18, dynamo=False)
    print('exported', out)


if __name__ == '__main__':
    main()
