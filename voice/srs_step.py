"""Smule Renaissance Small as a stateful step for ONNX: blocks of spectrogram frames in, the same blocks out
DELAY frames later, exactly as a whole-file run would give them.

Only the depthwise time convolutions look across frames: the band transformer works within a frame (attention
across the 80 bands), and the input features and output layers are per frame. So each time convolution keeps its
last 6·d input frames as state, the residual paths get delay lines, and frames outside [0, total) are zeroed before
every convolution, which is what the original's zero padding does. The receptive field is ±129 frames.

The band transformer is rewritten channel-last (its 1x1 convolutions become MatMuls, RMSNorm runs over the last
axis, RoPE is a product with a constant 8x8 matrix): the same weights and fp32 maths, and about twice as fast on
ONNX Runtime's CPU provider. The released RMSNorm rounds through fp16; here it's fp32 throughout.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DELAY = 129
N_FFT, HOP = 4096, 2048


def load(checkpoint):
    """The upstream model from its pinned files (model.py, smule-renaissance-small.pt), with fp32 RMSNorm."""
    sys.path.insert(0, str(Path(checkpoint)))
    import model as srs

    def rmsnorm_fp32(self, x):
        v = x.pow(2).mean(dim=1, keepdim=True)
        return x * torch.rsqrt(v + self.eps) * self.weight.unsqueeze(0).unsqueeze(-1)

    srs.RMSNorm.forward = rmsnorm_fp32
    m = srs.Renaissance().eval()
    m.load_state_dict(torch.load(Path(checkpoint) / 'smule-renaissance-small.pt', map_location='cpu'))
    return m


def rms_last(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * w


def band_forward(self, x):
    """The band transformer on (frames, bands, channels), channel-last."""
    BT, L, C = x.shape
    H, D = self.num_head, self.hidden_size
    qkv = rms_last(x, self.input_norm.weight, self.input_norm.eps) @ self.weight.weight[:, :, 0].t()
    qkv = qkv.view(BT, L, H, 3 * D).permute(0, 2, 1, 3)  # channels are ordered [head][q|k|v][d]
    q, k, v = qkv.split(D, dim=-1)
    cos, sin = self.cos_freq[:L], self.sin_freq[:L]
    q = (q * cos + (q @ self.rope_rot) * sin) * (D ** -0.5)  # scale q rather than the L x L scores
    k = k * cos + (k @ self.rope_rot) * sin
    att = torch.softmax(q @ k.transpose(-1, -2), dim=-1) @ v
    out = att.permute(0, 2, 1, 3).reshape(BT, L, H * D) @ self.output.weight[:, :, 0].t() + x
    norm, conv = self.MLP[0], self.MLP[1]
    a = F.silu(rms_last(out, norm.weight, norm.eps) @ conv.weight[:, :, 0].t())
    gate, z = a.chunk(2, dim=-1)
    return out + (F.silu(gate) * z) @ self.MLP_output.weight[:, :, 0].t()


def dilations(m):
    return [layer.seq_net.block2.dwconv.dilation[0] for layer in m.net]


def state_shapes(m):
    """Every state tensor's name and shape, in input order."""
    nb, C = m.nband, m.feature_dim
    shapes = [('features', (nb, C, DELAY))]
    for i, d in enumerate(dilations(m)):
        for j, dj in enumerate((1, d, 1)):
            shapes.append((f'l{i}_conv{j}', (nb, C, 6 * dj)))
        shapes.append((f'l{i}_res', (nb, C, 6 + 3 * d)))
    return shapes


def delay(buf, x):
    """A delay line: x delayed by buf's length, and the new buf. Time is the last axis."""
    both = torch.cat([buf, x], -1)
    n = x.shape[-1]
    return both[..., :n], both[..., n:]


class Step(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.dil = dilations(m)
        for layer in m.net:
            r = layer.band_net
            assert not r.causal
            D = r.hidden_size
            rot = torch.zeros(D, D)
            for i in range(D // 2):
                rot[2 * i + 1, 2 * i] = -1
                rot[2 * i, 2 * i + 1] = 1
            r.register_buffer('rope_rot', rot)

    def conv_block(self, blk, x, cache, d, mask):
        inp = torch.cat([cache, x * mask], -1)
        new_cache = inp[..., inp.shape[-1] - cache.shape[-1]:]
        y = F.conv1d(inp, blk.dwconv.weight, blk.dwconv.bias, dilation=d, groups=blk.dwconv.groups)
        return blk.pwconv2(blk.act(blk.pwconv1(blk.norm(y)))), new_cache

    def forward(self, spec, t0, total, *states):
        """spec (1, 2049, B, 2): frames t0..t0+B of a spectrogram `total` frames long (zeros past its end).
        Returns frames t0-DELAY..t0-DELAY+B of the output, and the next states."""
        m = self.m
        B = spec.shape[2]
        feats = m.feature_extraction(spec)[0]  # (bands, C, B)
        delayed_feats, f_buf = delay(states[0], feats)
        new_states = [f_buf]
        x = feats
        lag = 0  # how far behind the input block this layer's stream is
        idx = torch.arange(B, dtype=torch.int64)
        k = 1
        for li, layer in enumerate(m.net):
            x = band_forward(layer.band_net, x.permute(2, 0, 1)).permute(1, 2, 0)
            icb = layer.seq_net
            res, r_buf = delay(states[k + 3], x)
            y = x
            caches = []
            for j, (blk, dj) in enumerate(((icb.block1, 1), (icb.block2, self.dil[li]), (icb.block3, 1))):
                f = t0 - lag + idx
                mask = ((f >= 0) & (f < total)).to(y.dtype)
                y, c = self.conv_block(blk, y, states[k + j], dj, mask)
                caches.append(c)
                lag += 3 * dj
            x = y * icb.gamma.view(1, -1, 1) + res
            new_states += caches + [r_buf]
            k += 4
        x = x + delayed_feats
        out = []
        for i in range(m.nband):
            o = m.output_layers[i](x[i:i + 1])
            out.append(o.view(1, m.band_width[i], 2, B).permute(0, 1, 3, 2))
        return (torch.cat(out, 1), *new_states)

