"""NeMo log-mel features for Nemotron, as Transformers' NemotronAsrStreamingFeatureExtractor computes them for an
offline pass (center=True): pre-emphasis, a centred STFT with zero padding and a symmetric Hann window, Slaney mel
filters and log(x + 2^-24). The filters are built here the way librosa.filters.mel builds them, so librosa (and
numba) aren't needed. Rimlo computes the same features in TypeScript."""
import numpy as np
import torch

SR, N_FFT, HOP, WIN, MELS, PREEMPH = 16000, 512, 160, 400, 128, 0.97


def _hz_to_mel(f):
    f = np.asanyarray(f, dtype=np.float64)
    return np.where(f >= 1000.0, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0), f / (200.0 / 3))


def _mel_to_hz(m):
    m = np.asanyarray(m, dtype=np.float64)
    return np.where(m >= 15.0, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)), (200.0 / 3) * m)


def mel_filters() -> torch.Tensor:
    fftfreqs = np.linspace(0, SR / 2, 1 + N_FFT // 2)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(SR / 2), MELS + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fftfreqs[None, :]
    weights = np.maximum(0, np.minimum(-ramps[:-2] / fdiff[:-1, None], ramps[2:] / fdiff[1:, None]))
    weights *= (2.0 / (mel_f[2:MELS + 2] - mel_f[:MELS]))[:, None]
    return torch.from_numpy(weights.astype(np.float32))


FILTERS = mel_filters()


def features(audio: np.ndarray) -> torch.Tensor:
    """[1, frames, 128] log-mel for 16 kHz mono float audio."""
    x = torch.from_numpy(np.array(audio, dtype=np.float32))[None]
    x = torch.cat([x[:, :1], x[:, 1:] - PREEMPH * x[:, :-1]], dim=1)
    stft = torch.stft(x, N_FFT, hop_length=HOP, win_length=WIN, window=torch.hann_window(WIN, periodic=False),
                      return_complex=True, pad_mode='constant', center=True)
    mel = torch.log(FILTERS @ (stft.real.pow(2) + stft.imag.pow(2)) + 2 ** -24)
    return mel.permute(0, 2, 1)[:, :x.shape[1] // HOP]
