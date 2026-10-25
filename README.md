# Model builds for Rimlo

Machine-learning models that [Rimlo](https://rimlo.app) downloads on demand, converted and quantized by CI from pinned upstream checkpoints. None are built on a developer machine. Rimlo pins every file by SHA-256 from a release's `manifest.json`.

## Nemotron speech recognition

[NVIDIA Nemotron 3.5 ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) (600M parameters, OpenMDW-1.1) transcribes 32 language-locales, including Hindi and Hinglish, with punctuation and capitals, and detects the language itself.

Rimlo runs it on ONNX Runtime's CPU provider, one offline pass per segment of up to 30 s, cut at pauses. The upstream model is cache-aware streaming; in streaming mode it sometimes drops most of a short, noisy clip, and the offline pass doesn't. The offline pass is also 6–8× faster on a CPU.

| Build | Encoder weights | Download | Loaded | Speed (2 threads, M4) |
| --- | --- | --- | --- | --- |
| `int8` | int8 per channel | 664 MB | ~1.1 GB | ~39× real time |
| `int4` | feed-forward int4 (symmetric, blocks of 64), the rest int8 | 486 MB | ~0.9 GB | ~32× real time |

The decoder and joint network are int8 in both. Each segment adds about 0.25 GB while it's transcribed.

Accuracy, as word error rate (lower is better):

| | Hinglish (IndicVoices) | Noisy Hindi (IndicVoices) | Far-field English (AMI) | Phone English (Earnings-22) |
| --- | --- | --- | --- | --- |
| fp32 | 25.8 | 24.5 | 26.9 | 29.0 |
| `int8` | 27.2 | 25.6 | 28.7 | 29.2 |
| `int4` | 29.5 | 27.9 | 31.1 | 30.9 |

These were measured on 30–40 clips per set, so treat them as directional.

### Files

Each release has, named `nemotron-<version>-r<revision>-<file>`:

- `int8-encoder.onnx`, `int4-encoder.onnx`: log-mel `features` [1, frames, 128] and a language `prompt` id [1] → `encoded` [1, steps, 640]. The FastConformer encoder with the attention it was trained with (56 steps of left context, 3 of lookahead), the language-prompt fusion and the joint network's encoder projection.
- `decoder.onnx`: one prediction-network step: `token` [1, 1], `h` and `c` [2, 1, 640] → `decoded`, `h_out`, `c_out`.
- `joint.onnx`: `encoded` and `decoded` [1, 1, 640] → `logits` [1, 13088].
- `tokens.json`: the vocabulary by id, the language prompts (`auto` detects the language), and the feature and decoding constants.
- The model's licence and a notice of the changes, `SHA256SUMS` and `manifest.json`.

Features are NeMo's log-mel spectrogram (pre-emphasis 0.97, 512-point FFT of a 400-sample symmetric Hann window every 160 samples, centred with zero padding, 128 Slaney mel bins, log(x + 2⁻²⁴)), as `nemotron/features.py` computes them. Decoding is greedy RNN-T with at most 10 tokens per encoder step. Each encoder step is 80 ms.

### Building

`nemotron/versions.sh` pins the checkpoint by revision and every file by SHA-256. `nemotron/requirements.txt` pins the toolchain.

```sh
python3 -m venv venv
venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1
venv/bin/pip install -r nemotron/requirements.txt
PYTHON=venv/bin/python nemotron/build.sh work
```

`build.sh` exports fp32 ONNX (`export.py`), quantizes both builds (`quantize.py`), writes `tokens.json` (`package.py`) and verifies the result (`verify.py`):

- The fp32 encoder graph must match PyTorch within 2 × 10⁻³ on every test clip.
- Transcripts decoded from the graphs must match PyTorch's greedy decoding: exactly for fp32, and within a mean character error rate of 5% (`int8`) and 8% (`int4`).

It needs about 7 GB of memory and 10 GB of disk.

To release, bump `REVISION` in `versions.sh` and push a tag `nemotron-<MODEL_VERSION>-r<REVISION>`.

### Test clips

`nemotron/tests/clips` holds four clips from the [FLEURS](https://huggingface.co/datasets/google/fleurs) test split (English, Hindi, Spanish and Japanese), CC BY 4.0, by Google, converted to 16-bit WAV.

## Licences

The models keep their upstream licences, included in every release with a notice of the changes.
