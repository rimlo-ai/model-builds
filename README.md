# Model builds for Rimlo

Machine-learning models that [Rimlo](https://rimlo.app) downloads on demand, converted and quantized by CI from pinned upstream checkpoints. None are built on a developer machine. Rimlo pins every file by SHA-256 from a release's `manifest.json`.

## Nemotron speech recognition

[NVIDIA Nemotron 3.5 ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) (600M parameters, OpenMDW-1.1) transcribes 32 language-locales, including Hindi and Hinglish, with punctuation and capitals, and detects the language itself.

Rimlo runs it on ONNX Runtime's CPU provider, one offline pass per segment of up to 30 s, cut at pauses. The upstream model is cache-aware streaming; in streaming mode it sometimes drops most of a short, noisy clip, and the offline pass doesn't. The offline pass is also 6–8× faster on a CPU.

| Build | Encoder weights | Download | Loaded | Speed (2 threads, M4) |
| --- | --- | --- | --- | --- |
| `int8` | int8 per channel | 664 MB | ~1.1 GB | ~39× real time |
| `int4` | feed-forward int4 (symmetric, blocks of 64), the rest int8 | 486 MB | ~0.9 GB | ~32× real time |

The decoder and joint network are 8-bit in both. Each segment adds about 0.25 GB while it's transcribed.

Each build comes for two architectures, which differ only in how 8-bit weights are stored:

- **arm64**: signed. On Apple silicon this is about 1.5× as fast as unsigned and takes less memory.
- **x64**: unsigned. On x86 CPUs that have AVX2 but not VNNI (AMD Zen 2 and 3, Intel before Ice Lake), ONNX Runtime multiplies unsigned activations by signed weights with an instruction whose 16-bit sums saturate, and transcripts come out empty. Signed 7-bit weights would avoid that too, but cost 2–3 points of word error rate.

Accuracy, as word error rate (lower is better):

| | Hinglish (IndicVoices) | Noisy Hindi (IndicVoices) | Far-field English (AMI) | Phone English (Earnings-22) |
| --- | --- | --- | --- | --- |
| fp32 | 25.8 | 24.5 | 26.9 | 29.0 |
| `int8` | 27.2 | 25.6 | 28.7 | 29.2 |
| `int4` | 29.5 | 27.9 | 31.1 | 30.9 |

These were measured on 30–40 clips per set, so treat them as directional.

### Files

Each release has, named `nemotron-<version>-r<revision>-<file>` (with `arm64-` or `x64-` for the ONNX files):

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

`build.sh` exports fp32 ONNX (`export.py`), quantizes both builds for both architectures (`quantize.py`), writes `tokens.json` (`package.py`), and checks:

- `reference.py`: the fp32 encoder graph must match PyTorch within 2 × 10⁻³ on every test clip, and transcripts decoded from the fp32 graphs must equal PyTorch's greedy decoding. It saves the clips' features and PyTorch's transcripts.
- `check.py`: each architecture's builds, run on a CPU of that architecture, must stay within a mean character error rate of 5% (`int8`) and 8% (`int4`) of PyTorch's transcripts. CI checks x64 on an AMD EPYC (no VNNI) and arm64 on GitHub's Arm runner; `build.sh` checks the machine it runs on.

It needs about 7 GB of memory and 10 GB of disk.

To release, bump `REVISION` in `versions.sh` and push a tag `nemotron-<MODEL_VERSION>-r<REVISION>`.

### Test clips

`nemotron/tests/clips` holds four clips from the [FLEURS](https://huggingface.co/datasets/google/fleurs) test split (English, Hindi, Spanish and Japanese), CC BY 4.0, by Google, converted to 16-bit WAV.

## Voice: noise removal and Enhance voice

Two models for Rimlo's sound effects, which render a cleaned copy of a file's sound on request (not live):

- **Noise removal**: [Ceva's DPDFNet-2](https://huggingface.co/Ceva-IP/DPDFNet) for 48 kHz (Apache-2.0), as released. It takes one 10 ms frame (960-point STFT with a Vorbis window, 480 hop) and its state per step, and returns gains applied as a deep filter. Rimlo blends its output with the original to limit how far noise is turned down.
- **Enhance voice**: [Smule Renaissance Small](https://huggingface.co/smulelabs/Smule-Renaissance-Small) (MIT, 10.4M parameters), a generative restorer for degraded speech at 48 kHz. It's exported as a stateful step: a block of 16 frames of a 4096-point STFT (Hann, 2048 hop) in, the same frames 129 frames later out, with 37 state tensors carrying the time convolutions' history. Stepping a file through it gives the same output as running the whole file at once, without the memory that takes.

| Model | Download | Loaded, while running | Speed (M4) |
| --- | --- | --- | --- |
| Noise removal | 10 MB | ~110 MB | ~10× real time, 1 thread |
| Enhance voice | 43 MB | ~400 MB | ~10× real time, 4 threads |

Both are fp32, so one build serves every architecture.

### Files

Each release has, named `voice-r<revision>-<file>`:

- `denoise.onnx`: `spec` [1, 1, 481, 2] and `state_in` [56436] → `spec_e`, `state_out`.
- `enhance.onnx`: `spec` [1, 2049, 16, 2], `t0` and `total` (int64 scalars: the block's first frame, and the file's length in frames) and the states → `out` [1, 2049, 16, 2] (frames `t0 − 129` onwards) and `<state>_next`.
- `voice.json`: both models' STFT settings, DPDFNet's initial state (from its ONNX metadata) and the step model's state names and shapes.
- The models' licences and a notice of the changes, `SHA256SUMS` and `manifest.json`.

### Building

`voice/versions.sh` pins both checkpoints by revision and every file by SHA-256. `voice/requirements.txt` pins the toolchain.

```sh
python3 -m venv venv
venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1
venv/bin/pip install -r voice/requirements.txt
PYTHON=venv/bin/python voice/build.sh work
```

`build.sh` exports the step model (`export.py`, `srs_step.py`) and checks:

- `reference.py`: PyTorch's whole-clip output for each test clip, and DPDFNet's output on this machine.
- `check.py`: stepped through each clip, the step model must match PyTorch to within 10⁻³ of the peak (it's about 10⁻⁶), and DPDFNet must match the reference to the same tolerance. CI runs it on x64 and on GitHub's Arm runner.

It needs about 1 GB of memory and takes a minute.

To release, bump `REVISION` in `versions.sh` and push a tag `voice-r<REVISION>`.

The test clips are Nemotron's (`nemotron/tests/clips`).

## Licences

The models keep their upstream licences, included in every release with a notice of the changes.
