# Notice

`enhance.onnx` is derived from **Smule Renaissance Small** (`smulelabs/Smule-Renaissance-Small`) by Smule Labs, licensed under the MIT License (`LICENSE-MIT-Smule.txt`).

Source: https://huggingface.co/smulelabs/Smule-Renaissance-Small

Changes: the model was exported to ONNX as a stateful step that takes a block of spectrogram frames and returns the same frames 129 frames later, matching a whole-file run. Its band transformer was rewritten in an equivalent form (the same weights), and its RMSNorm computes in fp32 instead of fp16.

`denoise.onnx` is **DPDFNet-2 (48 kHz)** (`Ceva-IP/DPDFNet`, `onnx/dpdfnet2_48khz_hr.onnx`) by Ceva, licensed under the Apache License, version 2.0 (`LICENSE-Apache-2.0.txt`), unchanged.

Source: https://huggingface.co/Ceva-IP/DPDFNet

`voice.json` holds the settings both need to run, including DPDFNet's initial state from its ONNX metadata. The build is described at https://github.com/rimlo-ai/model-builds.
