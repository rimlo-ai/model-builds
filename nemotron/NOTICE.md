# Notice

These files are derived from **Nemotron 3.5 ASR** (`nvidia/nemotron-3.5-asr-streaming-0.6b`) by NVIDIA Corporation, licensed under the OpenMDW License Agreement, version 1.1 (`LICENSE-OpenMDW-1.1.txt`).

Source: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b

Changes: the model was exported to ONNX for one offline pass per segment (its encoder with 56 steps of left attention context and 3 of lookahead), and its weights were quantized (int8, and int4 for the encoder's feed-forward layers). The build is described at https://github.com/rimlo-ai/model-builds.
