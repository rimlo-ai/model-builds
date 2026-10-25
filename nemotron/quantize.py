"""Quantizes the fp32 export into the builds Rimlo offers, for each CPU architecture. All keep activations in fp32 and
quantize weights:

- int8: every weight matrix to 8 bits, per channel (dynamic quantization: MatMulInteger).
- int4: the feed-forward weights (two thirds of the encoder) to symmetric int4 in blocks of 64 (MatMulNBits), the
  rest to 8 bits as above. Plain int4 everywhere, or asymmetric int4, cost more accuracy (and asymmetric int4 runs at
  half speed in ONNX Runtime).

The decoder's embedding and both prediction-network and joint weights are 8-bit in every build.

8-bit weights are signed for arm64 and unsigned for x64. x86 CPUs that have AVX2 but not VNNI (AMD Zen 2 and 3, Intel
before Ice Lake) multiply unsigned activations by signed weights with an instruction whose 16-bit sums saturate, and
the transcripts come out empty; unsigned weights avoid it. On Apple silicon unsigned weights are half as fast again
and take more memory, and signed 7-bit weights (ONNX Runtime's other workaround) cost 2–3 points of word error rate.

    python quantize.py FP32_DIR OUT_DIR      -> OUT_DIR/<arch>/{int8,int4}/encoder.onnx, OUT_DIR/<arch>/{decoder,joint}.onnx
"""
import os
import re
import sys

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer

WEIGHTS = {'arm64': QuantType.QInt8, 'x64': QuantType.QUInt8}


def int8(source, target, weights):
    quantize_dynamic(source, target, weight_type=weights, per_channel=True, op_types_to_quantize=['MatMul'])


def int4(source, target, weights):
    model = onnx.load(source)
    initializers = {i.name for i in model.graph.initializer}
    matmuls = [n.name for n in model.graph.node if n.op_type == 'MatMul' and n.input[1] in initializers]
    feed_forward = [n for n in matmuls if re.search(r'/feed_forward\d/', n)]
    quantizer = MatMulNBitsQuantizer(
        model,
        algo_config=DefaultWeightOnlyQuantConfig(block_size=64, is_symmetric=True, accuracy_level=4),
        nodes_to_exclude=[n for n in matmuls if n not in feed_forward],
    )
    quantizer.process()
    staged = f'{target}.stage.onnx'
    onnx.save_model(quantizer.model.model, staged)
    int8(staged, target, weights)
    os.remove(staged)


def main():
    src, out = sys.argv[1:3]
    for arch, weights in WEIGHTS.items():
        for variant, encode in (('int8', int8), ('int4', int4)):
            os.makedirs(f'{out}/{arch}/{variant}', exist_ok=True)
            encode(f'{src}/encoder.onnx', f'{out}/{arch}/{variant}/encoder.onnx', weights)
            print('quantized', arch, variant)
        quantize_dynamic(f'{src}/decoder.onnx', f'{out}/{arch}/decoder.onnx', weight_type=weights,
                         op_types_to_quantize=['Gather', 'MatMul'])
        quantize_dynamic(f'{src}/joint.onnx', f'{out}/{arch}/joint.onnx', weight_type=weights, per_channel=True,
                         op_types_to_quantize=['MatMul', 'Gemm'])
    for root, _, files in sorted(os.walk(out)):
        for f in sorted(files):
            print(os.path.relpath(f'{root}/{f}', out), round(os.path.getsize(f'{root}/{f}') / 1e6, 1), 'MB')


if __name__ == '__main__':
    main()
