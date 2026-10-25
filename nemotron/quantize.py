"""Quantizes the fp32 export into the two builds Rimlo offers. Both keep activations in fp32 and quantize weights:

- int8: every weight matrix to int8, per channel (dynamic quantization: MatMulInteger).
- int4: the feed-forward weights (two thirds of the encoder) to symmetric int4 in blocks of 64 (MatMulNBits), the
  rest to int8 as above. Plain int4 everywhere, or asymmetric int4, cost more accuracy (and asymmetric int4 runs at
  half speed in ONNX Runtime).

The decoder's embedding and both prediction-network and joint weights are int8 in both builds.

    python quantize.py FP32_DIR OUT_DIR
"""
import os
import re
import sys

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer


def int8(source, target):
    quantize_dynamic(source, target, weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=['MatMul'])


def int4(source, target):
    model = onnx.load(source)
    weights = {i.name for i in model.graph.initializer}
    matmuls = [n.name for n in model.graph.node if n.op_type == 'MatMul' and n.input[1] in weights]
    feed_forward = [n for n in matmuls if re.search(r'/feed_forward\d/', n)]
    quantizer = MatMulNBitsQuantizer(
        model,
        algo_config=DefaultWeightOnlyQuantConfig(block_size=64, is_symmetric=True, accuracy_level=4),
        nodes_to_exclude=[n for n in matmuls if n not in feed_forward],
    )
    quantizer.process()
    staged = f'{target}.stage.onnx'
    onnx.save_model(quantizer.model.model, staged)
    int8(staged, target)
    os.remove(staged)


def main():
    src, out = sys.argv[1:3]
    for variant, encode in (('int8', int8), ('int4', int4)):
        os.makedirs(f'{out}/{variant}', exist_ok=True)
        encode(f'{src}/encoder.onnx', f'{out}/{variant}/encoder.onnx')
        print('quantized', variant)
    quantize_dynamic(f'{src}/decoder.onnx', f'{out}/decoder.onnx', weight_type=QuantType.QInt8,
                     op_types_to_quantize=['Gather', 'MatMul'])
    quantize_dynamic(f'{src}/joint.onnx', f'{out}/joint.onnx', weight_type=QuantType.QInt8, per_channel=True,
                     op_types_to_quantize=['MatMul', 'Gemm'])
    for f in sorted(os.listdir(out)):
        path = f'{out}/{f}'
        if os.path.isfile(path):
            print(f, round(os.path.getsize(path) / 1e6, 1), 'MB')


if __name__ == '__main__':
    main()
