"""Writes tokens.json: what Rimlo needs besides the ONNX graphs to transcribe with them. The vocabulary by id, the
language prompts, and the constants its feature extraction and decoding must match.

    python package.py CHECKPOINT_DIR OUT_DIR
"""
import json
import sys


def main():
    checkpoint, out = sys.argv[1:3]
    config = json.load(open(f'{checkpoint}/config.json'))
    processor = json.load(open(f'{checkpoint}/processor_config.json'))
    tokenizer = json.load(open(f'{checkpoint}/tokenizer.json'))
    pieces = {i: p for p, i in tokenizer['model']['vocab'].items()}
    for token in tokenizer.get('added_tokens', []):
        pieces[token['id']] = token['content']
    blank = config['blank_token_id']
    vocab = [pieces[i] for i in range(blank)]
    features = processor['feature_extractor']
    encoder = config['encoder_config']
    tokens = {
        'model': 'nvidia/nemotron-3.5-asr-streaming-0.6b',
        'blank': blank,
        'vocab': vocab,
        'prompts': processor['prompt_dictionary'],
        'autoPrompt': config['default_prompt_id'],
        'maxSymbolsPerStep': config['max_symbols_per_step'],
        'decoder': {'layers': config['num_decoder_layers'], 'hidden': config['decoder_hidden_size']},
        'features': {
            'sampleRate': features['sampling_rate'],
            'melBins': features['feature_size'],
            'fft': features['n_fft'],
            'window': features['win_length'],
            'hop': features['hop_length'],
            'preemphasis': features['preemphasis'],
        },
        'encoder': {
            'subsampling': encoder['subsampling_factor'],
            'leftContext': encoder['sliding_window'] - 1,
            'lookahead': 3,
        },
    }
    assert len(vocab) == blank and all(isinstance(p, str) for p in vocab)
    json.dump(tokens, open(f'{out}/tokens.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    print('tokens.json:', len(vocab), 'pieces,', len(tokens['prompts']), 'prompts')


if __name__ == '__main__':
    main()
