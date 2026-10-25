#!/usr/bin/env bash
# Turns build.sh's <dist> into release files in <out>: each named nemotron-<version>-r<revision>-<file> (per
# architecture where they differ), with SHA256SUMS, manifest.json (what Rimlo pins) and the release notes.
#
#   ./release.sh <dist> <out> <notes-file>
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
. "$here/versions.sh"

dist=$(cd "${1:?usage: release.sh <dist> <out> <notes-file>}" && pwd)
out=$(mkdir -p "${2:?usage: release.sh <dist> <out> <notes-file>}" && cd "$2" && pwd)
notes=${3:?usage: release.sh <dist> <out> <notes-file>}
tag=nemotron-$MODEL_VERSION-r$REVISION
repo=${GITHUB_REPOSITORY:-rimlo-ai/model-builds}

if [ -n "${GITHUB_REF_NAME:-}" ] && [ "$GITHUB_REF_NAME" != "$tag" ]; then
  echo "release.sh: tag $GITHUB_REF_NAME doesn't match versions.sh ($tag)" >&2
  exit 1
fi

for arch in arm64 x64; do
  for variant in int8 int4; do
    cp "$dist/$arch/$variant/encoder.onnx" "$out/$tag-$arch-$variant-encoder.onnx"
  done
  cp "$dist/$arch/decoder.onnx" "$out/$tag-$arch-decoder.onnx"
  cp "$dist/$arch/joint.onnx" "$out/$tag-$arch-joint.onnx"
done
for file in tokens.json LICENSE-OpenMDW-1.1.txt NOTICE.md; do
  cp "$dist/$file" "$out/$tag-$file"
done

cd "$out"
if command -v sha256sum >/dev/null; then sha256sum "$tag"-* >SHA256SUMS; else shasum -a 256 "$tag"-* >SHA256SUMS; fi

python3 - "$tag" "https://github.com/$repo/releases/download/$tag" "$CHECKPOINT_REPO" "$CHECKPOINT_REVISION" >manifest.json <<'PY'
import hashlib, json, os, sys

tag, base, checkpoint, revision = sys.argv[1:]
def entry(name):
    data = open(f'{tag}-{name}', 'rb').read()
    return {'url': f'{base}/{tag}-{name}', 'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data)}
shared = {role: entry(name) for role, name in (('tokens', 'tokens.json'), ('license', 'LICENSE-OpenMDW-1.1.txt'),
                                               ('notice', 'NOTICE.md'))}
variants = {
    v: {
        arch: {'encoder': entry(f'{arch}-{v}-encoder.onnx'), 'decoder': entry(f'{arch}-decoder.onnx'),
               'joint': entry(f'{arch}-joint.onnx'), **shared}
        for arch in ('arm64', 'x64')
    }
    for v in ('int8', 'int4')
}
json.dump({'tag': tag, 'checkpoint': {'repo': checkpoint, 'revision': revision}, 'variants': variants},
          sys.stdout, indent=2)
print()
PY

python3 - "$tag" "$CHECKPOINT_REPO" "$CHECKPOINT_REVISION" >"$notes" <<'PY'
import json, sys

tag, checkpoint, revision = sys.argv[1:]
manifest = json.load(open('manifest.json'))
print(f'Speech recognition for [Rimlo](https://rimlo.app): NVIDIA Nemotron 3.5 ASR, exported to ONNX for one offline '
      f'pass per segment and quantized. Built by CI from [{checkpoint}](https://huggingface.co/{checkpoint}/tree/{revision}) '
      f'at `{revision[:12]}`, and checked against PyTorch before release. OpenMDW-1.1.')
print()
print('| Build | arm64 download | x64 download |')
print('| --- | --- | --- |')
for name, archs in manifest['variants'].items():
    sizes = [f'{sum(f["size"] for f in archs[a].values()) / 1e6:.0f} MB' for a in ('arm64', 'x64')]
    print(f'| `{name}` | {sizes[0]} | {sizes[1]} |')
print()
print('8-bit weights are signed for arm64 and unsigned for x64, where signed ones saturate on CPUs without VNNI.')
PY
