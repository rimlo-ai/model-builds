#!/usr/bin/env bash
# Turns build.sh's <dist> into release files in <out>: each named voice-r<revision>-<file>, with SHA256SUMS,
# manifest.json (what Rimlo pins) and the release notes.
#
#   ./release.sh <dist> <out> <notes-file>
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
. "$here/versions.sh"

dist=$(cd "${1:?usage: release.sh <dist> <out> <notes-file>}" && pwd)
out=$(mkdir -p "${2:?usage: release.sh <dist> <out> <notes-file>}" && cd "$2" && pwd)
notes=${3:?usage: release.sh <dist> <out> <notes-file>}
tag=voice-r$REVISION
repo=${GITHUB_REPOSITORY:-rimlo-ai/model-builds}

if [ -n "${GITHUB_REF_NAME:-}" ] && [ "$GITHUB_REF_NAME" != "$tag" ]; then
  echo "release.sh: tag $GITHUB_REF_NAME doesn't match versions.sh ($tag)" >&2
  exit 1
fi

files="denoise.onnx enhance.onnx voice.json LICENSE-MIT-Smule.txt LICENSE-Apache-2.0.txt NOTICE.md"
for file in $files; do cp "$dist/$file" "$out/$tag-$file"; done

cd "$out"
if command -v sha256sum >/dev/null; then sha256sum "$tag"-* >SHA256SUMS; else shasum -a 256 "$tag"-* >SHA256SUMS; fi

python3 - "$tag" "https://github.com/$repo/releases/download/$tag" $files >manifest.json <<'PY'
import hashlib, json, sys

tag, base, *names = sys.argv[1:]
roles = {'denoise.onnx': 'denoise', 'enhance.onnx': 'enhance', 'voice.json': 'config',
         'LICENSE-MIT-Smule.txt': 'licenseMit', 'LICENSE-Apache-2.0.txt': 'licenseApache', 'NOTICE.md': 'notice'}
def entry(name):
    data = open(f'{tag}-{name}', 'rb').read()
    return {'url': f'{base}/{tag}-{name}', 'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data)}
json.dump({'tag': tag, 'files': {roles[n]: entry(n) for n in names}}, sys.stdout, indent=2)
print()
PY

python3 - "$SRS_REPO" "$SRS_REVISION" "$DPDFNET_REPO" "$DPDFNET_REVISION" >"$notes" <<'PY'
import json, sys

srs, srs_rev, dpdf, dpdf_rev = sys.argv[1:]
total = sum(f['size'] for f in json.load(open('manifest.json'))['files'].values())
print('Noise removal and Enhance voice for [Rimlo](https://rimlo.app), built by CI and checked against PyTorch '
      'before release:')
print()
print(f'- Enhance voice: [{srs}](https://huggingface.co/{srs}/tree/{srs_rev}) at `{srs_rev[:12]}` (MIT), exported '
      'to ONNX as an exact fp32 step model.')
print(f'- Noise removal: [{dpdf}](https://huggingface.co/{dpdf}/tree/{dpdf_rev}) at `{dpdf_rev[:12]}` '
      '(Apache-2.0), DPDFNet-2 48 kHz as released.')
print()
print(f'Download: {total / 1e6:.0f} MB, for every architecture.')
PY
