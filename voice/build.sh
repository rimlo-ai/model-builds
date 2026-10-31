#!/usr/bin/env bash
# Builds Rimlo's voice models from the pinned checkpoints: downloads them (checked by SHA-256), exports Enhance
# voice as a stateful ONNX step and takes noise removal's ONNX as released (export.py), writes the reference that
# check.py compares builds against, then checks on this machine.
# Expects the toolchain from requirements.txt on PATH's python3 (or $PYTHON).
#
#   ./build.sh <work-dir>      The release files end up in <work-dir>/dist.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
. "$here/versions.sh"

work=$(mkdir -p "${1:?usage: build.sh <work-dir>}" && cd "$1" && pwd)
python=${PYTHON:-python3}

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

# fetch <url> <target> <sha256>
fetch() {
  if [ ! -f "$2" ] || [ "$(sha256 "$2")" != "$3" ]; then
    curl -fsSL --retry 5 --retry-all-errors -o "$2" "$1"
  fi
  [ "$(sha256 "$2")" = "$3" ] || { echo "build.sh: $(basename "$2") checksum mismatch" >&2; exit 1; }
}

mkdir -p "$work/srs" "$work/dpdfnet"
echo "$SRS_FILES" | while read -r file sum; do
  [ -n "$file" ] || continue
  fetch "https://huggingface.co/$SRS_REPO/resolve/$SRS_REVISION/$file" "$work/srs/$file" "$sum"
done
fetch "https://huggingface.co/$DPDFNET_REPO/resolve/$DPDFNET_REVISION/$DPDFNET_FILE" \
  "$work/dpdfnet/dpdfnet2_48khz_hr.onnx" "$DPDFNET_SHA256"

rm -rf "$work/dist"
cd "$here"
"$python" export.py "$work/srs" "$work/dpdfnet/dpdfnet2_48khz_hr.onnx" "$work/dist"
cp "$work/srs/LICENSE" "$work/dist/LICENSE-MIT-Smule.txt"
cp LICENSE-Apache-2.0.txt NOTICE.md "$work/dist/"
"$python" reference.py "$work/srs" "$work/dist" ../nemotron/tests/clips "$work/dist/reference"
"$python" check.py "$work/dist"
