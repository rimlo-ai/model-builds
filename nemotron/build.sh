#!/usr/bin/env bash
# Builds Rimlo's speech-recognition models from NVIDIA's pinned checkpoint: downloads it (checked by SHA-256),
# exports fp32 ONNX and checks it against PyTorch, quantizes the int8 and int4 builds for each architecture, writes
# tokens.json and the reference that check.py compares builds against, then checks this machine's architecture.
# Expects the toolchain from requirements.txt on PATH's python3 (or $PYTHON).
#
#   ./build.sh <work-dir>      The release files end up in <work-dir>/dist.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
. "$here/versions.sh"

work=$(mkdir -p "${1:?usage: build.sh <work-dir>}" && cd "$1" && pwd)
python=${PYTHON:-python3}

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

checkpoint=$work/checkpoint
mkdir -p "$checkpoint"
echo "$CHECKPOINT_FILES" | while read -r file sum; do
  [ -n "$file" ] || continue
  target=$checkpoint/$file
  if [ ! -f "$target" ] || [ "$(sha256 "$target")" != "$sum" ]; then
    curl -fsSL --retry 5 --retry-all-errors -o "$target" \
      "https://huggingface.co/$CHECKPOINT_REPO/resolve/$CHECKPOINT_REVISION/$file"
  fi
  [ "$(sha256 "$target")" = "$sum" ] || { echo "build.sh: $file checksum mismatch" >&2; exit 1; }
done

rm -rf "$work/fp32" "$work/dist"
cd "$here"
"$python" export.py "$checkpoint" "$work/fp32"
"$python" quantize.py "$work/fp32" "$work/dist"
"$python" package.py "$checkpoint" "$work/dist"
cp LICENSE-OpenMDW-1.1.txt NOTICE.md "$work/dist/"
"$python" reference.py "$checkpoint" "$work/fp32" "$work/dist/tokens.json" tests "$work/dist/reference"
case "$(uname -m)" in arm64 | aarch64) arch=arm64 ;; *) arch=x64 ;; esac
"$python" check.py "$work/dist" "$arch"
