#!/usr/bin/env bash
# P3: one model in four formats, all made from the same F16 file.
#
#   bash labs/p3/models.sh
#
# Downloads Qwen2.5-1.5B-Instruct as F16 GGUF (Qwen's official repo, ~3.6 GB),
# then quantizes it with the pinned llama-quantize into Q8_0, Q4_0 and Q4_K_M.
# Quantizing from F16 (never from an already-quantized file) keeps the formats
# comparable: each one carries only its own rounding error.
#
# Output (gitignored, not synced to iCloud): labs/p3/models.nosync/
set -euo pipefail

URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-fp16.gguf"
SIZE=3560416288

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/models.nosync"
QUANT="$HERE/llama.nosync/build/bin/llama-quantize"
F16="$OUT/qwen2.5-1.5b-instruct-f16.gguf"
mkdir -p "$OUT"
[ -x "$QUANT" ] || { echo "build llama.cpp first: bash labs/p3/vendor_llama.sh" >&2; exit 1; }

if [ "$(stat -f %z "$F16" 2>/dev/null || echo 0)" != "$SIZE" ]; then
  echo "downloading F16 (resumable)..."
  curl -fL -C - --retry 3 -o "$F16" "$URL"
fi
[ "$(stat -f %z "$F16")" = "$SIZE" ] || { echo "size mismatch for $F16" >&2; exit 1; }

for TYPE in Q8_0 Q4_0 Q4_K_M; do
  DST="$OUT/qwen2.5-1.5b-instruct-$(echo "$TYPE" | tr 'A-Z' 'a-z').gguf"
  [ -s "$DST" ] || "$QUANT" "$F16" "$DST" "$TYPE" > "$OUT/quantize-$TYPE.log" 2>&1
done

( cd "$OUT" && shasum -a 256 *.gguf > SHA256SUMS && ls -l *.gguf )
