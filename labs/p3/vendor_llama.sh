#!/usr/bin/env bash
# P3: fetch and build a pinned llama.cpp, CPU backend only.
#
#   bash labs/p3/vendor_llama.sh
#
# Why pinned: llama.cpp changes daily; a benchmark is only reproducible against
# a fixed commit. Why CPU only: P3 asks how close llama.cpp's own CPU kernels
# get to the bandwidth limit measured in P1, so Metal (GPU), BLAS and Accelerate
# are switched off; every matmul runs in ggml's NEON / dotprod / i8mm code.
#
# Output (gitignored; ".nosync" keeps iCloud from syncing it):
#   labs/p3/llama.nosync/src      the checkout
#   labs/p3/llama.nosync/build    the build; binaries in build/bin
set -euo pipefail

TAG="b11562"
COMMIT="b2f036c0f138414dedc987446f1d9d802f920275"
REPO="https://github.com/ggml-org/llama.cpp"

HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HERE/llama.nosync"
SRC="$DEST/src"
BUILD="$DEST/build"
mkdir -p "$DEST"

if [ ! -d "$SRC/.git" ]; then
  git clone --quiet --depth 1 --branch "$TAG" "$REPO" "$SRC"
fi
HEAD="$(git -C "$SRC" rev-parse HEAD)"
if [ "$HEAD" != "$COMMIT" ]; then
  echo "pinned commit mismatch: tag $TAG is $HEAD, expected $COMMIT" >&2
  exit 1
fi

cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_METAL=OFF \
  -DGGML_BLAS=OFF \
  -DGGML_ACCELERATE=OFF \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_SERVER=OFF > "$DEST/cmake.log"
cmake --build "$BUILD" --target llama-bench llama-quantize llama-perplexity -j > "$DEST/build.log"

{
  echo "tag=$TAG"
  echo "commit=$HEAD"
  echo "built=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "compiler=$(cc --version | head -1)"
  echo "flags=GGML_NATIVE=ON GGML_METAL=OFF GGML_BLAS=OFF GGML_ACCELERATE=OFF"
} > "$DEST/BUILD-INFO"
cat "$DEST/BUILD-INFO"
"$BUILD/bin/llama-bench" --help >/dev/null && echo "llama-bench OK"
