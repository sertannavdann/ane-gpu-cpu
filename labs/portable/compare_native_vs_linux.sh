#!/usr/bin/env bash
# Same chip, two operating systems: run portable_lab.cpp natively on macOS, then
# inside Docker's Linux arm64 VM, and print both. What differs is not the silicon
# but what the OS and VM let a program use (page size, scheduler, SME, Metal, ANE).
#
#   bash labs/portable/compare_native_vs_linux.sh [seconds] [docker-image]
#
# The image needs g++; by default it reuses gRPC_llm's llm_service image (Debian,
# g++ 14) so nothing is downloaded. The source is mounted read-only; the binary is
# built in the container's /tmp and discarded with the container (--rm).
set -euo pipefail
SECONDS_PER_CASE="${1:-1}"
IMAGE="${2:-grpc_llm-llm_service:latest}"
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$(mktemp -d)"

echo "### macOS native (clang, -mcpu=native)"
clang++ -O3 -std=c++20 -mcpu=native "$DIR/portable_lab.cpp" -o "$OUT/portable_lab"
"$OUT/portable_lab" --seconds "$SECONDS_PER_CASE"

echo
echo "### Linux arm64 in Docker's VM (g++, -march=native), image $IMAGE"
docker run --rm --entrypoint sh -v "$DIR":/src:ro "$IMAGE" -c \
  "grep -m1 -i features /proc/cpuinfo | sed 's/^/   cpu /'; \
   g++ -O3 -std=c++20 -march=native -pthread /src/portable_lab.cpp -o /tmp/portable_lab && \
   /tmp/portable_lab --seconds $SECONDS_PER_CASE"
rm -rf "$OUT"
