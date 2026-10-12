#!/usr/bin/env bash
# What the labs leave on disk and in memory, and how to remove it.
#
#   bash labs/cleanup.sh                     # list only (default): sizes, running lab processes, containers
#   bash labs/cleanup.sh --remove models     # the downloaded and quantized GGUF models (~7 GB)
#   bash labs/cleanup.sh --remove llama      # the pinned llama.cpp checkout and build (~250 MB)
#   bash labs/cleanup.sh --remove builds     # the labs' own C++ build (~2 MB)
#   bash labs/cleanup.sh --remove results    # run outputs under results/ (evidence is copied into docs/labs/)
#   bash labs/cleanup.sh --remove all        # everything above
#
# Everything removed here can be recreated:
#   models  -> bash labs/p3/models.sh        (re-downloads 3.6 GB, then quantizes)
#   llama   -> bash labs/p3/vendor_llama.sh  (re-clones the pinned commit, ~15 s build)
#   builds  -> python3 labs/run_labs.py      (rebuilds automatically)
# Only fixed paths inside this worktree are ever deleted.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
declare -a NAMES=(models llama builds results)
path_for() {
  case "$1" in
    models)  echo "$ROOT/labs/p3/models.nosync" ;;
    llama)   echo "$ROOT/labs/p3/llama.nosync" ;;
    builds)  echo "$ROOT/labs/build" ;;
    results) echo "$ROOT/results" ;;
  esac
}
what_for() {
  case "$1" in
    models)  echo "Qwen2.5-1.5B GGUF: F16 download + Q8_0/Q4_K_M/Q4_0 made from it" ;;
    llama)   echo "llama.cpp b11562 source (shallow clone) + CPU-only build" ;;
    builds)  echo "C++ lab binaries (p1_*, p2_quant, test_kernels)" ;;
    results) echo "run outputs (records.jsonl, console.log, summary.md)" ;;
  esac
}

LAB_PROCS='labs/build/|llama-bench|llama-quantize|llama-perplexity|run_labs.py|llm_service_lab.py|llama_bench_lab.py|portable_lab'

list() {
  echo "== Disk used by the labs"
  total=0
  for n in "${NAMES[@]}"; do
    p="$(path_for "$n")"
    if [ -e "$p" ]; then
      kb=$(du -sk "$p" | cut -f1); total=$((total + kb))
      printf "  %-8s %8s  %s\n           %s\n" "$n" "$(du -sh "$p" | cut -f1)" "$(what_for "$n")" "${p#$ROOT/}"
    else
      printf "  %-8s %8s  (not present)\n" "$n" "-"
    fi
  done
  printf "  %-8s %8s\n" "total" "$((total / 1024)) MB"

  echo
  echo "== Lab processes still running (should be none when no lab is running)"
  pgrep -fl "$LAB_PROCS" | grep -v -e cleanup.sh -e "/bin/zsh -c" | cut -c1-110 | sed 's/^/  /' | grep . || echo "  none"

  echo
  echo "== Docker containers left by the labs (the portable lab uses --rm, so none expected)"
  if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
    docker ps -a --filter "volume=$ROOT/labs/portable" --format '  {{.Names}} {{.Status}}' | grep . || echo "  none"
  else
    echo "  docker not running"
  fi

  echo
  echo "== Not created by the labs, but measured by them (left alone)"
  echo "  gRPC_llm model servers (make llm-up / llm-down in gRPC_llm) hold their models in RAM:"
  pgrep -f "llm_service.llm_service" | while read -r pid; do
    ps -o rss=,command= -p "$pid" | awk '{printf "    %6.1f GB  %s\n", $1/1048576, substr($0, index($0,$2), 70)}'
  done
  [ -n "$(pgrep -f llm_service.llm_service)" ] || echo "    none running"
  echo "    (RSS = memory the process holds in its own pages. Weights an engine hands to the GPU"
  echo "     through Metal, as MLX does, are not counted here: check Activity Monitor > Memory.)"
}

remove() {
  local what="$1"
  case "$what" in models|llama|builds|results|all) ;; *) echo "unknown group: $what (use: ${NAMES[*]} all)" >&2; exit 2 ;; esac
  if pgrep -f "$LAB_PROCS" | grep -qv "$$"; then
    echo "a lab is still running; stop it first" >&2; exit 1
  fi
  local targets=()
  if [ "$what" = all ]; then targets=("${NAMES[@]}"); else targets=("$what"); fi
  for n in "${targets[@]}"; do
    p="$(path_for "$n")"
    [ -n "$p" ] || { echo "unknown group: $n (use: ${NAMES[*]} all)" >&2; exit 1; }
    case "$p" in "$ROOT"/*) ;; *) echo "refusing to delete outside the worktree: $p" >&2; exit 1 ;; esac
    if [ -e "$p" ]; then
      size=$(du -sh "$p" | cut -f1); rm -rf "$p"; echo "removed $n ($size): ${p#$ROOT/}"
    else
      echo "$n: already absent"
    fi
  done
}

case "${1:-}" in
  ""|--list) list ;;
  --remove) remove "${2:?--remove needs one of: ${NAMES[*]} all}" ;;
  *) echo "usage: bash labs/cleanup.sh [--list | --remove models|llama|builds|results|all]" >&2; exit 2 ;;
esac
