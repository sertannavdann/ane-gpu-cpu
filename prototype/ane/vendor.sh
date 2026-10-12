#!/usr/bin/env bash
# Vendor ane.cpp at a pinned commit into prototype/ane/vendor/ane.cpp and record provenance.
# Idempotent: a second run verifies the pin and rewrites PIN.json. Patches in vendor/patches/*.patch
# are applied in order after checkout (none expected for R0).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
URL="https://github.com/skyfallsin/ane.cpp"
PIN="b87bf57356c26025853a60cddfe4b58784ee4fda"
DST="$HERE/vendor/ane.cpp"
if [ ! -d "$DST/.git" ]; then
  rm -rf "$DST"; mkdir -p "$DST"
  git -C "$DST" init -q
  git -C "$DST" remote add origin "$URL"
  git -C "$DST" fetch -q --depth 1 origin "$PIN"
  git -C "$DST" checkout -q FETCH_HEAD
fi
HEAD="$(git -C "$DST" rev-parse HEAD)"
[ "$HEAD" = "$PIN" ] || { echo "pin mismatch: $HEAD != $PIN" >&2; exit 1; }
git -C "$DST" checkout -q -- . 2>/dev/null || true
PATCHES=()
for p in "$HERE"/vendor/patches/*.patch; do
  [ -e "$p" ] || continue
  git -C "$DST" apply "$p"; PATCHES+=("$(basename "$p")")
done
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
python3 - "$DST" "$PIN" "$URL" "$(sha "$DST/core/ane_runtime.cpp")" "$(sha "$DST/core/ane_runtime.h")" "$(git -C "$DST" log -1 --format=%cI)" "${PATCHES[*]:-}" > "$HERE/vendor/PIN.json" <<'PY'
import json, sys
dst, pin, url, cpp, h, date, patches = sys.argv[1:8]
lic = open(f"{dst}/LICENSE").read().splitlines()[0]
json.dump({"url": url, "commit": pin, "commit_date": date, "license_line": lic,
           "sha256": {"core/ane_runtime.cpp": cpp, "core/ane_runtime.h": h},
           "patches": patches.split() if patches else []}, sys.stdout, indent=2)
PY
echo "vendored $PIN -> $DST"; cat "$HERE/vendor/PIN.json"
