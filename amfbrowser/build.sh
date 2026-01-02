#! /bin/bash

# Build OCaml GTK3 amfbrowser and copy binary to project root
set -euo pipefail

OUT="amfbrowser"

rm -f "$OUT" 2>/dev/null || true
cd "sources"

# Build the executable with dune (target is amfbrowser.exe on all platforms)
if ! dune build amfbrowser.exe --profile release; then
  echo "Dune build failed. Log (tail):" >&2
  [ -f _build/log ] && tail -n 200 _build/log >&2 || true
  exit 1
fi

# Try standard path first
CANDIDATES=(
  "_build/default/amfbrowser.exe"
  "_build/default/amfbrowser"
  "_build/$(ocamlc -version 2>/dev/null || echo default)/amfbrowser.exe"
)

BIN_PATH=""
for c in "${CANDIDATES[@]}"; do
  if [ -f "$c" ]; then BIN_PATH="$c"; break; fi
done

# Fallback: search under _build for an executable named amfbrowser*
if [ -z "$BIN_PATH" ]; then
  FOUND=$(find _build -type f -name 'amfbrowser*' -perm -111 2>/dev/null | head -n1 || true)
  if [ -n "$FOUND" ]; then BIN_PATH="$FOUND"; fi
fi

cd ..

if [ -z "${BIN_PATH:-}" ] || [ ! -f "sources/${BIN_PATH}" ]; then
  echo "Build artifact not found. Showing dune log (tail):" >&2
  [ -f sources/_build/log ] && tail -n 200 sources/_build/log >&2 || true
  exit 1
fi

cp "sources/${BIN_PATH}" "$OUT"
chmod +x "$OUT"
echo "Built $OUT (from ${BIN_PATH})"
