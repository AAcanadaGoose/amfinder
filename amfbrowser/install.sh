#! /bin/bash

set -euo pipefail

OPAM="${OPAM:-$(command -v opam)}"
OCAML_SWITCH="${OCAML_SWITCH:-4.14.1}"

# Windows users may have to uncomment (remove #) --disable-sandboxing below:
$OPAM init #--disable-sandboxing || true
$OPAM switch create "$OCAML_SWITCH" "ocaml-base-compiler.$OCAML_SWITCH" || $OPAM switch "$OCAML_SWITCH"

eval $($OPAM env)

$OPAM install -y dune odoc lablgtk3 cairo2 magic-mime camlzip

eval $($OPAM env)

./build.sh

DIR="$HOME/.local/share/amfinder"

mkdir -p "$DIR"

cp -r data "$DIR"
