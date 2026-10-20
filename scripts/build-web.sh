#!/usr/bin/env bash
# Build a demo crate for the web: scripts/build-web.sh <crate-dir> [--features <list>]
# Output in dist/<basename of crate-dir>/ (pkg/, page files, common/, sizes.json).
set -euo pipefail

NODE="${NODE:-node}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  echo "usage: scripts/build-web.sh <crate-dir> [--features <list>]" >&2
  exit 2
}

[ $# -ge 1 ] || usage
CRATE_ARG="$1"
shift
FEATURES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --features)
      [ $# -ge 2 ] || usage
      FEATURES="$2"
      shift 2
      ;;
    --features=*)
      FEATURES="${1#--features=}"
      shift
      ;;
    *)
      echo "build-web.sh: unknown argument: $1" >&2
      usage
      ;;
  esac
done

if [ ! -d "$CRATE_ARG" ]; then
  echo "build-web.sh: crate directory does not exist: $CRATE_ARG" >&2
  exit 1
fi
CRATE_DIR="$(cd "$CRATE_ARG" && pwd)"
if [ ! -f "$CRATE_DIR/Cargo.toml" ]; then
  echo "build-web.sh: no Cargo.toml in crate directory: $CRATE_ARG" >&2
  exit 1
fi
NAME="$(basename "$CRATE_DIR")"

cd "$ROOT"

# Same toolchain as scripts/check.sh: the rustc on PATH, not a cargo config `build.rustc`
# that may lack the wasm32 std. An explicit RUSTC wins.
if [ -z "${RUSTC:-}" ] && command -v rustc >/dev/null; then
  RUSTC="$(command -v rustc)"
  export RUSTC
fi

# Package name and target directory from cargo metadata, matched on manifest_path.
META="$(cargo metadata --no-deps --format-version 1)"
read -r PKG TARGET_DIR < <(printf '%s' "$META" | "$NODE" -e '
  const fs = require("fs");
  const m = JSON.parse(fs.readFileSync(0, "utf8"));
  const want = fs.realpathSync(process.argv[1]);
  const p = m.packages.find((p) => fs.realpathSync(p.manifest_path) === want);
  if (!p) { console.error("build-web.sh: " + process.argv[1] + " is not a workspace member"); process.exit(1); }
  console.log(p.name + " " + m.target_directory);
' "$CRATE_DIR/Cargo.toml")
[ -n "${PKG:-}" ] || exit 1
CRATE_US="${PKG//-/_}"

CARGO_ARGS=(build --profile wasm-release --target wasm32-unknown-unknown -p "$PKG")
if [ -n "$FEATURES" ]; then
  CARGO_ARGS+=(--features "$FEATURES")
fi
echo "== cargo ${CARGO_ARGS[*]}"
cargo "${CARGO_ARGS[@]}"

WASM="$TARGET_DIR/wasm32-unknown-unknown/wasm-release/$CRATE_US.wasm"
if [ ! -f "$WASM" ]; then
  echo "build-web.sh: expected wasm not found: $WASM (is $PKG a cdylib?)" >&2
  exit 1
fi

OUT="$ROOT/dist/$NAME"
rm -rf "$OUT"
mkdir -p "$OUT/pkg"
echo "== wasm-bindgen --target web --out-dir dist/$NAME/pkg"
wasm-bindgen --target web --out-dir "$OUT/pkg" "$WASM"

if [ -d "$CRATE_DIR/web" ]; then
  cp -R "$CRATE_DIR/web/." "$OUT/"
fi
if [ -d "$ROOT/web/common" ]; then
  mkdir -p "$OUT/common"
  cp -R "$ROOT/web/common/." "$OUT/common/"
fi

# sizes.json (S7): {"wasm": "pkg/<crate>_bg.wasm", "raw", "gzip", "sha256"}
"$NODE" -e '
  const fs = require("fs");
  const zlib = require("zlib");
  const crypto = require("crypto");
  const [out, rel] = process.argv.slice(1);
  const buf = fs.readFileSync(out + "/" + rel);
  const sizes = {
    wasm: rel,
    raw: buf.length,
    gzip: zlib.gzipSync(buf, { level: 9 }).length,
    sha256: crypto.createHash("sha256").update(buf).digest("hex"),
  };
  fs.writeFileSync(out + "/sizes.json", JSON.stringify(sizes, null, 2) + "\n");
  console.log("wasm raw " + sizes.raw + " B, gzip " + sizes.gzip + " B");
' "$OUT" "pkg/${CRATE_US}_bg.wasm"
