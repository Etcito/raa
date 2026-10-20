#!/usr/bin/env bash
# Build the @etcito/raa package: scripts/build-package.sh (offline).
# raa-js wasm -> packages/raa/pkg/, dist/raa-sizes.json, dist/example-basic/,
# and npm pack -> dist/etcito-raa-0.1.0.tgz.
set -euo pipefail

NODE="${NODE:-/root/nmc/engines/bin/node}"
NPM="${NPM:-/root/nmc/engines/bin/npm}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Same toolchain as scripts/check.sh: the rustc on PATH, not a cargo config `build.rustc`
# that may lack the wasm32 std. An explicit RUSTC wins.
if [ -z "${RUSTC:-}" ] && command -v rustc >/dev/null; then
  RUSTC="$(command -v rustc)"
  export RUSTC
fi

TARGET_DIR="$(cargo metadata --no-deps --format-version 1 | "$NODE" -e '
  console.log(JSON.parse(require("fs").readFileSync(0, "utf8")).target_directory);
')"
PKG_DIR="$ROOT/packages/raa"
DIST="$ROOT/dist"

# write_sizes <wasm file> <wasm path recorded in the json> <json out> (S7 shape)
write_sizes() {
  "$NODE" -e '
    const fs = require("fs");
    const zlib = require("zlib");
    const crypto = require("crypto");
    const [file, rel, out] = process.argv.slice(1);
    const buf = fs.readFileSync(file);
    const sizes = {
      wasm: rel,
      raw: buf.length,
      gzip: zlib.gzipSync(buf, { level: 9 }).length,
      sha256: crypto.createHash("sha256").update(buf).digest("hex"),
    };
    fs.writeFileSync(out, JSON.stringify(sizes, null, 2) + "\n");
    console.log("wasm raw " + sizes.raw + " B, gzip " + sizes.gzip + " B");
  ' "$1" "$2" "$3"
}

echo "== cargo build --profile wasm-release --target wasm32-unknown-unknown -p raa-js"
cargo build --profile wasm-release --target wasm32-unknown-unknown -p raa-js

WASM="$TARGET_DIR/wasm32-unknown-unknown/wasm-release/raa_js.wasm"
if [ ! -f "$WASM" ]; then
  echo "build-package.sh: expected wasm not found: $WASM" >&2
  exit 1
fi

echo "== wasm-bindgen --target web --out-dir packages/raa/pkg"
rm -rf "$PKG_DIR/pkg"
wasm-bindgen --target web --out-dir "$PKG_DIR/pkg" "$WASM"
# wasm-bindgen writes pkg/.gitignore containing `*`, which would make npm pack drop pkg/.
# pkg/ stays out of git through the root .gitignore.
rm -f "$PKG_DIR/pkg/.gitignore"

echo "== sizes dist/raa-sizes.json"
mkdir -p "$DIST"
write_sizes "$PKG_DIR/pkg/raa_js_bg.wasm" "pkg/raa_js_bg.wasm" "$DIST/raa-sizes.json"

echo "== example dist/example-basic"
EX="$DIST/example-basic"
rm -rf "$EX"
mkdir -p "$EX/raa" "$EX/common"
cp examples/basic/index.html examples/basic/main.js examples/basic/nmc.gpu.json "$EX/"
cp "$PKG_DIR/package.json" "$PKG_DIR/index.js" "$PKG_DIR/index.d.ts" "$PKG_DIR/README.md" "$EX/raa/"
cp -R "$PKG_DIR/pkg" "$EX/raa/pkg"
cp -R web/common/. "$EX/common/"
write_sizes "$EX/raa/pkg/raa_js_bg.wasm" "raa/pkg/raa_js_bg.wasm" "$EX/sizes.json"

echo "== npm pack --pack-destination dist --offline"
TGZ="$DIST/etcito-raa-0.1.0.tgz"
rm -f "$TGZ"
(cd "$PKG_DIR" && "$NPM" pack --pack-destination ../../dist --offline)
if [ ! -f "$TGZ" ]; then
  echo "build-package.sh: npm pack did not produce $TGZ" >&2
  exit 1
fi

echo "== tar -tzf dist/etcito-raa-0.1.0.tgz"
LIST="$(tar -tzf "$TGZ")"
printf '%s\n' "$LIST"
MISSING=0
for f in package/package.json package/index.js package/index.d.ts package/README.md \
  package/pkg/raa_js_bg.wasm package/pkg/raa_js.js; do
  if ! printf '%s\n' "$LIST" | grep -qx "$f"; then
    echo "build-package.sh: $f missing from dist/etcito-raa-0.1.0.tgz" >&2
    MISSING=1
  fi
done
[ "$MISSING" -eq 0 ] || exit 1
echo "build-package.sh: ok"
