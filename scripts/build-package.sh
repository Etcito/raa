#!/usr/bin/env bash
# Build the @etcito/raa package: scripts/build-package.sh [--tag vX.Y.Z]
#
# raa-js wasm -> packages/raa/pkg/, dist/raa-sizes.json, dist/example-basic/,
# packages/raa/LICENSE, then npm pack -> release/etcito-raa-<version>.tgz and
# release/SHA256SUMS (the release asset set; release/ is gitignored).
#
# Tools come from PATH: node, npm, cargo, rustc (rustup, with the wasm32-unknown-unknown
# target) and wasm-bindgen, which must be the version pinned in Cargo.lock. Override the
# node/npm binaries with NODE=... / NPM=... Works on Linux and Windows git-bash.
#
# --tag vX.Y.Z  refuses to build unless the tag is v<packages/raa/package.json version>.
set -euo pipefail

usage() { echo "usage: scripts/build-package.sh [--tag vX.Y.Z]" >&2; }

TAG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG="${2:-}"; shift 2 ;;
    --tag=*) TAG="${1#--tag=}"; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "build-package.sh: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

NODE="${NODE:-node}"
NPM="${NPM:-npm}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PKG_DIR="$ROOT/packages/raa"
DIST="$ROOT/dist"
RELEASE="$ROOT/release"

VERSION="$("$NODE" -e 'console.log(require(process.argv[1]).version)' "$PKG_DIR/package.json")"
if [ -n "$TAG" ] && [ "$TAG" != "v$VERSION" ]; then
  echo "build-package.sh: --tag $TAG does not match the package version v$VERSION (packages/raa/package.json)" >&2
  exit 1
fi
TGZ_NAME="etcito-raa-$VERSION.tgz"

# The wasm-bindgen CLI must match the wasm-bindgen crate pinned in Cargo.lock, or the glue
# and the binary disagree. Fail before the (long) cargo build.
PINNED="$(awk '/^name = "wasm-bindgen"\r?$/ { getline; gsub(/\r/, ""); split($0, a, "\""); print a[2]; exit }' Cargo.lock)"
if [ -z "$PINNED" ]; then
  echo "build-package.sh: wasm-bindgen is not pinned in Cargo.lock" >&2
  exit 1
fi
HAVE="$(wasm-bindgen --version 2>/dev/null | awk '{ print $2 }' | tr -d '\r' || true)"
if [ "$HAVE" != "$PINNED" ]; then
  echo "build-package.sh: wasm-bindgen $PINNED is required (Cargo.lock), found '${HAVE:-none}'." >&2
  echo "  install it: cargo install wasm-bindgen-cli --version $PINNED --locked" >&2
  exit 1
fi

# Same toolchain as scripts/check.sh: the rustc on PATH, not a cargo config `build.rustc`
# that may lack the wasm32 std. An explicit RUSTC wins.
if [ -z "${RUSTC:-}" ] && command -v rustc >/dev/null; then
  RUSTC="$(command -v rustc)"
  export RUSTC
fi

TARGET_DIR="$(cargo metadata --no-deps --format-version 1 | "$NODE" -e '
  console.log(JSON.parse(require("fs").readFileSync(0, "utf8")).target_directory);
')"

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

echo "== LICENSE copy into packages/raa"
cp "$ROOT/LICENSE" "$PKG_DIR/LICENSE"

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

echo "== npm pack -> release/$TGZ_NAME"
rm -rf "$RELEASE"
mkdir -p "$RELEASE"
(cd "$PKG_DIR" && "$NPM" pack --pack-destination ../../release --offline)
if [ ! -f "$RELEASE/$TGZ_NAME" ]; then
  echo "build-package.sh: npm pack did not produce release/$TGZ_NAME" >&2
  exit 1
fi

echo "== tar -tzf release/$TGZ_NAME"
LIST="$(tar -tzf "$RELEASE/$TGZ_NAME")"
printf '%s\n' "$LIST"
MISSING=0
for f in package/package.json package/index.js package/index.d.ts package/README.md \
  package/LICENSE package/pkg/raa_js_bg.wasm package/pkg/raa_js.js; do
  if ! printf '%s\n' "$LIST" | grep -qx "$f"; then
    echo "build-package.sh: $f missing from release/$TGZ_NAME" >&2
    MISSING=1
  fi
done
[ "$MISSING" -eq 0 ] || exit 1

echo "== SHA256SUMS"
(cd "$RELEASE" && sha256sum "$TGZ_NAME" > SHA256SUMS && sha256sum -c SHA256SUMS)
echo "build-package.sh: ok ($TGZ_NAME, version v$VERSION)"
