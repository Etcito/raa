#!/usr/bin/env bash
# Build and stage the three local RAA bricks: scripts/build-bricks.sh [--out <dir>]
#
# Stages dist/bricks/raa (@etcito/raa from scripts/build-package.sh), dist/bricks/raa-sdfm
# (copied from packages/raa-sdfm, no wasm) and dist/bricks/raa-anim (raa-anim-js built
# with the wasm-release profile and wasm-bindgen --target web), then writes bricks.json
# (versions, source provenance, wasm raw/gzip/sha256) and the animation package's sorted
# types/all.d.ts. Every wasm is measured and both sizes printed; the animation brick must
# satisfy Q2 (gzip <= 300000 B). build-package.sh, Cargo.toml and the package sources are
# not modified; raa and raa-sdfm keep version 0.4.1.
#
# Tools from PATH: node, npm, cargo, rustc with wasm32-unknown-unknown and wasm-bindgen at
# the Cargo.lock version. NODE=... overrides node.
set -euo pipefail

usage() { echo "usage: scripts/build-bricks.sh [--out <dir>]" >&2; exit 2; }
OUT_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out | --out-dir | --output | --output-dir | -o)
      [ $# -ge 2 ] || usage
      OUT_ARG="$2"
      shift 2
      ;;
    --out=* | --out-dir=* | --output=* | --output-dir=* | -o=*) OUT_ARG="${1#*=}"; shift ;;
    *) echo "build-bricks.sh: unknown argument: $1" >&2; usage ;;
  esac
done
OUT_ARG="${OUT_ARG:-dist/bricks}"

NODE="${NODE:-node}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="$("$NODE" -e 'console.log(require("path").resolve(process.argv[1]))' "$OUT_ARG")"
case "$OUT" in
  "/" | "$ROOT" | "$ROOT/dist" | "$ROOT/dist/package" | "$ROOT/dist/package/raa" | \
    "$ROOT/packages" | "$ROOT/packages/raa" | "$ROOT/packages/raa-sdfm" | "$ROOT/packages/raa-anim")
    echo "build-bricks.sh: refusing --out $OUT_ARG (source directory)" >&2
    exit 1
    ;;
esac
case "$ROOT/" in
  "$OUT"/*) echo "build-bricks.sh: refusing --out $OUT_ARG (contains the repository)" >&2; exit 1 ;;
esac

# Same toolchain as scripts/check.sh: an explicit RUSTC wins over cargo config.
if [ -z "${RUSTC:-}" ] && command -v rustc >/dev/null; then RUSTC="$(command -v rustc)"; export RUSTC; fi
PINNED="$(awk '/^name = "wasm-bindgen"\r?$/ { getline; gsub(/\r/, ""); split($0, a, "\""); print a[2]; exit }' Cargo.lock)"
HAVE="$(wasm-bindgen --version 2>/dev/null | awk '{ print $2 }' | tr -d '\r' || true)"
if [ -z "$PINNED" ] || [ "$HAVE" != "$PINNED" ]; then
  echo "build-bricks.sh: wasm-bindgen ${PINNED:-?} is required (Cargo.lock), found '${HAVE:-none}'." >&2
  exit 1
fi

echo "== scripts/build-package.sh (@etcito/raa, @etcito/raa-sdfm)"
bash "$ROOT/scripts/build-package.sh"
RAA_STAGE="$ROOT/dist/package/raa"
[ -f "$RAA_STAGE/pkg/raa_js_bg.wasm" ] && [ -f "$RAA_STAGE/index.js" ] || {
  echo "build-bricks.sh: dist/package/raa is incomplete after build-package.sh" >&2
  exit 1
}

echo "== cargo build --profile wasm-release --target wasm32-unknown-unknown -p raa-anim-js"
TARGET_DIR="$(cargo metadata --no-deps --format-version 1 | "$NODE" -e '
  console.log(JSON.parse(require("fs").readFileSync(0, "utf8")).target_directory);')"
cargo build --profile wasm-release --target wasm32-unknown-unknown -p raa-anim-js
ANIM_WASM="$TARGET_DIR/wasm32-unknown-unknown/wasm-release/raa_anim_js.wasm"
[ -f "$ANIM_WASM" ] || { echo "build-bricks.sh: expected wasm not found: $ANIM_WASM" >&2; exit 1; }

echo "== stage $OUT/{raa,raa-sdfm,raa-anim}"
rm -rf "$OUT"
mkdir -p "$OUT/raa" "$OUT/raa-sdfm" "$OUT/raa-anim"
cp -R "$RAA_STAGE/." "$OUT/raa/"
SDFM="$ROOT/packages/raa-sdfm"
for f in package.json index.js index.d.ts README.md worker.js; do cp "$SDFM/$f" "$OUT/raa-sdfm/$f"; done
cp -R "$SDFM/src" "$OUT/raa-sdfm/src"
[ ! -f "$SDFM/LICENSE" ] || cp "$SDFM/LICENSE" "$OUT/raa-sdfm/LICENSE"
ANIM="$ROOT/packages/raa-anim"
for f in package.json index.js index.d.ts README.md LICENSE; do cp "$ANIM/$f" "$OUT/raa-anim/$f"; done
cp -R "$ANIM/types" "$OUT/raa-anim/types"
rm -f "$OUT/raa-anim/types/all.d.ts"
mkdir -p "$OUT/raa-anim/pkg"
echo "== wasm-bindgen --target web --out-dir $OUT/raa-anim/pkg"
wasm-bindgen --target web --out-dir "$OUT/raa-anim/pkg" "$ANIM_WASM"
rm -f "$OUT/raa-anim/pkg/.gitignore"

echo "== types/all.d.ts, entrypoints, Q2 sizes and $OUT/bricks.json"
WASM_GZIP_MAX=300000 BRICKS_WASM_BINDGEN="$PINNED" "$NODE" -e '
  const fs = require("fs"), path = require("path"), zlib = require("zlib"), crypto = require("crypto");
  const OUT = path.resolve(process.argv[1]);
  const MAX_GZIP = Number(process.env.WASM_GZIP_MAX);
  const BINDGEN = process.env.BRICKS_WASM_BINDGEN;
  const fail = (message) => { console.error("build-bricks.sh: " + message); process.exit(1); };
  const exists = (file) => fs.existsSync(file);
  const walk = (dir) => exists(dir) ? fs.readdirSync(dir, { recursive: true, withFileTypes: true })
    .filter((entry) => entry.isFile()).map((entry) => path.join(entry.parentPath, entry.name)) : [];

  const BRICKS = [
    { name: "raa", package: "@etcito/raa", dir: "raa", wasmGzipMax: null,
      source: { repository: "raa", path: "packages/raa", builder: "scripts/build-package.sh",
        crate: "raa-js", profile: "wasm-release", target: "wasm32-unknown-unknown" } },
    { name: "raa-sdfm", package: "@etcito/raa-sdfm", dir: "raa-sdfm", wasmGzipMax: null,
      source: { repository: "raa", path: "packages/raa-sdfm", builder: "copy" } },
    { name: "raa-anim", package: "@etcito/raa-anim", dir: "raa-anim", wasmGzipMax: MAX_GZIP,
      source: { repository: "raa", path: "packages/raa-anim", builder: "scripts/build-bricks.sh",
        crate: "raa-anim-js", profile: "wasm-release", target: "wasm32-unknown-unknown" } },
  ];

  // index.d.ts references types/all.d.ts: sorted hand-written declarations, then the
  // wasm-bindgen module (the low-level *.wasm.d.ts would collide with its exports).
  const animDir = path.join(OUT, "raa-anim");
  const decls = ["types", "pkg"].flatMap((sub) => {
    const dir = path.join(animDir, sub);
    return exists(dir) ? fs.readdirSync(dir).sort().filter((name) => name.endsWith(".d.ts") &&
      !(sub === "types" ? name === "all.d.ts" : name.endsWith(".wasm.d.ts")))
      .map((name) => path.join(dir, name)) : [];
  });
  if (decls.length === 0) fail("raa-anim: no declaration files to aggregate");
  let allDts = "";
  for (const file of decls) {
    allDts += fs.readFileSync(file, "utf8");
    if (!allDts.endsWith("\n")) allDts += "\n";
  }
  fs.writeFileSync(path.join(animDir, "types", "all.d.ts"), allDts);
  console.log("types/all.d.ts <- " + decls.map((file) => path.relative(animDir, file)).join(", "));

  function checkPackage(brick) {
    const dir = path.join(OUT, brick.dir);
    const pkg = JSON.parse(fs.readFileSync(path.join(dir, "package.json"), "utf8"));
    if (pkg.name !== brick.package) fail(`${brick.dir}: package name is not ${brick.package}`);
    const entrypoints = new Set([pkg.main, pkg.types, ...(pkg.files ?? [])]);
    for (const value of Object.values(pkg.exports ?? {})) {
      if (typeof value === "string") entrypoints.add(value);
      else for (const target of Object.values(value ?? {})) entrypoints.add(target);
    }
    for (const rel of entrypoints) {
      if (typeof rel === "string" && !rel.includes("*") &&
        !exists(path.join(dir, rel.replace(/^\.\//, "")))) fail(`${brick.dir}: missing entrypoint ${rel}`);
    }
    for (const file of walk(dir).filter((name) => name.endsWith(".d.ts"))) {
      for (const match of fs.readFileSync(file, "utf8").matchAll(/\/\/\/\s*<reference\s+path="([^"]+)"/g)) {
        if (!exists(path.resolve(path.dirname(file), match[1]))) fail(`${brick.dir}: missing reference ${match[1]}`);
      }
    }
    return pkg;
  }

  const ownerOf = (file) => BRICKS.find((brick) => file.startsWith(path.join(OUT, brick.dir) + path.sep));
  const sizes = new Map();
  for (const file of walk(OUT).filter((name) => name.endsWith(".wasm"))) {
    const bytes = fs.readFileSync(file);
    const size = { path: path.relative(OUT, file).split(path.sep).join("/"), raw: bytes.length,
      gzip: zlib.gzipSync(bytes, { level: 9 }).length,
      sha256: crypto.createHash("sha256").update(bytes).digest("hex") };
    sizes.set(file, size);
    const budget = ownerOf(file)?.wasmGzipMax ?? null;
    console.log(`wasm ${size.path}: raw ${size.raw} B, gzip ${size.gzip} B` +
      (budget === null ? "" : `, Q2 max ${budget} B`) + `, sha256 ${size.sha256}`);
    if (budget !== null && size.gzip > budget) fail(`${size.path}: gzip ${size.gzip} B exceeds Q2 ${budget} B`);
  }
  if (sizes.size === 0) fail("no staged wasm found");

  const bricks = BRICKS.map((brick) => {
    const pkg = checkPackage(brick);
    const wasm = walk(path.join(OUT, brick.dir)).filter((name) => name.endsWith(".wasm"))
      .map((file) => sizes.get(file));
    if (wasm.length === 0) console.log(`${brick.dir}: JavaScript package, no wasm`);
    return { name: brick.name, package: pkg.name, version: pkg.version, dir: brick.dir,
      wasmGzipMax: brick.wasmGzipMax,
      source: { ...brick.source, ...(wasm.length === 0 ? {} : { wasmBindgen: BINDGEN }) },
      wasm: wasm.length === 0 ? null : wasm.length === 1 ? wasm[0] : wasm };
  });

  const manifest = { format: "raa.bricks", version: 1, wasmGzipMax: MAX_GZIP, bricks };
  fs.writeFileSync(path.join(OUT, "bricks.json"), JSON.stringify(manifest, null, 2) + "\n");
  console.log(`bricks.json: ${bricks.map((brick) => `${brick.name} ${brick.version}`).join(", ")}`);
' "$OUT"

echo "build-bricks.sh: ok ($OUT)"
