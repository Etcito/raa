# Local bricks

`scripts/build-bricks.sh` builds and stages the three RAA bricks without GitHub CI or a
registry, using the local toolchain and Node built-ins: `raa` (`@etcito/raa`, built by the
existing `scripts/build-package.sh`), `raa-sdfm` (`@etcito/raa-sdfm`, copied from
`packages/raa-sdfm`; JavaScript only) and `raa-anim` (`@etcito/raa-anim`, `raa-anim-js`
built with the `wasm-release` profile and `wasm-bindgen --target web`). The `raa` and
`raa-sdfm` packages keep version 0.4.1; the animation package is 0.5.0, the contract's
brick version defined by `raa-anim-js` independently of the inherited Rust version.

## Clean rebuild

```sh
rm -rf dist/bricks
bash scripts/build-bricks.sh
```

The default output is `dist/bricks`; the script removes and recreates it, so removing the
stage is a clean rebuild (a fully clean rebuild may also remove `dist/package/`,
`dist/example-basic/` and `target/wasm32-unknown-unknown/wasm-release/`). It needs `node`,
`npm`, `cargo`, `rustc` with the `wasm32-unknown-unknown` target and the `wasm-bindgen`
CLI pinned in `Cargo.lock` (0.2.129 here). Pick another directory with `--out`,
`--out-dir`, `--output`, `--output-dir` or `-o`; an output that is or contains the repository or a
package source is refused (for example `bash scripts/build-bricks.sh --out target/bricks`).

## Output

```text
dist/bricks/
  bricks.json
  raa/            package.json index.js index.d.ts README.md LICENSE pkg/
  raa-sdfm/       package.json index.js index.d.ts worker.js README.md src/
  raa-anim/       package.json index.js index.d.ts README.md LICENSE
                  types/all.d.ts types/brick.d.ts pkg/
```

The animation `types/all.d.ts` is generated from the staged package: the hand-written
`types/*.d.ts` in sorted order, then the wasm-bindgen module `pkg/raa_anim_js.d.ts` (the
low-level `*.wasm.d.ts` is not merged; its names would collide). `index.d.ts` references
it. The script checks that every `package.json` entrypoint (`main`, `types`, `exports`,
`files`) and every d.ts `/// <reference path="...">` target exists.

## bricks.json

```json
{
  "format": "raa.bricks",
  "version": 1,
  "wasmGzipMax": 300000,
  "bricks": [
    {
      "name": "raa-anim",
      "package": "@etcito/raa-anim",
      "version": "0.5.0",
      "dir": "raa-anim",
      "wasmGzipMax": 300000,
      "source": {
        "repository": "raa", "path": "packages/raa-anim",
        "builder": "scripts/build-bricks.sh", "crate": "raa-anim-js",
        "profile": "wasm-release", "target": "wasm32-unknown-unknown",
        "wasmBindgen": "0.2.129"
      },
      "wasm": {
        "path": "raa-anim/pkg/raa_anim_js_bg.wasm",
        "raw": 30858, "gzip": 13653, "sha256": "<64 lowercase hex>"
      }
    }
  ]
}
```

`bricks` is ordered `raa`, `raa-sdfm`, `raa-anim`; `wasm` is `null` for a JavaScript-only
brick (and an array if a brick ever ships several). `raw` is bytes, `gzip` is
`zlib.gzipSync` at level 9 and `sha256` is over the raw bytes. Every staged wasm is
measured and both sizes are printed. `wasmGzipMax` is the Q2 budget enforced for a brick:
300000 bytes for the animation brick this milestone ships; the inherited `raa` wasm is
recorded with `null` (it is already above 300000 B gzip at 0.4.1, and its size belongs to
the engine milestones).

## File-based consumption

The stage is plain ES modules; no install is needed. Node reads the wasm with `fs`:

```js
import { loadRaaAnim } from './dist/bricks/raa-anim/index.js';
const anim = await loadRaaAnim(); // dist/bricks/raa-anim/pkg/raa_anim_js_bg.wasm
console.log(anim.brickInfo());    // { name: 'raa-anim', version: '0.5.0' }
```

A page served from the stage fetches the default wasm next to `index.js`:

```html
<script type="module">
  import { loadRaaAnim } from './raa-anim/index.js';
  self.animBrick = await loadRaaAnim();
</script>
```

Serve it on a secure origin, for example `node web/serve.mjs dist/bricks --port 0`
(127.0.0.1). `./raa/index.js` exports `loadRaa`; `./raa-sdfm/index.js` exports
`createSdfmLoader`. Serve or copy the whole stage so each `index.js` keeps its `pkg/`.

## Native output and canonical cases

`cargo run -p raa-det --example selftest` prints the native reference as one JSON array,
in canonical order:

```json
[{"case":"det-selftest","seed":0,"ticks":0,"hash":"811c9dc5"},{"case":"det-selftest","seed":1,"ticks":60,"hash":"935f1896"},{"case":"det-selftest","seed":305419896,"ticks":600,"hash":"77aeeadf"}]
```

Each record has exactly `case`, `seed`, `ticks` and `hash`; seed and ticks are u32 JSON
numbers and the hash is eight lowercase hexadecimal digits without a prefix. One
`selftest(case, seed, ticks)` call returns one such record. Compare all four fields in
Node, and in Chrome through `examples/anim-brick` (`run({case, seeds, ticks})`). The
capture `node scripts/anim-capture.mjs --case det-selftest` invokes the page separately
for each canonical tick count, keeps canonical order and commits
`proofs/results/anim/det-selftest.json`. `node proofs/m16-anim-brick.mjs` checks unit
tests, this staging and manifest, `scripts/check-det.sh`, Node/native parity and the
committed Chrome record.
