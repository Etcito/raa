# @etcito/raa-anim

The RAA animation brick as WebAssembly, package version **0.5.0**. It exposes the
versioned brick metadata and the deterministic simulation selftest of the `raa-anim-js`
crate (`crates/raa-anim-js` in the RAA repository).

The package is a plain ES module with no third-party dependencies. The generated wasm
glue and binary live in `pkg/` (`raa_anim_js.js` and `raa_anim_js_bg.wasm`), produced
from `crates/raa-anim-js` with `wasm-bindgen --target web`.

## Loading

`loadRaaAnim(source?)` initialises the wasm brick and resolves to its API. It reads the
wasm with `fs` in Node and fetches it in a browser. The default source is
`./pkg/raa_anim_js_bg.wasm` next to `index.js`.

```js
import { loadRaaAnim } from '@etcito/raa-anim';

const anim = await loadRaaAnim(); // pkg/raa_anim_js_bg.wasm next to index.js
```

Node filesystem paths and file URLs:

```js
const anim = await loadRaaAnim('/srv/raa/packages/raa-anim/pkg/raa_anim_js_bg.wasm');
const same = await loadRaaAnim(
  new URL('./pkg/raa_anim_js_bg.wasm', import.meta.url),
);
```

Browser fetch, with a source relative to the page or module:

```js
const anim = await loadRaaAnim(new URL('./vendor/raa-anim/pkg/raa_anim_js_bg.wasm', import.meta.url));
```

| `source` | Node | browser |
|---|---|---|
| omitted | `pkg/raa_anim_js_bg.wasm` next to `index.js`, read with `fs` | same file, fetched |
| `string` without a scheme | filesystem path to the wasm or its package directory (relative paths resolve against `process.cwd()`) | URL relative to `index.js`, fetched |
| `file:`, `http:` or `https:` URL / `URL` | `file:` read with `fs`; `http(s):` fetched | fetched |
| `ArrayBuffer` / `Uint8Array` | used as the wasm bytes | used as the wasm bytes |

Repeated loads reuse the instantiated wasm module and return a fresh frozen API object.
A loading failure rejects with the underlying filesystem, fetch or WebAssembly error.

## Brick metadata

`brickInfo()` returns a plain JavaScript object with exactly two own enumerable string
fields:

```js
anim.brickInfo();
// { name: 'raa-anim', version: '0.5.0' }
```

Both fields are typed as their string literals in `types/brick.d.ts`. The version is the
contract's `BRICK_VERSION` (0.5.0), defined explicitly by `raa-anim-js`; it is
**independent of the inherited Rust package version**. The crate inherits the workspace's
`Cargo.toml` version (0.4.1 at the time of this release), so the Rust package and the
brick contract release on different schedules. `brickInfo().version` always names the
brick contract and this npm package, never `CARGO_PKG_VERSION`.

## Selftest

`selftest(case, seed, ticks)` executes one case of the Rust simulation and returns exactly
one record:

```js
anim.selftest('det-selftest', 1, 60);
// { case: 'det-selftest', seed: 1, ticks: 60, hash: '935f1896' }
```

- `case` is `'det-selftest'`.
- `seed` and `ticks` are u32 values represented as JSON numbers: finite integers in
  `[0, 4294967295]`. Strings are not coerced and fractional values are not truncated.
- `hash` is the unsigned FNV-1a 32-bit result of the simulation, formatted as exactly
  eight zero-padded lowercase hexadecimal digits, without a `0x` prefix.

The canonical `DET_CASES`, in order, are:

| case | seed | ticks | reference hash |
|---|---|---|---|
| `det-selftest` | 0 | 0 | `811c9dc5` |
| `det-selftest` | 1 | 60 | `935f1896` |
| `det-selftest` | 305419896 | 600 | `77aeeadf` |

Every hash, including the zero-tick case, comes from the same Rust simulation; there are
no special-case fixtures. The zero-tick case hashes the empty stream, and the other two
exercise simulation updates. Native, Node and browser runs must execute the same
simulation and agree on all four record fields.

Invalid arguments throw a `TypeError`: `code: 'api.invalid_input'` for a non-string case
or a seed or tick count outside the u32 range, and `code: 'api.unknown_case'` for an
unknown case name.

## Types

`index.d.ts` is the public entry point and references the generated `types/all.d.ts`;
`types/brick.d.ts` declares `BrickInfo`, `SelftestCase`, `SelftestResult` and
`RaaAnimApi`.

## Validation

In the RAA repository:

```sh
cargo check
cargo test -p raa-anim-js
```

## License

MIT, see `LICENSE`.
