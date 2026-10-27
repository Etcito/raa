# RAA API: the @etcito/raa-anim brick (0.5.0)

`@etcito/raa-anim` is the animation brick package of the RAA S1-0 path: the generated
WebAssembly build of `crates/raa-anim-js`, which consumes `raa-det`. It exposes exactly
two calls, `brickInfo()` and `selftest(case, seed, ticks)`, through a plain ES module.
The package has no third-party dependency.

Files:

- `packages/raa-anim/index.js`: `loadRaaAnim(source?)`;
- `packages/raa-anim/index.d.ts`: the public entry point; it references the generated
  `types/all.d.ts` produced by the brick build;
- `packages/raa-anim/types/brick.d.ts`: the hand-written brick API declarations;
- `packages/raa-anim/pkg/`: generated wasm-bindgen output (`raa_anim_js.js`,
  `raa_anim_js_bg.wasm`), not checked in.

## Loading

```js
import { loadRaaAnim } from '@etcito/raa-anim';

const anim = await loadRaaAnim();
```

`loadRaaAnim(source?)` initialises the generated wasm package and resolves to a frozen
object with `brickInfo` and `selftest`. It reads the wasm with `fs` in Node and fetches it
in a browser. The default source is `./pkg/raa_anim_js_bg.wasm` next to `index.js`.

| `source` | Node | browser |
|---|---|---|
| omitted | default wasm next to `index.js`, read with `fs` | default wasm, fetched |
| `string`, no scheme | filesystem path to the wasm or its package directory; relative paths resolve against `process.cwd()` | URL relative to `index.js`, fetched |
| `file:`, `http:`, `https:`, `URL` | `file:` read with `fs`; `http(s):` fetched | fetched |
| `ArrayBuffer` / `Uint8Array` | used as wasm bytes | used as wasm bytes |

A second load reuses the already instantiated wasm module. Loading failures reject with
the underlying filesystem, fetch or WebAssembly error.

## `brickInfo()`

```js
anim.brickInfo(); // { name: 'raa-anim', version: '0.5.0' }
```

The result is a plain JavaScript object with exactly two own enumerable string fields,
`name` and `version`. In TypeScript both are string literals:

```ts
interface BrickInfo {
  readonly name: 'raa-anim';
  readonly version: '0.5.0';
}
```

### Brick version versus Rust package version

`brickInfo().version` is the contract's **BRICK_VERSION**, `0.5.0`, defined explicitly by
`raa-anim-js`. It is not derived from `CARGO_PKG_VERSION`. The crate inherits the
workspace's `Cargo.toml` version, which is 0.4.1 during this release, so the inherited
Rust package version and the brick contract version can differ. The npm package
`@etcito/raa-anim` releases as 0.5.0 with the brick contract: `brickInfo().version`
identifies the contract and the package, not the Rust workspace release.

## `selftest(case, seed, ticks)`

Executes one case of the Rust simulation and returns exactly one record:

```js
anim.selftest('det-selftest', 305419896, 600);
// { case: 'det-selftest', seed: 305419896, ticks: 600, hash: '77aeeadf' }
```

```ts
interface SelftestResult {
  readonly case: 'det-selftest';
  readonly seed: number;
  readonly ticks: number;
  readonly hash: string;
}
```

- `case` is the string `'det-selftest'`.
- `seed` and `ticks` are u32 values represented as JSON numbers: finite integers in
  `[0, 4294967295]`. No string coercion, wrapping or fractional truncation occurs.
- `hash` is the unsigned FNV-1a 32-bit result of `raa-det`'s simulation, formatted as
  exactly eight zero-padded lowercase hexadecimal digits, without a `0x` prefix. Its
  shape is `^[0-9a-f]{8}$`.

The canonical `DET_CASES`, in order, are:

| case | seed | ticks | reference hash |
|---|---|---|---|
| `det-selftest` | 0 | 0 | `811c9dc5` |
| `det-selftest` | 1 | 60 | `935f1896` |
| `det-selftest` | 305419896 | 600 | `77aeeadf` |

The zero-tick case exercises the simulation's zero-step behaviour (the empty hash
stream); the other two exercise simulation updates. All three are computed by the same
Rust code with no special-case hashes. Native, Node-wasm and browser-wasm runs execute
the same simulation and compare all four fields exactly; the browser proof assembles
these three records in canonical order.

Invalid arguments throw a `TypeError` carrying a stable `code`:

- `api.invalid_input`: `case` is not a string, or `seed`/`ticks` is not a finite integer
  in the u32 range;
- `api.unknown_case`: `case` names no registered case.

## Types and generated declarations

`index.d.ts` is the npm entry point (`exports["."].types`) and references the generated
`types/all.d.ts`. `types/brick.d.ts` declares `BrickInfo`, `SelftestCase`, `SelftestResult`
and `RaaAnimApi`; it uses string literal types for the brick metadata and the selftest
case. `index.d.ts` re-exports these shapes, declares
`RaaAnimSource = string | URL | ArrayBuffer | Uint8Array` and
`loadRaaAnim(source?: RaaAnimSource): Promise<RaaAnimApi>`.

## Building and validation

The generated `pkg/` directory is produced from `crates/raa-anim-js` with
`wasm-bindgen --target web`; the wasm-bindgen CLI must match the pinned crate version.
The Rust-side checks are:

```sh
cargo check
cargo test -p raa-anim-js
cargo check -p raa-anim-js --target wasm32-unknown-unknown
```
