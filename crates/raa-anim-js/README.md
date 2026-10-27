# raa-anim-js

Animation wasm brick backed by `raa-det`; also builds as a host-testable rlib.

`brickInfo()` returns a plain JavaScript object with exactly two own enumerable
string fields: `{name: "raa-anim", version: "0.5.0"}`. `BRICK_VERSION` is explicitly
defined in `src/api/brick.rs`, independently of `CARGO_PKG_VERSION`. It versions
the public brick contract and corresponding npm package, while the Rust package
inherits the workspace's `0.4.1` version. These release versions can differ;
updating the brick contract does not require changing the workspace version.

`selftest(case, seed, ticks)` returns exactly
`{case: "det-selftest", seed, ticks, hash}`. Seed and ticks are u32 JSON numbers;
hash is eight lowercase hexadecimal digits, zero-padded without a prefix.
The ordinary Rust simulation computes every result, including zero ticks.
Canonical calls follow `raa_det::DET_CASES`: `(0, 0)`, `(1, 60)`,
`(305419896, 600)` in that order.

JS arguments must be a string and two finite integer numbers in `[0, 4294967295]`.
No string coercion, wrapping or fractional truncation occurs. Invalid arguments
throw a TypeError with `code: "api.invalid_input"`; unknown case names throw a
TypeError with `code: "api.unknown_case"`. Host callers receive `ApiError` variants.

The build discovers `src/api/*.rs` in sorted filename order and generates modules,
`registered_cases()` and `api_files.txt`. Each module supplies
`register(&mut Vec<Case>)`; add cases there or add another snake_case API file
without editing central dispatch. Case names must be unique. JS conversion stays
behind the wasm target cfg; dispatch, validation and hex formatting run on hosts.

Validation: `cargo test -p raa-anim-js`, `cargo check`, and
`cargo check -p raa-anim-js --target wasm32-unknown-unknown`.
