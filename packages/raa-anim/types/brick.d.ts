// Hand-written types of the @etcito/raa-anim brick API (contract BRICK_VERSION 0.5.0).
// The wasm-bindgen declarations of the generated package are aggregated into
// types/all.d.ts by the brick build; this file names the stable public shapes.

/**
 * The versioned brick metadata returned by {@link RaaAnimApi.brickInfo}.
 *
 * The returned JavaScript object has exactly two own enumerable string fields, `name`
 * and `version`. `version` is the contract's BRICK_VERSION, independently of the
 * inherited Rust package version: the raa-anim-js crate inherits the workspace's
 * Cargo version, while the brick contract and this npm package release as 0.5.0.
 */
export interface BrickInfo {
  /** The brick name, always the string literal 'raa-anim'. */
  readonly name: 'raa-anim';
  /** The contract's brick version, always the string literal '0.5.0'. */
  readonly version: '0.5.0';
}

/** The selftest cases of the brick. Canonical DET_CASES use `'det-selftest'`. */
export type SelftestCase = 'det-selftest';

/**
 * One deterministic selftest record computed by the Rust simulation.
 *
 * `seed` and `ticks` are u32 values represented as JSON numbers (integers in
 * [0, 4294967295]); `hash` is exactly eight zero-padded lowercase hexadecimal digits
 * without a `0x` prefix.
 */
export interface SelftestResult {
  /** The executed case; `'det-selftest'` for every canonical DET_CASES entry. */
  readonly case: SelftestCase;
  /** The u32 simulation seed, as a JSON number. */
  readonly seed: number;
  /** The u32 tick count, as a JSON number. */
  readonly ticks: number;
  /** The unsigned FNV-1a 32-bit result as eight lowercase hex digits (no prefix). */
  readonly hash: string;
}

/**
 * The animation brick API resolved by `loadRaaAnim()`.
 */
export interface RaaAnimApi {
  /**
   * Returns the brick metadata object exactly `{name: 'raa-anim', version: '0.5.0'}`.
   * Both fields are string literals of the public contract.
   */
  brickInfo(): BrickInfo;

  /**
   * Executes one simulation case and returns exactly `{case, seed, ticks, hash}`.
   *
   * `seed` and `ticks` must be finite integer u32 JSON numbers. The canonical
   * DET_CASES, in order, are `(0, 0)`, `(1, 60)` and `(305419896, 600)`.
   *
   * @throws TypeError with `code: 'api.invalid_input'` for a non-string case or a seed
   *   or tick count outside the u32 range; `code: 'api.unknown_case'` for an unknown
   *   case name.
   */
  selftest(testCase: SelftestCase, seed: number, ticks: number): SelftestResult;
}
