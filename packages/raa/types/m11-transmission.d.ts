// Hand-written types of the chain milestone 5 additions (transmission, refraction with
// absorption, sorted translucency with per-pixel linked-list OIT). Documented in
// docs/api/m11-transmission.md.
//
// This file is appended to packages/raa/index.d.ts when the package is staged
// (scripts/build-package.sh): the `Engine` interface below merges with the v0 declaration of the
// same name.

/** Settings of the sorted transparency pass `transparent.oit`; fields merge over the current values. */
export interface TransparencySettings {
  /** Fragments kept per pixel by the resolve, integer 4..32. Default 16. */
  maxFragments?: number;
  /** Fragment pool size as a multiple of the pixel count, integer 2..32. Default 8. */
  poolFactor?: number;
}

/** Debugging information of the transparency pass; never a gate. */
export interface TransparencyStats {
  /** Nodes of the fragment pool currently allocated. */
  poolNodes: number;
  /** Frames so far in which the pool overflowed (the pool then grows for the next frame). */
  overflowFrames: number;
}

/** The chain milestone 5 methods, merged into the v0 Engine. */
export interface Engine {
  /**
   * Validates and applies transparency settings, merged over the current values. A refusal
   * changes nothing.
   * @throws RaaError `oit.invalid` (field not an integer or out of range; names the field),
   *   `api.invalid_argument` (argument is not an object, or a field is not a number).
   */
  setTransparency(settings: TransparencySettings): void;

  /** Returns the current pool size and overflow count. Information for debugging, never a gate. */
  transparencyStats(): TransparencyStats;
}
