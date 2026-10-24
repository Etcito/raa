// Hand-written types of the chain milestone 6 additions (animated liquid material
// functions and GPU particle emitters). Documented in docs/api/m13-liquids-particles.md.
//
// This file is appended to packages/raa/index.d.ts when the package is staged
// (scripts/build-package.sh): `Vec3` comes from there, and the `Engine` interface below
// merges with the v0 declaration of the same name.

/** Language of one emitter: its look and motion family. */
export type EmitterKind = 'splash' | 'snow' | 'confetti' | 'sparkles';

/**
 * One particle emitter for {@link Engine.addEmitter} and {@link Engine.setEmitter}.
 * `kind` and `position` are required; every other field overrides the kind's default.
 * Colours are sRGB in [0, 1], lengths metres, angles radians and times seconds.
 */
export interface EmitterDesc {
  /** Effect family. */
  kind: EmitterKind;
  /** Emitter centre in world space. */
  position: Vec3;
  /** Mean spawn direction, non-zero; the simulation normalises it. */
  direction?: Vec3;
  /** Cone half-angle around `direction`, radians in [0, pi]. */
  spread?: number;
  /** Continuous spawn rate in particles per second, >= 0. */
  rate?: number;
  /** Lifetime `[min, max]` in seconds, `0 < min <= max`. */
  lifetime?: [number, number];
  /** Initial speed in m/s: one number or `[min, max]`, `0 <= min <= max`. */
  speed?: number | [number, number];
  /** Particle size in metres: one number or `[min, max]`, `0 < min <= max`. */
  size?: number | [number, number];
  /** Downward acceleration in m/s²; a negative value floats the particles upwards. */
  gravity?: number;
  /** Velocity damping per second, >= 0. */
  drag?: number;
  /** Palette of 1 to 8 sRGB colours, each component in [0, 1]. */
  colors?: Vec3[];
  /** Emissive strength added to the palette colour, >= 0. */
  emissive?: number;
  /** Half extents of the spawn volume around `position`, metres, each >= 0. */
  volume?: Vec3;
  /** Ground plane height in metres; omitted disables the bounce. */
  groundY?: number;
  /** Hard cap on live particles of this emitter, integer 1..65536. */
  maxParticles?: number;
  /** Seed of the deterministic spawn sequence, a non-negative integer. */
  seed?: number;
}

/**
 * One ripple impact of a liquid surface for {@link Engine.setLiquidImpacts}. The four
 * fields are written to the instance params 0..3 of the instance.
 */
export interface LiquidImpact {
  /** Impact X in the liquid mesh's local space, metres. */
  x: number;
  /** Impact Z in the liquid mesh's local space, metres. */
  z: number;
  /** Start time in seconds on the engine clock (`render(timeMs) / 1000`). */
  time: number;
  /** Ripple amplitude; only a finite value > 0 is visible. */
  strength: number;
}

/** The chain milestone 6 methods, merged into the v0 Engine. */
export interface Engine {
  /**
   * Adds a particle emitter and returns its id (dense: the lowest free id is reused). The
   * description is validated before anything changes.
   * @throws RaaError `api.invalid_argument` (malformed JS, unknown kind),
   *   `particles.invalid` (a field is out of range; names it),
   *   `particles.capacity` (more than 64 emitters or more than 262144 particles in total),
   *   `api.feature_missing` (the particle feature is not installed).
   */
  addEmitter(desc: EmitterDesc): number;

  /**
   * Replaces the description of emitter `id` (its particles stay). A refusal changes
   * nothing.
   * @throws RaaError as {@link addEmitter}, plus `particles.unknown_emitter` for an id that
   *   does not exist.
   */
  setEmitter(id: number, desc: EmitterDesc): void;

  /**
   * Removes emitter `id` and its particles; the id becomes free for the next
   * `addEmitter`.
   * @throws RaaError `particles.unknown_emitter`, `api.feature_missing`.
   */
  removeEmitter(id: number): void;

  /**
   * Spawns `count` extra particles from emitter `id` at the next simulation step, on top
   * of the continuous `rate`.
   * @throws RaaError `particles.unknown_emitter`, `api.feature_missing`.
   */
  burst(id: number, count: number): void;

  /**
   * Sets up to 4 ripple impacts of an instance whose material is a liquid built-in
   * (`raa.liquid.water`, `raa.liquid.slime` or `raa.liquid.mud`). Each call replaces all
   * four rows: the unused ones are zeroed. `x` and `z` are in the liquid mesh's local
   * space; `time` is in seconds on the engine clock.
   * @throws RaaError `api.invalid_argument` (not an array, more than 4 items, a field
   *   missing or not finite), `api.unknown_handle` (unknown instance handle).
   */
  setLiquidImpacts(instance: number, impacts: LiquidImpact[]): void;
}
