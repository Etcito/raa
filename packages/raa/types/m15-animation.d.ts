// Hand-written types of the chain milestone 9 additions (squash and stretch, jiggle springs,
// per-instance tint and parameters, and the batched per-instance record). Documented in
// docs/api/m15-animation.md.
//
// This file is appended to packages/raa/index.d.ts when the package is staged
// (scripts/build-package.sh): `Vec3` comes from there, and the `Engine` interface below
// merges with the v0 declaration of the same name.

/** Four components [r, g, b, a]. `rgb` is sRGB and `a` a linear multiplier, see {@link Engine.setInstanceTint}. */
export type Rgba = [number, number, number, number];

/** Volume-preserving squash and stretch for {@link Engine.setSquash}. */
export interface SquashSettings {
  /** Local axis the instance squashes along, non-zero, normalised by the engine. Default [0, 1, 0]. */
  axis?: Vec3;
  /**
   * Squash amount, finite. A positive value squashes along `axis` and widens the other two axes so
   * the volume is kept; a negative value stretches. The shader clamps it to at least -0.9.
   * The squash is about the instance's local origin: build the mesh with its origin at the contact point.
   */
  amount: number;
}

/**
 * Jiggle spring for {@link Engine.setJiggle}. The spring is driven by the world motion of the
 * instance's node and is stepped by each `render` call; it is deterministic for a given sequence of
 * node transforms and does not move when the frame time step is 0.
 */
export interface JiggleSettings {
  /** Spring stiffness, finite, > 0. Default 120. */
  stiffness?: number;
  /** Damping ratio, finite, >= 0; 1 is critically damped. Default 0.35. */
  damping?: number;
  /** Scale of the spring's response, finite. Default 1. */
  amount?: number;
  /** Height in metres of the moving part, finite, >= 0. Written to the jiggle record as the height. */
  height: number;
}

/** The chain milestone 9 methods, merged into the v0 Engine. */
export interface Engine {
  /**
   * Sets the tint of an instance. `rgb` is sRGB (each component finite, >= 0; [1, 1, 1] is no tint)
   * and multiplies the base colour; `a` is a linear multiplier of the opacity, in [0, 1]. Default
   * [1, 1, 1, 1]. A refusal changes nothing.
   * @throws RaaError `api.unknown_handle` (instance), `api.invalid_argument` (not four finite numbers, or a
   *   component out of range).
   */
  setInstanceTint(instance: number, rgba: Rgba): void;

  /**
   * Sets the 16 free per-instance parameters, read by material functions as `instance_params`
   * (four vec4). Default all 0. A refusal changes nothing.
   * @throws RaaError `api.unknown_handle` (instance), `api.length_mismatch` (not 16 values),
   *   `api.invalid_argument` (a value is not finite).
   */
  setInstanceParams(instance: number, params: Float32Array | number[]): void;

  /**
   * Sets the squash and stretch of an instance, replacing the previous one; an `amount` of 0 is no
   * squash. The axis defaults to [0, 1, 0]. A refusal changes nothing.
   * @throws RaaError `api.unknown_handle` (instance), `api.invalid_argument` (not an object, or
   *   a field is not finite or the axis is not three numbers), `deform.invalid` (axis of zero length).
   */
  setSquash(instance: number, settings: SquashSettings): void;

  /**
   * Starts a jiggle spring on an instance, or removes it with `null`. Unset fields take the defaults
   * of {@link JiggleSettings}. The spring overwrites the jiggle record of the instance at every `render`.
   * After `null` the jiggle record keeps the last value written, until {@link Engine.setInstanceExtras}
   * writes it. A refusal changes nothing.
   * @throws RaaError `api.unknown_handle` (instance), `api.invalid_argument` (not an object or `null`,
   *   or a field is not finite), `deform.invalid` (stiffness <= 0, damping < 0, height < 0).
   */
  setJiggle(instance: number, settings: JiggleSettings | null): void;

  /**
   * Writes the animation record of many instances at once: 32 floats per instance, in the order of
   * `instances`. The layout is listed in docs/api/m15-animation.md. Every handle and value is checked
   * before anything is written. A jiggle record written here is overwritten at the next `render` for
   * an instance that has a {@link Engine.setJiggle} spring.
   * @throws RaaError `api.length_mismatch` (`data.length !== 32 * instances.length`),
   *   `api.unknown_handle` (an instance), `api.invalid_argument` (a value is not finite),
   *   `deform.invalid` (a squash axis of zero length).
   */
  setInstanceExtras(instances: Uint32Array, data: Float32Array): void;
}
