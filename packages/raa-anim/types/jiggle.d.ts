import type { AnimationRig } from './clips.js';

/**
 * Visual jiggle bones (secondary motion: a jester hat's horns). Visual only: run it after the main pose, skip it
 * under budget or by quality setting, and never feed its output back into a simulation.
 *
 * A bone jiggles when its rig entry carries `jiggle: { maxAngle?, stiffness?, damping?, gravity? }` (degrees 0..180,
 * then 0..1 each; defaults 30, 0.5, 0.3, 0). Its tip (first child's rest offset, else its own) is a spring-damper
 * point pulled toward the animated pose; the bone turns about its head toward it, clamped to maxAngle. Steps are
 * fixed 1/240 s substeps of the real frame time (at most 1/15 s per frame), so the motion does not depend on the
 * frame rate. A handle left out of one `step` call restarts at its animated pose the next time it is stepped.
 */
export class JiggleWorld {
  constructor();
  /** Register one instance of `rig`; the handle stays valid until `remove`. */
  add(rig: AnimationRig): number;
  /** Forget a handle (unknown handles are ignored). */
  remove(handle: number): void;
  /** Bones of the handle's rig (its palette holds 16 floats per bone). */
  boneCount(handle: number): number;
  /** Bones of the handle's rig that jiggle. */
  jiggleCount(handle: number): number;
  /** Animated pose: model matrices in the mount frame, 16 column-major floats per bone; empty restores rest. */
  setPose(handle: number, model: Float32Array): void;
  /** Animated pose as skin matrices (model * inverse bind, the palette drawn without jiggle); empty restores rest. */
  setPoseSkin(handle: number, skin: Float32Array): void;
  /** The handle's next step starts from its animated pose (teleport, respawn, camera cut). */
  reset(handle: number): void;
  /** Live handles. */
  readonly size: number;
  /**
   * One frame of `dt` real seconds for `handles`, in one call. `frames` holds 32 floats per handle: the instance
   * `world` matrix (the placement the renderer applies), then its `mount` matrix (the armature rest frame into the
   * character's posed frame; for a cosmetic mounted on bone B of a character: that character's skin matrix of B).
   * Returns the skin palettes in the instance frame, concatenated in handle order; an armature at rest returns
   * `mount` for each of its bones. Throws `jiggle.handle` for an unknown handle and `jiggle.input` for bad input.
   */
  step(dt: number, handles: Uint32Array, frames: Float32Array): Float32Array;
  free(): void;
}

/** Additive jiggle surface of the brick API. */
export interface RaaAnimJiggleApi {
  /** A new, empty jiggle world (one per view). */
  createJiggleWorld(): JiggleWorld;
}
