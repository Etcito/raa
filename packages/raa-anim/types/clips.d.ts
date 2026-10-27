/** Opaque wasm rig. Construct with loadRig; free once no longer needed. */
export interface AnimationRig { free(): void; }
/** Opaque wasm clip retaining its rig. Construct with loadClip. */
export interface AnimationClip { free(): void; }

export interface ClipSample {
  /** Bone order, 10 floats per bone: translation xyz, quaternion xyzw, scale xyz. */
  readonly local: Float32Array;
  /** Bone order, 16 column-major floats per model matrix. */
  readonly model: Float32Array;
  /** Bone order, 16 column-major floats per skin matrix (model * inverse bind). */
  readonly skin: Float32Array;
  /** Three floats: stateless selected-axis translation delta from tick - 1. */
  readonly rootDelta: Float32Array;
}

/**
 * Load strict S16 raa.rig v1 JSON using raa-anim validation.
 * MAX_BONES = 256 inclusive; 257 or more bones throws rig.cap_bones,
 * never truncates. All other S16 validation rules also apply.
 * Refusals are Error objects with code, path (JSON pointer), and message.
 */
export function loadRig(json: string): AnimationRig;
/**
 * Load strict S17 raa.clip v2 JSON against this rig, retaining the rig internally.
 * Inclusive caps: 256 tracks, 4096 keys/track, 1024 events; excess throws clip.cap
 * without truncation. No additional lengthTicks cap beyond S17 numeric rules.
 */
export function loadClip(rig: AnimationRig, json: string): AnimationClip;
/**
 * Sample a finite integer tick in [0, 4294967295]; otherwise throws TypeError
 * with code api.invalid_input. Returns fresh arrays independent of wasm memory.
 * Sampling is stateless and uses deterministic f32 arithmetic in raa-anim.
 * Extracted xz/xyz root axes are restored to rig rest values in local pose;
 * unselected axes retain sampled values. Delta is zero at tick 0 and for
 * zero-duration clips. Loop poses wrap, while root deltas remain continuous:
 * each seam at length L returns P(L)-P(L-1). Non-looping root deltas use
 * clamped adjacent ticks and are zero after completion. See anim-clips.md.
 */
export function sampleClip(clip: AnimationClip, tick: number): ClipSample;

/** Additive surface to intersect with the brick's RaaAnimApi. */
export interface RaaAnimClipsApi {
  loadRig: typeof loadRig;
  loadClip: typeof loadClip;
  sampleClip: typeof sampleClip;
  selftest(testCase: 'clip-sample', seed: number, ticks: number): {
    case: 'clip-sample'; seed: number; ticks: number; hash: string;
  };
}
