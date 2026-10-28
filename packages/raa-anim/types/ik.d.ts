/** Three finite components accepted as an Array or Float32Array at runtime. */
export type IkVec3 = readonly [number, number, number] | number[] | Float32Array;

/** S17 local pose input: ten floats per bone, in rig bone order. */
export type IkPoseInput = number[] | Float32Array | { readonly local: number[] | Float32Array };

/** Corrected S17 pose and rebuilt S16 matrices, each returned as an owned copy. */
export interface IkPoseOutput {
  /** 10 floats per bone: translation xyz, quaternion xyzw, scale xyz. */
  readonly local: Float32Array;
  /** 16 column-major model-matrix floats per bone. */
  readonly model: Float32Array;
  /** 16 column-major skin-matrix floats per bone: model * inverse_bind. */
  readonly skin: Float32Array;
  /** Present for ccd and fabrik. */
  readonly converged?: boolean;
  /** Number of passes performed, present for ccd and fabrik. */
  readonly passes?: number;
  /** Final tip-to-target distance in world units, present for ccd and fabrik. */
  readonly distance?: number;
  /** Pelvis drop in world units, present for footPlacement. */
  readonly drop?: number;
}

/** Ground point and surface normal, both in world coordinates. */
export interface IkGroundHit {
  readonly point: IkVec3;
  readonly normal: IkVec3;
}

/** Additive visual IK API on the animation brick. */
export interface RaaAnimIkApi {
  /** Solve the rig's named S16 twoBone chain toward a world-space target. */
  twoBone(rig: AnimationRig, pose: IkPoseInput, chain: string, target: IkVec3,
    pole?: IkVec3 | null, weight?: number | null): IkPoseOutput;
  /** Solve a named CCD chain. Omitted iteration/tolerance values use rig/default values. */
  ccd(rig: AnimationRig, pose: IkPoseInput, chain: string, target: IkVec3,
    iterations?: number | null, tolerance?: number | null): IkPoseOutput;
  /** Solve a named FABRIK chain. Omitted iteration/tolerance values use rig/default values. */
  fabrik(rig: AnimationRig, pose: IkPoseInput, chain: string, target: IkVec3,
    iterations?: number | null, tolerance?: number | null): IkPoseOutput;
  /** Place named twoBone legs on corresponding world-space ground hits. */
  footPlacement(rig: AnimationRig, pose: IkPoseInput, legs: readonly string[],
    hits: readonly IkGroundHit[]): IkPoseOutput;
  /** Solve a named chain toward a world-space target with a clamped blend weight. */
  handReach(rig: AnimationRig, pose: IkPoseInput, chain: string, target: IkVec3,
    weight?: number | null): IkPoseOutput;
  /** Turn a bone's local axis toward a world-space target, capped in degrees. */
  lookAt(rig: AnimationRig, pose: IkPoseInput, bone: string, target: IkVec3,
    axis: IkVec3, maxDeg?: number | null): IkPoseOutput;
}
