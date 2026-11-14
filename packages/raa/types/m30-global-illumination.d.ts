/**
 * The global-illumination debug views (S36). `none` draws the frame as shaded; the others replace
 * the shading with one diagnostic: the voxel cubes, the voxel radiance, the distance contours or
 * the nested clipmap levels.
 */
export type GiDebugView = 'none' | 'voxels' | 'radiance' | 'distance' | 'levels';

/**
 * The engine-wide global-illumination settings, returned by {@link Engine.globalIllumination}.
 * Every key is present; the record is a fresh object.
 */
export interface GlobalIlluminationSettings {
  /** Turns the voxel volume and its passes on. Default false. */
  enabled: boolean;
  /** Multiplier of the indirect term, a finite number in [0, 16]. Default 1. */
  intensity: number;
  /**
   * Scale applied over raa-ibl's authored diffuse term while GI is active, a finite number in
   * [0, 1]. Default 0: diffuse light comes from the traced irradiance alone.
   */
  environmentDiffuse: number;
  /** Local lights are shadowed by the GI volume while GI is active. Default true. */
  localLightShadows: boolean;
  /** Where screen traces miss, reflections fall back to a volume trace. Default true. */
  reflections: boolean;
  /** Screen probes trace the previous lit frame before the volume. Default true. */
  screenTraces: boolean;
  /** The debug view of the frame. Default 'none'. */
  debugView: GiDebugView;
}

/**
 * A patch of the global-illumination settings, accepted by {@link Engine.setGlobalIllumination}.
 * Every key is optional; an omitted key (`undefined`) keeps its value. `null`, a wrong type, an
 * unknown key, a non-finite number or an out-of-domain value throws `gi.invalid` with the message
 * `<key>: <reason>` for the first offending key in entry order; nothing changes.
 */
export type GlobalIlluminationPatch = Partial<GlobalIlluminationSettings>;

/**
 * The global-illumination quality, returned by {@link Engine.giQuality}. Defaults 96, 5 and 8.
 * A quality that differs from the current one reallocates the volume and screen resources at the
 * next active frame and rebuilds every level.
 */
export interface GiQuality {
  /** Voxels per axis of every level: 64 or 96. Default 96. */
  resolution: 64 | 96;
  /** Clipmap levels: 3, 4 or 5. Level i has voxel size 0.2 x 2^i m. Default 5. */
  levels: 3 | 4 | 5;
  /** Rays per side of a screen probe: 6 or 8. Default 8. */
  probeRays: 6 | 8;
}

/**
 * A patch of the quality, accepted by {@link Engine.setGiQuality}. Every key is optional and
 * merges as {@link GlobalIlluminationPatch} does. A number must be exactly one of the listed
 * integers; any other value, `null`, a wrong type or an unknown key throws `gi.invalid` with
 * `<key>: <reason>` and nothing changes.
 */
export type GiQualityPatch = Partial<GiQuality>;

/**
 * Why global illumination is not active, the first match of: `disabled` (`enabled` is false),
 * `passDisabled` (`gi.volume` is disabled), `multiView` (the last submitted frame had a view count
 * other than one), `noInstance` (no voxel member). Null exactly when active.
 */
export type GiInactiveReason = 'disabled' | 'passDisabled' | 'multiView' | 'noInstance';

/**
 * The global-illumination state of one instance group, returned by
 * {@link Engine.instanceGlobalIllumination}. The record is fresh.
 */
export interface InstanceGlobalIllumination {
  /** The explicit request, or `null` for the default membership. `false` opts the group out. */
  requested: boolean | null;
  /**
   * Whether the group's scene instances are voxelized in the latest prepared frame. `false`
   * before any frame.
   */
  voxelized: boolean;
  /**
   * `virtual` when the instances are voxelized by driver B, `forward` by driver A, and `null`
   * when they are not voxelized (including before any frame).
   */
  driver: 'forward' | 'virtual' | null;
  /**
   * The first refusal of the membership rank table: `gi.opted_out`, `gi.skinned_instance`,
   * `gi.deformed_instance`, `gi.non_opaque_material` or `gi.no_shadow_material`. `null` when none
   * applies, and before any frame.
   */
  code: string | null;
  /** The refusal message, or `null`. */
  message: string | null;
}

/**
 * Counters of global illumination, returned by {@link Engine.giStats}. The record is fresh. The
 * counters come from CPU state published on complete submission; there is no GPU readback.
 */
export interface GiStats {
  active: boolean;
  /** The first matching {@link GiInactiveReason}; `null` exactly when `active`. */
  inactiveReason: GiInactiveReason | null;
  /** The current quality. */
  quality: GiQuality;
  /** The finest voxel size in metres: 0.2. */
  finestVoxelMetres: number;
  /** Scene instances voxelized by driver A (forward) in the latest prepared frame. */
  voxelizedForwardInstances: number;
  /** Scene instances voxelized by driver B (virtualized) in the latest prepared frame. */
  voxelizedVirtualInstances: number;
  /** A rebuild slice is pending, or a round is outdated or waiting. */
  pendingRebuild: boolean;
  /**
   * Frames since the last rebuild round completed, saturating at 1000000; 0 while a rebuild is
   * pending.
   */
  framesSinceRebuild: number;
  /** Every GPU byte raa-gi owns; 0 when inactive. */
  allocatedBytes: number;
  /** The invocation id of the newest frame whose submission completed, or `null`. */
  statsFrameIndex: number | null;
  /** True while a prepared frame's submission is not yet observed. */
  pending: boolean;
}

/**
 * The global-illumination part of the Engine (S36). Every method throws `api.disposed` on a
 * disposed Engine. The instance methods throw `api.unknown_handle` for a handle that is not a live
 * instance group.
 */
export interface Engine {
  /**
   * Merges a settings patch atomically; see {@link GlobalIlluminationPatch}. Refusals throw
   * `gi.invalid` with `<key>: <reason>`.
   */
  setGlobalIllumination(patch: GlobalIlluminationPatch): void;
  /** The effective settings, a fresh {@link GlobalIlluminationSettings}. */
  globalIllumination(): GlobalIlluminationSettings;
  /** Merges a quality patch atomically; see {@link GiQualityPatch}. Refusals throw `gi.invalid`. */
  setGiQuality(patch: GiQualityPatch): void;
  /** The current quality, a fresh {@link GiQuality}. */
  giQuality(): GiQuality;
  /**
   * Sets the membership request of one live instance group: `true` records an explicit default
   * request, `false` opts the group out of the voxels (it still receives GI), `null` clears the
   * request. Any other value throws `gi.invalid` with `value: expected a boolean or null`.
   */
  setInstanceGlobalIllumination(instance: number, value: boolean | null): void;
  /** The state of one instance group, a fresh {@link InstanceGlobalIllumination}. */
  instanceGlobalIllumination(instance: number): InstanceGlobalIllumination;
  /** The latest published counters, a fresh {@link GiStats}. */
  giStats(): GiStats;
}
