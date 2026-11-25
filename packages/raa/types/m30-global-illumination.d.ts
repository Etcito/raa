/**
 * The global-illumination debug views (S36). `none` draws the frame as shaded; the others replace
 * the shading with one diagnostic: the voxel cubes, the voxel radiance, the distance contours or
 * the nested clipmap levels; and for the screen irradiance: `probes` (where this frame's screen
 * probes sit, magenta), `hitType` (what the rays hit: red screen, green voxels, blue sky),
 * `history` (green history kept, yellow clamped, red rejected, blue no irradiance yet),
 * `historyLength` (black 0 to white `historyMax` frames), `reservoirs` (ReSTIR GI confidence M,
 * black 0 to white `restirMCap`) and `rawDenoised` (this frame's raw irradiance on the left half,
 * the denoised one on the right).
 */
export type GiDebugView =
  | 'none'
  | 'voxels'
  | 'radiance'
  | 'distance'
  | 'levels'
  | 'probes'
  | 'hitType'
  | 'history'
  | 'historyLength'
  | 'reservoirs'
  | 'rawDenoised';

/**
 * A GI preset of {@link Engine.setGiPreset}: `performance` (6 probe rays, 32 frames of history,
 * clamp 2, 2 a-trous passes, M cap 10), `balanced` (8 rays, 64 frames, clamp 2.5, 3 passes, M cap
 * 20: the defaults) or `quality` (8 rays, 128 frames, clamp 3, 4 passes, M cap 30, full voxel
 * relight).
 */
export type GiPreset = 'performance' | 'balanced' | 'quality';

/**
 * The screen-space GI quality (`screenSpaceQuality`): horizon slices x steps per side of the
 * screen-space trace. `low` 2 slices x 6 steps, `medium` 4 x 8, `high` 6 x 12.
 */
export type GiScreenSpaceQuality = 'low' | 'medium' | 'high';

/**
 * The source of the screen irradiance (`indirect`): `probes`, one SH9 probe per 16-pixel tile
 * integrated per pixel, or `restir`, ReSTIR GI (Ouyang et al. 2021): one secondary sample per
 * pixel traced through the depth pyramid and the voxel volume, resampled over frames and
 * neighbouring pixels, then denoised. Screen-space GI (`screenSpace`) composes over either.
 */
export type GiIndirectMode = 'probes' | 'restir';

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
  /**
   * Screen-space indirect diffuse with a visibility bitmask adds the near-field bounce and contact
   * detail of the 2 m around each pixel on top of the volume and screen probes; what it finds
   * replaces the probe irradiance there, never adds to it. Default false.
   */
  screenSpace: boolean;
  /** The screen-space trace quality, see {@link GiScreenSpaceQuality}. Default 'medium'. */
  screenSpaceQuality: GiScreenSpaceQuality;
  /** The source of the screen irradiance, see {@link GiIndirectMode}. Default 'probes'. */
  indirect: GiIndirectMode;
  /**
   * The most frames the screen irradiance averages, an integer in [1, 256]. Each pixel blends at
   * max(1 / n, 1 / historyMax), n its history length: a still view keeps sharpening until
   * historyMax frames; a disoccluded pixel starts again at n = 1. Default 64.
   */
  historyMax: number;
  /**
   * Width of the history change test in standard deviations of the current frame's noise, finite
   * in [0.5, 10]: a history further than clampGamma sigma from the current light loses length
   * linearly, all of it at twice that (a lighting change is taken at once). Lower follows
   * lighting changes faster, higher keeps more history. Default 2.5.
   */
  clampGamma: number;
  /** Variance-guided a-trous passes over the accumulated irradiance, an integer in [0, 5]. Default 3. */
  spatialPasses: number;
  /** Confidence cap M of the ReSTIR GI reservoirs, finite in [1, 64]. Default 20. */
  restirMCap: number;
  /** Relight every voxel of the relit level each update instead of a checkerboard half. Default false. */
  voxelFullRelight: boolean;
  /**
   * The history length a pixel keeps at most after a lighting change, an integer in [1, 256]:
   * 1 takes the new light whole the next frame. Default 1.
   */
  historyMin: number;
  /**
   * The most updates a voxel's radiance averages (weight 1 / n), an integer in [1, 255]: a still
   * scene's light volume converges over this many updates; a changed light, a changed direct
   * light on the voxel or a scrolled-in cell restarts it. Default 32.
   */
  voxelHistoryMax: number;
  /** Cosine gather rays per voxel update, an integer in [1, 15]; a restarted voxel takes at least 8. Default 3. */
  voxelRays: number;
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
 * counters come from CPU state published on complete submission, except the three screen
 * accumulation figures, read back from the GPU a few frames late over one pixel in 16.
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
  /** At least one clipmap level needs further rebuild work. */
  pendingRebuild: boolean;
  /**
   * Frames with no rebuild pending, counting the completion frame as 1 and saturating at
   * 1000000. Zero while inactive or while a level still needs rebuilding.
   */
  framesSinceRebuild: number;
  /** Every GPU byte raa-gi owns; 0 when inactive. */
  allocatedBytes: number;
  /** The invocation id of the newest frame whose submission completed, or `null`. */
  statsFrameIndex: number | null;
  /** True while a prepared frame's submission is not yet observed. */
  pending: boolean;
  /** Mean history length of the screen irradiance in frames, or `null` before a readback. */
  historyLengthMean: number | null;
  /** Share of pixels whose history was rejected this frame (disocclusion), percent, or `null`. */
  rejectedPercent: number | null;
  /** Share of pixels whose history the neighbourhood clamp moved, percent, or `null`. */
  clampedPercent: number | null;
  /** Real-frame age of the lit HDR available to screen hits, or null if unavailable. */
  litHistoryAge: number | null;
  /** Unique levels with an unfinished slice, newer scene inputs or a different snapped origin. */
  rebuildPendingLevels: number;
  /** The cause of the rebuild recorded this frame, or null when no rebuild was recorded. */
  rebuildReason: 'allocation' | 'sceneChanged' | 'cameraMoved' | 'cameraAndSceneChanged' | null;
  /** Zero-based level recorded this frame, or null for a whole-volume build or no rebuild. */
  rebuildLevel: number | null;
  /** Flood passes still needed on that slice, or null when no slice was recorded. */
  rebuildFloodPassesRemaining: number | null;
  /** Levels relit this frame, regardless of rebuild activity. */
  relitLevels: number;
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
   * Applies a {@link GiPreset}: sets `historyMax`, `clampGamma`, `spatialPasses`, `restirMCap`,
   * `voxelFullRelight` and the probe rays, and leaves every other setting as it is. Another name
   * throws `gi.invalid` with `preset: expected one of performance, balanced, quality`.
   */
  setGiPreset(name: GiPreset): void;
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
  /**
   * Merges a spherical grid patch. The primary-view pass allocates GPU resources on first use.
   * Grid changes discard earlier captures. Debug changes preserve the capture cache.
   */
  setSphericalGi(patch: Partial<SphericalGiSettings>): void;
  /** Effective settings for this engine. The grid is disabled by default. */
  sphericalGi(): SphericalGiSettings;
  /** Live capture counters. Returns null before the first spherical frame is recorded. */
  sphericalGiStats(): SphericalGiStats | null;
  /** Polls or requests a diagnostic GPU texel readback from one ready spherical probe. */
  sphericalGiProbe(): SphericalGiProbe | null;
  /**
   * Registers or removes (`null`/`undefined`) per-vertex colours of a mesh for the GI voxelizer:
   * a Float32Array (linear) or Uint8Array (normalised by 255) of `components` (3 = rgb, 4 =
   * rgba, default 4) values per vertex, matching the mesh's vertex count. Omitted keeps the
   * albedo (base colour x instance tint). Bad buffers throw `gi.invalid`.
   */
  setMeshVertexColors(
    mesh: number,
    colors: Float32Array | Uint8Array | null | undefined,
    components?: 3 | 4,
  ): void;
  /**
   * As {@link Engine.setMeshVertexColors} for virtualized geometry: one colour per vertex of
   * section `ordinal` of the mesh's cluster store. The virtualized-geometry raster draws the
   * section with base colour x these colours (as the forward path draws its vertex colours) and
   * the virtualized GI voxelizer reads them; each installed feature of the two receives them.
   * With neither installed it throws `feature.missing`.
   */
  setMeshClusterVertexColors(
    mesh: number,
    ordinal: number,
    colors: Float32Array | Uint8Array | null | undefined,
    components?: 3 | 4,
  ): void;
}

export type SphericalGiDebug = 'off' | 'boundaries' | 'occupancy' | 'age' | 'light-hash';

export interface SphericalGiSettings {
  enabled: boolean;
  /** Integer from 0 to 6. Each cube face has 2 ** sphereSubdivision cells along each axis. */
  sphereSubdivision: number;
  /** Finite number from 0 to 2. Shell widths grow by a factor of 2 ** exponentialPower. */
  exponentialPower: number;
  /**
   * Inner radial boundary in world units. Must be nonnegative and smaller than zMax.
   * It is also the radius of the centre probe's sphere around the grid anchor.
   */
  clipStart: number;
  /**
   * Finite outer radial boundary in world units, strictly greater than clipStart.
   * The last shell center is zMax minus half that shell's width. Shell boundaries
   * must be distinct in float32.
   */
  zMax: number;
  debug: SphericalGiDebug;
  /** Bit mask for sky (1), sun (2), and other lights (4). Defaults to 7. */
  channelMask: number;
  /** Integer from 16 to 4096: resident free-space probes, the centre probe included. Defaults to 1024. */
  maxProbes: number;
  /**
   * Integer from 1 to 1024: fixed budget of near-shell cube faces captured per frame; far
   * shells get half. Used when captureMs is 0 or the device has no timestamp queries, and as
   * the first burst batch before a GPU timing exists. Defaults to 24.
   */
  captureFaces: number;
  /**
   * Steady-state capture target in GPU milliseconds per frame, from 0 to 50. Non-zero switches
   * to the GPU-time budget: faces are issued from one pool, most visible first. Defaults to 0.
   */
  captureMs: number;
  /** Capture target while visible probes are missing (after open or a camera move), 0 to 50. Defaults to 0 (= captureMs). */
  burstMs: number;
  /**
   * Integer from 0 to 4096: extra pool slots that keep published probes which left the grid,
   * lost a rekey collision or were evicted; a recenter adopts them back without recapture.
   * Cleared by any light, material or geometry change. Defaults to 0.
   */
  probeCache: number;
  /** Integer from 0 to 64: cosine bounce rays toward the sky per captured hit. Defaults to 1. */
  skySamples: number;
  /** Integer from 0 to 64: local lights sampled per captured hit (stratified); 0 = every light. Defaults to 0. */
  lightSamples: number;
}

export interface SphericalGiStats {
  active: boolean;
  cells: number;
  nearCells: number;
  farCells: number;
  readyCells: number;
  readyFaces: number;
  inFlightFaces: number;
  pendingFaces: number;
  originEpoch: number;
  /** Resident slots allocated by the opt-in reflected receiver priority. */
  reflectedCells: number;
  reflectedReadyCells: number;
  lastCellQueries: number;
  /** Cells with observed contributing-light metadata for every face. */
  lightingMetadataReadyCells: number;
  /** Cumulative cells and faces discarded after lighting changes. */
  lightInvalidatedCells: number;
  lightInvalidatedFaces: number;
  /** Bytes in the active spherical capture buffers. */
  gpuBytes: number;
  /** Total faces recorded since the active grid was allocated. */
  capturedFaces: number;
  captureFrames: number;
  /** Resident probes in the view cone (with the centre probe), and how many are published. */
  visibleCells: number;
  visiblePublishedCells: number;
  /** Discovery still has candidate cells to examine. */
  discoveryPending: boolean;
  /** Capture ran the burst target this frame (visible probes missing or undiscovered). */
  burst: boolean;
  /** Faces issued by the latest frame, and the latest measured capture GPU time (0 before a sample). */
  faceBudget: number;
  captureGpuMs: number;
  /** Probes held in the probe cache, cumulative cached probes adopted back, and bytes per pool slot. */
  cachedCells: number;
  adoptedCells: number;
  slotBytes: number;
  /** Revision of the global lighting inputs, encoded as hexadecimal. */
  lightRevision: string;
  /** World-space grid anchor retained across sub-threshold camera movement. */
  origin: [number, number, number];
  lastShell: { inner: number; center: number; outer: number };
}

export interface SphericalGiProbe {
  slot: number;
  frame: number;
  /** Hit counts in cube face order +X, -X, +Y, -Y, +Z, -Z. */
  faceHits: [number, number, number, number, number, number];
  /** Per-face counts of texels with valid world position and normal. */
  faceValidGeometry: SphericalGiProbe['faceHits'];
  /** Per-face counts of misses with cleared position and normal. */
  faceValidMisses: SphericalGiProbe['faceHits'];
  /** RGB sums for sky, sun, and other lights, in that order. */
  channelSums: [[number, number, number], [number, number, number], [number, number, number]];
  /** Hash of this cell's contributing-light state, encoded as hexadecimal. */
  lightHash: string;
  /** Stable IDs of lights observed by this cell, encoded as hexadecimal. */
  contributingLightIds: string[];
}
