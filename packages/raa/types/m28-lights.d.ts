/** The four supported local-light kinds. */
export type LocalLightKind = 'point' | 'spot' | 'directional' | 'area';

/** Stable positive safe-integer handle returned by {@link Engine.addLight}. */
export type LightHandle = number;

/** Description accepted by {@link Engine.addLight}; omitted values use S17 defaults. */
export interface LocalLightDesc {
  kind?: LocalLightKind;
  /** World position in metres, or local position while attached. Default [0, 0, 0]. */
  position?: [number, number, number];
  /** Nonzero travel direction; normalized on acceptance. Default [0, 0, -1]. */
  direction?: [number, number, number];
  /** sRGB colour in [0, 1], converted once to linear RGB. Default [1, 1, 1]. */
  color?: [number, number, number];
  /** Nonnegative intensity. Default 2. */
  intensity?: number;
  /** Positive range in metres. Default 6. */
  range?: number;
  /** Full spot-cone angle in degrees, [1, 179]. Default 35. */
  angleDeg?: number;
  /** Optional full inner spot-cone angle in degrees, [1, 179] and <= angleDeg. */
  innerAngleDeg?: number;
  /** Positive full width and height in metres. Default [1, 1]. */
  size?: [number, number];
  /** Nonnegative authoring metadata; does not control shadows. Default 0. */
  radius?: number;
  /** Authoring metadata; local-light shadows are not enabled. Default false. */
  shadows?: boolean;
}

/** Partial changes merged atomically onto an existing light. */
export type LocalLightChanges = Partial<LocalLightDesc>;

/** Current light counts and the latest completed tile-overflow readback. */
export interface LightStats {
  activeLights: number;
  attachedLights: number;
  tileColumns: number;
  tileRows: number;
  tileCapacity: 127;
  /** Excluded light/tile references, or null before a completed result exists. */
  droppedLightReferences: number | null;
  /** Invocation frame described by the completed result, or null before one exists. */
  statsFrameIndex: number | null;
  /** True while overflow readback is pending. */
  pending: boolean;
}

export interface Engine {
  /** Adds a local light; capacity is 256. The sun remains configured through setLight. */
  addLight(desc: LocalLightDesc): LightHandle;
  /** Merges and validates changes before mutation; explicit undefined resets a field. */
  updateLight(handle: LightHandle, changes: LocalLightChanges): void;
  /** Removes a light; its handle is never reused. */
  removeLight(handle: LightHandle): void;
  /** Attaches to an instance-group handle returned by addInstance/addMeshInstance. */
  attachLight(
    handle: LightHandle,
    instanceId: number,
    localOffset: [number, number, number],
  ): void;
  /** Detaches while preserving the latest world position and direction. */
  detachLight(handle: LightHandle): void;
  /** Returns synchronous counts and the latest fully completed asynchronous stats. */
  lightStats(): LightStats;
}
