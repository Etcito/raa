// M6 whole-tree billboards: asynchronous eight-direction baking, caller-owned proxy
// meshes and whole-tree LOD groups. scripts/build-package.sh appends this file to
// index.d.ts, and scripts/typecheck-types.mjs stages it the same way; the `Engine`
// interface below merges with the v0 declaration of the same name.
//
// The bake methods (`bakeBillboard`, `removeBillboardBake`, `createBillboardProxy`) are
// declared here for the shared API; their binding belongs to "Expose asynchronous
// billboard baking in RAA" and no consumer calls them before that card. `setBillboardGroup`
// and `billboardGroupState` are bound by the group registry card.

/** One direction tile of a bake: x, y, width, height in texels. */
export type BillboardTileRect = [number, number, number, number];

/**
 * Input of {@link Engine.bakeBillboard}. `instances` is the complete nonempty set of LOD0
 * draw handles; `root` is their common scene-node handle (the object-space frame of the
 * bake).
 */
export interface BillboardBakeDesc {
  /** The complete nonempty set of LOD0 instance handles. */
  instances: number[];
  /** The scene-node handle shared by `instances`; the bake's object-space frame. */
  root: number;
  /** Square tile resolution in texels. */
  resolution: 64 | 128 | 256 | 512;
}

/**
 * A completed bake. The image values are atlas handles; `center` and `size` are
 * object-space metres; `tiles` and `extent` are texels.
 */
export interface BillboardBake {
  /** Bake-store id, accepted by {@link Engine.removeBillboardBake} and
   *  {@link Engine.createBillboardProxy}. */
  id: number;
  /** Atlas handle of the diffuse image. */
  diffuse: number;
  /** Atlas handle of the normal image. */
  normal: number;
  /** Atlas handle of the opacity image. */
  opacity: number;
  /** Object-space centre of the baked bounds, in metres. */
  center: [number, number, number];
  /** Object-space width and height of the baked bounds, in metres. */
  size: [number, number];
  /** Number of baked directions: always 8. */
  directions: 8;
  /** The content rectangle of each direction, in texel rank order. */
  tiles: [
    BillboardTileRect,
    BillboardTileRect,
    BillboardTileRect,
    BillboardTileRect,
    BillboardTileRect,
    BillboardTileRect,
    BillboardTileRect,
    BillboardTileRect,
  ];
  /** The combined atlas image extent in texels, gutters included. */
  extent: [number, number];
}

/** The settings of one whole-tree billboard group. */
export interface BillboardGroupDesc {
  /** The scene-node handle shared by the group's instances. */
  root: number;
  /** The complete nonempty set of LOD0 instance handles; excludes `proxy`. */
  full: number[];
  /** The proxy instance handle. */
  proxy: number;
  /** The bake-store id the proxy draws; unresolved bakes keep LOD0 visible. */
  bake: number;
  /** Positive projected-size threshold in pixels. */
  switchPixels: number;
  /** Whole-number switch bias in -3..3. */
  bias: number;
  /** Automatic switching, forced LOD0 (0) or forced proxy (1). */
  level: 'auto' | 0 | 1;
}

/** The last prepared status of a group; a fresh group is not ready. */
export interface BillboardGroupState {
  /** The selected level: 0 (LOD0) or 1 (proxy). */
  level: 0 | 1;
  /** The selected baked direction, 0..7. */
  direction: number;
  /** Whether the group's bake was resolved and prepared. */
  ready: boolean;
  /** A preparation diagnostic, or null. */
  error: string | null;
}

export interface Engine {
  /**
   * Bakes the eight directions of the whole tree described by `desc`. Completion
   * publishes the three atlas handles atomically and releases the private offscreen
   * attachments and temporary camera/instance data after submitted GPU work completes.
   * @throws RaaError `billboard.invalid` (invalid handles, empty list, nonfinite settings
   *   or duplicate membership), `billboard.bake_failed` (atlas allocation or render
   *   failure, releasing partial resources), `api.disposed`.
   */
  bakeBillboard(desc: BillboardBakeDesc): Promise<BillboardBake>;

  /**
   * Releases the bake `id`. Dependent groups detach to LOD0 before its atlas images are
   * freed. Caller-owned proxy meshes and materials are never removed.
   * @throws RaaError `billboard.invalid` (unknown or stale bake), `api.disposed`.
   */
  removeBillboardBake(id: number): void;

  /**
   * Registers or replaces the whole-tree billboard group `id`, or removes it with a null
   * description. Invalid handles, an empty `full` list, nonfinite settings or membership
   * already taken by another group refuse before any change. Removal is idempotent and
   * never removes the caller's instances. Until per-view preparation is attached the
   * group keeps LOD0 visible.
   * @throws RaaError `billboard.invalid`, `api.disposed`.
   */
  setBillboardGroup(id: string, desc: BillboardGroupDesc | null): void;

  /**
   * The last prepared status of group `id`, or null when no such group is registered.
   * @throws RaaError `api.disposed`.
   */
  billboardGroupState(id: string): BillboardGroupState | null;

  /**
   * Uploads a caller-owned proxy mesh for the ready bake `bake`: one submesh, four
   * vertices and six indices, built from the bake's object-space centre and size. It
   * creates no instance and no material; install it with
   * {@link Engine.addInstance} and dispose the instance before
   * {@link Engine.removeMesh}. Removing the bake does not delete the mesh.
   * @throws RaaError `billboard.invalid` (invalid or stale bake before allocation),
   *   `api.disposed`.
   */
  createBillboardProxy(bake: number): number;
}
