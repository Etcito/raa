// S23 player tint additions; see docs/api/m20-player-tint.md. build-package.sh appends this
// file to index.d.ts. The `Engine` interface below merges with the v0 declaration.

/**
 * How the instance tint reaches a material. `multiply` (the default): the tint rgb multiplies
 * the base colour after the material. `graph`: the material graph reads `mat_instance_tint`
 * itself (S23 `input.player_color` node) and only the tint alpha multiplies the opacity.
 */
export type TintMode = 'multiply' | 'graph';

/** The m20 methods, merged into the v0 Engine. */
export interface Engine {
  /**
   * Sets the tint mode of a surface material and composes its pipelines again with the
   * matching shader. Materials start in `multiply`. A refusal changes nothing.
   * @throws RaaError `tint.material_range` (unknown material handle), `tint.mode_unknown`
   *   (`mode` is not `multiply` or `graph`).
   */
  setMaterialTintMode(material: number, mode: TintMode): void;

  /**
   * Resets the tint of an instance to [1, 1, 1, 1] and its `mat_instance_tint_set` flag to 0,
   * so a `graph` material shows its preview colour again. Squash, jiggle and parameters are kept.
   * @throws RaaError `api.unknown_handle` (instance), like setInstanceTint.
   */
  clearInstanceTint(instance: number): void;
}
