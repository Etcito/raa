/** Number of coarser Paramesh meshes made by default. */
export const LOD_LEVELS_DEFAULT = 2;

/**
 * Copy a canonical document with Paramesh quality halved `level` times. Missing settings
 * use MAMA's Paramesh defaults (16 subdivisions, 96 field samples). Geometry is still
 * produced by MAMA; this only changes its meshing settings.
 */
export function lodDocument(document, level) {
  if (!Number.isInteger(level) || level < 0) throw new TypeError('lodDocument: level must be a non-negative integer');
  const doc = structuredClone(document);
  doc.settings ??= {};
  doc.settings.paramesh ??= {};
  const quality = doc.settings.paramesh;
  const divisor = 2 ** level;
  quality.subdivisions = Math.max(1, Math.floor((quality.subdivisions ?? 16) / divisor));
  quality.fieldResolution = Math.max(16, Math.floor((quality.fieldResolution ?? 96) / divisor));
  return doc;
}

/**
 * Measure each level against the full mesh in Rust, then register the chain in one call.
 * `full` and `levels` are uploaded handles; the descriptions supply the geometry for
 * meshLodError. Engines without the LOD API return []; engine refusals propagate.
 */
export function attachLods(engine, full, fullDesc, levels, levelDescs) {
  if (typeof engine.setMeshLods !== 'function' || typeof engine.meshLodError !== 'function' || levels.length === 0) return [];
  const errors = levelDescs.map((desc) => engine.meshLodError(fullDesc, desc));
  engine.setMeshLods(full, levels.map((mesh, i) => ({ mesh, error: errors[i] })));
  return errors;
}
