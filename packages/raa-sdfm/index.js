// @etcito/raa-sdfm: sdfm.game-asset v1 files as RAA meshes and materials (meshed by MAMA).
export { createSdfmLoader } from './src/loader.js';
export { mapMaterial } from './src/map-material.js';
export { SdfmError } from './src/errors.js';
export { AO_QUALITY_LEVELS, applyRenderOptions } from './src/render-options.js';
export {
  FIELD_DATA_HEADER, FIELD_DATA_VERSION, FIELD_FLAG_DEBUG, FIELD_FLAG_SINGLE, FIELD_FRAME_WGSL, FIELD_NORMAL_WGSL, FIELD_OPERAND_WORDS,
  FIELD_PROGRAM_VERSION, MATERIAL_FUNCTION_MAX_BYTES, REST_TANGENT_AXIS, checkPartShading, fieldPartShading, fieldShadingCore, fieldSurfaceAttributes,
  restTangent,
} from './src/field-shading.js';
export {
  BAKED_DATA_VERSION, BAKED_DATA_WORDS, BAKED_FUNCTION, BAKED_MATERIAL_WGSL, BAKED_NO_TEXTURE, BAKE_ATLAS_BUDGET, BAKE_DILATE_PASSES,
  BAKE_DILATE_WGSL, BAKE_MIN_SIZE, atlasScale, bakedMaterialWords, createFieldBaker, fieldBakeWgsl, fitBakeSizes, partTangents,
} from './src/field-bake.js';
export { BAKE_BATCH, BAKE_RUN_SHARE, bakeTextureCount, createBakeScheduler } from './src/bake-scheduler.js';
