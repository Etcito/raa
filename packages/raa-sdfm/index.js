// @etcito/raa-sdfm: sdfm.game-asset v1 files as RAA meshes and materials (meshed by MAMA).
export { createSdfmLoader } from './src/loader.js';
export { mapMaterial } from './src/map-material.js';
export { SdfmError } from './src/errors.js';
export { AO_QUALITY_LEVELS, applyRenderOptions } from './src/render-options.js';
export {
  FIELD_DATA_HEADER, FIELD_DATA_VERSION, FIELD_FLAG_DEBUG, FIELD_FLAG_SINGLE, FIELD_FRAME_WGSL, FIELD_NORMAL_WGSL, FIELD_OPERAND_WORDS,
  FIELD_PROGRAM_VERSION, MATERIAL_FUNCTION_MAX_BYTES, REST_TANGENT_AXIS, checkPartShading, fieldPartShading, fieldSurfaceAttributes, restTangent,
} from './src/field-shading.js';
