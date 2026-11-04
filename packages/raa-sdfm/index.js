// @etcito/raa-sdfm: sdfm.game-asset v1 files as RAA meshes and materials (meshed by MAMA).
export { createSdfmLoader } from './src/loader.js';
export { mapMaterial } from './src/map-material.js';
export { SdfmError } from './src/errors.js';
export {
  FIELD_FRAME_WGSL, FIELD_NORMAL_WGSL, MATERIAL_FUNCTION_MAX_BYTES, REST_TANGENT_AXIS, checkPartShading, fieldPartMaterialWgsl,
  fieldSurfaceAttributes, restTangent,
} from './src/field-shading.js';
export {
  NORMAL_VOLUME_FORMAT, NORMAL_VOLUME_VERSIONS, VOLUME_FUNCTION_PREFIX, VOLUME_STANDARD_FUNCTION, checkNormalVolume,
  volumeInstanceParams, volumeMaterial, volumeStandardMaterial, volumeSurfaceAttributes, volumeTexture,
} from './src/normal-volume.js';
