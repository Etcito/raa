// @etcito/raa-sdfm refusal codes (S15). Each refusal is an SdfmError whose code is one of these.
export const CODES = Object.freeze([
  'sdfm.format',
  'sdfm.version',
  'sdfm.schema',
  'sdfm.not_finite',
  'sdfm.id',
  'sdfm.cap_document',
  'sdfm.hash_mismatch',
  'sdfm.mesher_refused',
  'sdfm.material_unmappable',
  'sdfm.worker',
  'sdfm.disposed',
]);

/**
 * A refusal of the sdfm adapter. The message is prefixed with `<code>: `, like RaaError.
 */
export class SdfmError extends Error {
  /**
   * @param {string} code one of CODES
   * @param {string} message human message
   * @param {string} [path] JSON pointer into the asset, when the refusal has one
   */
  constructor(code, message, path) {
    super(`${code}: ${message}`);
    this.name = 'SdfmError';
    this.code = code;
    if (path !== undefined) this.path = path;
  }
}
