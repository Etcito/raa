// MAMA browser-host viewport of the compare page (left pane).
// MAMA 0.1.0 @ 864ebc3 has no browser viewport host, so createMamaView refuses until it exists.
// No MAMA call is made here: a call would have to be listed in packages/raa-sdfm/README.md (## MAMA calls).
import { SdfmError } from '../../../packages/raa-sdfm/src/errors.js';

export const MAMA_HOST_UNAVAILABLE = 'MAMA_HOST_UNAVAILABLE';
export const MAMA_CHECKED_COMMIT = '864ebc3';

/**
 * Intended shape once MAMA ships a viewport host:
 * `{ load(asset), setCamera({ position, target, fovYDeg }), render(), probeNonEmpty(), dispose() }`.
 * @param {HTMLCanvasElement} canvas never touched while the host is unavailable
 * @param {{ mamaUrl: string }} opts never read while the host is unavailable
 * @returns {Promise<never>}
 */
export async function createMamaView(canvas, opts) {
  throw new SdfmError(
    MAMA_HOST_UNAVAILABLE,
    'SDF Modeler viewport unavailable: MAMA (checked against ' + MAMA_CHECKED_COMMIT + ') exports no browser ' +
      'viewport host. Missing: canvas mount, document load, camera, render-one-frame.',
  );
}
