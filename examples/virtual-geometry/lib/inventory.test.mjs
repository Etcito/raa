import test from 'node:test';
import assert from 'node:assert/strict';
import { validateRequiredFacts } from './inventory.js';

// Diagnostic of the 'statistics not published' failure: the thrown error carries the last
// virtualGeometryStats() snapshot the page read, or the literal 'none received'.
const ASSERTION = 'lod-triangles-fall-with-distance';
const row = { id: 'passes-lod-row', requiredFacts: [`assertion:${ASSERTION}`], pair: null };
const timedOut = { name: ASSERTION, passed: false, detail: 'statistics not published within 240 frames' };

function facts(stats) {
  return {
    settings: {},
    stats,
    meshes: [],
    instances: [],
    assertions: [timedOut],
    edits: [],
    passes: [],
    viewSizes: [],
    readiness: { status: 'ready', stage: 'passes', error: null },
  };
}

test('a statistics wait that times out names the last snapshot, or none received', () => {
  const snapshot = { statsFrameIndex: 57, pending: true, drawnClusters: 4096, drawnTriangles: 81920 };
  assert.throws(() => validateRequiredFacts(row, facts(snapshot), [ASSERTION]), error => {
    assert.match(error.message, /required assertion:lod-triangles-fall-with-distance is missing or invalid: statistics not published within 240 frames/);
    assert.match(error.message, /last statistics snapshot: /);
    assert.match(error.message, /"statsFrameIndex":57/);
    assert.match(error.message, /"pending":true/);
    assert.match(error.message, /"drawnClusters":4096/);
    assert.match(error.message, /"drawnTriangles":81920/);
    return true;
  });
  assert.throws(() => validateRequiredFacts(row, facts(null), [ASSERTION]), /last statistics snapshot: none received/);
});
