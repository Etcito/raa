import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createMamaView, MAMA_HOST_UNAVAILABLE } from '../lib/mama-view.js';

const untouchable = (label) =>
  new Proxy({}, {
    get(_t, key) { throw new Error(`${label} touched: get ${String(key)}`); },
    set(_t, key) { throw new Error(`${label} touched: set ${String(key)}`); },
    has(_t, key) { throw new Error(`${label} touched: has ${String(key)}`); },
  });

test('createMamaView refuses with MAMA_HOST_UNAVAILABLE and touches neither canvas nor options', async () => {
  await assert.rejects(
    createMamaView(untouchable('canvas'), untouchable('opts')),
    (err) => {
      assert.equal(err.code, MAMA_HOST_UNAVAILABLE);
      assert.equal(err.code, 'MAMA_HOST_UNAVAILABLE');
      assert.match(err.message, /SDF Modeler viewport unavailable/);
      for (const piece of ['canvas mount', 'document load', 'camera', 'render-one-frame', '864ebc3']) {
        assert.ok(err.message.includes(piece), `message names ${piece}`);
      }
      return true;
    },
  );
});
