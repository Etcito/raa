import { toMamaLight } from './mama-bridge.js';
import { SdfmError } from './errors.js';

/** @typedef {import('mama/procgen/output-types').ProcLight} ProcLight */
/** @typedef {import('@etcito/raa').LocalLightDesc} LocalLightDesc */
/** @typedef {{commit: () => void, rollback: () => void}} LightTransaction */
/** @typedef {{prepare: (records: readonly ProcLight[] | undefined) => LightTransaction,
 * setVisible: (visible: boolean) => void, dispose: () => void}} MamaLightOwner */

/**
 * M3_LIGHT_OWNER: one private evaluator-id-to-handle registry for a loaded scene, independent
 * of geometry keys. Mesh loading omits placement; trace loading passes options.placement.
 * Runtime receives this exact same owner. Each snapshot copies t and adds the fixed copied
 * translation once before toMamaLight (M3_LIGHT_MAPPING); direction, range and size are unchanged.
 * Undefined/empty records clear the set; callers skip prepare for an absent runtime snapshot.
 * Preparation maps every record before allocation and stages additions without illumination.
 * Apply external scene changes before commit; rollback releases preparations on external failure.
 * Commit updates surviving handles before removing obsolete ones. Failed updates restore authored
 * descriptors and release additions. Commit/rollback are terminal and idempotent; only one
 * preparation may be pending. Visibility retains authored descriptors; disposal aborts preparation.
 *
 * @param {import('@etcito/raa').Engine} engine
 * @param {[number, number, number]} [placement] finite translation in metres, default [0,0,0]
 * @returns {MamaLightOwner}
 * @throws {SdfmError} sdfm.scene_format for invalid placement/records or concurrent preparation;
 *   sdfm.disposed for mutations after disposal. S17 engine refusals retain their original identity.
 */
export function createMamaLightOwner(engine, placement = [0, 0, 0]) {
  const fail = (message, path) => { throw new SdfmError('sdfm.scene_format', message, path); };
  const translation = (value) => Array.isArray(value) && value.length === 3 && [...value].every(Number.isFinite);
  if (!translation(placement)) fail('placement must be three finite metres', '/placement');
  const offset = placement.slice();
  /** @type {Map<string, {handle: import('@etcito/raa').LightHandle, desc: LocalLightDesc}>} */
  let published = new Map();
  let pending = null;
  let visible = true;
  let disposed = false;
  const live = () => {
    if (disposed) throw new SdfmError('sdfm.disposed', 'light owner is disposed');
  };
  // Explicit resets prevent optional authored fields from surviving a complete snapshot.
  const changes = (desc) => ({
    angleDeg: undefined, innerAngleDeg: undefined, size: undefined, ...desc,
    intensity: visible ? desc.intensity : 0,
  });
  const release = (entries) => {
    for (const { handle } of entries.values()) engine.removeLight(handle);
  };
  return {
    prepare(records) {
      live();
      if (pending) fail('a light transaction is already pending');
      if (records !== undefined && !Array.isArray(records)) fail('lights must be an array', '/lights');
      const desired = new Map();
      for (const [index, light] of (records ?? []).entries()) {
        if (typeof light?.id !== 'string') fail('light id must be a string', `/lights/${index}/id`);
        if (desired.has(light.id)) fail(`duplicate light id '${light.id}'`, `/lights/${index}/id`);
        if (!translation(light.t)) fail('light t must be three finite metres', `/lights/${index}/t`);
        const t = light.t.map((value, axis) => value + offset[axis]);
        desired.set(light.id, toMamaLight({ ...light, t }));
      }
      const staged = new Map();
      try {
        for (const [id, desc] of desired) {
          if (published.has(id)) continue;
          const entry = { handle: engine.addLight(desc), desc };
          staged.set(id, entry);
          engine.updateLight(entry.handle, { intensity: 0 });
        }
      } catch (error) {
        release(staged);
        throw error;
      }
      let done = false;
      const transaction = {
        commit() {
          if (done) return;
          live();
          const updated = [];
          try {
            for (const [id, desc] of desired) {
              const entry = published.get(id) ?? staged.get(id);
              engine.updateLight(entry.handle, changes(desc));
              if (published.has(id)) updated.push(entry);
            }
          } catch (error) {
            for (const entry of updated.reverse()) engine.updateLight(entry.handle, changes(entry.desc));
            transaction.rollback();
            throw error;
          }
          for (const [id, entry] of published) {
            if (!desired.has(id)) engine.removeLight(entry.handle);
          }
          published = new Map([...desired].map(([id, desc]) => [id, {
            handle: (published.get(id) ?? staged.get(id)).handle, desc,
          }]));
          done = true;
          pending = null;
        },
        rollback() {
          if (done) return;
          release(staged);
          done = true;
          pending = null;
        },
      };
      pending = transaction;
      return transaction;
    },
    setVisible(next) {
      live();
      if (typeof next !== 'boolean') fail('visibility must be a boolean');
      if (next === visible) return;
      const updated = [];
      try {
        for (const entry of published.values()) {
          engine.updateLight(entry.handle, { intensity: next ? entry.desc.intensity : 0 });
          updated.push(entry);
        }
      } catch (error) {
        for (const entry of updated.reverse()) {
          engine.updateLight(entry.handle, { intensity: visible ? entry.desc.intensity : 0 });
        }
        throw error;
      }
      visible = next;
    },
    dispose() {
      if (disposed) return;
      pending?.rollback();
      release(published);
      published.clear();
      disposed = true;
    },
  };
}
