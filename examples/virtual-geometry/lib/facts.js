// Facts of the virtual-geometry page (contract 'vg-facts'). Pure ES module, node-importable.
// Every Engine getter is optional: a method the Engine does not have yields null (settings and
// stats: milestone 2; per-mesh clusters: contract 'm29-engine-methods'; per-instance state: milestone 2).
// A getter that exists is called without a catch: its error propagates to the caller.

// A fresh plain copy of a getter's result; null when the getter is absent or returns nothing.
function copyOf(engine, name, ...args) {
  if (typeof engine[name] !== 'function') return null;
  const value = engine[name](...args);
  return value == null ? null : { ...value };
}

// The fact record of one mesh: clusters from meshClusters for a numbered mesh, cuts and states
// copied from the page's own records (empty when absent).
function meshFact(engine, { name, mesh, cuts = [], states = [] }) {
  const clusters = typeof mesh === 'number' ? copyOf(engine, 'meshClusters', mesh) : null;
  return {
    name,
    mesh,
    clusters,
    cuts: cuts.map((cut) => ({ ...cut })),
    states: states.map((state) => ({ ...state })),
  };
}

// The fact record of one instance: its virtualized state, or null when the Engine has no getter.
function instanceFact(engine, { name, instance }) {
  return { name, instance, virtualGeometry: copyOf(engine, 'instanceVirtualGeometry', instance) };
}

export function collectVgFacts(engine, meshes, instances) {
  return {
    settings: copyOf(engine, 'virtualGeometry'),
    stats: copyOf(engine, 'virtualGeometryStats'),
    meshes: meshes.map((m) => meshFact(engine, m)),
    instances: instances.map((i) => instanceFact(engine, i)),
  };
}
