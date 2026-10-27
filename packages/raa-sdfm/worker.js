// Module worker of @etcito/raa-sdfm: meshes sdfm documents with MAMA (through src/mama-bridge.js).
// Protocol (ARCHITECTURE raa-sdfm-pkg):
//   in:  { type: 'init', mamaUrl }            loads MAMA once; later inits are ignored
//        { type: 'mesh', id, document, mesher } waits for init, then meshes
//   out: { type: 'meshed', id, groups, materials } (the bridge output; typed arrays transferred)
//        { type: 'error', id, code, message, path? } (an SdfmError; any other exception -> sdfm.worker)
import { importMama, meshDocument } from './src/mama-bridge.js';
import { SdfmError } from './src/errors.js';

/**
 * Loads MAMA in a worker: the files under MAMA's `dist/` URL `mamaUrl` (bare specifiers when
 * absent). `mama/browser` is not imported: inside the worker MAMA meshes on its inline pool.
 *
 * @param {string | URL | undefined} mamaUrl
 * @returns {Promise<object>} the MAMA object of importMama
 */
export function loadMama(mamaUrl) {
  return importMama(mamaUrl ?? undefined, { browser: false });
}

/**
 * The message handler of the worker, without the worker scope (node tests drive it directly).
 *
 * @param {{ post: (message: object, transfer: ArrayBuffer[]) => void,
 *   loadMama?: (mamaUrl: string | URL | undefined) => Promise<object> }} options
 * @returns {(message: unknown) => Promise<void>} resolves once the message is answered
 */
export function createWorkerHandler({ post, loadMama: load = loadMama }) {
  let initStarted = false;
  let markInit;
  const initRequested = new Promise((resolve) => {
    markInit = resolve;
  });
  // Mesh requests that arrive before init (or while it runs) wait on this promise.
  const ready = initRequested.then((mamaUrl) => load(mamaUrl));
  ready.catch(() => {}); // a failed init is reported per mesh request

  return async function onMessage(message) {
    const type = message && typeof message === 'object' ? message.type : undefined;
    if (type === 'init') {
      if (!initStarted) {
        initStarted = true;
        markInit(message.mamaUrl);
      }
      return;
    }
    if (type !== 'mesh') return;
    const { id, document } = message;
    try {
      const mama = await ready;
      // `mesher` is part of the protocol; the bridge's meshDocument has no input for it yet.
      const result = await meshDocument(mama, document, { pool: undefined });
      post({ type: 'meshed', id, groups: result.groups, materials: result.materials }, transferList(result));
    } catch (e) {
      post(errorMessage(id, e), []);
    }
  };
}

function errorMessage(id, e) {
  if (e instanceof SdfmError) {
    const prefix = `${e.code}: `;
    const message = e.message.startsWith(prefix) ? e.message.slice(prefix.length) : e.message;
    return { type: 'error', id, code: e.code, message, ...(e.path !== undefined ? { path: e.path } : {}) };
  }
  const message = e instanceof Error ? e.message : String(e);
  return { type: 'error', id, code: 'sdfm.worker', message: `worker failed: ${message}` };
}

/** The distinct ArrayBuffers of every typed array of a bridge result. */
function transferList(result) {
  const buffers = new Set();
  const add = (a) => {
    if (ArrayBuffer.isView(a) && a.buffer instanceof ArrayBuffer) buffers.add(a.buffer);
  };
  for (const group of result.groups) {
    for (const part of group.parts) {
      add(part.frame);
      for (const mesh of part.meshes) {
        add(mesh.positions);
        add(mesh.normals);
        add(mesh.uvs);
        add(mesh.indices);
        add(mesh.edges);
      }
    }
  }
  return [...buffers];
}

// Installed only inside a worker scope; importing this module elsewhere has no side effect.
if (typeof WorkerGlobalScope !== 'undefined' && globalThis instanceof WorkerGlobalScope) {
  const handle = createWorkerHandler({ post: (message, transfer) => globalThis.postMessage(message, transfer) });
  globalThis.addEventListener('message', (event) => {
    handle(event.data);
  });
}
