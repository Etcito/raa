// Pool of meshing workers (../worker.js protocol): round-robin dispatch, results keyed by
// request id, a crashed worker is replaced and its pending requests reject sdfm.worker.
import { SdfmError } from './errors.js';

/** min(4, navigator.hardwareConcurrency || 2). */
export function defaultPoolSize() {
  return Math.min(4, globalThis.navigator?.hardwareConcurrency || 2);
}

function defaultCreateWorker(url) {
  return new Worker(url, { type: 'module' });
}

/**
 * @param {{ size?: number, workerUrl: string | URL, mamaUrl?: string | URL,
 *   createWorker?: (url: string | URL) => Worker }} options
 * @returns {{ size: number, mesh(document: unknown, mesher: unknown): Promise<{ groups: object[], materials: object[] }>,
 *   terminate(): void }}
 */
export function createPool({ size, workerUrl, mamaUrl, createWorker = defaultCreateWorker }) {
  const count = size ?? defaultPoolSize();
  if (!Number.isInteger(count) || count < 1) {
    throw new RangeError(`worker pool size must be a positive integer, got ${count}`);
  }
  const init = { type: 'init', mamaUrl: mamaUrl === undefined ? undefined : String(mamaUrl) };
  const slots = [];
  let nextId = 1;
  let cursor = 0;
  let disposed = false;

  function spawn(index) {
    const slot = { worker: createWorker(workerUrl), pending: new Map() };
    const onMessage = (event) => {
      if (disposed || slots[index] !== slot) return;
      const data = event.data;
      const request = data ? slot.pending.get(data.id) : undefined;
      if (!request) return;
      slot.pending.delete(data.id);
      if (data.type === 'meshed') {
        request.resolve({ groups: data.groups, materials: data.materials });
      } else if (data.type === 'error') {
        request.reject(new SdfmError(data.code || 'sdfm.worker', data.message || 'worker error', data.path));
      } else {
        request.reject(new SdfmError('sdfm.worker', `unknown worker reply type '${data.type}'`));
      }
    };
    const onCrash = (event) => {
      if (disposed || slots[index] !== slot) return;
      if (typeof event?.preventDefault === 'function') event.preventDefault();
      const what = event?.type === 'messageerror' ? 'a reply could not be deserialised' : 'the worker crashed';
      const detail = event?.message ? `: ${event.message}` : '';
      fail(slot, new SdfmError('sdfm.worker', `${what}${detail}`));
      slot.worker.terminate();
      spawn(index);
    };
    slot.worker.addEventListener('message', onMessage);
    slot.worker.addEventListener('error', onCrash);
    slot.worker.addEventListener('messageerror', onCrash);
    slots[index] = slot;
    slot.worker.postMessage(init);
  }

  function fail(slot, error) {
    const requests = [...slot.pending.values()];
    slot.pending.clear();
    for (const request of requests) request.reject(error);
  }

  for (let i = 0; i < count; i += 1) spawn(i);

  return {
    size: count,
    mesh(document, mesher) {
      if (disposed) return Promise.reject(new SdfmError('sdfm.disposed', 'the worker pool was terminated'));
      const slot = slots[cursor];
      cursor = (cursor + 1) % count;
      const id = nextId;
      nextId += 1;
      return new Promise((resolve, reject) => {
        slot.pending.set(id, { resolve, reject });
        try {
          slot.worker.postMessage({ type: 'mesh', id, document, mesher });
        } catch (e) {
          slot.pending.delete(id);
          reject(new SdfmError('sdfm.worker', `cannot send the document to a worker: ${e instanceof Error ? e.message : String(e)}`));
        }
      });
    },
    terminate() {
      if (disposed) return;
      disposed = true;
      const error = new SdfmError('sdfm.disposed', 'the worker pool was terminated');
      for (const slot of slots) {
        fail(slot, error);
        slot.worker.terminate();
      }
    },
  };
}
