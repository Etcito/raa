// The bake scheduling of the 'baked' field shading (MAMA core/game/render-options.ts shading 'baked'), shared by every
// sdfm game renderer: Royal Fiesta's SDF port and SDF Modeler's Game view run the same code, so the same asset with the
// same options gets the same textures.
//
// A part is prepared when it uploads (prepare: its baked surface material, the ONE shared function BAKED_FUNCTION with a
// data block that names no texture yet, and its bake tangents), and queued when a scene first shows it (queue). One bake
// run works through everything queued since the last run: the sides MAMA's texel density asks for (textureSize) are
// fitted (fitBakeSizes) into BAKE_RUN_SHARE of what RAA's texture atlas has left, so a scene gets the texels and a later
// one (a late costume, the podium) still finds room; parts bake BAKE_BATCH at a time on the GPU (createFieldBaker), their
// textures go to the atlas (Engine.addTexture) and their ids into the part's data block (setMaterialData). Until then the
// renderer draws the part's vertex-colour copy; a failed bake (the atlas full, a GPU error) keeps that copy and is
// reported once through onFailed. release gives a part's textures and data block back.
import { SdfmError } from './errors.js';
import { BAKED_FUNCTION, BAKED_MATERIAL_WGSL, BAKED_NO_TEXTURE, BAKE_ATLAS_BUDGET, atlasScale, bakedMaterialWords, createFieldBaker, fitBakeSizes, partTangents } from './field-bake.js';

/** What one bake run may take of the texture atlas left; the rest stays for the parts a later scene brings. */
export const BAKE_RUN_SHARE = 0.75;
/** Parts baked at once (each holds two sets of three render targets and three readback buffers). */
export const BAKE_BATCH = 6;

/** Textures a part's bake uploads: normal and colour, plus emissive when an operand may glow (a graph or an emissive constant). */
export function bakeTextureCount(shading) {
  return shading.operands.some((o) => o.constant === undefined || o.constant.emissive.some((e) => e > 0)) ? 3 : 2;
}

const codeOf = (e) => (typeof e === 'object' && e !== null && typeof e.code === 'string' ? e.code : '');
const messageOf = (e) => (e instanceof Error ? e.message : String(e));

/**
 * A bake scheduler on one RAA engine.
 * @param {object} engine the RAA Engine (addMaterialFunction, addMaterialData, setMaterialData, removeMaterialData,
 *   addSurfaceMaterial, addTexture, removeTexture)
 * @param {object} options
 * @param {() => (GPUDevice | undefined)} options.device the GPUDevice RAA renders with (read when the first run starts)
 * @param {string} options.field MAMA's FIELD_PROGRAM_WGSL
 * @param {(scale: number) => number} options.textureSize the side MAMA asks for (bakeTextureSize(scale, options))
 * @param {(parts: object[]) => void} [options.onChange] after each batch, with the parts that became done or failed
 * @param {(part: object, why: string) => void} [options.onFailed] once per failed part
 * @param {(stats: object) => void} [options.onRun] after each run
 * @param {number} [options.budget] texels of the atlas the bakes may take (BAKE_ATLAS_BUDGET)
 * @param {number} [options.runShare] BAKE_RUN_SHARE
 * @param {number} [options.batch] BAKE_BATCH
 */
export function createBakeScheduler(engine, options) {
  if (!engine || typeof engine.addMaterialData !== 'function') throw new SdfmError('sdfm.schema', 'createBakeScheduler: pass a RAA engine');
  const { field, textureSize } = options ?? {};
  if (typeof field !== 'string' || field === '') throw new SdfmError('sdfm.schema', 'createBakeScheduler: pass MAMA\'s FIELD_PROGRAM_WGSL as field');
  if (typeof textureSize !== 'function') throw new SdfmError('sdfm.schema', 'createBakeScheduler: pass textureSize(scale)');
  const budget = options.budget ?? BAKE_ATLAS_BUDGET;
  const runShare = options.runShare ?? BAKE_RUN_SHARE;
  const batch = options.batch ?? BAKE_BATCH;
  const queue = [];
  const stats = { parts: 0, failed: 0, ms: 0, texels: 0, bytes: 0, textures: 0, pipelines: 0, resident: { textures: 0, texels: 0 } };
  let used = 0;
  let running = null;
  let baker = null;
  let functionAdded = false;
  let disposed = false;

  const fail = (part, why) => {
    part.state = 'failed';
    part.error = why;
    stats.failed++;
    options.onFailed?.(part, why);
  };

  async function bakeParts(parts) {
    const t0 = performance.now();
    const device = options.device();
    if (device === undefined || device === null) {
      for (const part of parts) fail(part, 'no GPU device to bake with');
      options.onChange?.(parts);
      return;
    }
    baker ??= options.baker ?? createFieldBaker(device);
    const wanted = parts.map((part) => {
      const scale = atlasScale(part.positions, part.uvs, part.indices);
      return { size: textureSize(scale), textures: bakeTextureCount(part.shading), scale };
    });
    const fitted = fitBakeSizes(wanted, Math.max(0, budget - used) * runShare);
    // Largest first, so the atlas's shelves pack whole.
    const order = parts.map((part, i) => ({ part, size: fitted[i] })).sort((a, b) => b.size - a.size);
    let full = false;
    for (let i = 0; i < order.length; i += batch) {
      const group = order.slice(i, i + batch);
      const changed = [];
      const baked = await Promise.all(group.map(async ({ part, size }) => {
        try {
          return { part, r: await baker.bake({ shading: part.shading, materials: part.materials, field, positions: part.positions,
            normals: part.normals, uvs: part.uvs, tangents: part.tangents, indices: part.indices, size }) };
        } catch (e) {
          if (part.state === 'queued') { fail(part, `bake failed: ${messageOf(e)}`); changed.push(part); }
          return null;
        }
      }));
      if (disposed) return;
      for (const entry of baked) {
        if (entry === null) continue;
        const { part, r } = entry;
        // Released while it baked: nothing to upload.
        if (part.state !== 'queued') continue;
        const textures = [];
        try {
          if (full) throw new Error('the texture atlas is full');
          for (const image of [r.normal, r.color, r.emissive]) if (image !== null) textures.push(engine.addTexture({ width: r.size, height: r.size, data: image }));
          engine.setMaterialData(part.data, bakedMaterialWords(textures[0], textures[1], textures[2]));
        } catch (e) {
          for (const id of textures) engine.removeTexture(id);
          if (codeOf(e) === 'texture.atlas_full') full = true;
          fail(part, messageOf(e));
          changed.push(part);
          continue;
        }
        part.textures = textures;
        part.size = r.size;
        part.texels = r.size * r.size * textures.length;
        part.state = 'done';
        used += part.texels;
        stats.parts++;
        stats.textures += textures.length;
        stats.texels += part.texels;
        stats.bytes += part.texels * 4;
        stats.resident.textures += textures.length;
        stats.resident.texels += part.texels;
        changed.push(part);
      }
      if (changed.length > 0) options.onChange?.(changed);
    }
    stats.pipelines = baker.pipelines;
    stats.ms += performance.now() - t0;
  }

  async function run() {
    try {
      // A scene creates its instances in one go: the sizes are fitted over everything it queued.
      await new Promise((resolve) => setTimeout(resolve, 0));
      while (queue.length > 0 && !disposed) {
        await bakeParts(queue.splice(0));
        if (!disposed) options.onRun?.(snapshot());
      }
    } finally {
      running = null;
    }
  }

  const snapshot = () => ({ ...stats, resident: { ...stats.resident }, used });

  return {
    /**
     * A part the 'baked' shading draws: its surface material (the shared baked function, a data block naming no texture
     * yet; `look` = roughness/metallic, the shading's by default) and its bake tangents. The renderer draws its mesh
     * (positions, normals, uvs, `tangents`, indices) with `material`, hidden until `state` is 'done'.
     */
    prepare({ shading, materials, positions, normals, uvs, indices, look = shading?.look, label = '' }) {
      if (disposed) throw new SdfmError('sdfm.disposed', 'the bake scheduler was disposed');
      if (!shading || !Array.isArray(shading.operands)) throw new SdfmError('sdfm.schema', `${label}: a baked part needs its field shading`);
      if (!(uvs instanceof Float32Array) || uvs.length !== positions.length / 3 * 2) throw new SdfmError('sdfm.schema', `${label}: a baked part needs its UV atlas (one uv per vertex)`);
      if (!functionAdded) {
        try { engine.addMaterialFunction(BAKED_FUNCTION, BAKED_MATERIAL_WGSL); } catch (e) { if (codeOf(e) !== 'material.fn_duplicate') throw e; }
        functionAdded = true;
      }
      const tangents = partTangents(positions, normals, uvs, indices);
      const data = engine.addMaterialData(bakedMaterialWords(BAKED_NO_TEXTURE, BAKED_NO_TEXTURE));
      const material = engine.addSurfaceMaterial({ function: BAKED_FUNCTION, data, roughness: look.roughness, metallic: look.metallic });
      return { state: 'idle', label, shading, materials, positions, normals, uvs, indices, tangents, data, material, textures: [], size: 0, texels: 0, error: null };
    },
    /** Queues the idle parts a scene shows; one run bakes them. */
    queue(parts) {
      let added = false;
      for (const part of parts) {
        if (part?.state !== 'idle') continue;
        part.state = 'queued';
        queue.push(part);
        added = true;
      }
      if (added && running === null && !disposed) running = run();
    },
    /** Frees a part: a waiting one leaves the queue, a baked one gives its textures and texels back; its data block goes. */
    release(part) {
      if (part.state === 'released') return;
      if (part.state === 'done') {
        for (const id of part.textures) engine.removeTexture(id);
        used -= part.texels;
        stats.resident.textures -= part.textures.length;
        stats.resident.texels -= part.texels;
      }
      const at = queue.indexOf(part);
      if (at >= 0) queue.splice(at, 1);
      part.state = 'released';
      part.textures = [];
      if (!disposed) engine.removeMaterialData(part.data);
    },
    /** Settles once every queued part is baked (or failed); at once when none waits. */
    done() { return running ?? Promise.resolve(); },
    /** Counts since creation (parts, failed, ms, texels, bytes, textures, pipelines), what stays resident, and `used` texels. */
    stats: snapshot,
    /** Stops the runs (the engine is going): nothing more is uploaded. */
    dispose() { disposed = true; queue.length = 0; },
  };
}
