// Consumer fixture of the shipped @etcito/raa declarations (npm run typecheck:types).
// Every v0 loadRaa option of packages/raa/index.js must be callable here without a cast.
import { loadRaa } from '@etcito/raa';
import type {
  Engine,
  FoliageAssetInfo,
  FoliageDesc,
  FoliageGroupDesc,
  FoliageMode,
  FoliageState,
  FrameTrace,
  RaaModule,
} from '@etcito/raa';

declare const bytes: ArrayBuffer;
declare const view: Uint8Array<ArrayBuffer>;
declare const module: WebAssembly.Module;

export async function consumers(): Promise<RaaModule[]> {
  return [
    await loadRaa(),
    await loadRaa({}),
    await loadRaa({ wasm: bytes }),
    await loadRaa({ wasm: view }),
    await loadRaa({ wasm: module }),
    await loadRaa({ wasmUrl: 'https://example.invalid/raa_js_bg.wasm' }),
    await loadRaa({ wasmUrl: new URL('https://example.invalid/raa_js_bg.wasm') }),
  ];
}

export function renderAfterKey(engine: Engine, event: KeyboardEvent, acceptedMs: number): FrameTrace {
  const inputCutoffMs = performance.now();
  return engine.renderInstrumented(inputCutoffMs, {
    viewId: 'main',
    trialId: 'trial-1',
    presentId: 'present-1',
    inputCutoffMs,
    eventId: 'blocker-step',
    eventMs: event.timeStamp,
    acceptedMs,
  });
}

// Negative cases: the declarations must refuse these. (A string for `wasm` is NOT a usable
// negative case: WebAssembly.Module is an empty interface in lib.dom, so a string matches it.)
export async function refused(): Promise<void> {
  // @ts-expect-error a misspelt option name is not a loadRaa option
  await loadRaa({ wasmURL: 'raa_js_bg.wasm' });
  // @ts-expect-error wasmUrl is a string or a URL, never a number
  await loadRaa({ wasmUrl: 42 });
}

// M7 foliage Engine API (packages/raa/types/m7-foliage.d.ts). Valid descriptors must type-check.
const foliageDesc: FoliageDesc = {
  instances: [1, 2, 3],
  foliageInstances: [2],
  root: 7,
  generation: 1,
  mode: 'stochastic',
  resolution: 32,
  triangles: 4096,
  seed: 0,
  views: 96,
  iterations: 64,
  silhouetteWeight: 1,
  normalWeight: 1,
  edgeWeight: 0.01,
  translucency: 0.35,
  cutoff: 0.5,
  material: 11,
};

const foliageGroup: FoliageGroupDesc = {
  asset: 4,
  full: [1, 2, 3],
  root: 7,
  switchPixels: 64,
  level: 'auto',
};

export async function foliageConsumers(engine: Engine): Promise<void> {
  const info: FoliageAssetInfo = await engine.prepareFoliage(foliageDesc);
  const id: number = info.id;
  const generation: number = info.generation;
  const mode: FoliageMode = info.mode;
  const volumeResolution: 16 | 32 = info.volumeResolution;
  engine.setFoliageGroup('tree', foliageGroup);
  engine.setFoliageGroup('tree', { ...foliageGroup, level: 1 });
  engine.setFoliageGroup('tree', null);
  engine.removeFoliage(id);
  const state: FoliageState | null = engine.foliageGroupState('tree');
  const phase: 'pending' | 'ready' | 'failed' | undefined = state?.phase;
  void [generation, mode, volumeResolution, phase];
}

export function foliageRefused(engine: Engine): void {
  // @ts-expect-error resolution is 16 or 32 only
  void engine.prepareFoliage({ ...foliageDesc, resolution: 24 });
  // @ts-expect-error mode is one of the six published modes
  void engine.prepareFoliage({ ...foliageDesc, mode: 'billboard' });
  // @ts-expect-error instances is a number array, never a string
  void engine.prepareFoliage({ ...foliageDesc, instances: 'all' });
  // @ts-expect-error level is 'auto', 0 or 1
  engine.setFoliageGroup('tree', { ...foliageGroup, level: 2 });
  // @ts-expect-error foliageGroupState takes a string id
  engine.foliageGroupState(5);
  // @ts-expect-error removeFoliage takes a numeric handle
  engine.removeFoliage('4');
  // @ts-expect-error a descriptor lacks the required material lease
  void engine.prepareFoliage({ ...foliageDesc, material: undefined });
}
