// Consumer fixture of the shipped @etcito/raa declarations (npm run typecheck:types).
// Every v0 loadRaa option of packages/raa/index.js must be callable here without a cast.
import { loadRaa } from '@etcito/raa';
import type { RaaModule } from '@etcito/raa';

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

// Negative cases: the declarations must refuse these. (A string for `wasm` is NOT a usable
// negative case: WebAssembly.Module is an empty interface in lib.dom, so a string matches it.)
export async function refused(): Promise<void> {
  // @ts-expect-error a misspelt option name is not a loadRaa option
  await loadRaa({ wasmURL: 'raa_js_bg.wasm' });
  // @ts-expect-error wasmUrl is a string or a URL, never a number
  await loadRaa({ wasmUrl: 42 });
}
