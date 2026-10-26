// Stage an editor proof using an existing package build and caller-supplied source asset.
// No source asset or dependency bundle is committed.
import { cpSync, mkdirSync, readFileSync, writeFileSync, existsSync, readdirSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { documentHash } from '../packages/raa-sdfm/src/canonical.js';
const [mamaArg, assetArg] = process.argv.slice(2);
if (!mamaArg || !assetArg) throw new Error('usage: node scripts/build-editor-demo.mjs <MAMA checkout> <asset.json>');
const mama=resolve(mamaArg), out=resolve('dist/editor');
if (!existsSync('packages/raa/pkg/raa_js_bg.wasm')) throw new Error('Build the raa-js wasm and run wasm-bindgen into packages/raa/pkg first.');
mkdirSync(out,{recursive:true});
cpSync('examples/editor',out,{recursive:true});
cpSync('packages/raa',join(out,'raa'),{recursive:true});
const additions=readdirSync('packages/raa/types').filter(f=>f.endsWith('.d.ts')).sort()
  .map(f=>readFileSync(join('packages/raa/types',f),'utf8')).join('\n');
writeFileSync(join(out,'raa/index.d.ts'),readFileSync('packages/raa/index.d.ts','utf8')+'\n'+additions);
cpSync('packages/raa-sdfm',join(out,'raa-sdfm'),{recursive:true,filter:p=>!p.includes('/test')});
const entries=['index.js','core/game/mesh-assets.js','core/material-graph/helpers.js','browser/index.js','browser/compute-worker.js'];
const result=spawnSync(join(mama,'node_modules/.bin/esbuild'),[...entries,'--bundle','--format=esm','--platform=browser','--external:node:*','--outbase=.',`--outdir=${join(out,'mama')}`,'--log-level=error'],{cwd:join(mama,'dist'),stdio:'inherit',timeout:55000});
if(result.status!==0)throw new Error('MAMA browser bundle failed');
for(const dir of ['','core/game','core/material-graph','browser']) {
  mkdirSync(join(out,'mama',dir),{recursive:true});
  cpSync(join(mama,'node_modules/manifold-3d/manifold.wasm'),join(out,'mama',dir,'manifold.wasm'));
  const highs=join(mama,'node_modules/highs/build/highs.wasm');
  if(existsSync(highs))cpSync(highs,join(out,'mama',dir,'highs.wasm'));
}
// Normalise only the export hash to this adapter's canonical-document convention.
// The source and the mesh/material document remain unchanged.
const asset=JSON.parse(readFileSync(assetArg,'utf8'));
const hash=await documentHash(asset.document);
if(asset.hash!==hash) console.log('Normalised staged asset hash to the adapter canonical-document convention.');
asset.hash=hash;
writeFileSync(join(out,'asset.json'),JSON.stringify(asset,null,2)+'\n');
console.log('Staged dist/editor; serve with: node web/serve.mjs dist/editor --port 8080');
