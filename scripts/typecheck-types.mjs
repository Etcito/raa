#!/usr/bin/env node
// Typecheck the shipped @etcito/raa declarations: node scripts/typecheck-types.mjs
// (npm run typecheck:types). Exits non-zero on any TypeScript error.
//
// Stages target/types-check/raa exactly as scripts/build-package.sh stages the package's
// types: index.d.ts = packages/raa/index.d.ts followed by every packages/raa/types/*.d.ts in
// C-locale file-name order except m27-cel.d.ts, which is copied to types/ and imported by
// index.d.ts. Then runs the repository-local TypeScript (devDependency of the root
// package.json, resolved by kong) on tools/types-check/tsconfig.json: the staged index.d.ts
// plus the consumer fixture tools/types-check/consumer.ts. Keep the staging in step with
// build-package.sh. Nothing is built, fetched or written outside target/types-check.
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PKG = path.join(ROOT, 'packages', 'raa');
const STAGE = path.join(ROOT, 'target', 'types-check', 'raa');
const REFERENCED = 'm27-cel.d.ts';

let tsc;
try {
  tsc = createRequire(path.join(ROOT, 'package.json')).resolve('typescript/bin/tsc');
} catch {
  console.error('typecheck-types: typescript is not installed; run `kong rules && kong use` at the repository root');
  process.exit(1);
}

fs.rmSync(STAGE, { recursive: true, force: true });
fs.mkdirSync(path.join(STAGE, 'types'), { recursive: true });
fs.copyFileSync(path.join(PKG, 'package.json'), path.join(STAGE, 'package.json'));
fs.copyFileSync(path.join(PKG, 'types', REFERENCED), path.join(STAGE, 'types', REFERENCED));

// Same as build-package.sh's append_dts: each source ends with a newline before the next.
const parts = [path.join(PKG, 'index.d.ts')];
const typeFiles = fs
  .readdirSync(path.join(PKG, 'types'), { withFileTypes: true })
  .filter((e) => e.isFile() && e.name.endsWith('.d.ts') && e.name !== REFERENCED)
  .map((e) => e.name)
  .sort((a, b) => (Buffer.from(a) < Buffer.from(b) ? -1 : Buffer.from(a) > Buffer.from(b) ? 1 : 0));
for (const name of typeFiles) parts.push(path.join(PKG, 'types', name));
let dts = '';
for (const file of parts) {
  const text = fs.readFileSync(file, 'utf8');
  dts += text.length && !text.endsWith('\n') ? text + '\n' : text;
}
fs.writeFileSync(path.join(STAGE, 'index.d.ts'), dts);
console.log(`== staged target/types-check/raa/index.d.ts from index.d.ts + ${typeFiles.length} types files`);

const project = path.join(ROOT, 'tools', 'types-check', 'tsconfig.json');
console.log('== tsc -p tools/types-check/tsconfig.json');
const r = spawnSync(process.execPath, [tsc, '-p', project, '--pretty', 'false'], { stdio: 'inherit' });
if (r.status !== 0) {
  console.error(`typecheck-types: tsc failed (exit ${r.status ?? r.signal})`);
  process.exit(1);
}
console.log('typecheck-types: ok');
