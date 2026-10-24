// Milestone M7 proof (ARCHITECTURE: proof-m12): @etcito/raa-sdfm, its isolation from MAMA and
// game names, its README API section and the hardware captures of the compare page.
// node proofs/m12-sdfm.mjs [--quick|--full] [--only <regex>]
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { check, run } from './lib/proof.mjs';
import { dtsNames, lookdevCheck, NEUTRALITY_RE } from './lib/p2.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const PKG = 'packages/raa-sdfm';

const read = (rel) => readFileSync(join(ROOT, rel), 'utf8');

// Repo-relative files under `rel` (recursive, node_modules excluded).
function filesUnder(rel) {
  return readdirSync(join(ROOT, rel), { recursive: true })
    .map(String)
    .filter((p) => !p.split('/').includes('node_modules'))
    .map((p) => `${rel}/${p}`)
    .filter((p) => statSync(join(ROOT, p)).isFile());
}

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\$]/g, '\\$&');

// The name is written in an inline backtick span, alone or as `Name.member`, `Name(`, `Name<`.
function inBackticks(doc, name) {
  return new RegExp('`(?:[A-Za-z_$][\\w$]*\\.)?' + escapeRe(name) + '(?=[`(<?:\\s.\\[])').test(doc);
}

await check('unit', () => {
  const r = run('node', ['--test', `${PKG}/test/*.test.js`], { cwd: ROOT, timeout: 300_000 });
  const text = r.stdout + r.stderr;
  const pass = Number((text.match(/^(?:# |ℹ )pass (\d+)$/m) ?? [])[1] ?? NaN);
  const fail = Number((text.match(/^(?:# |ℹ )fail (\d+)$/m) ?? [])[1] ?? NaN);
  const measured = `exit ${r.code}, ${pass} pass, ${fail} fail`;
  return {
    measured,
    threshold: 'exit 0, 0 fail, at least 1 pass',
    ok: r.code === 0 && Number.isFinite(fail) && fail === 0 && pass > 0,
  };
});

await check('isolation', () => {
  const crateFiles = filesUnder('crates');
  const mamaHits = crateFiles.filter((f) => /mama/i.test(read(f)));
  const pkgFiles = filesUnder(PKG);
  const nameHits = pkgFiles.filter((f) => NEUTRALITY_RE.test(read(f)));
  const hits = [...mamaHits, ...nameHits];
  return {
    measured: `crates/ ${crateFiles.length} files, mama hits ${mamaHits.length}; ${PKG} ${pkgFiles.length} files, game-name hits ${nameHits.length}${hits.length ? ` (${hits.join(', ')})` : ''}`,
    threshold: 'no MAMA name under crates/, no game name under packages/raa-sdfm',
    ok: hits.length === 0,
  };
});

await check('api-doc', () => {
  const names = [...dtsNames(read(`${PKG}/index.d.ts`))].sort();
  const readme = read(`${PKG}/README.md`);
  const missing = names.filter((n) => !inBackticks(readme, n));
  const parts = [`${names.length - missing.length}/${names.length} exports in backticks`];
  if (missing.length) parts.push(`missing: ${missing.join(', ')}`);
  return {
    measured: parts.join(', '),
    threshold: `every export of ${PKG}/index.d.ts in backticks in ${PKG}/README.md`,
    ok: names.length > 0 && missing.length === 0,
  };
});

await check('captures', () => lookdevCheck('m12', { page: 'sdfm-compare' }));
