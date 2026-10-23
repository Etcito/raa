// P2 proof helpers (ARCHITECTURE.md, module lookdev-tools). Each check returns
// {measured, threshold, ok} for `check(label, fn)` of proofs/lib/proof.mjs. Node built-ins only.
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { readJson, run } from './proof.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const CARGO_TIMEOUT_MS = 30 * 60 * 1000;
const TAG_RE = /^m\d{2}$/;
export const NEUTRALITY_RE = /royal|fiesta|monster|racer/i;
export const RECORDS_DIR = 'proofs/results/lookdev';
const SCENE_DIRS = { lookdev: 'examples/lookdev/scenes', 'sdfm-compare': 'examples/sdfm-compare/scenes' };

function tail(text, lines = 12) {
  return text.trim().split(/\r?\n/).slice(-lines).join(' / ');
}

function cargo(args) {
  return run(process.env.CARGO || 'cargo', args, { cwd: ROOT, timeout: CARGO_TIMEOUT_MS });
}

function assertTag(tag) {
  if (!TAG_RE.test(tag)) throw new Error(`tag must match mNN, got ${JSON.stringify(tag)}`);
}

// Repo-relative files of `dir` named `<tag>-*<ext>`, sorted.
function tagFiles(dir, tag, ext) {
  const abs = join(ROOT, dir);
  if (!existsSync(abs)) return [];
  return readdirSync(abs)
    .filter((f) => f.startsWith(`${tag}-`) && f.endsWith(ext))
    .sort()
    .map((f) => `${dir}/${f}`);
}

// `cargo test -p a -p b ...` once; measured = summed test results.
export function cargoTest(crates) {
  if (!Array.isArray(crates) || crates.length === 0) throw new Error('cargoTest needs crate names');
  const args = ['test'];
  for (const c of crates) args.push('-p', c);
  const r = cargo(args);
  let passed = 0;
  let failed = 0;
  for (const m of r.stdout.matchAll(/test result: \w+\. (\d+) passed; (\d+) failed/g)) {
    passed += Number(m[1]);
    failed += Number(m[2]);
  }
  const measured = `exit ${r.code}, ${passed} passed, ${failed} failed (${crates.join(',')})`;
  return {
    measured: r.code === 0 ? measured : `${measured}: ${tail(r.stderr + r.stdout)}`,
    threshold: 'exit 0, 0 failed',
    ok: r.code === 0 && failed === 0,
  };
}

// `cargo check --target wasm32-unknown-unknown -p raa-js`.
export function wasmCheck() {
  const r = cargo(['check', '--target', 'wasm32-unknown-unknown', '-p', 'raa-js']);
  return {
    measured: r.code === 0 ? 'exit 0' : `exit ${r.code}: ${tail(r.stderr + r.stdout)}`,
    threshold: 'exit 0',
    ok: r.code === 0,
  };
}

function stripComments(src) {
  return src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
}

// Body of every `interface Engine { ... }` (exported or inside `declare module`), brace-matched.
function engineBodies(src) {
  const bodies = [];
  for (const m of src.matchAll(/\binterface\s+Engine\b[^{]*\{/g)) {
    let depth = 1;
    let i = m.index + m[0].length;
    const start = i;
    for (; i < src.length && depth > 0; i++) {
      if (src[i] === '{') depth++;
      else if (src[i] === '}') depth--;
    }
    if (depth !== 0) throw new Error('unbalanced braces in interface Engine');
    bodies.push(src.slice(start, i - 1));
  }
  return bodies;
}

// Member names at nesting depth 0 of an interface body (methods and properties).
function memberNames(body) {
  let flat = '';
  let depth = 0;
  for (const ch of body) {
    if ('{(['.includes(ch)) depth++;
    if (depth === 0) flat += ch;
    else if (depth === 1 && '{(['.includes(ch)) flat += ch === '(' ? '(' : ' ';
    if ('})]'.includes(ch)) depth = Math.max(0, depth - 1);
  }
  const names = [];
  const re = /(?:^|[;,\n])\s*(?:readonly\s+)?([A-Za-z_$][\w$]*)\s*\??\s*(?:<[^():;]*>)?\s*[(:]/g;
  for (const m of flat.matchAll(re)) names.push(m[1]);
  return names;
}

// Exported names of a .d.ts text: interfaces, types, functions, classes, consts, enums, and
// `export { a, b as c }` lists, plus Engine member names.
export function dtsNames(src) {
  const code = stripComments(src);
  const names = new Set();
  const decl = /^\s*export\s+(?:declare\s+)?(?:interface|type|function|class|const|let|enum)\s+([A-Za-z_$][\w$]*)/gm;
  for (const m of code.matchAll(decl)) names.add(m[1]);
  for (const m of code.matchAll(/^\s*export\s+(?:type\s+)?\{([^}]*)\}/gm)) {
    for (const part of m[1].split(',')) {
      const n = part.trim().split(/\s+as\s+/).pop().trim();
      if (/^[A-Za-z_$][\w$]*$/.test(n) && n !== 'default') names.add(n);
    }
  }
  for (const body of engineBodies(code)) for (const n of memberNames(body)) names.add(n);
  return names;
}

// True when `name` appears in an inline backtick span, alone or as `x.name(...)`, `name(...)`.
function inBackticks(doc, name) {
  const prose = doc.replace(/^```[\s\S]*?^```/gm, '');
  const esc = name.replace(/[$]/g, '\\$');
  return new RegExp('`(?:[A-Za-z_$][\\w$]*\\.)?' + esc + '(?=[`(<?:\\s.\\[])').test(prose);
}

// Every export and Engine member of packages/raa/types/<tag>-*.d.ts is in backticks in
// docs/api/<tag>-*.md; neither side matches the neutrality regex.
export function apiDocCheck(tag) {
  assertTag(tag);
  const threshold = `every ${tag} name in backticks in docs/api/${tag}-*.md, neutrality 0 hits`;
  const typeFiles = tagFiles('packages/raa/types', tag, '.d.ts');
  const docFiles = tagFiles('docs/api', tag, '.md');
  if (typeFiles.length === 0) return { measured: `no packages/raa/types/${tag}-*.d.ts`, threshold, ok: false };
  if (docFiles.length === 0) return { measured: `no docs/api/${tag}-*.md`, threshold, ok: false };
  const read = (f) => readFileSync(join(ROOT, f), 'utf8');
  const names = new Set();
  for (const f of typeFiles) for (const n of dtsNames(read(f))) names.add(n);
  const doc = docFiles.map(read).join('\n');
  const missing = [...names].filter((n) => !inBackticks(doc, n)).sort();
  const neutral = [...typeFiles, ...docFiles].filter((f) => NEUTRALITY_RE.test(read(f)));
  const parts = [`${names.size - missing.length}/${names.size} documented`];
  if (missing.length) parts.push(`missing: ${missing.join(',')}`);
  parts.push(`neutrality hits ${neutral.length}${neutral.length ? ` (${neutral.join(',')})` : ''}`);
  return {
    measured: parts.join(', '),
    threshold,
    ok: names.size > 0 && missing.length === 0 && neutral.length === 0,
  };
}

// Imports the scene modules of `tag` for `page` (S14 default exports), sorted by file name.
export async function loadScenes(tag, { page = 'lookdev' } = {}) {
  assertTag(tag);
  const dir = SCENE_DIRS[page];
  if (!dir) throw new Error(`unknown page ${JSON.stringify(page)} (lookdev|sdfm-compare)`);
  const scenes = [];
  for (const file of tagFiles(dir, tag, '.js')) {
    const mod = await import(pathToFileURL(join(ROOT, file)).href);
    const s = mod.default;
    if (!s || typeof s !== 'object' || typeof s.id !== 'string' || !Array.isArray(s.variants)) {
      throw new Error(`${file}: default export is not an S14 scene (id, variants)`);
    }
    if (s.milestone !== tag || `${s.id}.js` !== file.slice(dir.length + 1)) {
      throw new Error(`${file}: id ${s.id} / milestone ${s.milestone} do not match the file name`);
    }
    scenes.push({ file, scene: s });
  }
  return scenes;
}

// The committed record of one (scene, variant), or null when absent; throws on invalid JSON.
export function readRecord(scene, variant) {
  const path = join(ROOT, RECORDS_DIR, `${scene}--${variant}.json`);
  return existsSync(path) ? readJson(path) : null;
}

// Every declared variant of the tag's scenes has a record nonEmpty and hardware true, and every
// `<f>-on` variant has its `<f>-off` in the same scene.
export async function lookdevCheck(tag, { page = 'lookdev' } = {}) {
  const threshold = 'every variant recorded nonEmpty and hardware, every <f>-on has <f>-off';
  const scenes = await loadScenes(tag, { page });
  if (scenes.length === 0) return { measured: `no ${tag}-*.js scene for page ${page}`, threshold, ok: false };
  let total = 0;
  let good = 0;
  const problems = [];
  for (const { scene } of scenes) {
    const ids = new Set(scene.variants.map((v) => v.id));
    for (const v of scene.variants) {
      total++;
      const key = `${scene.id}--${v.id}`;
      if (typeof v.id !== 'string' || !/^[a-z0-9-]+$/.test(v.id)) {
        problems.push(`${key}: bad variant id`);
        continue;
      }
      if (v.id.endsWith('-on') && !ids.has(`${v.id.slice(0, -3)}-off`)) {
        problems.push(`${key}: no ${v.id.slice(0, -3)}-off`);
      }
      let r;
      try {
        r = readRecord(scene.id, v.id);
      } catch (e) {
        problems.push(`${key}: invalid record (${e.message})`);
        continue;
      }
      if (!r) problems.push(`${key}: no record`);
      else if (r.nonEmpty !== true) problems.push(`${key}: nonEmpty ${r.nonEmpty}`);
      else if (r.hardware !== true) problems.push(`${key}: ${r.hardware === false ? 'software' : 'hardware unstated'}`);
      else good++;
    }
  }
  const parts = [`${good}/${total} variants recorded nonEmpty hardware in ${scenes.length} scene(s)`];
  if (problems.length) parts.push(problems.join('; '));
  return { measured: parts.join(', '), threshold, ok: total > 0 && problems.length === 0 };
}

// Every `judge: true` variant has a committed record whose judge.exit == 0.
export async function judgeCheck(tag, { page = 'lookdev' } = {}) {
  const threshold = 'every judge:true variant: judge.exit == 0 (min score 6)';
  const scenes = await loadScenes(tag, { page });
  const results = [];
  for (const { scene } of scenes) {
    for (const v of scene.variants.filter((x) => x.judge === true)) {
      const key = `${scene.id}--${v.id}`;
      let r = null;
      try {
        r = readRecord(scene.id, v.id);
      } catch {
        r = null;
      }
      const j = r && r.judge && typeof r.judge === 'object' ? r.judge : null;
      const pass = j !== null && j.exit === 0;
      const score = j ? (j.score ?? 'null') : r ? 'not judged' : 'no record';
      results.push({ key, pass, text: `${key}=${score}${pass ? '' : ' FAIL'}` });
    }
  }
  const passed = results.filter((x) => x.pass).length;
  const scores = results.map((x) => x.text).join(', ');
  return {
    measured: `${passed}/${results.length} passed${scores ? ` (${scores})` : ''}`,
    threshold,
    ok: results.length > 0 && passed === results.length,
  };
}
