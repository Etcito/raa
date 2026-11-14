#!/usr/bin/env node
// Build one example page into dist/<topic> and write its ready BuildReceipt last.
//
//   node scripts/build-page.mjs <topic> --stages <a,b,...> [--shared <repo path>]... [--root <dir>]
//
// The page root holds the delivered entry files, the staged package, common/, lib/,
// scenes/, shared/ and scenes.json; build.json is written only after staging, the import
// rewrites and scenes.json succeeded, so a failed build leaves no receipt.
import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';

const USAGE = 'usage: build-page <topic> --stages <a,b,...> [--shared <repo path>]... [--root <dir>]';
const TOPIC = /^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$/;
const TEST = /\.test\.(?:m?js)$/;
const IMPORT = /(?:from\s*|import\s*\(\s*)['"]([^'"]+)['"]|(?:^|\n)\s*import\s+['"]([^'"]+)['"]/g;
const STALE = /sky-weather\/|web\/common\/|examples\//;
const SHARED = ['capture.js', 'measure.js', 'readiness.js', 'evidence.mjs'];
const MAIN = [['../sky-weather/lib/', './shared/'], ['../../web/common/', './common/']];
const NESTED = [['../../sky-weather/lib/', '../shared/'], ['../../../web/common/', '../common/']];
const COMMON = [['../../examples/sky-weather/lib/', '../shared/']];
const SCRIPT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const refuse = (message) => new Error(`build-page: ${message}`);
const posix = (value) => value.split(path.sep).join('/');

function requireFile(file, label) {
  if (!fs.existsSync(file) || !fs.statSync(file).isFile()) throw refuse(`missing ${label}`);
}

function walk(directory, prefix = '') {
  const names = [];
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const name = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) names.push(...walk(path.join(directory, entry.name), name));
    else if (entry.isFile()) names.push(name);
    else throw refuse(`staged output ${name} is not a regular file`);
  }
  return names;
}

function relativeSpecifier(from, to) {
  const relative = posix(path.relative(from, to));
  return relative.startsWith('.') ? relative : `./${relative}`;
}

function rewrite(file, out, shared) {
  const stagedDir = path.dirname(file.staged);
  const sourceDir = path.dirname(file.source);
  const pairs = file.group === 'main' ? MAIN : file.group === 'common' ? COMMON : NESTED;
  const replacements = [...shared].map(([basename, source]) => ({
    from: relativeSpecifier(sourceDir, source),
    to: relativeSpecifier(stagedDir, path.join(out, 'shared', basename)),
  }));
  const text = fs.readFileSync(file.staged, 'utf8');
  IMPORT.lastIndex = 0;
  const rewritten = text.replace(IMPORT, (match, first, second) => {
    const specifier = first ?? second;
    let next = specifier;
    for (const [from, to] of pairs) {
      if (next.startsWith(from)) next = `${to}${next.slice(from.length)}`;
    }
    const replacement = replacements.find((entry) => next === entry.from);
    if (replacement) next = replacement.to;
    if (next === specifier) return match;
    const at = match.lastIndexOf(specifier);
    return `${match.slice(0, at)}${next}${match.slice(at + specifier.length)}`;
  });
  if (rewritten !== text) fs.writeFileSync(file.staged, rewritten);
  if (file.group === 'common' && TEST.test(file.relative)) return;
  IMPORT.lastIndex = 0;
  for (const match of rewritten.matchAll(IMPORT)) {
    const specifier = match[1] ?? match[2];
    if (STALE.test(specifier) || replacements.some((entry) => entry.from === specifier)) {
      throw refuse(`${file.relative}: staged import ${specifier} still names a source path`);
    }
  }
}

/** @returns {{ topic: string, stages: string[], shared: string[], root: string | null }} */
export function parseArgs(argv) {
  const options = { topic: null, stages: [], shared: [], root: null };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--stages') {
      const value = argv[(index += 1)];
      if (value === undefined) throw refuse(`--stages needs a comma-separated list; ${USAGE}`);
      options.stages = value.split(',');
      if (options.stages.includes('')) throw refuse(`--stages needs nonempty stage names; ${USAGE}`);
    } else if (argument === '--shared') {
      const value = argv[(index += 1)];
      if (value === undefined || value === '') throw refuse(`--shared needs a repository path; ${USAGE}`);
      options.shared.push(value);
    } else if (argument === '--root') {
      const value = argv[(index += 1)];
      if (value === undefined || value === '') throw refuse(`--root needs a directory; ${USAGE}`);
      options.root = value;
    } else if (argument.startsWith('-')) {
      throw refuse(`unknown option ${argument}; ${USAGE}`);
    } else if (options.topic === null) {
      options.topic = argument;
    } else {
      throw refuse(`unexpected argument ${argument}; ${USAGE}`);
    }
  }
  if (options.topic === null) throw refuse(`missing topic; ${USAGE}`);
  if (!TOPIC.test(options.topic)) throw refuse(`invalid topic ${JSON.stringify(options.topic)}; ${USAGE}`);
  if (options.stages.length === 0) throw refuse(`missing --stages; ${USAGE}`);
  return options;
}

async function writeScenes(out, topic, stages) {
  const directory = path.join(out, 'scenes');
  const files = fs.readdirSync(directory).filter((name) => name.endsWith('.js')).sort();
  const scenes = [];
  const ids = new Set();
  for (const name of files) {
    const relative = `scenes/${name}`;
    const url = `${pathToFileURL(path.join(directory, name)).href}?build=${randomUUID()}`;
    const scene = (await import(url))?.default;
    if (!scene || typeof scene !== 'object' || Array.isArray(scene)) {
      throw refuse(`${relative}: no default export object`);
    }
    if (typeof scene.id !== 'string' || !TOPIC.test(scene.id)) {
      throw refuse(`${relative}: invalid scene id ${JSON.stringify(scene.id)}`);
    }
    if (!stages.includes(scene.milestone)) {
      throw refuse(`${relative}: milestone ${JSON.stringify(scene.milestone)} is not one of ${stages.join(',')}`);
    }
    if (ids.has(scene.id)) throw refuse(`${relative}: duplicate scene id ${scene.id}`);
    ids.add(scene.id);
    scenes.push({ id: scene.id, milestone: scene.milestone, file: relative });
  }
  if (scenes.length === 0) throw refuse(`examples/${topic}/scenes has no delivered scene module`);
  scenes.sort((left, right) => (left.id < right.id ? -1 : left.id > right.id ? 1 : 0));
  fs.writeFileSync(path.join(out, 'scenes.json'), `${JSON.stringify(scenes, null, 2)}\n`);
  return scenes;
}

/**
 * Stage the page under `<root>/dist/<topic>`, rewrite its imports, write `scenes.json`
 * and write a ready `build.json` last; resolves to `{ out, scenes, receipt }`.
 * Handoff: `buildPage(options)` in `scripts/build-page.mjs` is called by
 * `page-kit: scripts/build-page.test.mjs` on a temporary root with a stub `captureInputs`.
 */
export async function buildPage(options) {
  const root = path.resolve(options.root ?? SCRIPT_ROOT);
  const { topic, stages } = options;
  const out = path.join(root, 'dist', topic);
  const packageRoot = path.join(root, 'dist/package/raa');
  const pageRoot = path.join(root, 'examples', topic);
  for (const name of ['index.js', 'index.d.ts', 'pkg/raa_js_bg.wasm']) {
    requireFile(path.join(packageRoot, name), `dist/package/raa/${name}`);
  }
  for (const name of ['index.html', 'main.js', 'nmc.gpu.json']) {
    requireFile(path.join(pageRoot, name), `examples/${topic}/${name}`);
  }

  fs.rmSync(out, { recursive: true, force: true });
  fs.mkdirSync(out, { recursive: true });
  const scripts = [];
  const stage = (source, relative, group = null) => {
    const staged = path.join(out, relative);
    fs.mkdirSync(path.dirname(staged), { recursive: true });
    fs.copyFileSync(source, staged);
    if (group) scripts.push({ relative, source, staged, group });
    return staged;
  };
  stage(path.join(pageRoot, 'index.html'), 'index.html');
  stage(path.join(pageRoot, 'main.js'), 'main.js', 'main');
  stage(path.join(pageRoot, 'nmc.gpu.json'), 'nmc.gpu.json');
  fs.cpSync(packageRoot, path.join(out, 'raa'), { recursive: true });
  const stageTree = (sourceRoot, prefix, group) => {
    for (const name of walk(sourceRoot)) stage(path.join(sourceRoot, name), `${prefix}/${name}`, group);
  };
  const commonRoot = path.join(root, 'web/common');
  if (fs.existsSync(commonRoot)) stageTree(commonRoot, 'common', 'common');
  const libRoot = path.join(pageRoot, 'lib');
  if (fs.existsSync(libRoot)) stageTree(libRoot, 'lib', 'nested');
  const sceneRoot = path.join(pageRoot, 'scenes');
  fs.mkdirSync(path.join(out, 'scenes'), { recursive: true });
  if (fs.existsSync(sceneRoot)) {
    for (const name of fs.readdirSync(sceneRoot).filter((entry) => entry.endsWith('.js')).sort()) {
      const source = path.join(sceneRoot, name);
      if (fs.statSync(source).isFile()) stage(source, `scenes/${name}`, 'nested');
    }
  }
  const shared = new Map();
  const addShared = (source) => {
    const basename = path.basename(source);
    if (shared.has(basename)) throw refuse(`two shared files named ${basename}`);
    requireFile(source, posix(path.relative(root, source)));
    shared.set(basename, source);
    stage(source, `shared/${basename}`);
  };
  for (const name of SHARED) addShared(path.join(root, 'examples/sky-weather/lib', name));
  for (const entry of options.shared ?? []) addShared(path.resolve(root, entry));

  for (const file of scripts) {
    if (/\.(?:m?js)$/.test(file.relative)) rewrite(file, out, shared);
  }
  const scenes = await writeScenes(out, topic, stages);
  const evidence = await import(pathToFileURL(path.join(root, 'examples/sky-weather/lib/evidence.mjs')).href);
  const captureInputs = options.captureInputs ?? evidence.captureInputs;
  const preparationId = randomUUID();
  const receipt = {
    buildId: randomUUID(),
    preparationId,
    inputs: captureInputs(root, preparationId, topic),
    builtAt: new Date().toISOString(),
    page: path.resolve(out),
    files: walk(out).sort().filter((name) => name !== 'build.json'),
    status: 'ready',
    error: null,
  };
  evidence.writeJsonRecord(evidence.containedPath(out, 'build.json'), receipt);
  return { out: path.resolve(out), scenes, receipt };
}

async function main(argv) {
  const parsed = parseArgs(argv);
  const root = parsed.root === null ? SCRIPT_ROOT : path.resolve(parsed.root);
  const { out, scenes, receipt } = await buildPage({ ...parsed, root });
  process.stdout.write(`build-page: ${out}: ${scenes.length} scene(s), ${receipt.files.length} staged file(s)\n`);
}

if (process.argv[1] !== undefined && pathToFileURL(process.argv[1]).href === import.meta.url) {
  try {
    await main(process.argv.slice(2));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`${message.startsWith('build-page: ') ? message : `build-page: ${message}`}\n`);
    process.exitCode = 1;
  }
}
