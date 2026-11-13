import fs from 'node:fs';
import path from 'node:path';
import childProcess from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';

/**
 * @typedef {Object} InputReceipt
 * @property {'raa'} repository
 * @property {string} revision
 * @property {string} preparationId Builder-owned preparation UUID.
 * @property {Array<{repository:string, revision:string, preparationId:string}>} dependencies
 * @property {Array<{path:string, text:string}>} sources RAA-root-relative input text.
 * @property {{rustc:string, wasmBindgen:string, node:string}} toolchain
 * @property {string} commandsText
 * @property {string} criteriaText
 */

function ensure(condition, message) {
  if (!condition) throw new Error(message);
}

function lexical(value) {
  ensure(typeof value === 'string' && value.trim() !== ''
    && !/[\\\0]/.test(value) && !/^[A-Za-z]:/.test(value)
    && !value.split('/').includes('..'), `unsafe path: ${value}`);
}

function inspect(file) {
  let cursor = path.parse(file).root;
  const parts = file.slice(cursor.length).split('/').filter(Boolean);
  for (const [index, part] of parts.entries()) {
    cursor = path.join(cursor, part);
    let stat;
    try { stat = fs.lstatSync(cursor); }
    catch (error) {
      if (error.code === 'ENOENT') return;
      throw error;
    }
    ensure(!stat.isSymbolicLink(), `symlink path: ${cursor}`);
    ensure(index === parts.length - 1 || stat.isDirectory(), `not a directory: ${cursor}`);
  }
}

/** Resolve a name without creating anything; missing output directories are allowed. */
export function containedPath(root, relativePath) {
  lexical(root);
  lexical(relativePath);
  ensure(!path.posix.isAbsolute(relativePath) && !path.win32.isAbsolute(relativePath)
    && relativePath.split('/').every(part => part !== '' && part !== '.'),
  `unsafe relative path: ${relativePath}`);
  const directory = path.resolve(root);
  inspect(directory);
  if (fs.existsSync(directory)) ensure(fs.statSync(directory).isDirectory(), `not a directory: ${directory}`);
  const file = path.join(directory, relativePath);
  inspect(file);
  return file;
}

function recordPath(file) {
  lexical(file);
  const absolute = path.resolve(file);
  return containedPath(path.dirname(absolute), path.basename(absolute));
}

function readText(file) {
  file = recordPath(file);
  ensure(fs.statSync(file).isFile(), `not a regular file: ${file}`);
  const fd = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW);
  try { return new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(fs.readFileSync(fd)); }
  finally { fs.closeSync(fd); }
}

/** @returns {unknown} */
export function readJsonRecord(file) {
  try { return JSON.parse(readText(file)); }
  catch (cause) { throw new Error(`read JSON ${file}: ${cause.message}`, { cause }); }
}

/**
 * Complete-document replacement only. Callers own record schemas and lifecycle:
 * the builder writes page/build.json, and attempt allocators freeze selection/build
 * copies before publishing stage/current.json. No status or identity is inferred.
 */
export function writeJsonRecord(file, value) {
  file = recordPath(file);
  const text = JSON.stringify(value, null, 2);
  ensure(text !== undefined, `JSON document required: ${file}`);
  recordPath(file);
  const directory = path.dirname(file);
  fs.mkdirSync(directory, { recursive: true });
  const temporary = containedPath(directory, `.${path.basename(file)}.${randomUUID()}.tmp`);
  let fd;
  let created = false;
  try {
    fd = fs.openSync(temporary, 'wx', 0o600);
    created = true;
    fs.writeFileSync(fd, `${text}\n`);
    fs.fsyncSync(fd);
    fs.closeSync(fd);
    fd = undefined;
    recordPath(file);
    recordPath(temporary);
    fs.renameSync(temporary, file);
    created = false;
  } catch (cause) {
    const errors = [cause];
    try { if (fd !== undefined) fs.closeSync(fd); } catch (error) { errors.push(error); }
    try { if (created) fs.unlinkSync(recordPath(temporary)); } catch (error) { errors.push(error); }
    if (errors.length > 1) throw new AggregateError(errors, `replace JSON ${file}: ${cause.message}`, { cause });
    throw new Error(`replace JSON ${file}: ${cause.message}`, { cause });
  }
}

function run(root, program, args) {
  return childProcess.execFileSync(program, args, {
    cwd: root, encoding: 'utf8', timeout: 60000, maxBuffer: 32 * 1024 * 1024,
    stdio: ['ignore', 'pipe', 'pipe'],
  }).trimEnd();
}

function git(root, ...args) { return run(root, 'git', args); }

function revisionAt(root) {
  ensure(git(root, 'rev-parse', '--show-toplevel') === root, `not a repository root: ${root}`);
  const revision = git(root, 'rev-parse', '--verify', 'HEAD');
  ensure(/^[0-9a-f]{40,64}$/.test(revision), `invalid revision: ${root}`);
  return revision;
}

const PAGE_NAME = /^[a-z0-9][a-z0-9-]*$/;

function inputFiles(root, page) {
  // Include untracked delivered sources, never generated package/WASM/dist outputs.
  const names = git(root, 'ls-files', '-z', '--cached', '--others', '--exclude-standard', '--',
    'Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'rustfmt.toml', 'package.json',
    'package-lock.json', 'pnpm-lock.yaml', 'yarn.lock', '.cargo/config', '.cargo/config.toml',
    'crates', 'demos', 'tests', 'packages', 'scripts', 'examples/sky-weather',
    'examples/lookdev/lib', 'web/common', `examples/${page}`).split('\0').filter(Boolean);
  return [...new Set([...names, 'Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', '.nmc/commands.json'])].sort();
}

/** The scene inputs of one page: its inventory, its scenes and its GPU page description. */
function pageCriteria(sources, page) {
  const inventory = `examples/${page}/lib/inventory.js`;
  const scenes = `examples/${page}/scenes/`;
  const gpu = `examples/${page}/nmc.gpu.json`;
  const criteria = sources.filter(source => source.path === inventory
    || source.path.startsWith(scenes) || source.path === gpu);
  ensure(criteria.some(source => source.path === inventory),
    `missing ${page} inventory criteria: ${inventory}`);
  return criteria;
}

/** The page a receipt was captured for, read from its recorded criteria. */
function receiptPage(receipt) {
  let criteria;
  try { criteria = JSON.parse(receipt.criteriaText); }
  catch (cause) { throw new Error('invalid input receipt criteria', { cause }); }
  ensure(Array.isArray(criteria), 'invalid input receipt criteria');
  const pages = new Set();
  for (const source of criteria) {
    const match = typeof source?.path === 'string'
      ? /^examples\/([a-z0-9][a-z0-9-]*)\/lib\/inventory\.js$/.exec(source.path) : null;
    if (match) pages.add(match[1]);
  }
  ensure(pages.size === 1, 'invalid input receipt criteria');
  return [...pages][0];
}

function linkedPreparations(root, sources) {
  const metadata = JSON.parse(run(root, 'cargo', ['metadata', '--no-deps', '--locked', '--format-version', '1']));
  ensure(metadata.workspace_root === root, `wrong Cargo workspace: ${metadata.workspace_root}`);
  const targets = metadata.packages.flatMap(pkg => pkg.dependencies.map(dep => dep.path).filter(Boolean));
  for (const source of sources.filter(source => path.basename(source.path) === 'package.json')) {
    const manifest = JSON.parse(source.text);
    for (const group of ['dependencies', 'devDependencies', 'optionalDependencies']) {
      for (const [name, specifier] of Object.entries(manifest[group] ?? {})) {
        if (typeof specifier !== 'string' || !specifier.startsWith('file:')) continue;
        const target = path.resolve(root, path.dirname(source.path), specifier.slice(5));
        const link = path.join(root, path.dirname(source.path), 'node_modules', name);
        ensure(fs.realpathSync(link) === target, `wrong linked package: ${link}`);
        targets.push(target);
      }
    }
  }
  const dependencies = new Map();
  for (const target of targets) {
    inspect(target);
    if (target === root || target.startsWith(`${root}/`)) continue;
    const owner = git(target, 'rev-parse', '--show-toplevel');
    ensure(path.dirname(owner) === path.dirname(root), `dependency outside isolated siblings: ${owner}`);
    if (dependencies.has(owner)) continue;
    const revision = revisionAt(owner);
    const marker = `.nmc-built-${revision}`;
    const prepared = readText(containedPath(owner, marker)).trim();
    ensure(prepared === revision, `stale preparation: ${owner}/${marker}`);
    ensure(git(owner, 'status', '--porcelain', '--untracked-files=no') === '', `changed dependency: ${owner}`);
    const untracked = git(owner, 'ls-files', '--others', '--exclude-standard', '--',
      'src', 'crates', 'packages', 'scripts', 'Cargo.toml', 'Cargo.lock', 'package.json', 'package-lock.json');
    ensure(untracked === '', `unprepared dependency inputs: ${owner}: ${untracked}`);
    // Existing host marker is the preparation identity, not a consumer-generated UUID.
    dependencies.set(owner, { repository: path.basename(owner), revision, preparationId: marker });
  }
  return [...dependencies.values()].sort((a, b) => a.repository.localeCompare(b.repository));
}

/**
 * Capture before staging; the builder supplies the identity of that preparation.
 * `page` names the example page whose build the receipt records.
 * Handoff: examples/virtual-geometry: passes stage scene module (seven presets) takes the page
 * input collection from this card: `captureInputs` records every source under `examples/<page>`
 * and takes the criteria from that page's `lib/inventory.js`, `scenes/` and `nmc.gpu.json`
 * through `pageCriteria`.
 * @param {string} root
 * @param {string} preparationId
 * @param {string} [page]
 * @returns {InputReceipt}
 */
export function captureInputs(root, preparationId, page = 'sky-weather') {
  ensure(typeof preparationId === 'string'
    && /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(preparationId),
  'missing or invalid preparationId (expected builder preparation UUID)');
  ensure(typeof page === 'string' && PAGE_NAME.test(page), 'invalid page name');
  root = path.dirname(containedPath(root, 'Cargo.toml'));
  const revision = revisionAt(root);
  const sources = inputFiles(root, page).map(name => ({ path: name, text: readText(containedPath(root, name)) }));
  const dependencies = linkedPreparations(root, sources);
  const criteria = pageCriteria(sources, page);
  return {
    repository: 'raa', revision, preparationId, dependencies, sources,
    toolchain: {
      rustc: run(root, process.env.RUSTC || 'rustc', ['--version']),
      wasmBindgen: run(root, 'wasm-bindgen', ['--version']),
      node: process.version,
    },
    commandsText: sources.find(source => source.path === '.nmc/commands.json').text,
    criteriaText: JSON.stringify(criteria),
  };
}

export function assertCurrentInputs(root, receipt) {
  ensure(receipt && typeof receipt === 'object' && receipt.repository === 'raa', 'invalid RAA input receipt');
  const current = captureInputs(root, receipt.preparationId, receiptPage(receipt));
  ensure(Array.isArray(receipt.sources), 'invalid input receipt sources');
  const oldSources = new Map(receipt.sources.map(source => [source?.path, source?.text]));
  ensure(oldSources.size === receipt.sources.length, 'duplicate input receipt sources');
  for (const source of current.sources) {
    ensure(oldSources.get(source.path) === source.text, `changed input: ${source.path}`);
  }
  for (const key of Object.keys(current)) {
    ensure(isDeepStrictEqual(receipt[key], current[key]), `changed inputs: ${key}`);
  }
  ensure(isDeepStrictEqual(Object.keys(receipt).sort(), Object.keys(current).sort()), 'invalid input receipt fields');
}
