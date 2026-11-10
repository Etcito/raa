#!/usr/bin/env node
// P2 sky-weather page protocol: verify the staged build, select the cumulative
// inventory and allocate immutable, non-destructive render attempts.
import fs from 'node:fs';
import { publishGpuPage, runGpuPageOp } from './gpu-job.mjs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { isDeepStrictEqual, parseArgs as parse } from 'node:util';
import { containedPath, readJsonRecord, writeJsonRecord, assertCurrentInputs } from '../../examples/sky-weather/lib/evidence.mjs';
import { validateInventory } from '../../examples/sky-weather/lib/inventory.js';

/**
 * @typedef {Object} BuildReceipt
 * @property {string} buildId UUID
 * @property {string} preparationId UUID
 * @property {object} inputs InputReceipt from evidence.mjs
 * @property {string} builtAt ISO time
 * @property {string} page canonical built page path
 * @property {string[]} files page-relative staged outputs including scenes.json
 * @property {'ready'|'failed'} status
 * @property {string|null} error
 *
 * @typedef {Object} Selection
 * @property {string} selectionId UUID
 * @property {'foundation'|'atmosphere'|'weather'} stage
 * @property {string} buildId
 * @property {string} preparationId
 * @property {object} inputs Frozen InputReceipt
 * @property {object[]} inventory Frozen SkyWeatherInventoryEntry rows
 *
 * @typedef {Object} Attempt
 * @property {string} attemptId UUID
 * @property {string} selectionId
 * @property {string} buildId
 * @property {string} directory stage-relative attempts/<UUID>
 * @property {'allocated'|'running'|'passed'|'failed'} status
 * @property {string|null} error
 *
 * @typedef {Object} RunRecord
 * @property {string} selectionId
 * @property {string} attemptId
 * @property {string} buildId
 * @property {string} preparationId
 * @property {string} pageRef
 * @property {Array<{jobId:string,clientId:string,scenario:string,mode:string,result:object|null,error:string|null}>} jobs
 * @property {Array<{id:string,filename:string,path:string,jobId:string,clientId:string,observation:string}>} images
 * @property {'passed'|'failed'} status
 * @property {string[]} errors
 *
 * @typedef {Object} FreshRecord
 * @property {string} selectionId
 * @property {string} attemptId
 * @property {string} buildId
 * @property {'passed'|'failed'} status
 * @property {string[]} errors
 *
 * @typedef {Object} JudgeRecord
 * @property {string} selectionId
 * @property {string} attemptId
 * @property {string} buildId
 * @property {Array<{id:string,filename:string,description:string,minScore:6,exit:number,score:number|null,output:string,judgedAt:string}>} images
 * @property {'passed'|'failed'} status
 * @property {string[]} errors
 */

/**
 * P2-EVIDENCE-RECORDS paths. The stage root is the --out directory.
 *   BuildReceipt <page>/build.json, page-relative staged `files` including scenes.json.
 *   Selection    <out>/selection.json
 *   current      <out>/current.json (pointer only: selectionId, attemptId, directory)
 *   Attempt      <out>/attempts/<UUID>/attempt.json
 *   RunRecord    <out>/attempts/<UUID>/run.json
 *   FreshRecord  <out>/attempts/<UUID>/fresh.json
 *   JudgeRecord  <out>/attempts/<UUID>/judge.json
 *   jobs         <out>/attempts/<UUID>/jobs/<index>/{input,result}.json
 *   images       <out>/attempts/<UUID>/images/<inventory filename>
 * Later shares (render/fresh/judge) extend this module and must retain these
 * operations and record shapes. A pointer or an allocated attempt is never a pass.
 */
export const RECORD_FIELDS = Object.freeze({
  buildReceipt: Object.freeze(['buildId', 'preparationId', 'inputs', 'builtAt', 'page', 'files', 'status', 'error']),
  selection: Object.freeze(['selectionId', 'stage', 'buildId', 'preparationId', 'inputs', 'inventory']),
  current: Object.freeze(['selectionId', 'attemptId', 'directory']),
  attempt: Object.freeze(['attemptId', 'selectionId', 'buildId', 'directory', 'status', 'error']),
  run: Object.freeze(['selectionId', 'attemptId', 'buildId', 'preparationId', 'pageRef', 'jobs', 'images', 'status', 'errors']),
  fresh: Object.freeze(['selectionId', 'attemptId', 'buildId', 'status', 'errors']),
  judge: Object.freeze(['selectionId', 'attemptId', 'buildId', 'images', 'status', 'errors']),
});

export const STAGES = Object.freeze(['foundation', 'atmosphere', 'weather']);
export const ACTIONS = Object.freeze(['check', 'select', 'clean', 'render']);

/** Record names inside an allocated `<out>/attempts/<UUID>` directory. */
export function attemptRecords(directory) {
  ensure(nonblank(directory) && !path.isAbsolute(directory) && !directory.includes('\\')
    && !directory.split('/').includes('..'), `unsafe attempt directory: ${directory}`);
  return Object.freeze({
    directory,
    selection: `${directory}/selection.json`,
    build: `${directory}/build.json`,
    attempt: `${directory}/attempt.json`,
    run: `${directory}/run.json`,
    fresh: `${directory}/fresh.json`,
    judge: `${directory}/judge.json`,
    image: (filename) => `${directory}/images/${filename}`,
    input: (index) => `${directory}/jobs/${jobIndex(index)}/input.json`,
    result: (index) => `${directory}/jobs/${jobIndex(index)}/result.json`,
  });
}

const USAGE = 'usage: node web/common/sky-weather-job.mjs <check|select|clean|render> --page dist/sky-weather --out out/sky-weather/<stage> --stage <foundation|atmosphere|weather>';
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function ensure(condition, message) { if (!condition) throw new Error(message); }
function nonblank(value) { return typeof value === 'string' && value.trim() !== ''; }
function isRecord(value) { return value !== null && typeof value === 'object' && !Array.isArray(value); }
function isUuid(value) { return nonblank(value) && UUID.test(value); }
function isIso(value) {
  return nonblank(value) && Number.isFinite(Date.parse(value)) && new Date(value).toISOString() === value;
}
function jobIndex(value) {
  ensure(Number.isSafeInteger(value) && value >= 0, `invalid job index: ${value}`);
  return value;
}
function exact(value, keys, context) {
  ensure(isRecord(value), `${context} must be an object`);
  const actual = Object.keys(value).sort(), expected = [...keys].sort();
  ensure(actual.length === expected.length && actual.every((key, index) => key === expected[index]),
    `${context} fields must be exactly ${expected.join(', ')}`);
}
function stagedFile(page, name, context) {
  ensure(nonblank(name), `${context}: staged file name is blank`);
  const file = containedPath(page, name);
  ensure(fs.existsSync(file), `${context}: staged file ${name} is missing`);
  const stat = fs.lstatSync(file);
  ensure(stat.isFile() && stat.size > 0, `${context}: staged file ${name} is not a nonempty regular file`);
  return file;
}

/** The RAA repository root owning the built page. */
export function repositoryRoot(page) {
  let cursor = path.resolve(page);
  for (;;) {
    if (fs.existsSync(path.join(cursor, 'Cargo.toml')) && fs.existsSync(path.join(cursor, '.git'))) return cursor;
    const parent = path.dirname(cursor);
    ensure(parent !== cursor, `${path.resolve(page)} is not inside the RAA repository`);
    cursor = parent;
  }
}

function validateInputReceipt(inputs, context) {
  exact(inputs, ['repository', 'revision', 'preparationId', 'dependencies', 'sources', 'toolchain',
    'commandsText', 'criteriaText'], context);
  ensure(inputs.repository === 'raa', `${context}: repository must be raa`);
  ensure(nonblank(inputs.revision), `${context}: revision is missing`);
  ensure(isUuid(inputs.preparationId), `${context}: preparationId is not a UUID`);
  ensure(Array.isArray(inputs.dependencies), `${context}: dependencies must be an array`);
  for (const dependency of inputs.dependencies) {
    exact(dependency, ['repository', 'revision', 'preparationId'], `${context}: dependency`);
    ensure(nonblank(dependency.repository) && nonblank(dependency.revision) && isUuid(dependency.preparationId),
      `${context}: invalid dependency identity`);
  }
  ensure(Array.isArray(inputs.sources), `${context}: sources must be an array`);
  for (const source of inputs.sources) {
    exact(source, ['path', 'text'], `${context}: source`);
    ensure(nonblank(source.path) && typeof source.text === 'string', `${context}: invalid source record`);
  }
  exact(inputs.toolchain, ['rustc', 'wasmBindgen', 'node'], `${context}: toolchain`);
  ensure(nonblank(inputs.toolchain.rustc) && nonblank(inputs.toolchain.wasmBindgen) && nonblank(inputs.toolchain.node),
    `${context}: invalid toolchain identity`);
  ensure(typeof inputs.commandsText === 'string' && typeof inputs.criteriaText === 'string',
    `${context}: command/criteria text is missing`);
}

/**
 * Read and verify dist/sky-weather/build.json before any scene module import.
 * Only the builder writes this receipt; selection copies it into an attempt.
 */
export function readBuildReceipt(page) {
  page = path.resolve(page);
  const root = repositoryRoot(page);
  const file = containedPath(page, 'build.json');
  ensure(fs.existsSync(file), `${file} is missing: build the page before checking or selecting it`);
  const build = readJsonRecord(file);
  exact(build, RECORD_FIELDS.buildReceipt, `${file}`);
  ensure(isUuid(build.buildId), `${file}: buildId is not a UUID`);
  ensure(isUuid(build.preparationId), `${file}: preparationId is not a UUID`);
  validateInputReceipt(build.inputs, `${file}: inputs`);
  ensure(build.inputs.preparationId === build.preparationId, `${file}: preparation identity differs from its input receipt`);
  ensure(isIso(build.builtAt), `${file}: builtAt is not an ISO time`);
  ensure(nonblank(build.page) && path.resolve(build.page) === page, `${file}: page does not name the selected page`);
  ensure(build.status === 'ready' || build.status === 'failed', `${file}: status must be ready or failed`);
  ensure(build.error === null || typeof build.error === 'string', `${file}: error must be null or a message`);
  ensure(build.status === 'ready', `build ${build.buildId} is ${build.status}: ${build.error ?? 'no error recorded'}`);
  ensure(build.error === null, `ready build ${build.buildId} records an error`);
  ensure(Array.isArray(build.files) && build.files.length > 0, `${file}: files must list staged outputs`);
  const files = new Set();
  for (const name of build.files) {
    ensure(!files.has(name), `${file}: duplicate staged file ${name}`);
    files.add(name);
    stagedFile(page, name, file);
  }
  ensure(files.has('scenes.json'), `${file}: scenes.json is not a staged build output`);
  assertCurrentInputs(root, build.inputs);
  return build;
}

/** Import only modules listed by the delivered scenes.json. */
export async function loadScenes(page, build) {
  page = path.resolve(page);
  const staged = new Set(build.files);
  const indexFile = containedPath(page, 'scenes.json');
  const index = readJsonRecord(indexFile);
  ensure(Array.isArray(index) && index.length > 0, `${indexFile} must list at least one delivered scene`);
  const ids = index.map((entry) => entry?.id);
  ensure(ids.every((id, i) => nonblank(id) && (i === 0 || ids[i - 1] < id)), `${indexFile} must be sorted by unique id`);
  ensure(new Set(ids).size === ids.length, `${indexFile} contains duplicate scene ids`);
  const namespaces = [];
  for (const entry of index) {
    exact(entry, ['id', 'milestone', 'file'], `${indexFile}: scene ${entry?.id}`);
    ensure(STAGES.includes(entry.milestone), `${indexFile}: scene ${entry.id} has unknown milestone ${entry.milestone}`);
    ensure(staged.has(entry.file), `${indexFile}: scene ${entry.id} module ${entry.file} is not a staged build output`);
    const module = stagedFile(page, entry.file, indexFile);
    let namespace;
    try { namespace = await import(pathToFileURL(module).href); }
    catch (cause) { throw new Error(`${indexFile}: scene ${entry.id} module import failed: ${cause.message}`, { cause }); }
    const scene = namespace.default;
    ensure(isRecord(scene) && scene.id === entry.id && scene.milestone === entry.milestone,
      `${indexFile}: scene ${entry.id} metadata differs from its module`);
    namespaces.push(namespace);
  }
  return namespaces;
}

/** Build receipt, delivered namespaces and the browser-free cumulative inventory. */
export async function verifyPage(page, stage) {
  ensure(STAGES.includes(stage), `invalid stage: ${stage ?? '<missing>'}`);
  page = path.resolve(page);
  const build = readBuildReceipt(page);
  const namespaces = await loadScenes(page, build);
  const inventory = validateInventory(namespaces, stage);
  return { page, build, namespaces, inventory };
}

export async function check(page, stage) {
  return (await verifyPage(page, stage)).inventory;
}

export async function select(page, out, stage) {
  const { build, inventory } = await verifyPage(page, stage);
  const selection = {
    selectionId: randomUUID(), stage, buildId: build.buildId, preparationId: build.preparationId,
    inputs: build.inputs, inventory,
  };
  writeJsonRecord(containedPath(out, 'selection.json'), selection);
  return selection;
}

function validateSelection(value, stage, build, inventory, context) {
  exact(value, RECORD_FIELDS.selection, context);
  ensure(isUuid(value.selectionId), `${context}: selectionId is not a UUID`);
  ensure(value.stage === stage, `${context}: selection stage ${value.stage} does not match ${stage}`);
  ensure(value.buildId === build.buildId, `${context}: selection belongs to build ${value.buildId}, current build is ${build.buildId}; select again`);
  ensure(value.preparationId === build.preparationId, `${context}: preparation differs from the current build; select again`);
  validateInputReceipt(value.inputs, `${context}: inputs`);
  ensure(isDeepStrictEqual(value.inputs, build.inputs), `${context}: build inputs changed; select again`);
  ensure(Array.isArray(value.inventory) && isDeepStrictEqual(value.inventory, inventory),
    `${context}: inventory changed; select again`);
}

/** Allocate a fresh immutable attempt; earlier attempts are never deleted. */
export async function clean(page, out, stage) {
  const { build, inventory } = await verifyPage(page, stage);
  const selectionFile = containedPath(out, 'selection.json');
  ensure(fs.existsSync(selectionFile), `${selectionFile} is missing: select the produced page first`);
  const selection = readJsonRecord(selectionFile);
  validateSelection(selection, stage, build, inventory, selectionFile);
  const attempts = containedPath(out, 'attempts');
  fs.mkdirSync(attempts, { recursive: true });
  const attemptId = randomUUID();
  const directory = `attempts/${attemptId}`;
  fs.mkdirSync(containedPath(out, directory));
  const records = attemptRecords(directory);
  writeJsonRecord(containedPath(out, records.selection), selection);
  writeJsonRecord(containedPath(out, records.build), build);
  const attempt = {
    attemptId, selectionId: selection.selectionId, buildId: build.buildId, directory,
    status: 'allocated', error: null,
  };
  writeJsonRecord(containedPath(out, records.attempt), attempt);
  writeJsonRecord(containedPath(out, 'current.json'), {
    selectionId: selection.selectionId, attemptId, directory,
  });
  return attempt;
}

/** Render once; transport owns queue, running and download clocks independently. */
export async function render(page, out, stage, transport = { publishGpuPage, runGpuPageOp }) {
  const { build, inventory } = await verifyPage(page, stage);
  const read = (name) => readJsonRecord(containedPath(out, name));
  const write = (name, value) => writeJsonRecord(containedPath(out, name), value);
  const pointer = read('current.json');
  exact(pointer, RECORD_FIELDS.current, 'current pointer');
  ensure(isUuid(pointer.attemptId) && isUuid(pointer.selectionId)
    && pointer.directory === `attempts/${pointer.attemptId}`, 'invalid current attempt identity');
  const records = attemptRecords(pointer.directory);
  const selection = read(records.selection);
  validateSelection(selection, stage, build, inventory, 'frozen selection');
  ensure(isDeepStrictEqual(read('selection.json'), selection), 'current selection differs from frozen selection');
  ensure(isDeepStrictEqual(read(records.build), build), 'frozen build differs from current build');
  const attempt = read(records.attempt);
  exact(attempt, RECORD_FIELDS.attempt, 'attempt');
  ensure(attempt.attemptId === pointer.attemptId && attempt.selectionId === pointer.selectionId
    && attempt.selectionId === selection.selectionId && attempt.buildId === build.buildId
    && attempt.directory === pointer.directory, 'frozen attempt identities differ');
  ensure(attempt.status === 'allocated' && attempt.error === null, 'attempt already started; clean before rendering');
  ensure(!fs.existsSync(containedPath(out, records.run)), 'attempt already has a run; clean before rendering');
  // Exclusive directory creation also prevents concurrent render callers from claiming this attempt.
  fs.mkdirSync(containedPath(out, `${pointer.directory}/jobs`));
  const run = {
    selectionId: selection.selectionId, attemptId: attempt.attemptId, buildId: build.buildId,
    preparationId: build.preparationId, pageRef: '', jobs: [], images: [], status: 'failed', errors: [],
  };
  const message = (error) => String(error?.message ?? error);
  const current = () => ensure(isDeepStrictEqual(read('current.json'), pointer), 'current pointer was superseded');
  const saveCaptures = (job, rows) => {
    for (const image of Array.isArray(job.result?.images) ? job.result.images : []) {
      try {
        const row = rows.find((row) => row.id === image.id && row.filename === image.filename);
        ensure(row && image.mime === 'image/png' && typeof image.observation === 'string', 'unexpected capture identity or format');
        ensure(!run.images.some((saved) => saved.id === image.id), `duplicate capture ${image.id}`);
        ensure(typeof image.data === 'string' && /^[A-Za-z0-9+/]+={0,2}$/.test(image.data), `invalid PNG payload ${image.id}`);
        const bytes = Buffer.from(image.data, 'base64');
        ensure(bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])), `invalid PNG signature ${image.id}`);
        const file = containedPath(out, records.image(row.filename));
        fs.mkdirSync(path.dirname(file), { recursive: true });
        fs.writeFileSync(file, bytes, { flag: 'wx' });
        run.images.push({ id: image.id, filename: row.filename, path: `images/${row.filename}`,
          jobId: job.jobId, clientId: job.clientId, observation: image.observation });
      } catch (error) { run.errors.push(message(error)); }
    }
  };
  try {
    write(records.attempt, { ...attempt, status: 'running' });
    current();
    const published = await transport.publishGpuPage(build.page, { timeoutMs: 300000 });
    ensure(nonblank(published?.pageRef), 'publication returned no pageRef');
    run.pageRef = published.pageRef;
    const groups = new Map();
    for (const row of inventory) {
      const key = JSON.stringify([row.scenario, row.mode]);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(row);
    }
    for (const rows of groups.values()) {
      current();
      const { scenario, mode } = rows[0];
      const index = run.jobs.length;
      const input = { selectionId: run.selectionId, attemptId: run.attemptId, buildId: run.buildId,
        scenario, mode, width: 1280, height: 720, remainingMs: 270000 };
      write(records.input(index), input);
      const job = { jobId: '', clientId: '', scenario, mode, result: null, error: null };
      run.jobs.push(job);
      try {
        const reply = await transport.runGpuPageOp(run.pageRef, 'run', input, {
          timeoutMs: 300000, out: containedPath(out, `${pointer.directory}/jobs/${index}`),
        });
        job.jobId = nonblank(reply.jobId) ? reply.jobId : '';
        job.clientId = nonblank(reply.clientId) ? reply.clientId : '';
        job.result = reply.result ?? null;
        write(records.result(index), job.result);
        ensure(nonblank(job.jobId) && nonblank(job.clientId), 'missing job/client identity');
        ensure(isRecord(job.result), 'missing or malformed direct operation result');
      } catch (error) {
        job.error = message(error);
        // The transport leaves the actual host reply here even when download/operation fails.
        const file = containedPath(out, records.result(index));
        if (job.result === null && fs.existsSync(file)) {
          try {
            const reply = readJsonRecord(file);
            if (Object.hasOwn(reply ?? {}, 'result')) {
              job.jobId = String(reply.job_id ?? '');
              job.clientId = String(reply.client_id ?? '');
              job.result = reply.result;
              write(records.result(index), job.result);
            }
          } catch (recovery) { run.errors.push(message(recovery)); }
        }
        if (!fs.existsSync(file)) write(records.result(index), null);
        run.errors.push(job.error);
      }
      saveCaptures(job, rows);
      if (job.error) throw new Error(job.error);
      const expected = { selectionId: run.selectionId, attemptId: run.attemptId, buildId: run.buildId, scenario, mode };
      ensure(isDeepStrictEqual(job.result.provenance, expected), 'operation provenance differs from input');
      ensure(rows.every((row) => run.images.some((image) => image.id === row.id)), 'operation is missing selected captures');
      for (const kind of ['console', 'page', 'gpu']) {
        ensure(Array.isArray(job.result.errors?.[kind]), `missing ${kind} error observations`);
        run.errors.push(...job.result.errors[kind].map((error) => `${kind}: ${error}`));
      }
      write(records.run, run);
      ensure(run.errors.length === 0, run.errors.join('\n'));
    }
    current();
    run.status = 'passed';
  } catch (error) {
    if (!run.errors.includes(message(error))) run.errors.push(message(error));
  }
  write(records.run, run);
  write(records.attempt, { ...attempt, status: run.status, error: run.errors.length ? run.errors.join('\n') : null });
  ensure(run.status === 'passed', run.errors.join('\n'));
  return run;
}

export function parseArgs(argv) {
  const { values, positionals } = parse({ args: argv, allowPositionals: true,
    options: { page: { type: 'string' }, out: { type: 'string' }, stage: { type: 'string' } } });
  ensure(positionals.length === 1 && ACTIONS.includes(positionals[0]), USAGE);
  const action = positionals[0];
  ensure(nonblank(values.page), USAGE);
  ensure(STAGES.includes(values.stage), `invalid stage: ${values.stage ?? '<missing>'}`);
  if (action === 'check' && values.out === undefined) return { action, page: values.page, out: undefined, stage: values.stage };
  ensure(nonblank(values.out), USAGE);
  return { action, page: values.page, out: values.out, stage: values.stage };
}

export async function main(argv = process.argv.slice(2)) {
  const { action, page, out, stage } = parseArgs(argv);
  if (action === 'check') {
    const inventory = await check(page, stage);
    process.stdout.write(`sky-weather-job: check ${stage}: ${inventory.length} image(s)\n`);
  } else if (action === 'select') {
    const selection = await select(page, out, stage);
    process.stdout.write(`sky-weather-job: select ${stage}: ${selection.inventory.length} image(s), selection ${selection.selectionId}\n`);
  } else if (action === 'render') {
    const run = await render(page, out, stage);
    process.stdout.write(`sky-weather-job: render ${stage}: attempt ${run.attemptId}: ${run.status}\n`);
  } else {
    const attempt = await clean(page, out, stage);
    process.stdout.write(`sky-weather-job: clean ${stage}: attempt ${attempt.attemptId} at ${attempt.directory}\n`);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch((error) => {
    process.stderr.write(`sky-weather-job: ${error?.message ?? error}\n`);
    process.exit(1);
  });
}
