#!/usr/bin/env node
// P2 sky-weather page protocol: verify the staged build, select the cumulative
// inventory, allocate immutable render attempts, render bounded managed jobs,
// verify actual GPU evidence and judge every selected capture.
import fs from 'node:fs';
import { spawn } from 'node:child_process';
import { publishGpuPage, runGpuPageOp } from './gpu-job.mjs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { isDeepStrictEqual, parseArgs as parse } from 'node:util';
import { containedPath, readJsonRecord, writeJsonRecord, assertCurrentInputs } from '../../examples/sky-weather/lib/evidence.mjs';
import { validateInventory, validateRequiredFacts } from '../../examples/sky-weather/lib/inventory.js';
import { validateFeatureMeasurement } from '../../examples/sky-weather/lib/measure.js';

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
 * @property {Array<{id:string,filename:string,attemptId:string,selectionId:string,buildId:string}>} kept
 *   images copied from an earlier attempt of the same build, with the identities of the attempt
 *   whose managed operation produced them (P2-RESUMABLE-RENDER ORIGIN); empty when nothing is kept
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
 * P2-RESUMABLE-RENDER: two builds are the same build when their buildKey() strings are
 * equal (buildId/preparationId are redrawn by every build). render keeps a (scenario, mode)
 * group from the first earlier attempt directory (ascending name, any status) of the same
 * build whose selection holds the identical rows, whose job passes the per-job checks
 * against its origin identities and whose images are intact regular files; it copies that
 * job, its inputs/results and images into the new attempt, lists them in run.kept, renders
 * only the other groups and publishes only when it renders one. clean still allocates a
 * fresh attempt and deletes no record; it also deletes the image files of earlier attempts
 * whose frozen build has a different build key or cannot be read.
 * P2-RESUMABLE-RENDER FRESH: a job whose images are listed in run.kept passes the per-job checks
 * against its kept entries' origin identities, and that origin attempt must still hold its frozen
 * build of the current build key, the group's selection rows and the identical job record (and must
 * not merely have copied the group). Every other job passes against the current identities. Coverage,
 * image bytes and pair coverage apply to the union; the two states of a pair may come from different
 * attempts of the same build.
 * P2-RESUMABLE-RENDER JUDGE: a kept image reuses an earlier passed verdict of the byte-identical
 * image with the identical description (exit 0, score >= 6), keeping its judgedAt; every other image
 * is judged, and every verdict is written before failing.
 */
export const RECORD_FIELDS = Object.freeze({
  buildReceipt: Object.freeze(['buildId', 'preparationId', 'inputs', 'builtAt', 'page', 'files', 'status', 'error']),
  selection: Object.freeze(['selectionId', 'stage', 'buildId', 'preparationId', 'inputs', 'inventory']),
  current: Object.freeze(['selectionId', 'attemptId', 'directory']),
  attempt: Object.freeze(['attemptId', 'selectionId', 'buildId', 'directory', 'status', 'error']),
  run: Object.freeze(['selectionId', 'attemptId', 'buildId', 'preparationId', 'pageRef', 'jobs', 'images', 'kept', 'status', 'errors']),
  fresh: Object.freeze(['selectionId', 'attemptId', 'buildId', 'status', 'errors']),
  judge: Object.freeze(['selectionId', 'attemptId', 'buildId', 'images', 'status', 'errors']),
});

export const STAGES = Object.freeze(['foundation', 'atmosphere', 'weather']);
export const ACTIONS = Object.freeze(['check', 'select', 'clean', 'render', 'fresh', 'judge']);

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

const USAGE = 'usage: node web/common/sky-weather-job.mjs <check|select|clean|render|fresh|judge> --page dist/sky-weather --out out/sky-weather/<stage> --stage <foundation|atmosphere|weather>';
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const PNG_SIGNATURE = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
const ADAPTER_FIELDS = ['vendor', 'architecture', 'device', 'description'];
const SOFTWARE_ADAPTER = /software|swiftshader|llvmpipe|lavapipe|softpipe|fallback|basic render|microsoft basic|warp\b/i;

function ensure(condition, message) { if (!condition) throw new Error(message); }
function nonblank(value) { return typeof value === 'string' && value.trim() !== ''; }
function isRecord(value) { return value !== null && typeof value === 'object' && !Array.isArray(value); }
function isUuid(value) { return nonblank(value) && UUID.test(value); }
function message(error) { return String(error?.message ?? error); }
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

/** P2-RESUMABLE-RENDER SAME BUILD identity: the receipt inputs without their preparationId. */
export function buildKey(build) {
  const { revision, dependencies, sources, toolchain, commandsText, criteriaText } = build.inputs;
  return JSON.stringify({ revision, dependencies, sources, toolchain, commandsText, criteriaText });
}

function sameBuild(build, key) {
  try { return buildKey(build) === key; } catch { return false; }
}

/** Earlier attempt directories of the stage root, in ascending directory name order. */
function earlierAttempts(out, currentDirectory) {
  const root = containedPath(out, 'attempts');
  if (!fs.existsSync(root)) return [];
  return fs.readdirSync(root).sort().map((name) => `attempts/${name}`)
    .filter((directory) => directory !== currentDirectory && isUuid(directory.slice('attempts/'.length)));
}

/** Readable records of earlier attempts frozen with the given build key. */
function keptCandidates(out, currentDirectory, key) {
  const candidates = [];
  for (const directory of earlierAttempts(out, currentDirectory)) {
    try {
      const records = attemptRecords(directory);
      const build = readJsonRecord(containedPath(out, records.build));
      if (!sameBuild(build, key)) continue;
      candidates.push({ directory, records, selection: readJsonRecord(containedPath(out, records.selection)),
        run: readJsonRecord(containedPath(out, records.run)) });
    } catch { /* an unreadable attempt supplies nothing */ }
  }
  return candidates;
}

function readRegularFile(file) {
  const descriptor = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW);
  try {
    ensure(fs.fstatSync(descriptor).isFile(), `${file} is not a regular file`);
    return fs.readFileSync(descriptor);
  } finally { fs.closeSync(descriptor); }
}

/** KEPT GROUP: the first earlier attempt whose evidence of this group passes every check. */
function keptGroup(out, candidates, rows, namespaces) {
  const { scenario, mode } = rows[0];
  const factNames = namespaces.find((entry) => entry.default?.id === scenario)?.factNames;
  for (const candidate of candidates) {
    try {
      const { selection, run, records } = candidate;
      ensure(rows.every((row) => selection.inventory.some((entry) => isDeepStrictEqual(entry, row))), 'changed rows');
      const index = run.jobs.findIndex((job) => job?.scenario === scenario && job?.mode === mode);
      ensure(index >= 0, 'no job');
      const job = run.jobs[index];
      const listed = (Array.isArray(run.kept) ? run.kept : []).filter((entry) => rows.some((row) => row.id === entry?.id));
      const origin = listed.length
        ? { selectionId: listed[0].selectionId, attemptId: listed[0].attemptId, buildId: listed[0].buildId }
        : { selectionId: run.selectionId, attemptId: run.attemptId, buildId: run.buildId };
      ensure(listed.length === 0 || (listed.length === rows.length
        && listed.every((entry) => isDeepStrictEqual({ selectionId: entry.selectionId, attemptId: entry.attemptId,
          buildId: entry.buildId }, origin))), 'partial kept origin');
      const captures = checkJob(job, `${candidate.directory} job ${index}`, rows, origin, namespaces, factNames, new Map());
      const images = rows.map((row) => {
        const image = run.images.find((entry) => entry?.id === row.id && entry.filename === row.filename);
        ensure(image, 'no image entry');
        const bytes = readRegularFile(containedPath(out, records.image(row.filename)));
        ensure(bytes.equals(captures.get(row.id).bytes), 'damaged image');
        return { row, image, bytes };
      });
      const input = readJsonRecord(containedPath(out, records.input(index)));
      return { candidate, job, input, origin, images };
    } catch { /* not kept: rendered again */ }
  }
  return null;
}

/** Allocate a fresh immutable attempt; earlier records are never deleted, older-build images are. */
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
  const key = buildKey(build);
  for (const earlier of earlierAttempts(out, directory)) {
    let frozen = null;
    try { frozen = readJsonRecord(containedPath(out, attemptRecords(earlier).build)); } catch { /* unreadable */ }
    if (frozen !== null && sameBuild(frozen, key)) continue;
    let images;
    try { images = containedPath(out, `${earlier}/images`); } catch { continue; }
    if (!fs.existsSync(images) || !fs.lstatSync(images).isDirectory()) continue;
    for (const name of fs.readdirSync(images)) {
      const file = path.join(images, name);
      if (!fs.lstatSync(file).isDirectory()) fs.unlinkSync(file);
    }
  }
  return attempt;
}

/**
 * Render once; transport owns queue, running and download clocks independently. Groups kept
 * from an earlier attempt of the same build are copied, not rendered (P2-RESUMABLE-RENDER).
 */
export async function render(page, out, stage, transport = { publishGpuPage, runGpuPageOp }) {
  const { build, namespaces, inventory } = await verifyPage(page, stage);
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
    preparationId: build.preparationId, pageRef: '', jobs: [], images: [], kept: [], status: 'failed', errors: [],
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
    const groups = new Map();
    for (const row of inventory) {
      const key = JSON.stringify([row.scenario, row.mode]);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(row);
    }
    const candidates = keptCandidates(out, pointer.directory, buildKey(build));
    const pending = [];
    let firstKept = null;
    for (const rows of groups.values()) {
      const kept = keptGroup(out, candidates, rows, namespaces);
      if (!kept) { pending.push(rows); continue; }
      if (run.jobs.length === 0 && pending.length === 0) firstKept = kept;
      const index = run.jobs.length;
      write(records.input(index), kept.input);
      write(records.result(index), kept.job.result);
      run.jobs.push(kept.job);
      for (const { row, image, bytes } of kept.images) {
        const file = containedPath(out, records.image(row.filename));
        fs.mkdirSync(path.dirname(file), { recursive: true });
        fs.writeFileSync(file, bytes, { flag: 'wx' });
        run.images.push(image);
        run.kept.push({ id: row.id, filename: row.filename, ...kept.origin });
      }
    }
    write(records.run, run);
    if (pending.length > 0) {
      current();
      const published = await transport.publishGpuPage(build.page, { timeoutMs: 300000 });
      ensure(nonblank(published?.pageRef), 'publication returned no pageRef');
      run.pageRef = published.pageRef;
    } else {
      ensure(nonblank(firstKept.candidate.run.pageRef), 'kept attempt has no managed page reference');
      run.pageRef = firstKept.candidate.run.pageRef;
    }
    for (const rows of pending) {
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

/** Resolve the stage pointer to the frozen attempt and its current page build. */
async function resolveAttempt(page, out, stage) {
  const { build, namespaces, inventory } = await verifyPage(page, stage);
  const pointerFile = containedPath(out, 'current.json');
  ensure(fs.existsSync(pointerFile), `${pointerFile} is missing: render an attempt before verifying it`);
  const pointer = readJsonRecord(pointerFile);
  exact(pointer, RECORD_FIELDS.current, 'current pointer');
  ensure(isUuid(pointer.selectionId) && isUuid(pointer.attemptId)
    && pointer.directory === `attempts/${pointer.attemptId}`, 'invalid current attempt identity');
  const records = attemptRecords(pointer.directory);
  const selection = readJsonRecord(containedPath(out, records.selection));
  validateSelection(selection, stage, build, inventory, 'frozen selection');
  ensure(isDeepStrictEqual(readJsonRecord(containedPath(out, 'selection.json')), selection),
    'current selection differs from frozen selection');
  ensure(isDeepStrictEqual(readJsonRecord(containedPath(out, records.build)), build),
    'frozen build differs from current build');
  const attempt = readJsonRecord(containedPath(out, records.attempt));
  exact(attempt, RECORD_FIELDS.attempt, 'attempt');
  ensure(attempt.attemptId === pointer.attemptId && attempt.selectionId === pointer.selectionId
    && attempt.selectionId === selection.selectionId && attempt.buildId === build.buildId
    && attempt.directory === pointer.directory, 'frozen attempt identities differ');
  return { page, out, stage, build, namespaces, inventory, pointer, records, selection, attempt };
}

function sceneVariant(namespaces, scenario, mode, label) {
  const namespace = namespaces.find((entry) => entry.default?.id === scenario);
  ensure(namespace, `${label}: scene ${scenario} is not a delivered namespace`);
  const variant = namespace.default.variants.find((entry) => entry.id === mode);
  ensure(variant, `${label}: scene ${scenario} has no variant ${mode}`);
  return variant;
}

function decodePng(data, label) {
  ensure(typeof data === 'string' && data.length % 4 === 0 && /^[A-Za-z0-9+/]+={0,2}$/.test(data),
    `${label}: invalid base64 PNG payload`);
  const bytes = Buffer.from(data, 'base64');
  ensure(bytes.length > PNG_SIGNATURE.length && bytes.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE),
    `${label}: invalid PNG signature`);
  return bytes;
}

/** Actual Chrome on Windows driving a physical (non-fallback) GPU adapter. */
function validateProvenance(label, result) {
  ensure(isRecord(result.browser), `${label}: browser provenance is missing`);
  exact(result.browser, ['userAgent', 'platform', 'devicePixelRatio'], `${label}: browser`);
  const browser = result.browser;
  ensure(/\b(?:Headless)?Chrome\/\d+/.test(browser.userAgent)
    && /Windows NT/.test(browser.userAgent) && !/Edg\/|OPR\//.test(browser.userAgent)
    && /^Win(?:32|64|dows)$/.test(browser.platform), `${label}: actual Chrome on Windows is required`);
  ensure(browser.devicePixelRatio === 1, `${label}: device pixel ratio must be 1`);
  ensure(isRecord(result.adapter), `${label}: adapter provenance is missing`);
  exact(result.adapter, [...ADAPTER_FIELDS, 'isFallbackAdapter'], `${label}: adapter`);
  ensure(ADAPTER_FIELDS.every((key) => typeof result.adapter[key] === 'string'),
    `${label}: adapter identity must be text`);
  ensure(ADAPTER_FIELDS.some((key) => nonblank(result.adapter[key])),
    `${label}: adapter identity is blank`);
  ensure(result.hardware === true && result.adapter.isFallbackAdapter === false,
    `${label}: a physical GPU adapter is required; fallback or software adapters are refused`);
  ensure(!ADAPTER_FIELDS.some((key) => SOFTWARE_ADAPTER.test(result.adapter[key])),
    `${label}: software or fallback adapter is refused`);
}

function validateJobErrors(label, errors) {
  ensure(isRecord(errors), `${label}: console/page/GPU error observations are missing`);
  exact(errors, ['console', 'page', 'gpu'], `${label}: errors`);
  for (const kind of ['console', 'page', 'gpu']) {
    ensure(Array.isArray(errors[kind]) && errors[kind].every((entry) => typeof entry === 'string'),
      `${label}: ${kind} errors must be an array of messages`);
    ensure(errors[kind].length === 0, `${label}: ${kind} errors: ${errors[kind].join('; ')}`);
  }
}

/** Exact measurement records declared by the selected variant's measurement. */
function validateMeasurements(declared, result, label, seen) {
  const expected = declared ? [declared] : [];
  ensure(Array.isArray(result.measurements), `${label}: measurements must be an array`);
  ensure(result.measurements.length === expected.length,
    `${label}: expected ${expected.length} measurement record(s), found ${result.measurements.length}`);
  for (const measurement of result.measurements) {
    validateFeatureMeasurement(measurement);
    const spec = expected.find((entry) => entry.pair === measurement.pair && entry.state === measurement.state);
    ensure(spec && measurement.preset === spec.preset && isDeepStrictEqual(measurement.features, spec.features),
      `${label}: measurement differs from the declared ${spec?.pair ?? 'unknown'} ${spec?.state ?? ''} record`);
    const key = `${measurement.pair}\0${measurement.state}`;
    ensure(!seen.has(key), `${label}: duplicate measurement ${measurement.pair} ${measurement.state}`);
    seen.set(key, measurement);
  }
}

const JOB_FIELDS = ['jobId', 'clientId', 'scenario', 'mode', 'result', 'error'];
const KEPT_FIELDS = ['id', 'filename', 'attemptId', 'selectionId', 'buildId'];

/**
 * P2-RESUMABLE-RENDER PER-JOB CHECKS of one managed job for its (scenario, mode) group
 * against the given identities {selectionId, attemptId, buildId}. Returns the decoded
 * captures by image id; `seen` collects measurements keyed by pair and state.
 */
function checkJob(job, label, rows, identities, namespaces, factNames, seen) {
  exact(job, JOB_FIELDS, label);
  ensure(nonblank(job.jobId) && nonblank(job.clientId), `${label}: managed job/client identity is missing`);
  ensure(job.error === null, `${label}: ${job.error}`);
  ensure(isRecord(job.result), `${label}: a direct GPU operation result is required`);
  const result = job.result;
  ensure(isDeepStrictEqual(result.provenance, {
    selectionId: identities.selectionId, attemptId: identities.attemptId, buildId: identities.buildId,
    scenario: job.scenario, mode: job.mode,
  }), `${label}: operation provenance differs from the frozen attempt`);
  const scoped = `${label} (${job.scenario}/${job.mode})`;
  validateProvenance(scoped, result);
  validateJobErrors(scoped, result.errors);
  ensure(isRecord(result.facts), `${scoped}: scene facts are missing`);
  const readiness = result.facts.readiness;
  ensure(readiness?.status === 'ready' && readiness.error === null,
    `${scoped}: preparation is not ready (${readiness?.status ?? 'missing'}${readiness?.error ? `: ${readiness.error}` : ''})`);
  ensure(Array.isArray(factNames), `${scoped}: scene ${job.scenario} has no factNames registry`);
  ensure(Array.isArray(result.images), `${scoped}: operation returned no image inventory`);
  ensure(result.images.length === rows.length,
    `${scoped}: returned ${result.images.length} image(s) for ${rows.length} selected capture(s)`);
  const captures = new Map();
  for (const image of result.images) {
    exact(image, ['id', 'filename', 'mime', 'data', 'observation'], `${scoped}: image`);
    const row = rows.find((entry) => entry.id === image.id && entry.filename === image.filename);
    ensure(row, `${scoped}: unexpected capture ${image.id} (${image.filename})`);
    ensure(!captures.has(image.id), `${scoped}: duplicate capture ${image.id}`);
    ensure(image.mime === 'image/png', `${scoped}: capture ${image.id} is not image/png`);
    ensure(nonblank(image.observation), `${scoped}: capture ${image.id} has no observation`);
    captures.set(image.id, { row, image, bytes: decodePng(image.data, `${scoped}: capture ${image.id}`) });
  }
  for (const row of rows) {
    ensure(captures.has(row.id), `${scoped}: missing selected capture ${row.id}`);
    validateRequiredFacts(row, result.facts, factNames);
  }
  const variant = sceneVariant(namespaces, job.scenario, job.mode, scoped);
  validateMeasurements(variant.measurement ?? null, result, scoped, seen);
  return captures;
}

/**
 * P2-RESUMABLE-RENDER: the originating attempt named by a kept entry must still hold its frozen
 * build of the current build key, the selection rows of the group and the identical job record.
 * An attempt that only copied the group (its run.kept lists these images) is not the origin.
 */
function verifyOrigin(ctx, origin, job, rows, label) {
  ensure(isUuid(origin.attemptId) && `attempts/${origin.attemptId}` !== ctx.pointer.directory,
    `${label}: kept origin ${origin.attemptId} is not an earlier attempt`);
  const records = attemptRecords(`attempts/${origin.attemptId}`);
  const read = (name) => {
    const file = containedPath(ctx.out, name);
    ensure(fs.existsSync(file), `${label}: kept origin attempt ${origin.attemptId} is missing ${path.basename(name)}`);
    return readJsonRecord(file);
  };
  const build = read(records.build);
  ensure(sameBuild(build, buildKey(ctx.build)) && build.buildId === origin.buildId,
    `${label}: kept origin attempt ${origin.attemptId} belongs to another build`);
  const selection = read(records.selection);
  ensure(selection.selectionId === origin.selectionId
    && rows.every((row) => selection.inventory.some((entry) => isDeepStrictEqual(entry, row))),
  `${label}: kept origin attempt ${origin.attemptId} does not select the rows of ${job.scenario}/${job.mode}`);
  const run = read(records.run);
  ensure(run.attemptId === origin.attemptId && run.selectionId === origin.selectionId && run.buildId === origin.buildId,
    `${label}: kept origin attempt ${origin.attemptId} has other identities`);
  ensure(!(Array.isArray(run.kept) ? run.kept : []).some((entry) => rows.some((row) => row.id === entry?.id)),
    `${label}: kept origin ${origin.attemptId} only copied this group; it is not the originating attempt`);
  ensure(Array.isArray(run.jobs) && run.jobs.some((entry) => isDeepStrictEqual(entry, job)),
    `${label}: kept origin attempt ${origin.attemptId} does not hold the identical job record`);
}

/**
 * One managed job of the current attempt. A job whose images are listed in run.kept is checked
 * against the identities of its kept entries (its origin); every other job against the current ones.
 */
function validateJob(job, index, ctx, state) {
  const label = `run job ${index}`;
  exact(job, JOB_FIELDS, label);
  const key = `${job.scenario}\0${job.mode}`;
  ensure(!state.jobs.has(key), `${label}: duplicate scenario/mode ${job.scenario}/${job.mode}`);
  const rows = state.groups.get(key);
  ensure(rows, `${label}: unexpected scenario/mode ${job.scenario}/${job.mode}`);
  let identities = {
    selectionId: ctx.selection.selectionId, attemptId: ctx.attempt.attemptId, buildId: ctx.build.buildId,
  };
  const listed = state.kept.filter((entry) => rows.some((row) => row.id === entry.id));
  if (listed.length > 0) {
    ensure(rows.every((row) => listed.some((entry) => entry.id === row.id && entry.filename === row.filename)),
      `${label}: kept images must cover exactly the selected images of ${job.scenario}/${job.mode}`);
    identities = { selectionId: listed[0].selectionId, attemptId: listed[0].attemptId, buildId: listed[0].buildId };
    ensure(listed.every((entry) => isDeepStrictEqual({
      selectionId: entry.selectionId, attemptId: entry.attemptId, buildId: entry.buildId,
    }, identities)), `${label}: kept images name different origin attempts`);
    verifyOrigin(ctx, identities, job, rows, label);
  }
  const captures = checkJob(job, label, rows, identities, ctx.namespaces, state.factNames.get(job.scenario),
    state.measurements);
  state.jobs.set(key, { job, captures });
}

/** Run images must exactly cover the selected inventory and match the direct payloads. */
function validateRunImages(ctx, run, state) {
  ensure(run.images.length === ctx.inventory.length,
    `run records ${run.images.length} image(s) for ${ctx.inventory.length} selected image(s)`);
  const byId = new Map(), filenames = new Set();
  for (const image of run.images) {
    exact(image, ['id', 'filename', 'path', 'jobId', 'clientId', 'observation'], 'run image');
    const row = ctx.inventory.find((entry) => entry.id === image.id && entry.filename === image.filename);
    ensure(row, `run image ${image.id} is not a selected capture`);
    ensure(!byId.has(image.id), `duplicate run image ${image.id}`);
    ensure(!filenames.has(image.filename), `duplicate run image filename ${image.filename}`);
    ensure(image.path === `images/${image.filename}`, `run image ${image.id} has an unexpected path`);
    let capture = null;
    for (const entry of state.jobs.values()) {
      const candidate = entry.captures.get(image.id);
      if (candidate && entry.job.jobId === image.jobId && entry.job.clientId === image.clientId) capture = candidate;
    }
    ensure(capture, `run image ${image.id} has no matching managed job attribution`);
    ensure(capture.image.observation === image.observation,
      `run image ${image.id} observation differs from the direct operation result`);
    const file = containedPath(ctx.out, ctx.records.image(image.filename));
    ensure(fs.existsSync(file) && fs.lstatSync(file).isFile(), `run image ${image.id} is missing at ${image.path}`);
    ensure(fs.readFileSync(file).equals(capture.bytes),
      `run image ${image.id} differs from the direct operation payload`);
    byId.set(image.id, image);
    filenames.add(image.filename);
  }
  for (const row of ctx.inventory) ensure(byId.has(row.id), `selected image ${row.id} is missing from the run`);
  const imagesDirectory = containedPath(ctx.out, `${ctx.pointer.directory}/images`);
  if (fs.existsSync(imagesDirectory)) {
    for (const name of fs.readdirSync(imagesDirectory)) {
      ensure(filenames.has(name), `unexpected capture file images/${name}`);
    }
  }
}

/** Every declared measurement pair must have been collected in both states. */
function validatePairCoverage(ctx, state) {
  const scenes = new Set(ctx.inventory.map((row) => row.scenario));
  const declared = new Map();
  for (const scenario of scenes) {
    const namespace = ctx.namespaces.find((entry) => entry.default?.id === scenario);
    for (const variant of namespace.default.variants) {
      const measurement = variant.measurement;
      if (!measurement) continue;
      const states = declared.get(measurement.pair) ?? new Set();
      states.add(measurement.state);
      declared.set(measurement.pair, states);
    }
  }
  for (const [pair, states] of declared) {
    ensure(states.has('on') && states.has('off'), `measurement pair ${pair} does not declare both on and off states`);
    ensure(state.measurements.has(`${pair}\0on`) && state.measurements.has(`${pair}\0off`),
      `measurement pair ${pair} was not collected in both on and off states`);
  }
}

function validateFreshRecord(ctx) {
  ensure(ctx.attempt.status === 'passed' && ctx.attempt.error === null,
    `attempt ${ctx.attempt.attemptId} is ${ctx.attempt.status}${ctx.attempt.error ? `: ${ctx.attempt.error}` : ''}: a completed GPU render is required`);
  const runFile = containedPath(ctx.out, ctx.records.run);
  ensure(fs.existsSync(runFile), `${runFile} is missing: render must complete before freshness`);
  const run = readJsonRecord(runFile);
  exact(run, RECORD_FIELDS.run, 'run record');
  ensure(run.selectionId === ctx.selection.selectionId && run.attemptId === ctx.attempt.attemptId
    && run.buildId === ctx.build.buildId && run.preparationId === ctx.build.preparationId,
  'run identity differs from the frozen attempt');
  ensure(nonblank(run.pageRef), 'run has no managed page reference');
  ensure(run.status === 'passed', `run is ${run.status}${run.errors?.length ? `: ${run.errors.join('; ')}` : ''}`);
  ensure(Array.isArray(run.errors) && run.errors.length === 0, `run errors: ${run.errors.join('; ')}`);
  ensure(Array.isArray(run.jobs) && run.jobs.length > 0, 'run has no managed jobs');
  ensure(Array.isArray(run.images), 'run has no image inventory');
  ensure(Array.isArray(run.kept), 'run has no kept image list');
  for (const entry of run.kept) {
    exact(entry, KEPT_FIELDS, 'kept image');
    ensure(ctx.inventory.some((row) => row.id === entry.id && row.filename === entry.filename),
      `kept image ${entry.id} is not a selected capture`);
  }
  ensure(new Set(run.kept.map((entry) => entry.id)).size === run.kept.length, 'run lists a kept image twice');
  const state = { groups: new Map(), jobs: new Map(), measurements: new Map(), factNames: new Map(), kept: run.kept };
  for (const row of ctx.inventory) {
    const key = `${row.scenario}\0${row.mode}`;
    if (!state.groups.has(key)) state.groups.set(key, []);
    state.groups.get(key).push(row);
  }
  ensure(run.jobs.length === state.groups.size,
    `run has ${run.jobs.length} job(s) for ${state.groups.size} selected scenario/mode group(s)`);
  for (const namespace of ctx.namespaces) state.factNames.set(namespace.default.id, namespace.factNames);
  for (const [index, job] of run.jobs.entries()) validateJob(job, index, ctx, state);
  ensure(state.jobs.size === state.groups.size, 'run is missing selected scenario/mode jobs');
  validateRunImages(ctx, run, state);
  validatePairCoverage(ctx, state);
  return run;
}

function assertCurrentAttempt(ctx, run) {
  ensure(isDeepStrictEqual(readJsonRecord(containedPath(ctx.out, 'current.json')), ctx.pointer),
    'current pointer was superseded during freshness');
  ensure(isDeepStrictEqual(readJsonRecord(containedPath(ctx.out, ctx.records.selection)), ctx.selection),
    'frozen selection changed during freshness');
  ensure(isDeepStrictEqual(readJsonRecord(containedPath(ctx.out, ctx.records.build)), ctx.build),
    'frozen build changed during freshness');
  ensure(isDeepStrictEqual(readJsonRecord(containedPath(ctx.out, ctx.records.attempt)), ctx.attempt),
    'frozen attempt changed during freshness');
  ensure(isDeepStrictEqual(readJsonRecord(containedPath(ctx.out, ctx.records.run)), run),
    'run changed during freshness');
}

/** Verify the current attempt's actual GPU results and write attempt fresh.json. */
export async function verifyFresh(page, out, stage) {
  const ctx = await resolveAttempt(page, out, stage);
  const record = {
    selectionId: ctx.selection.selectionId, attemptId: ctx.attempt.attemptId, buildId: ctx.build.buildId,
    status: 'failed', errors: [],
  };
  let run = null;
  try {
    run = validateFreshRecord(ctx);
    assertCurrentAttempt(ctx, run);
    record.status = 'passed';
  } catch (error) {
    record.errors.push(message(error));
  }
  writeJsonRecord(containedPath(out, ctx.records.fresh), record);
  ensure(record.status === 'passed', record.errors.join('\n'));
  return { ...ctx, record, run };
}

export async function fresh(page, out, stage) {
  return (await verifyFresh(page, out, stage)).record;
}

function judgeScore(text) {
  for (const chunk of [text.trim(), ...text.split(/\r?\n/).reverse()]) {
    if (!chunk.trim().startsWith('{')) continue;
    try { const score = JSON.parse(chunk).score; if (typeof score === 'number' && Number.isFinite(score)) return score; }
    catch { /* progress output */ }
  }
  const verdicts = [...text.matchAll(/\bJUDGE (?:PASS|FAIL): (\d+(?:\.\d+)?)\/10\b/g)];
  if (verdicts.length) return Number(verdicts.at(-1)[1]);
  const scores = [...text.matchAll(/(?<![-\w])score\W{0,4}(\d+(?:\.\d+)?)/gi)];
  return scores.length ? Number(scores.at(-1)[1]) : null;
}

/** One judge process on one capture; no reference image is ever passed. */
export function judgeCapture(capture, description) {
  const client = process.env.NMC_JUDGE_CLIENT;
  ensure(nonblank(client), 'NMC_JUDGE_CLIENT is missing');
  return new Promise((resolveVerdict, rejectVerdict) => {
    const child = spawn('python3', [client, capture, '--description', description, '--min-score', '6'],
      { stdio: ['ignore', 'pipe', 'pipe'], shell: false });
    let stdout = '', stderr = '';
    child.stdout.on('data', (data) => { stdout += data; });
    child.stderr.on('data', (data) => { stderr += data; });
    child.on('error', (cause) => rejectVerdict(new Error(`judge process could not start: ${cause.message}`, { cause })));
    child.on('close', (code, signal) => {
      const output = [stdout, stderr].filter((text) => text.trim()).join('\n').trimEnd();
      resolveVerdict({ exit: typeof code === 'number' ? code : -1, score: judgeScore(`${stdout}\n${stderr}`),
        output: signal ? `${output}\njudge terminated by ${signal}`.trim() : output });
    });
  });
}

/**
 * P2-RESUMABLE-RENDER JUDGE reuse: an earlier passed verdict of the byte-identical capture with the
 * identical description (exit 0, minScore 6, score >= 6) from an attempt of the same build. The byte
 * equality only establishes that the reused verdict is about the same file. Null when none exists.
 */
function reusableVerdict(out, currentDirectory, key, row, capture) {
  let bytes;
  try { bytes = readRegularFile(capture); } catch { return null; }
  const sameBytes = (file) => { try { return readRegularFile(file).equals(bytes); } catch { return false; } };
  for (const directory of earlierAttempts(out, currentDirectory)) {
    try {
      const records = attemptRecords(directory);
      if (!sameBuild(readJsonRecord(containedPath(out, records.build)), key)) continue;
      for (const entry of readJsonRecord(containedPath(out, records.judge)).images) {
        const passed = entry?.id === row.id && entry.filename === row.filename
          && entry.description === row.expectedVisibleBehavior && entry.minScore === 6
          && entry.exit === 0 && typeof entry.score === 'number' && entry.score >= 6;
        if (passed && sameBytes(containedPath(out, records.image(row.filename)))) return { ...entry };
      }
    } catch { /* an unreadable attempt supplies no verdict */ }
  }
  return null;
}

/**
 * Judge every selected capture and retain every verdict before failing. A kept image reuses its
 * earlier passed verdict (reusableVerdict, judgedAt retained); new and previously failed images
 * are judged with the inventory behaviour and no reference image.
 */
export async function judge(page, out, stage, options = {}) {
  const runJudge = options.judge ?? judgeCapture;
  ensure(typeof runJudge === 'function', 'judge runner must be a function');
  const ctx = await resolveAttempt(page, out, stage);
  const record = {
    selectionId: ctx.selection.selectionId, attemptId: ctx.attempt.attemptId, buildId: ctx.build.buildId,
    images: [], status: 'failed', errors: [],
  };
  const write = () => writeJsonRecord(containedPath(out, ctx.records.judge), record);
  let verified;
  try {
    verified = await verifyFresh(page, out, stage);
  } catch (error) {
    record.errors.push(message(error));
    write();
    throw error;
  }
  const images = new Map(verified.run.images.map((image) => [image.id, image]));
  const kept = new Set(verified.run.kept.map((entry) => entry.id));
  const key = buildKey(verified.build);
  for (const row of verified.inventory) {
    const capture = containedPath(out, verified.records.image(row.filename));
    const reused = kept.has(row.id) ? reusableVerdict(out, verified.pointer.directory, key, row, capture) : null;
    if (reused) {
      record.images.push(reused);
      write();
      continue;
    }
    const entry = {
      id: row.id, filename: row.filename, description: row.expectedVisibleBehavior, minScore: 6,
      exit: -1, score: null, output: '', judgedAt: new Date().toISOString(),
    };
    try {
      const image = images.get(row.id);
      ensure(image && image.filename === row.filename, `selected image ${row.id} is missing from the run`);
      ensure(fs.existsSync(capture) && fs.lstatSync(capture).isFile(), `capture ${row.filename} is missing`);
      const verdict = await runJudge(capture, row.expectedVisibleBehavior);
      ensure(isRecord(verdict), `judge returned no verdict for ${row.filename}`);
      ensure(Number.isInteger(verdict.exit), `judge verdict for ${row.filename} has no exit code`);
      ensure(typeof verdict.output === 'string', `judge verdict for ${row.filename} has no output`);
      ensure(verdict.score === null || (typeof verdict.score === 'number' && Number.isFinite(verdict.score)),
        `judge verdict for ${row.filename} has an invalid score`);
      Object.assign(entry, { exit: verdict.exit, score: verdict.score, output: verdict.output });
      ensure(entry.exit === 0, `judge exited ${entry.exit}: ${entry.output}`);
      ensure(entry.score !== null, 'judge returned no parseable score verdict');
      ensure(entry.score >= 6, `judge score ${entry.score} is below the minimum 6`);
    } catch (error) {
      if (!entry.output) entry.output = message(error);
      record.errors.push(`${row.filename}: ${message(error)}`);
    }
    record.images.push(entry);
    write();
  }
  try {
    assertCurrentAttempt(verified, verified.run);
    ensure(record.images.length === verified.inventory.length,
      `judged ${record.images.length} of ${verified.inventory.length} selected images`);
  } catch (error) {
    record.errors.push(message(error));
  }
  record.status = record.errors.length === 0 ? 'passed' : 'failed';
  write();
  ensure(record.status === 'passed',
    `image judgment failures (${record.errors.length}):\n${record.errors.join('\n')}`);
  return record;
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
  } else if (action === 'fresh') {
    const record = await fresh(page, out, stage);
    process.stdout.write(`sky-weather-job: fresh ${stage}: attempt ${record.attemptId}: ${record.status}\n`);
  } else if (action === 'judge') {
    const record = await judge(page, out, stage);
    process.stdout.write(`sky-weather-job: judge ${stage}: ${record.images.length} image(s): ${record.status}\n`);
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
