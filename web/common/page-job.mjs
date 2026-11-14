// Shared page capture job protocol: verify the staged build, select the cumulative
// inventory, allocate immutable render attempts, render bounded managed jobs,
// verify actual GPU evidence and judge every selected capture.
import fs from 'node:fs';
import zlib from 'node:zlib';
import { spawn } from 'node:child_process';
import { publishGpuPage, runGpuPageOp } from './gpu-job.mjs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { isDeepStrictEqual, parseArgs as parse } from 'node:util';
import { containedPath, readJsonRecord, writeJsonRecord, assertCurrentInputs } from '../../examples/sky-weather/lib/evidence.mjs';

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
 * P2-RESUMABLE-RENDER: an image already rendered is kept by every later attempt of the stage, in
 * any worktree, while its inventory row (id, scenario, mode, filename, requiredFacts) is unchanged;
 * a page-code change alone does not invalidate it. render keeps a (scenario, mode) group from the
 * first earlier attempt directory (the stage root's attempts, then the store's; ascending name, any
 * status) whose selection holds the same rows, whose job passes the per-job checks against its
 * origin identities and whose images are intact, non-blank regular files; it copies that job, its
 * inputs/results and images into the new attempt, lists them in run.kept, renders only the other
 * groups and publishes only when it renders one. <envPrefix>_RERENDER_ALL=1 keeps nothing.
 * clean allocates a fresh attempt and deletes no record and no image.
 * P2-RESUMABLE-RENDER STORE: storeRoot() is $NMC_CACHE_DIR/<topic>/<stage>, the per-project
 * directory every worktree shares (the stage root itself when NMC_CACHE_DIR is unset). As soon as
 * a rendered job passes, render saves it there in the attempt layout (selection.json, build.json,
 * run.json listing only the jobs that attempt rendered and passed, jobs/<i>/, images/); judge
 * saves its verdicts there with the judged images. A failing job never discards the saved ones.
 * P2-RESUMABLE-RENDER ORDER: render dispatches the jobs whose last dispatch failed (the store's
 * failed-jobs.json, blank refusals included) first, then never-rendered jobs, then the rest.
 * P2-RESUMABLE-RENDER BLANK: a capture that is fully transparent or one flat colour (blankCapture)
 * fails render as soon as its job returns, before the next job is dispatched, and fails fresh (so
 * judge calls no judge client) listing every blank filename with its scenario/mode.
 * P2-RESUMABLE-RENDER FRESH: a job whose images are listed in run.kept passes the per-job checks
 * against its kept entries' origin identities, and that origin attempt (stage root or store) must
 * still hold its frozen build of that buildId, the group's rows and the identical job record (and
 * must not merely have copied the group). Every other job passes against the current identities.
 * Coverage, image bytes and pair coverage apply to the union; the two states of a pair may come
 * from different attempts.
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

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const PNG_SIGNATURE = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
/** The GPU canvas of every job; a capture is that canvas with the page status band below it. */
const CANVAS = Object.freeze({ width: 1280, height: 720 });
const ADAPTER_FIELDS = ['vendor', 'architecture', 'device', 'description'];
const SOFTWARE_ADAPTER = /software|swiftshader|llvmpipe|lavapipe|softpipe|fallback|basic render|microsoft basic|warp\b/i;
const FAILED_JOBS = 'failed-jobs.json';
const JOB_FIELDS = ['jobId', 'clientId', 'scenario', 'mode', 'result', 'error'];
const KEPT_FIELDS = ['id', 'filename', 'attemptId', 'selectionId', 'buildId'];

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
 * Read and verify dist/<topic>/build.json before any scene module import.
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

/** The build identity of a receipt: its inputs without the per-preparation id. */
export function buildKey(receipt) {
  const { preparationId: _preparationId, ...inputs } = receipt.inputs;
  return inputs;
}

/**
 * P2-RESUMABLE-RENDER BLANK: why a PNG capture shows nothing (fully transparent, or one flat
 * colour: every channel's standard deviation below one level), or null when it shows something.
 * Only the first `rows` rows are read: the GPU canvas, not the status text band the page draws
 * below it. A local, deterministic check of the decoded pixels; a layout the decoder does not
 * cover (interlaced, bit depth below 8) is not called blank here. Throws on an undecodable PNG.
 */
export function blankCapture(bytes, rows = CANVAS.height) {
  ensure(Buffer.isBuffer(bytes) && bytes.subarray(0, 8).equals(PNG_SIGNATURE), 'not a PNG capture');
  let header = null, palette = null, alphas = null;
  const data = [];
  for (let at = 8; at + 8 <= bytes.length;) {
    const length = bytes.readUInt32BE(at), type = bytes.toString('latin1', at + 4, at + 8);
    const body = bytes.subarray(at + 8, at + 8 + length);
    ensure(body.length === length, `truncated PNG ${type} chunk`);
    if (type === 'IHDR') header = { width: body.readUInt32BE(0), height: body.readUInt32BE(4), depth: body[8], color: body[9], interlace: body[12] };
    else if (type === 'PLTE') palette = body;
    else if (type === 'tRNS') alphas = body;
    else if (type === 'IDAT') data.push(body);
    else if (type === 'IEND') break;
    at += 12 + length;
  }
  ensure(header && header.width > 0 && header.height > 0, 'PNG has no image header');
  const channels = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 }[header.color];
  ensure(channels !== undefined, `unknown PNG colour type ${header.color}`);
  if (header.interlace !== 0 || (header.depth !== 8 && header.depth !== 16)) return null;
  ensure(header.color !== 3 || palette, 'palette PNG has no palette');
  const raw = zlib.inflateSync(Buffer.concat(data));
  const step = channels * (header.depth / 8), stride = header.width * step;
  ensure(raw.length >= (stride + 1) * header.height, 'PNG pixel data is truncated');
  let previous = new Uint8Array(stride), current = new Uint8Array(stride);
  const sum = [0, 0, 0], squares = [0, 0, 0];
  let visible = 0;
  for (let y = 0; y < Math.min(header.height, rows); y++) {
    const line = (stride + 1) * y, filter = raw[line];
    for (let x = 0; x < stride; x++) {
      const left = x >= step ? current[x - step] : 0, up = previous[x], corner = x >= step ? previous[x - step] : 0;
      let predicted = 0;
      if (filter === 1) predicted = left;
      else if (filter === 2) predicted = up;
      else if (filter === 3) predicted = (left + up) >> 1;
      else if (filter === 4) {
        const p = left + up - corner, pa = Math.abs(p - left), pb = Math.abs(p - up), pc = Math.abs(p - corner);
        predicted = pa <= pb && pa <= pc ? left : pb <= pc ? up : corner;
      } else ensure(filter === 0, `unknown PNG filter ${filter}`);
      current[x] = (raw[line + 1 + x] + predicted) & 255;
    }
    for (let x = 0; x < stride; x += step) {
      const sample = (index) => current[x + index * (header.depth / 8)];
      let rgb, alpha = 255;
      if (header.color === 3) {
        const index = current[x];
        rgb = [palette[index * 3] ?? 0, palette[index * 3 + 1] ?? 0, palette[index * 3 + 2] ?? 0];
        if (alphas && index < alphas.length) alpha = alphas[index];
      } else if (header.color === 0 || header.color === 4) {
        rgb = [sample(0), sample(0), sample(0)];
        if (header.color === 4) alpha = sample(1);
      } else {
        rgb = [sample(0), sample(1), sample(2)];
        if (header.color === 6) alpha = sample(3);
      }
      if (alpha === 0) continue;
      visible += 1;
      for (let c = 0; c < 3; c++) { sum[c] += rgb[c]; squares[c] += rgb[c] * rgb[c]; }
    }
    [previous, current] = [current, previous];
  }
  if (visible === 0) return 'fully transparent';
  const mean = sum.map((value) => value / visible);
  const deviation = Math.max(...squares.map((value, c) => Math.sqrt(Math.max(0, value / visible - mean[c] * mean[c]))));
  if (deviation >= 1) return null;
  return `one flat colour rgb(${mean.map((value) => Math.round(value)).join(', ')})`;
}

/** Throws one error naming every blank capture of the given {row, bytes} list. */
function refuseBlank(captures, context) {
  const blank = [];
  for (const { row, bytes } of captures) {
    let reason;
    try { reason = blankCapture(bytes); } catch (error) { reason = `not a decodable PNG: ${message(error)}`; }
    if (reason) blank.push(`${row.filename} (${row.scenario}/${row.mode}): ${reason}`);
  }
  ensure(blank.length === 0, `${context}: blank capture(s): ${blank.join('; ')}`);
}

function readRegularFile(file) {
  const descriptor = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW);
  try {
    ensure(fs.fstatSync(descriptor).isFile(), `${file} is not a regular file`);
    return fs.readFileSync(descriptor);
  } finally { fs.closeSync(descriptor); }
}

/** The inventory fields an earlier image must share with the selected row to be kept. */
function sameRow(a, b) {
  return ['id', 'scenario', 'mode', 'filename', 'requiredFacts'].every((key) => isDeepStrictEqual(a?.[key], b?.[key]));
}

function sameJob(a, b) { return a?.scenario === b?.scenario && a?.mode === b?.mode; }

/** Earlier attempt directories of every root, each root in ascending directory name order. */
function earlierAttempts(roots, currentDirectory) {
  const found = [];
  for (const root of roots) {
    let names = [];
    try {
      const attempts = containedPath(root, 'attempts');
      if (fs.existsSync(attempts)) names = fs.readdirSync(attempts).sort();
    } catch { /* an unreadable root supplies nothing */ }
    for (const name of names) {
      const directory = `attempts/${name}`;
      if (directory === currentDirectory || !isUuid(name)) continue;
      found.push({ root, directory });
    }
  }
  return found;
}

/** Readable records of earlier attempts. */
function keptCandidates(roots, currentDirectory) {
  const candidates = [];
  for (const { root, directory } of earlierAttempts(roots, currentDirectory)) {
    try {
      const records = attemptRecords(directory);
      const build = readJsonRecord(containedPath(root, records.build));
      candidates.push({ root, directory, records, build, selection: readJsonRecord(containedPath(root, records.selection)),
        run: readJsonRecord(containedPath(root, records.run)) });
    } catch { /* an unreadable attempt supplies nothing */ }
  }
  return candidates;
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

/** Run images must exactly cover the selected inventory and match the direct payloads. */
function validateRunImages(ctx, run, state) {
  ensure(run.images.length === ctx.inventory.length,
    `run records ${run.images.length} image(s) for ${ctx.inventory.length} selected image(s)`);
  const byId = new Map(), filenames = new Set(), captures = [];
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
    captures.push({ row, bytes: capture.bytes });
  }
  for (const row of ctx.inventory) ensure(byId.has(row.id), `selected image ${row.id} is missing from the run`);
  // Every blank capture of the selection is named before any judge client is called.
  refuseBlank(captures, 'fresh');
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

export function judgeScore(text) {
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

/**
 * A judge process still running this long after the judge budget ended is killed, so one hung judge
 * cannot hold the step until its command timeout (<envPrefix>_JUDGE_GRACE_MS or 150 s).
 */
export const JUDGE_GRACE_MS = 150000;

/** The judge aspects a row's reference may be scored by (inventory `judgeAspect`). */
export const JUDGE_ASPECTS = Object.freeze(['look', 'functional', 'design']);

/**
 * One judge process on one capture. A row without a reference passes no `--reference` or
 * `--aspect`, as before. `reference` is an absolute .png path that must exist when the judge
 * starts (`judge: reference <path> is missing` otherwise, and no process starts).
 */
export function judgeCapture(capture, description, { killAt, heldConstant, reference, aspect } = {}) {
  const client = process.env.NMC_JUDGE_CLIENT;
  ensure(nonblank(client), 'NMC_JUDGE_CLIENT is missing');
  if (heldConstant !== undefined) {
    ensure(Array.isArray(heldConstant) && heldConstant.every(nonblank),
      'judge: heldConstant must be an array of fixture names');
  }
  if (reference !== undefined) {
    ensure(nonblank(reference) && path.isAbsolute(reference) && reference.endsWith('.png'),
      'judge: reference must be an absolute .png path');
    ensure(referenceExists(reference), `judge: reference ${reference} is missing`);
  }
  ensure(aspect === undefined || JUDGE_ASPECTS.includes(aspect),
    'judge: aspect must be look, functional or design');
  const args = [client, capture, '--description', description];
  for (const fixture of heldConstant ?? []) args.push('--held-constant', fixture);
  if (reference !== undefined) args.push('--reference', reference);
  if (aspect !== undefined) args.push('--aspect', aspect);
  args.push('--min-score', '6');
  return new Promise((resolveVerdict, rejectVerdict) => {
    const child = spawn('python3', args,
      { stdio: ['ignore', 'pipe', 'pipe'], shell: false });
    let killed = false;
    const timer = Number.isFinite(killAt) ? setTimeout(() => { killed = true; child.kill('SIGKILL'); },
      Math.max(0, killAt - Date.now())) : null;
    child.on('exit', () => { if (timer) clearTimeout(timer); });
    let stdout = '', stderr = '';
    child.stdout.on('data', (data) => { stdout += data; });
    child.stderr.on('data', (data) => { stderr += data; });
    child.on('error', (cause) => rejectVerdict(new Error(`judge process could not start: ${cause.message}`, { cause })));
    child.on('close', (code, signal) => {
      const output = [stdout, stderr].filter((text) => text.trim()).join('\n').trimEnd();
      const ended = killed ? 'judge killed: still running when the judge budget and its grace ended'
        : signal ? `judge terminated by ${signal}` : '';
      resolveVerdict({ exit: typeof code === 'number' && !killed ? code : -1,
        score: killed ? null : judgeScore(`${stdout}\n${stderr}`),
        output: ended ? `${output}\n${ended}`.trim() : output });
    });
  });
}

/** Whether a judge reference exists as a regular file. */
function referenceExists(reference) {
  try { return fs.lstatSync(reference).isFile(); } catch { return false; }
}

/** Captures judged at once: <envPrefix>_JUDGE_CONCURRENCY (1..16) or 4. */
export const JUDGE_CONCURRENCY = 4;

/**
 * Judge calls start only within this budget (--budget-s, <envPrefix>_JUDGE_BUDGET_MS or 480 s), so
 * the step ends before its command timeout with every verdict it has written.
 */
export const JUDGE_BUDGET_MS = 480000;

/**
 * Whether a verdict was judged with the row's reference and aspect. An entry written before the
 * keys existed counts as null, so it only matches a row without a reference.
 */
function judgeSame(entry, row) {
  return (entry.reference ?? null) === (row.judgeReference ?? null)
    && (entry.aspect ?? null) === (row.judgeAspect ?? null);
}

/**
 * P2-RESUMABLE-RENDER JUDGE reuse: an earlier passed verdict of the byte-identical capture with the
 * identical description (exit 0, minScore 6, score >= 6) from an earlier attempt (stage root or
 * store). The byte equality only establishes that the reused verdict is about the same file. Null
 * when none exists.
 */
function reusableVerdict(roots, currentDirectory, row, capture) {
  let bytes;
  try { bytes = readRegularFile(capture); } catch { return null; }
  const sameBytes = (file) => { try { return readRegularFile(file).equals(bytes); } catch { return false; } };
  for (const { root, directory } of earlierAttempts(roots, currentDirectory)) {
    try {
      const records = attemptRecords(directory);
      for (const entry of readJsonRecord(containedPath(root, records.judge)).images) {
        const passed = entry?.id === row.id && entry.filename === row.filename
          && entry.description === row.expectedVisibleBehavior && entry.minScore === 6
          && judgeSame(entry, row) && entry.exit === 0 && typeof entry.score === 'number' && entry.score >= 6;
        if (passed && sameBytes(containedPath(root, records.image(row.filename)))) return { ...entry };
      }
    } catch { /* an unreadable attempt supplies no verdict */ }
  }
  return null;
}

/**
 * Build a page capture job from one configuration of the page-specific literals.
 * `name` is the file stem and output-line prefix, `topic` the store and page directory name,
 * `stages` the cumulative stage list in order, `envPrefix` the environment-name prefix, and the
 * validators/measurement checker come from the page's own modules.
 */
export function createPageJob(config) {
  validateConfig(config);
  const STAGES = Object.freeze([...config.stages]);
  const USAGE = `usage: node web/common/${config.name}.mjs <check|select|clean|render|fresh|judge> --page dist/${config.topic} --out out/${config.topic}/<stage> --stage <${config.stages.join('|')}>`;

  function validateConfig(value) {
    ensure(isRecord(value), 'page job config: config: must be an object');
    for (const field of ['name', 'topic', 'envPrefix']) {
      ensure(nonblank(value[field]), `page job config: ${field}: must be a nonblank string`);
    }
    ensure(Array.isArray(value.stages) && value.stages.length > 0,
      'page job config: stages: must be a non-empty array');
    ensure(value.stages.every(nonblank), 'page job config: stages: must contain only nonblank strings');
    ensure(new Set(value.stages).size === value.stages.length, 'page job config: stages: must be distinct');
    ensure(typeof value.validators === 'function', 'page job config: validators: must be a function');
    ensure(typeof value.validateMeasurement === 'function', 'page job config: validateMeasurement: must be a function');
    ensure(value.report === undefined || typeof value.report === 'function',
      'page job config: report: must be a function');
    for (const field of ['keepSameBuildOnly', 'judgeHeldConstant']) {
      ensure(value[field] === undefined || typeof value[field] === 'boolean',
        `page job config: ${field}: must be a boolean`);
    }
  }

  /**
   * The stdout lines of the default report: the six forms the sky-weather CLI printed before the
   * report option existed. `result` is the value the action returned, `inventory` the stage's rows.
   */
  function defaultReport(action, stage, result) {
    if (action === 'check') return [`${config.name}: check ${stage}: ${result.length} image(s)`];
    if (action === 'select') {
      return [`${config.name}: select ${stage}: ${result.inventory.length} image(s), selection ${result.selectionId}`];
    }
    if (action === 'render') return [`${config.name}: render ${stage}: attempt ${result.attemptId}: ${result.status}`];
    if (action === 'fresh') return [`${config.name}: fresh ${stage}: attempt ${result.attemptId}: ${result.status}`];
    if (action === 'judge') return [`${config.name}: judge ${stage}: ${result.images.length} image(s): ${result.status}`];
    return [`${config.name}: clean ${stage}: attempt ${result.attemptId} at ${result.directory}`];
  }

  const report = config.report ?? defaultReport;
  // The two behaviours S33 asks of the virtual-geometry job; both are off for the sky-weather job.
  const keepSameBuildOnly = config.keepSameBuildOnly === true;
  const judgeHeldConstant = config.judgeHeldConstant === true;

  /** The frozen build receipt of an earlier attempt, looked up in the stage root and then the store. */
  function originBuild(roots, attemptId) {
    const records = attemptRecords(`attempts/${attemptId}`);
    for (const root of roots) {
      const file = containedPath(root, records.build);
      if (!fs.existsSync(file)) continue;
      try { return readJsonRecord(file); } catch { return null; }
    }
    return null;
  }

  /** The inventory of `<out>/selection.json`, read after the action (check returns its own rows). */
  function selectionInventory(out) {
    const file = containedPath(path.resolve(out), 'selection.json');
    const selection = readJsonRecord(file);
    ensure(Array.isArray(selection.inventory), `${file}: inventory must be an array`);
    return selection.inventory;
  }

  /** Import only modules listed by the delivered scenes.json. */
  async function loadScenes(page, build) {
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
  async function verify(page, stage) {
    ensure(STAGES.includes(stage), `invalid stage: ${stage ?? '<missing>'}`);
    page = path.resolve(page);
    const build = readBuildReceipt(page);
    const namespaces = await loadScenes(page, build);
    const validators = await config.validators(page);
    const inventory = validators.validateInventory(namespaces, stage);
    return { page, build, namespaces, inventory, validators };
  }

  /** Build receipt, delivered namespaces and the browser-free cumulative inventory. */
  async function verifyPage(page, stage) {
    const { page: resolved, build, namespaces, inventory } = await verify(page, stage);
    return { page: resolved, build, namespaces, inventory };
  }

  async function check(page, stage) {
    return (await verify(page, stage)).inventory;
  }

  async function select(page, out, stage) {
    const { build, inventory } = await verify(page, stage);
    const selection = {
      selectionId: randomUUID(), stage, buildId: build.buildId, preparationId: build.preparationId,
      inputs: build.inputs, inventory,
    };
    writeJsonRecord(containedPath(out, 'selection.json'), selection);
    return selection;
  }

  /** P2-RESUMABLE-RENDER STORE root of a stage: shared by every worktree of the project. */
  function storeRoot(out, stage) {
    const cache = process.env.NMC_CACHE_DIR;
    return nonblank(cache) ? path.join(path.resolve(cache), config.topic, stage) : path.resolve(out);
  }

  /** The roots holding earlier attempts: the stage root, then the store when it is elsewhere. */
  function attemptRoots(out, stage) {
    const roots = [path.resolve(out), storeRoot(out, stage)];
    return roots[0] === roots[1] ? [roots[0]] : roots;
  }

  /** P2-RESUMABLE-RENDER ORDER: the store's record of the jobs whose last dispatch failed. */
  function failedJobs(store) {
    try {
      const jobs = readJsonRecord(containedPath(store, FAILED_JOBS)).jobs;
      return Array.isArray(jobs) ? jobs.filter((job) => nonblank(job?.scenario) && nonblank(job?.mode)) : [];
    } catch { return []; }
  }
  function saveFailedJobs(store, jobs) {
    try { writeJsonRecord(containedPath(store, FAILED_JOBS), { jobs }); }
    catch (error) { process.stderr.write(`${config.name}: failed jobs not recorded in ${store}: ${message(error)}\n`); }
  }

  /**
   * P2-RESUMABLE-RENDER STORE: save one rendered job that passed, with its input, result and images,
   * under the store's copy of the attempt. Nothing to do when the store is the stage root itself.
   */
  function storeJob(store, ctx, job, input, images) {
    if (store === path.resolve(ctx.out)) return;
    const records = attemptRecords(ctx.directory);
    const write = (name, value) => writeJsonRecord(containedPath(store, name), value);
    if (!ctx.stored) {
      write(records.selection, ctx.selection);
      write(records.build, ctx.build);
      ctx.stored = { ...ctx.run, jobs: [], images: [], kept: [], status: 'passed', errors: [] };
    }
    const index = ctx.stored.jobs.length;
    write(records.input(index), input);
    write(records.result(index), job.result);
    for (const { image, bytes } of images) {
      const file = containedPath(store, records.image(image.filename));
      fs.mkdirSync(path.dirname(file), { recursive: true });
      fs.writeFileSync(file, bytes, { flag: 'wx' });
    }
    ctx.stored.jobs.push(job);
    ctx.stored.images.push(...images.map(({ image }) => image));
    ctx.stored.pageRef = ctx.run.pageRef;
    write(records.run, ctx.stored);
  }

  /**
   * KEPT GROUP: the first earlier attempt whose evidence of this group passes every check. With
   * keepSameBuildOnly the attempt that produced the group must also hold the current build key; a
   * candidate that only copied the group is compared through the origin its kept entries name.
   */
  function keptGroup(candidates, rows, namespaces, validators, options) {
    const { scenario, mode } = rows[0];
    const factNames = namespaces.find((entry) => entry.default?.id === scenario)?.factNames;
    for (const candidate of candidates) {
      try {
        const { selection, run, records, root } = candidate;
        ensure(rows.every((row) => selection.inventory.some((entry) => sameRow(entry, row))), 'changed rows');
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
        if (options.keepSameBuildOnly) {
          const own = `attempts/${origin.attemptId}` === candidate.directory;
          const build = own ? candidate.build : originBuild(options.roots, origin.attemptId);
          ensure(build && isDeepStrictEqual(buildKey(build), buildKey(options.build)), 'other sources');
        }
        const captures = checkJob(job, `${candidate.directory} job ${index}`, rows, origin, namespaces, factNames,
          new Map(), validators);
        const images = rows.map((row) => {
          const image = run.images.find((entry) => entry?.id === row.id && entry.filename === row.filename);
          ensure(image, 'no image entry');
          const bytes = readRegularFile(containedPath(root, records.image(row.filename)));
          ensure(bytes.equals(captures.get(row.id).bytes), 'damaged image');
          return { row, image, bytes };
        });
        refuseBlank(images, 'kept group');
        const input = readJsonRecord(containedPath(root, records.input(index)));
        return { candidate, job, input, origin, images };
      } catch { /* not kept: rendered again */ }
    }
    return null;
  }

  /** Allocate a fresh immutable attempt; no earlier record or image is ever deleted. */
  async function clean(page, out, stage) {
    const { build, inventory } = await verify(page, stage);
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

  /**
   * Render once; transport owns queue, running and download clocks independently. Groups kept
   * from an earlier attempt of the same build are copied, not rendered (P2-RESUMABLE-RENDER).
   */
  async function render(page, out, stage, transport = { publishGpuPage, runGpuPageOp }) {
    const { build, namespaces, inventory, validators } = await verify(page, stage);
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
    const store = storeRoot(out, stage);
    const storeCtx = { out, directory: pointer.directory, selection, build, run, stored: null };
    let failed = failedJobs(store), dispatched = null;
    const rendered = new Set();
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
      const roots = attemptRoots(out, stage);
      const candidates = process.env[`${config.envPrefix}_RERENDER_ALL`] === '1' ? []
        : keptCandidates(roots, pointer.directory);
      const keepOptions = { keepSameBuildOnly, build, roots };
      const pending = [];
      let firstKept = null;
      for (const rows of groups.values()) {
        const kept = keptGroup(candidates, rows, namespaces, validators, keepOptions);
        if (!kept) {
          pending.push(rows);
          if (!failed.some((entry) => sameJob(entry, rows[0]))
            && candidates.some((candidate) => Array.isArray(candidate.run?.jobs)
              && candidate.run.jobs.some((job) => sameJob(job, rows[0])))) rendered.add(rows);
          continue;
        }
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
      // P2-RESUMABLE-RENDER ORDER: the jobs that failed last time first (a still-broken job fails the
      // stage before other GPU time is spent), then the never-rendered ones, then the rest.
      const rank = (rows) => (failed.some((entry) => sameJob(entry, rows[0])) ? 0 : rendered.has(rows) ? 2 : 1);
      pending.sort((a, b) => rank(a) - rank(b));
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
          scenario, mode, width: CANVAS.width, height: CANVAS.height, remainingMs: 270000 };
        write(records.input(index), input);
        const job = { jobId: '', clientId: '', scenario, mode, result: null, error: null };
        run.jobs.push(job);
        dispatched = { scenario, mode };
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
        const captures = rows.map((row) => ({ row, image: run.images.find((image) => image.id === row.id),
          bytes: readRegularFile(containedPath(out, records.image(row.filename))) }));
        // A blank capture fails at once: no further GPU job is dispatched for this attempt.
        refuseBlank(captures, `${scenario}/${mode}`);
        for (const kind of ['console', 'page', 'gpu']) {
          ensure(Array.isArray(job.result.errors?.[kind]), `missing ${kind} error observations`);
          run.errors.push(...job.result.errors[kind].map((error) => `${kind}: ${error}`));
        }
        write(records.run, run);
        ensure(run.errors.length === 0, run.errors.join('\n'));
        try {
          storeJob(store, storeCtx, job, input, captures);
        } catch (error) {
          process.stderr.write(`${config.name}: ${scenario}/${mode} passed but was not saved to ${store}: ${message(error)}\n`);
        }
        dispatched = null;
        if (failed.some((entry) => sameJob(entry, job))) {
          failed = failed.filter((entry) => !sameJob(entry, job));
          saveFailedJobs(store, failed);
        }
      }
      current();
      run.status = 'passed';
    } catch (error) {
      if (!run.errors.includes(message(error))) run.errors.push(message(error));
      if (dispatched) {
        failed = [{ ...dispatched, attemptId: attempt.attemptId, error: message(error) },
          ...failed.filter((entry) => !sameJob(entry, dispatched))];
        saveFailedJobs(store, failed);
      }
    }
    write(records.run, run);
    write(records.attempt, { ...attempt, status: run.status, error: run.errors.length ? run.errors.join('\n') : null });
    ensure(run.status === 'passed', run.errors.join('\n'));
    return run;
  }

  /** Resolve the stage pointer to the frozen attempt and its current page build. */
  async function resolveAttempt(page, out, stage) {
    const { build, namespaces, inventory, validators } = await verify(page, stage);
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
    return { page, out, stage, build, namespaces, inventory, validators, pointer, records, selection, attempt };
  }

  /** Exact measurement records declared by the selected variant's measurement. */
  function validateMeasurements(declared, result, label, seen) {
    const expected = declared ? [declared] : [];
    ensure(Array.isArray(result.measurements), `${label}: measurements must be an array`);
    ensure(result.measurements.length === expected.length,
      `${label}: expected ${expected.length} measurement record(s), found ${result.measurements.length}`);
    for (const measurement of result.measurements) {
      config.validateMeasurement(measurement);
      const spec = expected.find((entry) => entry.pair === measurement.pair && entry.state === measurement.state);
      ensure(spec && measurement.preset === spec.preset && isDeepStrictEqual(measurement.features, spec.features),
        `${label}: measurement differs from the declared ${spec?.pair ?? 'unknown'} ${spec?.state ?? ''} record`);
      const key = `${measurement.pair}\0${measurement.state}`;
      ensure(!seen.has(key), `${label}: duplicate measurement ${measurement.pair} ${measurement.state}`);
      seen.set(key, measurement);
    }
  }

  /**
   * P2-RESUMABLE-RENDER PER-JOB CHECKS of one managed job for its (scenario, mode) group
   * against the given identities {selectionId, attemptId, buildId}. Returns the decoded
   * captures by image id; `seen` collects measurements keyed by pair and state.
   */
  function checkJob(job, label, rows, identities, namespaces, factNames, seen, validators) {
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
      validators.validateRequiredFacts(row, result.facts, factNames);
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
    const roots = attemptRoots(ctx.out, ctx.stage);
    const root = roots.find((entry) => fs.existsSync(containedPath(entry, records.build))) ?? roots[0];
    const read = (name) => {
      const file = containedPath(root, name);
      ensure(fs.existsSync(file), `${label}: kept origin attempt ${origin.attemptId} is missing ${path.basename(name)}`);
      return readJsonRecord(file);
    };
    const build = read(records.build);
    ensure(build.buildId === origin.buildId,
      `${label}: kept origin attempt ${origin.attemptId} belongs to another build`);
    if (keepSameBuildOnly) {
      ensure(isDeepStrictEqual(buildKey(build), buildKey(ctx.build)),
        `${label}: kept origin attempt ${origin.attemptId} was built from other sources`);
    }
    const selection = read(records.selection);
    ensure(selection.selectionId === origin.selectionId
      && rows.every((row) => selection.inventory.some((entry) => sameRow(entry, row))),
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
      state.measurements, ctx.validators);
    state.jobs.set(key, { job, captures });
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
  async function verifyFresh(page, out, stage) {
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

  async function fresh(page, out, stage) {
    return (await verifyFresh(page, out, stage)).record;
  }

  function judgeGraceMs() {
    const raw = process.env[`${config.envPrefix}_JUDGE_GRACE_MS`] ?? JUDGE_GRACE_MS;
    const value = Number(raw);
    ensure(Number.isFinite(value) && value >= 0, `invalid judge grace: ${raw}`);
    return value;
  }

  function judgeConcurrency(requested) {
    const raw = requested ?? process.env[`${config.envPrefix}_JUDGE_CONCURRENCY`] ?? JUDGE_CONCURRENCY;
    const value = Number(raw);
    ensure(Number.isInteger(value) && value >= 1 && value <= 16, `invalid judge concurrency: ${raw}`);
    return value;
  }

  function judgeBudgetMs(requested) {
    const raw = requested ?? process.env[`${config.envPrefix}_JUDGE_BUDGET_MS`] ?? JUDGE_BUDGET_MS;
    const value = Number(raw);
    ensure(Number.isFinite(value) && value >= 0, `invalid judge budget: ${raw}`);
    return value;
  }

  /**
   * Judge every selected capture and retain every verdict before failing. An image reuses an earlier
   * passed verdict of its identical bytes and description (its own attempt's, or reusableVerdict's,
   * judgedAt retained); the others are judged with the inventory behaviour and no reference image.
   * Images left when the budget ends are named, and the next run judges only those.
   */
  async function judge(page, out, stage, options = {}) {
    const runJudge = options.judge ?? judgeCapture;
    ensure(typeof runJudge === 'function', 'judge runner must be a function');
    const ctx = await resolveAttempt(page, out, stage);
    const record = {
      selectionId: ctx.selection.selectionId, attemptId: ctx.attempt.attemptId, buildId: ctx.build.buildId,
      images: [], status: 'failed', errors: [],
    };
    let earlier = [];
    try { earlier = readJsonRecord(containedPath(out, ctx.records.judge)).images ?? []; } catch { /* first judgment */ }
    if (!Array.isArray(earlier)) earlier = [];
    const store = storeRoot(out, stage);
    const saveVerdicts = () => {
      // P2-RESUMABLE-RENDER STORE: the verdicts and the judged images, for attempts in other worktrees.
      if (store === path.resolve(out)) return;
      try {
        for (const entry of record.images) {
          const file = containedPath(store, ctx.records.image(entry.filename));
          if (fs.existsSync(file)) continue;
          fs.mkdirSync(path.dirname(file), { recursive: true });
          fs.writeFileSync(file, readRegularFile(containedPath(out, ctx.records.image(entry.filename))), { flag: 'wx' });
        }
        writeJsonRecord(containedPath(store, ctx.records.judge), record);
      } catch (error) {
        process.stderr.write(`${config.name}: verdicts not saved to ${store}: ${message(error)}\n`);
      }
    };
    const write = () => { writeJsonRecord(containedPath(out, ctx.records.judge), record); saveVerdicts(); };
    let verified;
    try {
      verified = await verifyFresh(page, out, stage);
    } catch (error) {
      record.errors.push(message(error));
      write();
      throw error;
    }
    const images = new Map(verified.run.images.map((image) => [image.id, image]));
    const roots = attemptRoots(out, stage);
    const budgetMs = judgeBudgetMs(options.budgetMs);
    const deadline = Date.now() + budgetMs;
    const unjudged = [];
    // A passed verdict this attempt already wrote is about its own immutable capture.
    const ownVerdict = (row) => {
      const entry = earlier.find((value) => value?.id === row.id);
      return entry && entry.filename === row.filename && entry.description === row.expectedVisibleBehavior
        && entry.minScore === 6 && judgeSame(entry, row) && entry.exit === 0
        && typeof entry.score === 'number' && entry.score >= 6 ? { ...entry } : null;
    };
    const log = options.log ?? ((line) => process.stdout.write(`${line}\n`));
    const total = verified.inventory.length;
    // Verdicts land in inventory order whatever order the judges finish in; only finished slots are written.
    const slots = new Array(total).fill(null);
    const errors = new Array(total).fill(null);
    const settle = (index, entry, error, reusedVerdict = false) => {
      slots[index] = entry;
      errors[index] = error;
      record.images = slots.filter(Boolean);
      record.errors = errors.filter(Boolean);
      write();
      const done = record.images.length;
      log(`${config.name}: judge ${stage}: [${done}/${total}] ${entry.filename}: `
        + (error ? `FAIL ${error}` : `pass ${entry.score}${reusedVerdict ? ' (reused verdict)' : ''}`).split('\n')[0]);
    };
    const judgeRow = async (row, index) => {
      const capture = containedPath(out, verified.records.image(row.filename));
      const reused = ownVerdict(row) ?? reusableVerdict(roots, verified.pointer.directory, row, capture);
      if (reused) {
        settle(index, reused, null, true);
        return;
      }
      if (Date.now() >= deadline) {
        unjudged.push(row.filename);
        return;
      }
      const entry = {
        id: row.id, filename: row.filename, description: row.expectedVisibleBehavior, minScore: 6,
        reference: row.judgeReference ?? null, aspect: row.judgeAspect ?? null,
        exit: -1, score: null, output: '', judgedAt: new Date().toISOString(),
      };
      let failure = null;
      try {
        const image = images.get(row.id);
        ensure(image && image.filename === row.filename, `selected image ${row.id} is missing from the run`);
        ensure(fs.existsSync(capture) && fs.lstatSync(capture).isFile(), `capture ${row.filename} is missing`);
        const killAt = deadline + judgeGraceMs();
        if (row.judgeReference !== undefined) {
          // Checked before any judge process starts, whatever runner judges.
          ensure(referenceExists(row.judgeReference), `judge: reference ${row.judgeReference} is missing`);
        }
        // Rows without the reference fields pass exactly the options they always passed.
        const verdict = await runJudge(capture, row.expectedVisibleBehavior, {
          killAt,
          ...(judgeHeldConstant ? { heldConstant: row.heldConstant ?? [] } : {}),
          ...(row.judgeReference === undefined ? {} : { reference: row.judgeReference }),
          ...(row.judgeAspect === undefined ? {} : { aspect: row.judgeAspect }),
        });
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
        failure = `${row.filename}: ${message(error)}`;
      }
      settle(index, entry, failure);
    };
    // Each capture is judged by an independent process (several model votes each); judging the
    // cumulative weather inventory one by one does not fit the judge entry's budget, so a bounded
    // pool judges JUDGE_CONCURRENCY captures at a time.
    const concurrency = judgeConcurrency(options.concurrency);
    let next = 0;
    const worker = async () => {
      while (next < total) {
        const index = next++;
        await judgeRow(verified.inventory[index], index);
      }
    };
    await Promise.all(Array.from({ length: Math.min(concurrency, total) }, worker));
    if (unjudged.length > 0) {
      record.errors.push(`${unjudged.length} image(s) not judged within the ${budgetMs / 1000} s judge budget: `
        + `${unjudged.join(', ')}; every verdict above is saved and the next judge run judges only the rest`);
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

  function parseArgs(argv) {
    const { values, positionals } = parse({ args: argv, allowPositionals: true,
      options: { page: { type: 'string' }, out: { type: 'string' }, stage: { type: 'string' },
        'budget-s': { type: 'string' } } });
    ensure(positionals.length === 1 && ACTIONS.includes(positionals[0]), USAGE);
    const action = positionals[0];
    ensure(nonblank(values.page), USAGE);
    ensure(STAGES.includes(values.stage), `invalid stage: ${values.stage ?? '<missing>'}`);
    if (action === 'check' && values.out === undefined) return { action, page: values.page, out: undefined, stage: values.stage };
    ensure(nonblank(values.out), USAGE);
    const parsed = { action, page: values.page, out: values.out, stage: values.stage };
    if (values['budget-s'] !== undefined) {
      ensure(action === 'judge', '--budget-s applies to judge only');
      parsed.budgetMs = judgeBudgetMs(Number(values['budget-s']) * 1000);
    }
    return parsed;
  }

  async function main(argv = process.argv.slice(2)) {
    const { action, page, out, stage, budgetMs } = parseArgs(argv);
    let result;
    if (action === 'check') result = await check(page, stage);
    else if (action === 'select') result = await select(page, out, stage);
    else if (action === 'render') result = await render(page, out, stage);
    else if (action === 'fresh') result = await fresh(page, out, stage);
    else if (action === 'judge') result = await judge(page, out, stage, { budgetMs });
    else result = await clean(page, out, stage);
    const inventory = action === 'check' ? result : selectionInventory(out);
    const lines = report(action, stage, result, inventory);
    ensure(Array.isArray(lines) && lines.every((line) => typeof line === 'string'),
      `${config.name}: report must return an array of lines`);
    for (const line of lines) process.stdout.write(`${line}\n`);
  }

  const pageJob = {
    STAGES, ACTIONS, RECORD_FIELDS, USAGE, attemptRecords, repositoryRoot, readBuildReceipt, buildKey,
    loadScenes, verifyPage, check, select, storeRoot, clean, render, verifyFresh, fresh, judge,
    parseArgs, main,
  };
  Object.defineProperty(pageJob, 'name', { value: config.name, enumerable: true });
  return pageJob;
}

/**
 * Entry guard of a page job CLI: when `process.argv[1]` resolves to the file of `metaUrl`, run the
 * job's `main`; a rejection writes `<name>: <message>` to stderr and exits with code 1. Otherwise it
 * does nothing. A thin configuration ends its file with `runCli(job, import.meta.url)`.
 */
export function runCli(job, metaUrl) {
  const entry = process.argv[1];
  if (!nonblank(entry)) return;
  if (metaUrl !== pathToFileURL(path.resolve(entry)).href) return;
  job.main().catch((error) => {
    process.stderr.write(`${job.name}: ${message(error)}\n`);
    process.exit(1);
  });
}

