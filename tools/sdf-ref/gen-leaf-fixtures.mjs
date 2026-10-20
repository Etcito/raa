// One-off generator of the S8 SDF reference fixture (crates/raa-prims/tests/fixtures/sdf-leaf-reference.json).
// It evaluates SDF Modeler's own leafGeometry maths, read-only through the sdfm alias checkout, at every RAA quality
// with the count rules of mapping v1 (tools/sdf-ref/README.md §4). Never imported by the engine; never writes inside
// the reference checkout; never writes a partial fixture.
//
//   node tools/sdf-ref/gen-leaf-fixtures.mjs --ref <sdfm checkout root> --out <fixture>
import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { registerHooks } from 'node:module'
import { dirname, isAbsolute, relative, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const REGENERATE = 'node tools/sdf-ref/gen-leaf-fixtures.mjs --ref <sdfm checkout root> --out crates/raa-prims/tests/fixtures/sdf-leaf-reference.json'
const USAGE = 'gen-leaf-fixtures.mjs --ref <sdfm checkout root> --out <fixture path>'
const GENERATED_BY = 'tools/sdf-ref/gen-leaf-fixtures.mjs'
const README = fileURLToPath(new URL('./README.md', import.meta.url))
const BASIC_PATH = 'src/core/sdf/primitives/basic.ts'
const QUALITIES = [8, 24, 64] // QUALITY LOW / LOD0 / HIGH segments
const LOD0 = 24
const SETS = ['A', 'B']
const AXES = ['x', 'y', 'z']
const WELD_EPS_FACTOR = 1e-6

// S3 reference parameter sets, in PrimKind::ALL order; object keys are the S3 params keys in S3 order.
const S3 = {
  sphere: { A: { radius: 0.8 }, B: { radius: 1.7 } },
  box: { A: { halfExtents: [0.7, 0.6, 0.5] }, B: { halfExtents: [1.2, 0.3, 2.0] } },
  roundBox: { A: { halfExtents: [0.7, 0.6, 0.5], radius: 0.15 }, B: { halfExtents: [1.0, 1.0, 0.4], radius: 0.35 } },
  capsule: { A: { radius: 0.4, halfHeight: 0.5 }, B: { radius: 0.25, halfHeight: 1.2 } },
  cylinder: { A: { radius: 0.6, halfHeight: 0.7 }, B: { radius: 1.5, halfHeight: 0.2 } },
  cone: { A: { radius: 0.7, halfHeight: 0.8 }, B: { radius: 0.3, halfHeight: 1.5 } },
  torus: { A: { majorRadius: 0.6, minorRadius: 0.22 }, B: { majorRadius: 1.5, minorRadius: 0.1 } },
  ellipsoid: { A: { radii: [0.9, 0.6, 0.5] }, B: { radii: [0.3, 1.4, 0.8] } },
}
const KINDS = Object.keys(S3)

// How leafGeometry's switch (leaf-mesh.ts:695-703) reaches each generator, keyed by LEAF_KINDS entry. Path 'call'
// passes these shape arguments followed by the explicit counts; the LOD0 assertion checks this table against
// unmodified leafGeometry.
const GENERATORS = {
  sphere: { fn: 'ellipsoid', args: (p) => [p.radius, p.radius, p.radius] },
  ellipsoid: { fn: 'ellipsoid', args: (p) => [...p.radii] },
  box: { fn: 'box', args: (p) => [...p.size] },
  roundBox: { fn: 'roundBox', args: (p) => [...p.size, p.radius] },
  capsule: { fn: 'capsule', args: (p) => [p.height, p.radius] },
  cylinder: { fn: 'frustum', args: (p) => [p.height, p.radius, p.radius] },
  cone: { fn: 'frustum', args: (p) => [p.height, p.radiusBottom, p.radiusTop] },
  torus: { fn: 'torus', args: (p) => [p.majorRadius, p.minorRadius] },
}

class Refusal extends Error {}

function fail(messages) {
  for (const m of [].concat(messages)) process.stderr.write(`sdf-ref: ${m}\n`)
  process.exit(1)
}

const isObj = (v) => typeof v === 'object' && v !== null && !Array.isArray(v)
const isPosInt = (v) => Number.isInteger(v) && v >= 1

function parseArgs(argv) {
  const opts = {}
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a !== '--ref' && a !== '--out') fail(`unknown argument '${a}' (usage: ${USAGE})`)
    if (i + 1 >= argv.length) fail(`${a} needs a value (usage: ${USAGE})`)
    opts[a.slice(2)] = argv[++i]
  }
  if (!opts.ref) fail(`missing --ref <sdfm checkout root> (usage: ${USAGE})`)
  if (!opts.out) fail(`missing --out <fixture path> (usage: ${USAGE})`)
  return opts
}

/** The fenced json block of README §4, parsed (embedded verbatim as the fixture's `mapping`). */
function readMapping() {
  const text = readFileSync(README, 'utf8')
  const head = text.indexOf('\n## 4. mapping v1')
  const open = head < 0 ? -1 : text.indexOf('\n```json\n', head)
  const close = open < 0 ? -1 : text.indexOf('\n```\n', open + 9)
  const next = text.indexOf('\n## ', head + 1)
  if (head < 0 || open < 0 || close < 0 || (next >= 0 && close > next)) {
    throw new Refusal(`${README}: no fenced json block in section '## 4. mapping v1'`)
  }
  try {
    return JSON.parse(text.slice(open + 9, close + 1))
  } catch (e) {
    throw new Refusal(`${README}: mapping v1 is not valid JSON (${e.message})`)
  }
}

function git(root, args) {
  try {
    return execFileSync('git', ['-C', root, ...args], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim()
  } catch (e) {
    throw new Refusal(`git ${args.join(' ')} failed in ${root}: ${String(e.stderr || e.message).trim()}`)
  }
}

// count(s) = max(min, multiple * round(ref * s / (24 * multiple))), round half away from zero (mapping v1).
const roundHalfAway = (v) => Math.sign(v) * Math.round(Math.abs(v))
const countAt = (rule, s) => Math.max(rule.min, rule.multiple * roundHalfAway((rule.ref * s) / (LOD0 * rule.multiple)))
const clampSegments = (q) => Math.min(256, Math.max(3, q))

/**
 * Validates mapping v1 (the rules of README §4, the same ones raa-prims::reference applies) against LEAF_KINDS and
 * the reference's own parameter specs. Returns { errors, axis, plans }: axis[k] = { sdf, sign } for RAA axis k, and
 * plans[kind] = { sdfKind, entries, constants, segments }.
 */
function validateMapping(m, leafKinds, specsByKind) {
  const errors = []
  const exactKeys = (obj, keys, where) => {
    for (const k of Object.keys(obj)) if (!keys.includes(k)) errors.push(`${where}: unknown key '${k}'`)
    for (const k of keys) if (!(k in obj)) errors.push(`${where}: missing key '${k}'`)
  }
  if (!isObj(m)) return { errors: ['mapping is not a JSON object'] }
  exactKeys(m, ['version', 'reference', 'axisMap', 'paramMap'], 'mapping')
  if (m.version !== 1) errors.push(`mapping.version is ${JSON.stringify(m.version)}, expected 1`)
  const ref = m.reference
  if (!isObj(ref)) errors.push('mapping.reference is not an object')
  else {
    exactKeys(ref, ['repo', 'path', 'commit'], 'mapping.reference')
    if (ref.repo !== 'sdfm') errors.push(`mapping.reference.repo is ${JSON.stringify(ref.repo)}, expected "sdfm"`)
    if (ref.path !== 'src/core/hybrid/leaf-mesh.ts') errors.push(`mapping.reference.path is ${JSON.stringify(ref.path)}, expected "src/core/hybrid/leaf-mesh.ts"`)
    if (typeof ref.commit !== 'string' || !/^[0-9a-f]{40}$/.test(ref.commit)) errors.push('mapping.reference.commit is not 40 lowercase hex')
  }

  // axisMap: RAA axis <- signed SDF axis, a signed permutation whose determinant is `det`.
  const axis = []
  let permutation = false
  const am = m.axisMap
  if (!isObj(am)) errors.push('mapping.axisMap is not an object')
  else {
    exactKeys(am, ['x', 'y', 'z', 'det'], 'mapping.axisMap')
    for (const k of AXES) {
      const hit = typeof am[k] === 'string' ? /^([+-])([xyz])$/.exec(am[k]) : null
      if (!hit) errors.push(`mapping.axisMap.${k} is ${JSON.stringify(am[k])}, expected one of +x -x +y -y +z -z`)
      else axis.push({ sdf: AXES.indexOf(hit[2]), sign: hit[1] === '-' ? -1 : 1 })
    }
    if (axis.length === 3) {
      const perm = axis.map((a) => a.sdf)
      if (new Set(perm).size !== 3) errors.push('mapping.axisMap is not a permutation (an SDF axis is used twice)')
      else {
        permutation = true
        const inversions = (perm[0] > perm[1]) + (perm[0] > perm[2]) + (perm[1] > perm[2])
        const det = axis.reduce((d, a) => d * a.sign, inversions % 2 === 0 ? 1 : -1)
        if (am.det !== det) errors.push(`mapping.axisMap.det is ${JSON.stringify(am.det)}, the signed permutation has det ${det}`)
      }
    }
  }

  const plans = {}
  const pm = m.paramMap
  if (!isObj(pm)) return { errors: [...errors, 'mapping.paramMap is not an object'] }
  for (const k of Object.keys(pm)) if (!KINDS.includes(k)) errors.push(`mapping.paramMap.${k}: not an S3 kind`)
  for (const kind of KINDS) {
    const where = `mapping.paramMap.${kind}`
    const km = pm[kind]
    if (!isObj(km)) { errors.push(`${where}: missing kind`); continue }
    exactKeys(km, ['sdfKind', 'params', 'constants', 'segments'], where)
    const sdfKind = km.sdfKind
    if (!leafKinds.includes(sdfKind)) { errors.push(`${where}.sdfKind: ${JSON.stringify(sdfKind)} is absent from LEAF_KINDS`); continue }
    const specs = specsByKind.get(sdfKind)
    if (!specs || !GENERATORS[sdfKind]) { errors.push(`${where}.sdfKind: '${sdfKind}' is not a basic primitive leafGeometry generates from parameters`); continue }
    const s3 = S3[kind].A
    const used = new Map() // S3 key or 'key.component' -> SDF entry that consumes it
    const covered = new Map() // SDF param -> set of covered SDF axes ('' for a scalar)
    const entries = []
    if (!isObj(km.params)) errors.push(`${where}.params is not an object`)
    // One linear term {from, component, scale} of an SDF parameter with axis suffix `sdfAxis`: the component rules
    // of README §4, shared by the main term and the optional additive term. Returns the RAA component index (null for
    // a scalar), or undefined after recording an error.
    const term = (e, w, sdfAxis, spec, scaleRule) => {
      if (!Number.isFinite(e.scale) || !scaleRule.ok(e.scale)) errors.push(`${w}.scale: must be ${scaleRule.text}`)
      if (!(e.from in s3)) { errors.push(`${w}.from: ${JSON.stringify(e.from)} is not an S3 key of ${kind}`); return undefined }
      if (Array.isArray(s3[e.from])) {
        if (spec.type !== 'vec3' || !sdfAxis) { errors.push(`${w}: vector S3 key '${e.from}' needs a vec3 SDF parameter written '<param>.x|.y|.z'`); return undefined }
        if (!AXES.includes(e.component)) { errors.push(`${w}.component: must be "x", "y" or "z" for vector S3 key '${e.from}'`); return undefined }
        const raa = AXES.indexOf(e.component)
        const expected = axis.findIndex((a) => a.sdf === AXES.indexOf(sdfAxis))
        if (permutation && raa !== expected) errors.push(`${w}.component: SDF axis ${sdfAxis} is RAA component ${AXES[expected]} through axisMap, not ${e.component}`)
        return raa
      }
      if (spec.type !== 'number' || sdfAxis) { errors.push(`${w}: scalar S3 key '${e.from}' needs a scalar (number) SDF parameter`); return undefined }
      if (e.component !== null) { errors.push(`${w}.component: must be null for scalar S3 key '${e.from}'`); return undefined }
      return null
    }
    for (const [key, e] of Object.entries(isObj(km.params) ? km.params : {})) {
      const w = `${where}.params.${key}`
      const hit = /^([A-Za-z_]\w*)(?:\.([xyz]))?$/.exec(key)
      if (!hit) { errors.push(`${w}: malformed key`); continue }
      if (!isObj(e)) { errors.push(`${w}: not an object`); continue }
      exactKeys(e, 'add' in e ? ['from', 'component', 'scale', 'add'] : ['from', 'component', 'scale'], w)
      const [, name, sdfAxis] = hit
      const spec = specs.find((s) => s.name === name)
      if (!spec) { errors.push(`${w}: '${name}' is not a parameter of '${sdfKind}'`); continue }
      if (spec.advanced) { errors.push(`${w}: '${name}' is an advanced parameter (its presence makes leafGeometry return null)`); continue }
      const raa = term(e, w, sdfAxis, spec, { ok: (v) => v > 0, text: 'a finite number > 0' })
      if (raa === undefined) continue
      // Optional additive term (amendment Q5): an extra use of an S3 key, outside the each-key-once coverage rule. A bad
      // term is reported on its own; the main term still counts toward coverage, so no follow-on error hides it.
      let add = null
      if ('add' in e) {
        const a = e.add
        if (!isObj(a)) errors.push(`${w}.add: not an object`)
        else if ('add' in a) errors.push(`${w}.add: nested 'add' is not allowed (an entry has at most one additive term)`)
        else {
          exactKeys(a, ['from', 'component', 'scale'], `${w}.add`)
          const addRaa = term(a, `${w}.add`, sdfAxis, spec, { ok: (v) => v !== 0, text: 'a finite non-zero number' })
          if (addRaa !== undefined) add = { from: a.from, raa: addRaa, scale: a.scale }
        }
      }
      const s3Key = raa === null ? e.from : `${e.from}.${e.component}`
      if (used.has(s3Key)) errors.push(`${w}: S3 key '${s3Key}' is already used by '${used.get(s3Key)}'`)
      used.set(s3Key, key)
      const axes = covered.get(name) ?? new Set()
      if (axes.has(sdfAxis ?? '')) errors.push(`${w}: SDF parameter covered twice`)
      covered.set(name, axes.add(sdfAxis ?? ''))
      entries.push({ key, name, sdfAxis: sdfAxis ? AXES.indexOf(sdfAxis) : null, from: e.from, raa, scale: e.scale, add })
    }
    if (!isObj(km.constants)) errors.push(`${where}.constants is not an object`)
    const constants = isObj(km.constants) ? km.constants : {}
    for (const [name, v] of Object.entries(constants)) {
      const w = `${where}.constants.${name}`
      const spec = specs.find((s) => s.name === name)
      if (!spec) { errors.push(`${w}: '${name}' is not a parameter of '${sdfKind}'`); continue }
      if (spec.advanced) { errors.push(`${w}: '${name}' is an advanced parameter (its presence makes leafGeometry return null)`); continue }
      if (covered.has(name)) errors.push(`${w}: also set by params`)
      const typed = spec.type === 'number' ? Number.isFinite(v) : spec.type === 'enum' ? spec.options?.includes(v) : spec.type === 'boolean' && typeof v === 'boolean'
      if (!typed) errors.push(`${w}: ${JSON.stringify(v)} does not fit the '${spec.type}' parameter`)
      covered.set(name, new Set(['']))
    }
    for (const spec of specs) {
      if (spec.advanced) continue
      const axes = covered.get(spec.name)
      if (spec.type === 'vec3' ? !axes || axes.size !== 3 : !axes) {
        errors.push(`${where}: SDF parameter '${spec.name}' of '${sdfKind}' has no linear S3 source in params (and no constant)${spec.type === 'vec3' ? ': all of .x .y .z are required' : ''}`)
      }
    }
    for (const [key, value] of Object.entries(s3)) {
      for (const k of Array.isArray(value) ? AXES.map((c) => `${key}.${c}`) : [key]) {
        if (!used.has(k)) errors.push(`${where}: S3 key '${k}' is not used by any SDF parameter`)
      }
    }
    const segments = Array.isArray(km.segments) ? km.segments : []
    if (!Array.isArray(km.segments)) errors.push(`${where}.segments is not a list`)
    const names = new Set()
    segments.forEach((r, i) => {
      const w = `${where}.segments[${i}]`
      if (!isObj(r)) { errors.push(`${w}: not an object`); return }
      exactKeys(r, ['count', 'ref', 'min', 'multiple', 'source'], w)
      if (typeof r.count !== 'string' || !/^[A-Za-z_]\w*$/.test(r.count)) errors.push(`${w}.count: not an identifier`)
      else if (names.has(r.count)) errors.push(`${w}.count: '${r.count}' listed twice`)
      names.add(r.count)
      if (!isPosInt(r.ref) || !isPosInt(r.min) || !isPosInt(r.multiple)) errors.push(`${w}: ref, min and multiple must be integers >= 1`)
      else if (countAt(r, LOD0) !== r.ref) errors.push(`${w}: the count rule gives ${countAt(r, LOD0)} at LOD0, not the reference default ${r.ref}`)
      const src = typeof r.source === 'string' ? /^(.+):([1-9]\d*)$/.exec(r.source) : null
      if (!src || src[1] !== ref?.path) errors.push(`${w}.source: must be '${ref?.path}:<line>'`)
    })
    plans[kind] = { sdfKind, entries, constants, segments }
  }
  return { errors, axis, plans }
}

/**
 * leafGeometry params of `kind` for S3 values `s3`: sdf_value = scale x a, or scale x a + add.scale x b with the
 * additive term (f64, in that order), then the constants.
 */
function sdfParams(plan, s3) {
  const out = {}
  const pick = (from, raa) => (raa === null ? s3[from] : s3[from][raa])
  for (const e of plan.entries) {
    const value = e.add ? e.scale * pick(e.from, e.raa) + e.add.scale * pick(e.add.from, e.add.raa) : e.scale * pick(e.from, e.raa)
    if (e.sdfAxis === null) out[e.name] = value
    else (out[e.name] ??= [null, null, null])[e.sdfAxis] = value
  }
  return Object.assign(out, plan.constants)
}

/**
 * Decides per kind how the counts reach the maths (README §1.2): 'call' when every count is a parameter of the
 * kind's generator whose default literal on the `source` line is `ref` (the generator is then called with explicit
 * counts), 'literal' when every count is a unique `ref` literal on its `source` line (an in-memory copy with exactly
 * those literals substituted). A kind without counts calls unmodified leafGeometry ('call' with no count).
 */
function planPaths(lines, plans) {
  const errors = []
  for (const kind of KINDS) {
    const plan = plans[kind]
    const gen = GENERATORS[plan.sdfKind]
    plan.order = []
    if (plan.segments.length === 0) { plan.path = 'call'; plan.fn = null; continue }
    const found = plan.segments.map((r) => {
      const line = Number(r.source.split(':').pop())
      const text = lines[line - 1]
      if (text === undefined) { errors.push(`${kind}.${r.count}: ${r.source} is past the end of the file`); return null }
      const sig = /\bfunction\s+(\w+)\s*\(([^)]*)\)/.exec(text)
      const params = sig ? sig[2].split(',').map((p) => /^\s*(\w+)\s*(?::[^=]*?)?\s*(?:=\s*(.*?))?\s*$/.exec(p)) : []
      const at = params.findIndex((p) => p?.[1] === r.count)
      if (sig && at >= 0 && params[at][2] === String(r.ref)) return { path: 'call', fn: sig[1], params: params.map((p) => p?.[1]), line }
      const hits = [...text.matchAll(new RegExp(`(?<![\\w.])${r.ref}(?![\\w.])`, 'g'))]
      if (hits.length === 1) return { path: 'literal', line, column: hits[0].index, rule: r }
      errors.push(`${kind}.${r.count}: ${r.source} has neither a generator parameter '${r.count} = ${r.ref}' nor exactly one literal ${r.ref}`)
      return null
    })
    if (found.includes(null)) continue
    const paths = new Set(found.map((f) => f.path))
    if (paths.size !== 1) { errors.push(`${kind}: counts mix paths 'call' and 'literal'`); continue }
    plan.path = found[0].path
    if (plan.path === 'literal') { plan.literals = found; continue }
    const fns = new Set(found.map((f) => f.fn))
    if (fns.size !== 1 || !fns.has(gen.fn)) { errors.push(`${kind}: counts are parameters of ${[...fns].join(', ')}, but leafGeometry('${plan.sdfKind}') calls ${gen.fn}`); continue }
    const shape = gen.args(sdfParams(plan, S3[kind].A)).length
    const tail = found[0].params.slice(shape)
    const listed = plan.segments.map((r) => r.count)
    for (const name of tail) if (!listed.includes(name)) errors.push(`${kind}: count '${name}' of ${gen.fn} (line ${found[0].line}) is not in the mapping`)
    for (const name of listed) if (!tail.includes(name)) errors.push(`${kind}: '${name}' is not a count parameter after the ${shape} shape arguments of ${gen.fn}`)
    plan.fn = gen.fn
    plan.order = tail
  }
  return errors
}

/** In-memory copy of leaf-mesh.ts with only the `source` literals of `plan` replaced by `counts`. */
function literalCopy(lines, plan, counts) {
  const out = [...lines]
  const byLine = [...plan.literals].sort((a, b) => a.line - b.line || b.column - a.column)
  for (const { line, column, rule } of byLine) {
    const t = out[line - 1]
    out[line - 1] = t.slice(0, column) + String(counts[rule.count]) + t.slice(column + String(rule.ref).length)
  }
  return out.join('\n')
}

/**
 * Loads the reference with node's built-in type stripping. The hooks only (1) retry a relative '.js' specifier that
 * does not exist as its '.ts' sibling (the sources import '.ts' files as '.js'), (2) replace an npm package that is not
 * installed in the checkout by a module that throws on any use, so it can never alter a mesh, and (3) serve the
 * in-memory copies registered in `copies` (URL -> source); nothing is written anywhere.
 */
function installLoader(refRoot, copies) {
  const srcRoot = pathToFileURL(resolve(refRoot, 'src')).href + '/'
  const missing = new Set()
  registerHooks({
    resolve(specifier, context, nextResolve) {
      try {
        return nextResolve(specifier, context)
      } catch (e) {
        if (e?.code !== 'ERR_MODULE_NOT_FOUND' || !context.parentURL?.startsWith(srcRoot)) throw e
        if (/^\.\.?\//.test(specifier) && specifier.endsWith('.js')) {
          const url = new URL(`${specifier.slice(0, -3)}.ts`, context.parentURL)
          if (existsSync(fileURLToPath(url))) return { url: url.href, format: 'module-typescript', shortCircuit: true }
        }
        if (/^(@[\w.-]+\/)?[\w.-]+(\/[\w./-]*)?$/.test(specifier)) {
          missing.add(specifier)
          return { url: `sdf-ref-missing:${specifier}`, shortCircuit: true }
        }
        throw e
      }
    },
    load(url, context, nextLoad) {
      if (url.startsWith('sdf-ref-missing:')) {
        const message = JSON.stringify(`npm package '${url.slice(16)}' is not installed in the reference checkout and was used`)
        const source = `const use = () => { throw new Error(${message}) }\nexport default new Proxy(function () {}, { get: use, apply: use, construct: use })\n`
        return { format: 'module', source, shortCircuit: true }
      }
      if (copies.has(url)) return { format: 'module-typescript', source: copies.get(url), shortCircuit: true }
      return nextLoad(url, context)
    },
  })
  return missing
}

const sameBytes = (a, b) => a.constructor === b.constructor && a.length === b.length &&
  Buffer.compare(Buffer.from(a.buffer, a.byteOffset, a.byteLength), Buffer.from(b.buffer, b.byteOffset, b.byteLength)) === 0
const sameMesh = (a, b) => sameBytes(a.positions, b.positions) && sameBytes(a.normals, b.normals) && sameBytes(a.indices, b.indices)

/** Positions and normals through axisMap (p_raa[k] = sign x p_sdf[axis]); triangles reversed when det = -1. */
function toRaaFrame(g, axis, det) {
  const map = (src) => {
    const out = new Float32Array(src.length)
    for (let v = 0; v < src.length; v += 3) for (let k = 0; k < 3; k++) out[v + k] = axis[k].sign * src[v + axis[k].sdf]
    return out
  }
  const indices = Uint32Array.from(g.indices)
  if (det < 0) for (let t = 0; t < indices.length; t += 3) [indices[t], indices[t + 2]] = [indices[t + 2], indices[t]]
  return { positions: map(g.positions), normals: map(g.normals), indices }
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const length = (a) => Math.sqrt(dot(a, a))

/**
 * S8 metrics in f64, mirroring raa_geom::analysis::analyze: bbox, positions welded at WELD_EPS (quantised grid,
 * first-seen ids), welded-edge closedness, area and signed volume (divergence theorem); plus the 1e-6 rounding count
 * of unique_positions_1e6, which must agree with the welded count.
 */
function metrics(mesh) {
  const pts = []
  for (let v = 0; v < mesh.positions.length; v += 3) pts.push([mesh.positions[v], mesh.positions[v + 1], mesh.positions[v + 2]])
  const bboxMin = [Infinity, Infinity, Infinity]
  const bboxMax = [-Infinity, -Infinity, -Infinity]
  for (const p of pts) for (let k = 0; k < 3; k++) { bboxMin[k] = Math.min(bboxMin[k], p[k]); bboxMax[k] = Math.max(bboxMax[k], p[k]) }
  const eps = WELD_EPS_FACTOR * Math.max(length(sub(bboxMax, bboxMin)), 1)
  const grid = new Map()
  const reps = []
  const ids = pts.map((p) => {
    const cell = p.map((x) => Math.floor(x / eps))
    let found = -1
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
      for (const id of grid.get(`${cell[0] + dx},${cell[1] + dy},${cell[2] + dz}`) ?? []) {
        if (length(sub(reps[id], p)) <= eps && (found < 0 || id < found)) found = id
      }
    }
    if (found >= 0) return found
    const key = cell.join(',')
    grid.set(key, [...(grid.get(key) ?? []), reps.length])
    reps.push(p)
    return reps.length - 1
  })
  const rounded = new Set(pts.map((p) => p.map((x) => roundHalfAway(x * 1e6) || 0).join(',')))
  const edges = new Map()
  let area = 0
  let volume = 0
  for (let t = 0; t < mesh.indices.length; t += 3) {
    const idx = [mesh.indices[t], mesh.indices[t + 1], mesh.indices[t + 2]]
    if (idx.some((i) => i >= pts.length)) throw new Refusal(`triangle ${t / 3} has an out-of-range index`)
    const [p0, p1, p2] = idx.map((i) => pts[i])
    area += 0.5 * length(cross(sub(p1, p0), sub(p2, p0)))
    volume += dot(p0, cross(p1, p2)) / 6
    const w = idx.map((i) => ids[i])
    if (w[0] === w[1] || w[1] === w[2] || w[0] === w[2]) continue
    for (let k = 0; k < 3; k++) {
      const [a, b] = [w[k], w[(k + 1) % 3]]
      const key = `${Math.min(a, b)},${Math.max(a, b)}`
      const e = edges.get(key) ?? [0, 0]
      edges.set(key, [e[0] + 1, e[1] + (a < b ? 1 : 0)])
    }
  }
  const closed = [...edges.values()].every(([uses, forward]) => uses === 2 && forward === 1)
  return { uniquePositions: reps.length, unique1e6: rounded.size, triangles: mesh.indices.length / 3, bboxMin, bboxMax, area, volume, closed }
}

async function main() {
  const opts = parseArgs(process.argv.slice(2))
  const refRoot = resolve(opts.ref)
  const out = resolve(opts.out)
  const leafPath = resolve(refRoot, 'src/core/hybrid/leaf-mesh.ts')
  if (!existsSync(refRoot)) throw new Refusal(`--ref ${opts.ref}: no such directory`)
  if (!existsSync(leafPath)) throw new Refusal(`${leafPath} is missing (is --ref the sdfm checkout root?)`)
  const outRel = relative(refRoot, out)
  if (!outRel.startsWith('..') && !isAbsolute(outRel)) throw new Refusal(`--out ${opts.out} is inside the read-only reference checkout`)
  const mapping = readMapping()
  const bytes = readFileSync(leafPath)
  const sha256 = createHash('sha256').update(bytes).digest('hex')
  const text = bytes.toString('utf8')
  const lines = text.split('\n')

  const copies = new Map()
  const missing = installLoader(refRoot, copies)
  const leafUrl = pathToFileURL(leafPath).href
  let leaf, basic
  try {
    leaf = await import(leafUrl)
    basic = await import(pathToFileURL(resolve(refRoot, BASIC_PATH)).href)
  } catch (e) {
    throw new Refusal(`cannot load ${leafPath} with node type stripping: ${e.message}`)
  }
  if (!Array.isArray(leaf.LEAF_KINDS) || typeof leaf.leafGeometry !== 'function' || !Array.isArray(basic.basicPrimitives)) {
    throw new Refusal('the reference no longer exports LEAF_KINDS, leafGeometry (leaf-mesh.ts) and basicPrimitives (basic.ts)')
  }
  const specsByKind = new Map(basic.basicPrimitives.map((p) => [p.kind, p.params]))
  const { errors, axis, plans } = validateMapping(mapping, [...leaf.LEAF_KINDS], specsByKind)
  if (errors.length) fail(errors.map((e) => `invalid mapping v1 (tools/sdf-ref/README.md §4): ${e}`))
  const head = git(refRoot, ['rev-parse', 'HEAD'])
  if (head !== mapping.reference.commit) throw new Refusal(`the checkout is at ${head}, mapping v1 is bound to ${mapping.reference.commit}`)
  const dirty = git(refRoot, ['status', '--porcelain', '--', 'src'])
  if (dirty) throw new Refusal(`the checkout has local changes under src/:\n${dirty}`)
  const pathErrors = planPaths(lines, plans)
  if (pathErrors.length) fail(pathErrors)

  // Path 'call': one in-memory copy whose only change is an appended export of the generators (README §1.2).
  const callFns = [...new Set(KINDS.map((k) => plans[k].path === 'call' && plans[k].fn).filter(Boolean))]
  let call = null
  if (callFns.length) {
    const url = `${leafUrl}?sdf-ref=call`
    copies.set(url, `${text}${text.endsWith('\n') ? '' : '\n'}export { ${callFns.join(', ')} }\n`)
    call = await import(url)
  }
  const referenceMesh = async (kind, plan, input, counts, s) => {
    if (plan.path === 'literal') {
      const url = `${leafUrl}?sdf-ref=literal&kind=${kind}&segments=${s}`
      if (!copies.has(url)) copies.set(url, literalCopy(lines, plan, counts))
      return (await import(url)).leafGeometry(plan.sdfKind, input)
    }
    if (!plan.fn) return leaf.leafGeometry(plan.sdfKind, input)
    return call[plan.fn](...GENERATORS[plan.sdfKind].args(input), ...plan.order.map((c) => counts[c]))
  }

  const cases = []
  for (const kind of KINDS) {
    const plan = plans[kind]
    for (const set of SETS) {
      const params = structuredClone(S3[kind][set])
      const input = sdfParams(plan, params)
      for (const quality of QUALITIES) {
        const s = clampSegments(quality)
        const counts = Object.fromEntries(plan.segments.map((r) => [r.count, countAt(r, s)]))
        const where = `${kind} set ${set} quality ${quality}`
        const g = await referenceMesh(kind, plan, structuredClone(input), counts, s)
        if (!g) throw new Refusal(`${where}: leafGeometry returned null`)
        if (s === LOD0 && !sameMesh(g, leaf.leafGeometry(plan.sdfKind, structuredClone(input)))) {
          throw new Refusal(`${where}: the path '${plan.path}' mesh differs from unmodified leafGeometry output`)
        }
        const m = metrics(toRaaFrame(g, axis, mapping.axisMap.det))
        if (!(m.volume > 0)) throw new Refusal(`${where}: signed volume ${m.volume} is not positive`)
        if (m.uniquePositions !== m.unique1e6) {
          throw new Refusal(`${where}: ${m.uniquePositions} positions welded at WELD_EPS but ${m.unique1e6} after rounding to 1e-6`)
        }
        cases.push({
          kind, set, quality, params, refParams: input, sdfInput: { kind: plan.sdfKind, params: input }, counts,
          uniquePositions: m.uniquePositions, triangles: m.triangles, bboxMin: m.bboxMin, bboxMax: m.bboxMax,
          area: m.area, volume: m.volume, closed: m.closed,
        })
        const shown = Object.entries(counts).map(([c, v]) => `${c}=${v}`).join(',') || '-'
        console.log(`${kind.padEnd(9)} ${set} q${String(quality).padEnd(2)} ${plan.path.padEnd(7)} counts ${shown.padEnd(15)} ` +
          `tris ${String(m.triangles).padStart(5)} unique ${String(m.uniquePositions).padStart(5)} closed ${m.closed} volume ${m.volume.toFixed(6)}`)
      }
    }
    for (const set of SETS) {
      const tris = new Set(cases.filter((c) => c.kind === kind && c.set === set).map((c) => c.triangles))
      if (plan.segments.length && tris.size !== QUALITIES.length) throw new Refusal(`${kind} set ${set}: qualities ${QUALITIES.join('/')} do not give distinct meshes`)
    }
  }
  const keys = new Set(cases.map((c) => `${c.kind}/${c.set}/${c.quality}`))
  if (cases.length !== 48 || keys.size !== 48) throw new Refusal(`expected 48 distinct cases, got ${keys.size}`)
  const open = cases.filter((c) => !c.closed)
  if (open.length) console.log(`warning: reference meshes not closed: ${open.map((c) => `${c.kind}/${c.set}/${c.quality}`).join(' ')}`)

  const fixture = {
    generatedBy: GENERATED_BY,
    command: REGENERATE,
    regenerate: REGENERATE,
    sdfModelerRevision: mapping.reference.commit,
    leafMeshSha256: sha256,
    axisMap: axis.map((a) => ({ axis: AXES[a.sdf], sign: a.sign })),
    paramMap: Object.fromEntries(KINDS.map((k) => [k, [
      ...plans[k].entries.map((e) => ({
        ref: e.name, refIndex: e.sdfAxis, raa: e.from, index: e.raa, scale: e.scale, value: null,
        ...(e.add && { add: { raa: e.add.from, index: e.add.raa, scale: e.add.scale } }),
      })),
      ...Object.entries(plans[k].constants).map(([name, v]) => ({ ref: name, refIndex: null, raa: null, index: null, scale: 1, value: v })),
    ]])),
    mapping,
    kinds: Object.fromEntries(KINDS.map((k) => [k, { sdfKind: plans[k].sdfKind, path: plans[k].path, commit: mapping.reference.commit, sha256 }])),
    cases,
  }
  const json = `${JSON.stringify(fixture, null, 2)}\n`
  mkdirSync(dirname(out), { recursive: true })
  const tmp = `${out}.tmp-${process.pid}`
  writeFileSync(tmp, json)
  renameSync(tmp, out)
  if (missing.size) console.log(`note: not installed in the checkout, replaced by throwing stubs and never used: ${[...missing].sort().join(', ')}`)
  console.log(`wrote ${out}: ${cases.length} cases, sha256 ${createHash('sha256').update(json).digest('hex')}`)
}

main().catch((e) => fail(e instanceof Refusal ? e.message : `unexpected failure: ${e?.stack ?? e}`))
