// Independent f64 reference for S17 clip sampling and S16 pose matrices.
// Reads the canonical rig/clip fixtures (never Rust or wasm results) and writes
// crates/raa-anim/tests/fixtures/samples.json as one compact JSON line plus LF.
// Usage: node tools/anim-ref/gen-clip-samples.mjs
//        SKETCHBOOK_DIR=<dir or unified.ts> node tools/anim-ref/gen-clip-samples.mjs
import { readFileSync, statSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const FIXTURES = join(ROOT, 'crates/raa-anim/tests/fixtures')
const RIG = 'rig-basic.json'
const CLIPS = ['clip-bezier.json', 'clip-linear-constant.json', 'clip-rotation-flip.json']
const WIDTH = { translation: 3, rotation: 4, scale: 3 }
const TARGET = { translation: 'bone-pos', rotation: 'bone-rot', scale: 'bone-scale' }
const CHANNELS = { translation: ['X', 'Y', 'Z'], rotation: ['X', 'Y', 'Z', 'W'], scale: ['X', 'Y', 'Z'] }
const TOLERANCE = 1e-9
function fail(message) {
  process.stderr.write(`anim-ref: ${message}\n`)
  process.exit(1)
}
const finite = (v) => Array.isArray(v) && v.every(Number.isFinite)
function vector(v, n, path) {
  if (!finite(v) || v.length !== n) fail(`${path}: expected ${n} finite numbers`)
  return v.slice()
}
function readJson(file) {
  try { return JSON.parse(readFileSync(join(FIXTURES, file), 'utf8')) } catch (error) { fail(`${file}: ${error.message}`) }
}
function loadRig(file) {
  const doc = readJson(file)
  if (doc.format !== 'raa.rig' || doc.version !== 1) fail(`${file}: expected raa.rig v1`)
  if (!Array.isArray(doc.bones) || doc.bones.length === 0 || doc.bones.length > 256) fail(`${file}: expected 1..256 bones`)
  const index = new Map(), bones = [], rest = []
  let roots = 0
  doc.bones.forEach((bone, i) => {
    const path = `${file}: /bones/${i}`
    if (typeof bone.name !== 'string' || index.has(bone.name)) fail(`${path}: invalid or duplicate name`)
    let parent = null
    if (bone.parent === null) roots += 1
    else if (index.has(bone.parent)) parent = index.get(bone.parent)
    else fail(`${path}: parent must name an earlier bone`)
    const t = vector(bone.rest?.t, 3, `${path}/rest/t`)
    const r = vector(bone.rest?.r, 4, `${path}/rest/r`)
    const s = vector(bone.rest?.s, 3, `${path}/rest/s`)
    if (Math.abs(Math.sqrt(r.reduce((sum, x) => sum + x * x, 0)) - 1) > 1e-3) fail(`${path}/rest/r: not unit`)
    if (s.some((x) => !(x > 0))) fail(`${path}/rest/s: scale must be positive`)
    index.set(bone.name, i)
    bones.push({ name: bone.name, parent }); rest.push({ t, r, s })
  })
  if (roots !== 1) fail(`${file}: exactly one root is required`)
  return { name: doc.name, bones, rest, index }
}
function handle(key, name, width, path) {
  const h = key[name]
  if (h === undefined) return new Array(width + 1).fill(0)
  if (!finite(h) || h.length !== width + 1) fail(`${path}/${name}: expected ${width + 1} finite numbers`)
  return h.slice()
}
function prepareTrack(track, path) {
  const width = WIDTH[track.channel]
  if (!width) fail(`${path}: unknown channel ${track.channel}`)
  if (!Array.isArray(track.keys) || track.keys.length === 0) fail(`${path}: empty tracks are not sampled`)
  const keys = []; let previous = null
  track.keys.forEach((key, i) => {
    const kp = `${path}/keys/${i}`
    if (!Number.isInteger(key.tick) || key.tick < 0) fail(`${kp}/tick: expected a non-negative integer`)
    if (keys.length && key.tick <= keys[keys.length - 1].tick) fail(`${kp}/tick: ticks must strictly increase`)
    if (!['CONSTANT', 'LINEAR', 'BEZIER'].includes(key.interp)) fail(`${kp}/interp: unknown interpolation`)
    let v = vector(key.v, width, `${kp}/v`), hl = handle(key, 'hl', width, kp), hr = handle(key, 'hr', width, kp)
    if (key.interp !== 'BEZIER' && (key.hl !== undefined || key.hr !== undefined)) fail(`${kp}: handles require BEZIER`)
    if (track.channel === 'rotation' && previous) {
      const dot = ((previous[0] * v[0] + previous[1] * v[1]) + previous[2] * v[2]) + previous[3] * v[3]
      if (dot < 0) {
        v = v.map((x) => -x)
        hl = [hl[0], ...hl.slice(1).map((x) => -x)]
        hr = [hr[0], ...hr.slice(1).map((x) => -x)]
      }
    }
    keys.push({ tick: key.tick, interp: key.interp, v, hl, hr })
    if (track.channel === 'rotation') previous = v
  })
  return { bone: track.bone, channel: track.channel, width, keys }
}
function loadClip(file, rig) {
  const doc = readJson(file)
  if (doc.format !== 'raa.clip' || doc.version !== 2) fail(`${file}: expected raa.clip v2`)
  if (doc.rig !== rig.name) fail(`${file}: rig ${doc.rig} does not match ${rig.name}`)
  if (!Number.isInteger(doc.lengthTicks) || doc.lengthTicks <= 0) fail(`${file}: invalid lengthTicks`)
  if (typeof doc.loop !== 'boolean' || doc.rootMotion !== 'none') fail(`${file}: unsupported loop or rootMotion`)
  if (!Array.isArray(doc.tracks)) fail(`${file}: tracks must be an array`)
  const seen = new Set()
  const tracks = doc.tracks.map((track, i) => {
    const prepared = prepareTrack(track, `${file}: /tracks/${i}`)
    if (!rig.index.has(prepared.bone)) fail(`${file}: /tracks/${i}/bone: bone is not in the rig`)
    const id = `${prepared.bone}:${prepared.channel}`
    if (seen.has(id)) fail(`${file}: /tracks/${i}: duplicate bone and channel`)
    seen.add(id)
    return prepared
  })
  return { name: doc.name, file, lengthTicks: doc.lengthTicks, loop: doc.loop, rootMotion: doc.rootMotion, tracks }
}
const address = (clip, tick) => (clip.loop ? tick % clip.lengthTicks : tick)
function sampleRaw(track, tick) {
  const keys = track.keys
  if (tick <= keys[0].tick) return keys[0].v.slice()
  if (tick >= keys[keys.length - 1].tick) return keys[keys.length - 1].v.slice()
  let i = 0
  while (keys[i + 1].tick <= tick) i += 1
  const [a, b] = [keys[i], keys[i + 1]]
  const u = (tick - a.tick) / (b.tick - a.tick)
  if (a.interp === 'CONSTANT') return a.v.slice()
  if (a.interp === 'LINEAR') return a.v.map((av, c) => av + (b.v[c] - av) * u)
  const mt = 1 - u
  return a.v.map((av, c) => ((mt * mt * mt) * av + (3 * mt * mt * u) * (av + a.hr[c + 1])) +
    (3 * mt * u * u) * (b.v[c] + b.hl[c + 1]) + (u * u * u) * b.v[c])
}
function normalize4(v) {
  const squared = v.reduce((sum, x) => sum + x * x, 0)
  if (!Number.isFinite(squared) || squared === 0) fail(`rotation has squared length ${squared}`)
  const inverse = 1 / Math.sqrt(squared)
  return v.map((x) => x * inverse)
}
const finish = (channel, values) => (channel === 'rotation' ? normalize4(values) : values)
function sampleTicks(clip) {
  const ticks = new Set([0, clip.lengthTicks, clip.lengthTicks + 1])
  for (const track of clip.tracks) {
    const kt = track.keys.map((key) => key.tick)
    for (const tick of kt) ticks.add(tick)
    for (let i = 0; i < kt.length - 1; i += 1) {
      const [a, b] = [kt[i], kt[i + 1]]
      for (const tick of [Math.floor((a + b) / 2), Math.floor((3 * a + b) / 4), Math.floor((a + 3 * b) / 4)]) ticks.add(tick)
    }
    if (kt[0] > 0) ticks.add(kt[0] - 1)
    ticks.add(kt[kt.length - 1] + 1)
    if (clip.loop) ticks.add(clip.lengthTicks + 2).add(2 * clip.lengthTicks)
  }
  return [...ticks].filter((tick) => tick >= 0).sort((a, b) => a - b)
}
function trs(t, q, s) {
  const [x, y, z, w] = q
  const [xx, yy, zz, xy, xz, yz, wx, wy, wz] = [x * x, y * y, z * z, x * y, x * z, y * z, w * x, w * y, w * z]
  return [
    (1 - 2 * (yy + zz)) * s[0], 2 * (xy + wz) * s[0], 2 * (xz - wy) * s[0], 0,
    2 * (xy - wz) * s[1], (1 - 2 * (xx + zz)) * s[1], 2 * (yz + wx) * s[1], 0,
    2 * (xz + wy) * s[2], 2 * (yz - wx) * s[2], (1 - 2 * (xx + yy)) * s[2], 0,
    t[0], t[1], t[2], 1,
  ]
}
function multiply(a, b) {
  const out = new Array(16).fill(0)
  for (let c = 0; c < 4; c += 1) for (let r = 0; r < 4; r += 1) {
    out[c * 4 + r] = ((a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1]) + a[8 + r] * b[c * 4 + 2]) + a[12 + r] * b[c * 4 + 3]
  }
  return out
}
const restPose = (rig) => rig.rest.map((bone) => ({ t: bone.t.slice(), r: normalize4(bone.r), s: bone.s.slice() }))
function inverseBinds(rig, rest) {
  const inverse = []
  for (let i = 0; i < rig.bones.length; i += 1) {
    const { t, r, s } = rest[i]
    const m = trs([0, 0, 0], [-r[0], -r[1], -r[2], r[3]], [1, 1, 1])
    for (let row = 0; row < 3; row += 1) {
      for (let col = 0; col < 3; col += 1) m[col * 4 + row] /= s[row]
      m[12 + row] = -(m[row] * t[0] + m[4 + row] * t[1] + m[8 + row] * t[2])
    }
    inverse.push(rig.bones[i].parent === null ? m : multiply(m, inverse[rig.bones[i].parent]))
  }
  return inverse
}
function poseAt(rig, clip, tick, inverse) {
  const local = restPose(rig)
  for (const track of clip.tracks) {
    const values = finish(track.channel, sampleRaw(track, address(clip, tick)))
    const bone = local[rig.index.get(track.bone)]
    if (track.channel === 'rotation') bone.r = values
    else if (track.channel === 'translation') bone.t = values
    else bone.s = values
  }
  const model = [], skin = []
  for (let i = 0; i < local.length; i += 1) {
    const m = trs(local[i].t, local[i].r, local[i].s)
    const parent = rig.bones[i].parent
    const world = parent === null ? m : multiply(model[parent], m)
    model.push(world); skin.push(multiply(world, inverse[i]))
  }
  return { tick, local, model, skin }
}
function makeRecord(rig, clip) {
  const ticks = sampleTicks(clip)
  const inverse = inverseBinds(rig, restPose(rig))
  return {
    name: clip.name, file: `crates/raa-anim/tests/fixtures/${clip.file}`, lengthTicks: clip.lengthTicks,
    loop: clip.loop, rootMotion: clip.rootMotion, ticks,
    tracks: clip.tracks.map((track) => ({
      bone: track.bone, channel: track.channel,
      values: ticks.map((tick) => finish(track.channel, sampleRaw(track, address(clip, tick)))),
    })),
    poses: ticks.map((tick) => poseAt(rig, clip, tick, inverse)),
  }
}
function sketchbookFile(dir) {
  const target = resolve(dir)
  try {
    if (statSync(target).isFile()) return target
    const unified = join(target, 'unified.ts')
    if (statSync(unified).isFile()) return unified
  } catch { /* fall through */ }
  fail(`SKETCHBOOK_DIR=${dir}: expected unified.ts or a directory containing it`)
}
const close = (a, b) => Math.abs(a - b) <= TOLERANCE * Math.max(1, Math.abs(a), Math.abs(b))
async function crossCheck(clips, records, dir) {
  const file = sketchbookFile(dir)
  let module
  try { module = await import(pathToFileURL(file).href) } catch (error) { fail(`SKETCHBOOK_DIR: cannot import ${file}: ${error.message}`) }
  if (typeof module.sampleTrack !== 'function') fail(`SKETCHBOOK_DIR: ${file} does not export sampleTrack`)
  let checks = 0
  for (const [clipIndex, clip] of clips.entries()) for (const track of clip.tracks) {
    const pairs = (h) => Array.from({ length: track.width }, (_, c) => [h[0], h[c + 1]])
    const atrack = {
      id: `${track.bone}:${track.channel}`, label: `${track.bone}.${track.channel}`, channelNames: CHANNELS[track.channel],
      target: { kind: TARGET[track.channel], boneName: track.bone },
      keyframes: track.keys.map((key) => ({ time: key.tick, v: key.v.slice(), interp: key.interp, hl: pairs(key.hl), hr: pairs(key.hr) })),
    }
    for (const tick of records[clipIndex].ticks) {
      const addressed = address(clip, tick)
      const expected = sampleRaw(track, addressed)
      const actual = module.sampleTrack(atrack, addressed)
      if (!Array.isArray(actual) || actual.length !== track.width) fail(`SKETCHBOOK_DIR: bad width for ${clip.name} ${track.bone}/${track.channel}`)
      for (let c = 0; c < track.width; c += 1) {
        if (!close(expected[c], actual[c])) fail(`SKETCHBOOK_DIR mismatch ${clip.name} ${track.bone}/${track.channel} tick ${tick} component ${c}: reference ${expected[c]}, sampleTrack ${actual[c]}`)
      }
      checks += 1
    }
  }
  process.stdout.write(`anim-ref: Sketchbook sampleTrack matched ${checks} track samples\n`)
}
const rig = loadRig(RIG)
const clips = CLIPS.map((file) => loadClip(file, rig))
const records = clips.map((clip) => makeRecord(rig, clip))
if (process.env.SKETCHBOOK_DIR !== undefined) await crossCheck(clips, records, process.env.SKETCHBOOK_DIR)
const document = {
  format: 'raa.anim.samples', version: 1, precision: 'f64', generatedBy: 'tools/anim-ref/gen-clip-samples.mjs',
  rig: { name: rig.name, file: `crates/raa-anim/tests/fixtures/${RIG}` }, clips: records,
}
writeFileSync(join(FIXTURES, 'samples.json'), `${JSON.stringify(document)}\n`)
process.stdout.write(`anim-ref: wrote samples.json (${records.length} clips, ${records.reduce((n, r) => n + r.ticks.length, 0)} ticks)\n`)
