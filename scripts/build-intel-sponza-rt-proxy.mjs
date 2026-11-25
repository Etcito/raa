import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { existsSync } from 'node:fs'
import { chmod, copyFile, link, mkdir, mkdtemp, readFile, rename, rm, stat, writeFile } from 'node:fs/promises'
import { createRequire } from 'node:module'
import { endianness } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const root = fileURLToPath(new URL('..', import.meta.url))
const sourceRoot = resolve(process.argv[2] ?? join(root, 'out/benchmark-assets/intel-sponza'))
const output = resolve(process.argv[3] ?? join(sourceRoot, 'rt-proxy-245k'))
const variant = 'Intel Sponza Base — RT proxy (245k target)'
const filename = 'Intel_Sponza_RT_Proxy_245k'
const triangleBudget = 245_000
const traceLimit = 250_000
const keepSmall = 32
const errorLimit = 1
const flags = ['Permissive', 'RegularizeLight', 'PreserveFolds', 'ErrorClamped']
const weights = [0.5, 0.5, 0.5, 0.15, 0.15]
const simplifyVersion = '1.3.0'
const arities = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 }

assert.ok(!existsSync(output), `Output already exists: ${output}`)
const sha = value => createHash('sha256').update(value).digest('hex')
const json = value => JSON.stringify(value, null, 2) + '\n'
const fromMama = createRequire(resolve(root, '../MAMA/package.json'))
const simplifierPath = fromMama.resolve('meshoptimizer/simplifier')
const packageInfo = JSON.parse(await readFile(join(dirname(simplifierPath), 'package.json')))
assert.equal(packageInfo.version, simplifyVersion, 'Use the existing Kong MAMA MeshoptSimplifier version')
const { MeshoptSimplifier } = await import(pathToFileURL(simplifierPath))
await MeshoptSimplifier.ready
assert.equal(endianness(), 'LE', 'glTF buffers are little-endian')

const manifest = JSON.parse(await readFile(join(sourceRoot, 'manifest.json')))
const modelBytes = await readFile(join(sourceRoot, manifest.scene))
assert.equal(sha(modelBytes), manifest.files[manifest.scene].sha256)
const source = JSON.parse(modelBytes)
assert.equal(source.buffers.length, 1)
const originalBin = join(dirname(manifest.scene), source.buffers[0].uri)
const data = await readFile(join(sourceRoot, originalBin))
assert.equal(sha(data), manifest.files[originalBin].sha256)
assert.equal(data.length, source.buffers[0].byteLength)

function packed(index, ArrayType) {
  const accessor = source.accessors[index]
  const view = source.bufferViews[accessor.bufferView]
  const components = arities[accessor.type]
  const stride = components * ArrayType.BYTES_PER_ELEMENT
  const offset = (view.byteOffset ?? 0) + (accessor.byteOffset ?? 0)
  const length = accessor.count * components
  assert.ok(components && !accessor.sparse && !accessor.normalized && view.buffer === 0)
  assert.equal(accessor.componentType, ArrayType === Uint16Array ? 5123 : 5126)
  assert.equal(view.byteStride ?? stride, stride, 'Intel release uses tightly packed accessors')
  assert.ok(offset + length * ArrayType.BYTES_PER_ELEMENT <= (view.byteOffset ?? 0) + view.byteLength)
  return new ArrayType(data.buffer, data.byteOffset + offset, length)
}

const jobs = source.meshes.flatMap((mesh, meshIndex) => mesh.primitives.map((primitive, primitiveIndex) => {
  assert.equal(primitive.mode ?? 4, 4)
  assert.ok(!primitive.targets && !primitive.extensions)
  const indices = Uint32Array.from(packed(primitive.indices, Uint16Array))
  assert.ok(indices.length > 0 && indices.length % 3 === 0)
  const triangles = indices.length / 3
  return { meshIndex, primitiveIndex, primitive, indices, triangles, minimum: Math.min(keepSmall, triangles) }
}))
const total = jobs.reduce((sum, job) => sum + job.triangles, 0)
assert.equal(total, manifest.sceneInventory.instancedTriangles)
const reserved = jobs.reduce((sum, job) => sum + job.minimum, 0)
assert.ok(reserved <= triangleBudget && triangleBudget <= traceLimit && traceLimit < total)

for (const job of jobs) {
  job.target = job.minimum + Math.floor((job.triangles - job.minimum) * (triangleBudget - reserved) / (total - reserved))
  job.result = job.indices
  job.method = 'unchanged'
  if (job.target === job.triangles) continue

  const attributes = job.primitive.attributes
  const positions = packed(attributes.POSITION, Float32Array)
  const normals = packed(attributes.NORMAL, Float32Array)
  const uv = packed(attributes.TEXCOORD_0, Float32Array)
  const vertices = positions.length / 3
  assert.equal(normals.length, vertices * 3)
  assert.equal(uv.length, vertices * 2)
  const features = new Float32Array(vertices * weights.length)
  for (let vertex = 0; vertex < vertices; vertex++) {
    features.set(normals.subarray(vertex * 3, (vertex + 1) * 3), vertex * weights.length)
    features.set(uv.subarray(vertex * 2, (vertex + 1) * 2), vertex * weights.length + 3)
  }
  const [indices] = MeshoptSimplifier.simplifyWithAttributes(
    job.indices, positions, 3, features, weights.length, weights, null, job.target * 3, errorLimit, flags,
  )
  if (indices.length && indices.length < job.indices.length) {
    job.result = indices
    job.method = 'attribute-aware'
  }
}

let realized = jobs.reduce((sum, job) => sum + job.result.length / 3, 0)
const overflow = jobs.filter(job => job.result.length > job.target * 3).sort(
  (a, b) => (b.result.length - b.target * 3) - (a.result.length - a.target * 3),
)
for (const job of overflow) {
  if (realized <= triangleBudget) break
  const positions = packed(job.primitive.attributes.POSITION, Float32Array)
  const [indices] = MeshoptSimplifier.simplifySloppy(job.indices, positions, 3, null, job.target * 3, errorLimit)
  if (!indices.length || indices.length >= job.result.length) continue
  realized -= (job.result.length - indices.length) / 3
  job.result = indices
  job.method = 'position-only fallback'
}
assert.ok(realized <= triangleBudget, `Meshoptimizer output ${realized} exceeds the ${triangleBudget} proxy budget`)

const chunks = []
let offset = 0
const views = []
const accessors = []

function append(values, accessor, target) {
  const padding = (4 - offset % 4) % 4
  if (padding) chunks.push(Buffer.alloc(padding))
  offset += padding
  chunks.push(Buffer.from(values.buffer, values.byteOffset, values.byteLength))
  views.push({ buffer: 0, byteOffset: offset, byteLength: values.byteLength, target })
  offset += values.byteLength
  return accessors.push({ ...accessor, bufferView: views.length - 1 }) - 1
}

const meshes = source.meshes.map(mesh => ({ ...mesh, primitives: [] }))
const coverage = jobs.map(({ meshIndex, primitiveIndex, primitive, triangles, target, result, method }) => {
  const originalVertices = source.accessors[primitive.attributes.POSITION].count
  const remap = new Int32Array(originalVertices).fill(-1)
  const retained = []
  const indices = Uint16Array.from(result, vertex => {
    assert.ok(vertex < originalVertices && vertex <= 65535)
    if (remap[vertex] < 0) remap[vertex] = retained.push(vertex) - 1
    return remap[vertex]
  })
  const attributes = {}
  for (const [semantic, index] of Object.entries(primitive.attributes)) {
    const { componentType, type, name } = source.accessors[index]
    const components = arities[type]
    const from = packed(index, Float32Array)
    assert.equal(from.length, originalVertices * components)
    const to = new Float32Array(retained.length * components)
    const min = Array(components).fill(Infinity)
    const max = Array(components).fill(-Infinity)
    for (const [vertex, original] of retained.entries()) {
      for (let component = 0; component < components; component++) {
        const value = from[original * components + component]
        to[vertex * components + component] = value
        if (semantic === 'POSITION') {
          min[component] = Math.min(min[component], value)
          max[component] = Math.max(max[component], value)
        }
      }
    }
    attributes[semantic] = append(to, {
      componentType, type, name, count: retained.length, ...(semantic === 'POSITION' ? { min, max } : {}),
    }, 34962)
  }
  const index = append(indices, { componentType: 5123, type: 'SCALAR', count: indices.length }, 34963)
  meshes[meshIndex].primitives.push({ ...primitive, attributes, indices: index })
  return { mesh: meshIndex, primitive: primitiveIndex, material: primitive.material,
    sourceTriangles: triangles, targetTriangles: target, proxyTriangles: indices.length / 3, method }
})

const bin = Buffer.concat(chunks)
const gltf = {
  ...source,
  asset: { ...source.asset, extras: { ...source.asset.extras, benchmarkVariant: variant } },
  meshes, accessors, bufferViews: views,
  buffers: [{ uri: filename + '.bin', byteLength: bin.length }],
}
const model = json(gltf)
const materialCoverage = source.materials.map((material, index) => {
  const rows = coverage.filter(row => row.material === index)
  return { index, name: material.name, primitives: rows.length,
    sourceTriangles: rows.reduce((sum, row) => sum + row.sourceTriangles, 0),
    proxyTriangles: rows.reduce((sum, row) => sum + row.proxyTriangles, 0) }
})
const retainedFiles = Object.fromEntries(Object.entries(manifest.files)
  .filter(([name]) => name !== manifest.scene && name !== originalBin)
  .map(([name, entry]) => [name.slice(dirname(manifest.scene).length + 1), entry]))
const files = {
  ...retainedFiles,
  [filename + '.gltf']: { sha256: sha(model), bytes: Buffer.byteLength(model) },
  [filename + '.bin']: { sha256: sha(bin), bytes: bin.length },
}
const report = {
  variant,
  source: { archiveSha256: manifest.archive.sha256, gltfSha256: sha(modelBytes), binarySha256: sha(data) },
  creditsAndLicense: 'credits_license.txt',
  changeNotice: 'Meshopt triangle reduction and vertex compaction. Original node transforms, materials, images, cameras and lights.',
  recipe: { simplifier: 'meshoptimizer/simplifier', version: simplifyVersion,
    implementationSha256: sha(await readFile(simplifierPath)), flags, normalUvWeights: weights, errorLimit, keepSmall,
    traceLimit, triangleBudget, targetTriangleSum: jobs.reduce((sum, job) => sum + job.target, 0) },
  coverage: {
    sourceTriangles: total, proxyTriangles: realized, triangleFraction: realized / total,
    traceHeadroomTriangles: traceLimit - realized, sourcePrimitives: jobs.length, proxyPrimitives: coverage.length,
    omittedPrimitives: coverage.filter(row => row.proxyTriangles === 0).length,
    simplifiedPrimitives: coverage.filter(row => row.proxyTriangles < row.sourceTriangles).length,
    positionOnlyFallbackPrimitives: coverage.filter(row => row.method === 'position-only fallback').length,
    meshes: meshes.length, nodes: source.nodes.length, materials: source.materials.length,
    materialCoverage, images: source.images.length, cameras: source.cameras.length,
    punctualLights: manifest.sceneInventory.punctualLights,
    primitives: coverage,
  },
  files,
}
assert.equal(coverage.reduce((sum, row) => sum + row.proxyTriangles, 0), realized)
assert.ok(coverage.every(row => row.proxyTriangles > 0))
await mkdir(dirname(output), { recursive: true })
const stage = await mkdtemp(join(dirname(output), '.sponza-proxy-'))
try {
  for (const [name, entry] of Object.entries(retainedFiles)) {
    const original = join(sourceRoot, dirname(manifest.scene), name)
    const target = join(stage, name)
    assert.equal((await stat(original)).size, entry.bytes)
    await mkdir(dirname(target), { recursive: true })
    await link(original, target).catch(error => {
      if (error.code !== 'EXDEV') throw error
      return copyFile(original, target)
    })
  }
  await writeFile(join(stage, filename + '.gltf'), model)
  await writeFile(join(stage, filename + '.bin'), bin)
  await writeFile(join(stage, 'manifest.json'), json(report))
  await writeFile(join(stage, 'SHA256SUMS'), Object.keys(files).sort().map(name => `${files[name].sha256}  ${name}\n`).join(''))
  await chmod(stage, 0o755)
  await rename(stage, output)
} finally {
  await rm(stage, { recursive: true, force: true })
}
console.log(`${variant}: ${realized}/${total} triangles, ${coverage.length} primitives → ${output}`)
