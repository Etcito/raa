const INPUTS = ['red', 'green', 'blue', 'normalX', 'normalY', 'normalZ', 'metricDepth',
  'roughness', 'sourceValid', 'sourceConfidence', 'confidenceValid', 'freshlyTraced'];
const TENSORS = [
  ['hidden.weight', 0, [8, 12]], ['hidden.bias', 384, [8]],
  ['output.weight', 416, [3, 8]], ['output.bias', 512, [3]],
];
export const PATH_FIXTURE = 'gallery-mesh-specular/2';
const MODEL_FOLDER = 'lib/gallery-path-fit';
const MANIFEST = `${MODEL_FOLDER}/manifest.json`;
const WEIGHTS = `${MODEL_FOLDER}/weights.fp32le`;
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

export async function digest(data) {
  const buffer = await crypto.subtle.digest('SHA-256', data);
  return Array.from(new Uint8Array(buffer), byte => byte.toString(16).padStart(2, '0')).join('');
}

export async function validateGalleryFit(fit, bytes) {
  const source = fit?.captureSource;
  if (fit?.format !== 'raa.optional-head-cpu-fit/1' || fit.signal !== 'indirect-specular'
      || source?.kind !== 'windows-render' || source.fixture !== PATH_FIXTURE || !source.buildId
      || !source.facts?.hardware || !/Windows.+Chrome\//.test(source.facts.browser)
      || source.facts.adapter?.isFallbackAdapter !== false || !/^[0-9a-f]{64}$/.test(fit.captureSha256)
      || fit.layout !== 'HWC' || fit.dtype !== 'f32le' || !same(fit.inputChannels, INPUTS)
      || !same(fit.outputChannels, INPUTS.slice(0, 3)) || fit.hiddenChannels !== 8
      || fit.hiddenActivation !== 'relu' || fit.outputActivation !== 'softplus' || fit.backgroundValue !== 0
      || !same(fit.normalizationScales, INPUTS.map(name => name === 'metricDepth' ? 0.125 : 1))) {
    throw new Error('Expected a physical gallery specular fit with the mesh path input ABI');
  }
  const descriptor = fit.weights;
  if (descriptor?.file !== 'weights.fp32le' || descriptor.byteLength !== 524 || bytes.byteLength !== 524
      || descriptor.tensors?.length !== TENSORS.length || await digest(bytes) !== descriptor.sha256) {
    throw new Error('Invalid gallery path weights or SHA-256');
  }
  const values = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let offset = 0; offset < bytes.byteLength; offset += 4) {
    if (!Number.isFinite(values.getFloat32(offset, true))) throw new Error('Non-finite gallery path weight');
  }
  for (const [index, [name, offset, shape]] of TENSORS.entries()) {
    const tensor = descriptor.tensors[index];
    const length = shape.reduce((a, b) => a * b) * 4;
    if (tensor?.name !== name || tensor.offset !== offset || !same(tensor.shape, shape)
        || await digest(bytes.subarray(offset, offset + length)) !== tensor.sha256) {
      throw new Error(`Invalid gallery path tensor ${name}`);
    }
  }
  return { fit, bytes };
}

export async function loadGalleryFit(fetchFile = fetch) {
  const read = async file => {
    const response = await fetchFile(new URL(`../${file}`, import.meta.url));
    if (!response.ok) throw new Error(`Cannot load gallery path ${file} (HTTP ${response.status})`);
    return response;
  };
  const receipt = await (await read('build.json')).json();
  if (receipt.status !== 'ready') throw new Error('Gallery build receipt is not ready');
  if (!receipt.files.includes(MANIFEST) && !receipt.files.includes(WEIGHTS)) return null;
  if (!receipt.files.includes(MANIFEST) || !receipt.files.includes(WEIGHTS)) {
    throw new Error('Gallery build receipt is missing a model file');
  }
  const [manifest, weights] = await Promise.all([read(MANIFEST), read(WEIGHTS)]);
  return validateGalleryFit(await manifest.json(), new Uint8Array(await weights.arrayBuffer()));
}
