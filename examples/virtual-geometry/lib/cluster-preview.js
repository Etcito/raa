// Cluster preview of the virtual-geometry page. Pure ES module (no DOM, no engine, importable by
// node). cutPreviewMesh asks the module's uniform cut (raa.clusterCutMesh, S31) and returns its
// arrays unchanged plus one linear colour per vertex: every cluster has its own hashed colour,
// drawn on the existing forward path through the S13 vertex colours.

const GOLDEN_RATIO_CONJUGATE = 0.61803398875;
const SATURATION = 0.65;
const VALUE = 0.9;

// sRGB transfer to linear, per channel (the exact piecewise EOTF).
function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

// HSV (h in [0, 1), s, v) to sRGB components in [0, 1].
function hsvToRgb(h, s, v) {
  const h6 = h * 6;
  const i = Math.floor(h6);
  const f = h6 - i;
  const p = v * (1 - s);
  const q = v * (1 - s * f);
  const t = v * (1 - s * (1 - f));
  switch (i % 6) {
    case 0: return [v, t, p];
    case 1: return [q, v, p];
    case 2: return [p, v, t];
    case 3: return [p, q, v];
    case 4: return [t, p, v];
    default: return [v, p, q];
  }
}

// The linear colour of a cluster index: hue (index * golden ratio conjugate) mod 1, saturation
// 0.65, value 0.9, converted from sRGB to linear.
export function clusterColor(clusterIndex) {
  if (!Number.isSafeInteger(clusterIndex) || clusterIndex < 0) {
    throw new RangeError(`vg cluster preview: clusterIndex must be a non-negative integer, got ${clusterIndex}`);
  }
  const hue = (clusterIndex * GOLDEN_RATIO_CONJUGATE) % 1;
  return hsvToRgb(hue, SATURATION, VALUE).map(srgbToLinear);
}

// The uniform cut at errorMetres as a MeshDesc with colors (4 linear floats per vertex, alpha 1).
// A vertex belongs to the cluster of any triangle that uses it (cluster vertices are never shared
// across clusters). A vertex no triangle uses has no cluster and takes cluster 0's colour.
export function cutPreviewMesh(raa, section, errorMetres) {
  const cut = raa.clusterCutMesh(section, errorMetres);
  const vertexCount = cut.positions.length / 3;
  const colors = new Float32Array(4 * vertexCount);
  const clusterOfVertex = new Uint32Array(vertexCount);
  const assigned = new Uint8Array(vertexCount);
  for (let t = 0; t < cut.clusterOfTriangle.length; t++) {
    for (let k = 0; k < 3; k++) {
      const v = cut.indices[3 * t + k];
      if (!assigned[v]) {
        assigned[v] = 1;
        clusterOfVertex[v] = cut.clusterOfTriangle[t];
      }
    }
  }
  const palette = new Map();
  for (let v = 0; v < vertexCount; v++) {
    const c = clusterOfVertex[v];
    if (!palette.has(c)) palette.set(c, clusterColor(c));
    colors.set(palette.get(c), 4 * v);
    colors[4 * v + 3] = 1;
  }
  return {
    positions: cut.positions,
    normals: cut.normals,
    uvs: cut.uvs,
    indices: cut.indices,
    colors,
  };
}
