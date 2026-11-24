// Keep the typed screen stages and the area-light route from the production graph.
export async function reflectionGraph() {
  const response = await fetch(new URL('../shared/real-frame-model.json', import.meta.url));
  if (!response.ok) throw new Error(`SSR capture graph returned HTTP ${response.status}`);
  const graph = await response.json();
  graph.name = 'Offline screen incident capture';
  const excluded = new Set(graph.nodes.filter(node => node.type.startsWith('neural.')).map(node => node.id));
  graph.nodes = graph.nodes.filter(node => !excluded.has(node.id));
  graph.links = graph.links.filter(link => [link.from, link.to].every(port => !excluded.has(port.split('.')[0])));
  graph.links.push({ from: 'trace.raw', to: 'mask.src' }, { from: 'incident.incident', to: 'replace.incident' },
    { from: 'replace.color', to: 'tonemap.color' });
  return graph;
}

export function reflectionInputs(incident, meta, trace, minimum = 25) {
  const pixels = incident.length / 4;
  const input = new Float32Array(pixels * 6);
  let accepted = 0;
  for (let p = 0; p < pixels; p++) {
    const lane = p * 4;
    const valid = meta[lane] === 1 && meta[lane + 1] === 1 && incident[lane + 3] > 0;
    if (valid) {
      input.set([...incident.subarray(lane, lane + 3), 1, meta[lane], meta[lane + 3]], p * 6);
      accepted++;
    }
  }
  if (accepted < minimum) throw new Error(`Only ${accepted} accepted SSR screen hits`);
  return { input, trace, accepted };
}

function texel(level, x, y, c) {
  const i = Math.max(0, Math.min(level.w - 1, x)), j = Math.max(0, Math.min(level.h - 1, y));
  return level.rgb[(i + j * level.w) * 3 + c];
}

function linear(level, u, v, c) {
  const x = u * level.w - 0.5, y = v * level.h - 0.5, i = Math.floor(x), j = Math.floor(y);
  const a = x - i, b = y - j;
  return (1 - b) * ((1 - a) * texel(level, i, j, c) + a * texel(level, i + 1, j, c))
    + b * ((1 - a) * texel(level, i, j + 1, c) + a * texel(level, i + 1, j + 1, c));
}

const coneTaps = [
  [0, 0, 0.125], [-1, -1, 0.125], [1, -1, 0.125], [-1, 1, 0.125], [1, 1, 0.125],
  [0, -2, 0.0625], [-2, 0, 0.0625], [2, 0, 0.0625], [0, 2, 0.0625],
  [-2, -2, 0.03125], [2, -2, 0.03125], [-2, 2, 0.03125], [2, 2, 0.03125],
];

// Match ssr_hiz.wgsl's 13-tap cone chain on the scene grid. Level zero samples
// the independent 2x teacher grid after removing the student's camera jitter.
function coneLevels(hdr, sceneSize, displaySize, jitter) {
  const [w, h] = sceneSize, [dw, dh] = displaySize;
  const high = { w: dw, h: dh, rgb: hdr };
  const first = Float32Array.from({ length: w * h * 3 }, (_, i) => {
    const p = Math.floor(i / 3);
    return linear(high, (p % w + 0.5 - jitter[0]) / w, (Math.floor(p / w) + 0.5 - jitter[1]) / h, i % 3);
  });
  const levels = [{ w, h, rgb: first }];
  while (levels.at(-1).w > 1 || levels.at(-1).h > 1) {
    const prior = levels.at(-1), sw = Math.max(1, prior.w >> 1), sh = Math.max(1, prior.h >> 1);
    const rgb = Float32Array.from({ length: sw * sh * 3 }, (_, i) => {
      const p = Math.floor(i / 3), u = (p % sw + 0.5) / sw, v = (Math.floor(p / sw) + 0.5) / sh;
      return coneTaps.reduce((sum, [x, y, weight]) =>
        sum + weight * linear(prior, u + x / prior.w, v + y / prior.h, i % 3), 0);
    });
    levels.push({ w: sw, h: sh, rgb });
  }
  return levels;
}

export function reflectionTeacher(guides, independentBatches, sceneSize, displaySize, jitter, tolerance) {
  const target = new Float32Array(guides.trace.length / 4 * 3);
  if (guides.accepted === 0) return { target, logRmse: null, linearRmse: null, tolerance, accepted: 0 };
  const chains = independentBatches.map(hdr => coneLevels(hdr, sceneSize, displaySize, jitter));
  let logError = 0, linearError = 0;
  for (let p = 0; p < target.length / 3; p++) {
    if (guides.input[p * 6 + 3] !== 1) continue;
    const [u, v, , cone] = guides.trace.subarray(p * 4, p * 4 + 4);
    const mip = Math.max(0, Math.min(chains[0].length - 1, cone));
    const lo = Math.floor(mip), hi = Math.ceil(mip), blend = mip - lo;
    for (let c = 0; c < 3; c++) {
      const [a, b] = chains.map(levels => (1 - blend) * linear(levels[lo], u, v, c) + blend * linear(levels[hi], u, v, c));
      target[p * 3 + c] = (a + b) / 2;
      logError += (Math.log1p(Math.max(0, a)) - Math.log1p(Math.max(0, b))) ** 2;
      linearError += (a - b) ** 2;
    }
  }
  const lanes = guides.accepted * 3;
  const logRmse = Math.sqrt(logError / lanes), linearRmse = Math.sqrt(linearError / lanes);
  if (!(logRmse <= tolerance)) throw new Error(`Independent incident batches disagreed: ${logRmse} > ${tolerance}`);
  return { target, logRmse, linearRmse, tolerance, accepted: guides.accepted };
}
