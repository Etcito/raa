const refreshMs = 500;
const unavailable = '—';

const number = (value, digits = 0) => Number.isFinite(value) ? value.toFixed(digits) : unavailable;
const extent = (width, height) => `${width} × ${height}`;

function optional(read) {
  try {
    return read();
  } catch {
    return undefined;
  }
}

function passLines(images, width, height) {
  const passes = new Map();
  for (const image of images ?? []) {
    const { pass, port, width: w, height: h, layers, mips } = image;
    if (w === width && h === height && layers === 1 && mips === 1) continue;
    if (!passes.has(pass)) passes.set(pass, []);
    let size = extent(w, h);
    if (layers > 1) size += ` × ${layers}`;
    if (mips > 1) size += ` (${mips} mips)`;
    passes.get(pass).push(`${port} ${size}`);
  }
  return Array.from(passes, ([pass, ports]) => `${pass}  ${ports.join(' · ')}`);
}

export function createLiveStats(container) {
  let currentEngine;
  let sample;
  let outputSize;
  let fps;

  function reset() {
    currentEngine = undefined;
    sample = undefined;
    outputSize = undefined;
    fps = undefined;
    container.textContent = '';
  }

  // Call after each rendered frame with a monotonic wall-clock timestamp in milliseconds.
  function update(engine, timeMs, width, height) {
    if (engine !== currentEngine) {
      reset();
      currentEngine = engine;
    }
    if (!Number.isFinite(timeMs)) return;
    const size = extent(width, height);
    const elapsed = sample ? timeMs - sample.timeMs : 0;
    if (sample && elapsed >= 0 && elapsed < refreshMs && size === outputSize) return;

    const runtime = engine.stats();
    const frames = runtime.frames;
    if (!sample || elapsed < 0 || frames < sample.frames || !Number.isFinite(frames)) {
      fps = undefined;
      sample = { timeMs, frames };
    } else if (elapsed >= refreshMs) {
      fps = Number.isFinite(sample.frames) ? 1000 * (frames - sample.frames) / elapsed : undefined;
      sample = { timeMs, frames };
    }
    outputSize = size;

    const lines = [
      `Rendered FPS ${number(fps, 1)}  ·  Output ${size}  ·  Submitted frames ${number(frames)}`,
      `CPU frame median ${number(runtime.cpuMsMedian, 2)} ms  ·  p95 ${number(runtime.cpuMsP95, 2)} ms`,
      `Instances ${number(runtime.instances)}  ·  Draw calls ${number(runtime.drawCalls)}  ·  Triangles ${number(runtime.triangles)}`,
    ];
    const gi = optional(() => engine.giStats());
    if (gi) {
      const state = gi.active ? (gi.pendingRebuild ? 'rebuilding' : 'active') : (gi.inactiveReason ?? 'inactive');
      lines.push(`GI volume ${state}  ·  GI memory ${number(gi.allocatedBytes / (1024 * 1024), 1)} MiB`);
    }
    const images = passLines(optional(() => engine.passImageExtents()), width, height);
    if (images.length) lines.push('Internal pass textures', ...images);
    container.textContent = lines.join('\n');
  }

  return { update, reset };
}
