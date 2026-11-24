const refreshMs = 500;
const unavailable = '—';
const rebuildReasons = {
  allocation: 'volume allocation',
  sceneChanged: 'scene change',
  cameraMoved: 'camera recenter',
  cameraAndSceneChanged: 'camera recenter and scene change',
};

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

function gpuTimingLines(timing) {
  if (timing?.status === 'pending') return ['GPU timings pending'];
  if (timing?.status !== 'sample') {
    const reason = timing?.reason ?? 'timestamp data unavailable';
    return [`GPU timings unavailable (${reason})`];
  }
  const { frame, width, height, passes } = timing;
  if (!passes.some(({ ms }) => Number.isFinite(ms) && ms >= 0)) {
    return ['GPU timings unavailable (invalid timestamp data)'];
  }
  return [
    `GPU timings · frame ${frame} · ${extent(width, height)}`,
    ...passes.map(({ pass, ms }) => `${pass}  ${Number.isFinite(ms) && ms >= 0 ? `${number(ms, 3)} ms` : 'unavailable'}`),
  ];
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
      if (gi.active && Number.isInteger(gi.rebuildPendingLevels)) {
        const details = [`GI levels pending ${gi.rebuildPendingLevels}`];
        const reason = rebuildReasons[gi.rebuildReason];
        if (reason) {
          let rebuild = `GI rebuild ${reason}`;
          if (gi.rebuildLevel != null) {
            rebuild += ` · level ${gi.rebuildLevel} · ${gi.rebuildFloodPassesRemaining} flood passes left`;
          }
          details.push(rebuild);
        }
        details.push(`GI levels relit ${number(gi.relitLevels)}`);
        lines.push(details.join('  ·  '));
      }
    }
    const tracing = optional(() => engine.meshTracingStats());
    if (tracing?.referencePixels) {
      lines.push(`Path-traced reference · ${number(tracing.referencePixels)} pixels × ${tracing.referenceSpp} spp · ${tracing.pathBounces} path bounces`);
    } else if (tracing?.hybridLobeSpp) {
      lines.push(`Hybrid path · model ${number(tracing.modelPixels)} · traced refresh ${number(tracing.refreshPixels)} · mirror screen ${number(tracing.acceptedScreen)}`);
      lines.push(`Diffuse evidence missing ${number(tracing.diffuseMissingPixels)} · diffuse rays ${number(tracing.diffuseRays)} · specular rays ${number(tracing.specularRays)}`);
    } else if (tracing?.eligible) {
      const share = 100 * tracing.traced / tracing.eligible;
      lines.push(`Hybrid reflections · GPU traced ${tracing.traced}/${tracing.eligible} (${number(share, 1)}%) · screen accepted ${tracing.acceptedScreen}`);
    }
    lines.push(...gpuTimingLines(optional(() => engine.passGpuTimings())));
    const images = passLines(optional(() => engine.passImageExtents()), width, height);
    if (images.length) lines.push('Internal pass textures', ...images);
    container.textContent = lines.join('\n');
  }

  return { update, reset };
}
