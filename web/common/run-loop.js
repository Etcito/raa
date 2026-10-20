// Measured frame loop shared by demo pages (Conventions: Timing definitions, CAPTURE).
// Returns the BrowserResult (S6) page fields.

const RESIZE_TO = [960, 540];
const RESIZE_FRAMES = 30;

// Headless Chrome in nomorechat's GPU client has no begin-frame source and delivers
// requestAnimationFrame only every few seconds, so a timer stands in when rAF is late.
// Two 60 Hz vsyncs of slack keep the timer from racing a working rAF.
const FRAME_FALLBACK_MS = 34;

export function nextFrame() {
  return new Promise((resolve) => {
    let done = false;
    let timer = 0;
    const fire = (now) => {
      if (done) return;
      done = true;
      cancelAnimationFrame(id);
      clearTimeout(timer);
      resolve(now);
    };
    const id = requestAnimationFrame(fire);
    timer = setTimeout(() => fire(performance.now()), FRAME_FALLBACK_MS);
  });
}

function setSize(demo, canvas, width, height) {
  // devicePixelRatio 1: backing store equals CSS size.
  canvas.width = width;
  canvas.height = height;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  demo.resize(width, height);
}

function median(sorted) {
  const n = sorted.length;
  if (n === 0) return null;
  const mid = n >> 1;
  return n % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function percentile(sorted, p) {
  const n = sorted.length;
  if (n === 0) return null;
  // Nearest-rank percentile.
  const rank = Math.min(n, Math.max(1, Math.ceil((p / 100) * n)));
  return sorted[rank - 1];
}

async function runFrames(demo, count) {
  let ok = true;
  for (let i = 0; i < count; i++) {
    const now = await nextFrame();
    try {
      demo.frame(now);
    } catch (e) {
      ok = false;
      console.error('frame failed during resize cycle', e);
    }
  }
  return ok;
}

export async function runMeasured(demo, canvas, args) {
  const width = args.width;
  const height = args.height;
  const warmup = args.warmup;
  const frames = args.frames;
  setSize(demo, canvas, width, height);

  for (let i = 0; i < warmup; i++) {
    const now = await nextFrame();
    demo.frame(now);
  }

  const cpuMs = new Array(frames);
  for (let i = 0; i < frames; i++) {
    const now = await nextFrame();
    const t0 = performance.now();
    demo.frame(now);
    cpuMs[i] = performance.now() - t0;
  }
  const sorted = cpuMs.slice().sort((a, b) => a - b);

  let resize = null;
  if (args.resize) {
    setSize(demo, canvas, RESIZE_TO[0], RESIZE_TO[1]);
    const okTo = await runFrames(demo, RESIZE_FRAMES);
    setSize(demo, canvas, width, height);
    const okBack = await runFrames(demo, RESIZE_FRAMES);
    resize = { to: [RESIZE_TO[0], RESIZE_TO[1]], back: [width, height], ok: okTo && okBack };
  }

  const nonEmpty = await demo.probe_non_empty(performance.now());
  const stats = demo.stats();

  return {
    adapter: stats.adapter,
    hardware: stats.hardware,
    canvas: [width, height],
    nonEmpty: nonEmpty === true,
    firstFrameMs: stats.firstFrameMs,
    warmup,
    frames,
    cpuMsMedian: median(sorted),
    cpuMsP95: percentile(sorted, 95),
    instances: stats.instances,
    drawCalls: stats.drawCalls,
    triangles: stats.triangles,
    allocsPerFrameTotal: stats.allocsPerFrameTotal ?? null,
    resize,
  };
}
