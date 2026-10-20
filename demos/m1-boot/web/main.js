// m1-boot page: clear + triangle, driven by requestAnimationFrame, with the `run` and
// `render` ops for nomorechat's GPU client (window.nmcGpu.ops, see common/nmc-gpu.js).
import init, { start } from './pkg/m1_boot.js';
import { nextFrame, runMeasured } from './common/run-loop.js';
import { registerNmcGpu } from './common/nmc-gpu.js';

const canvas = document.getElementById('c');

function showError(error) {
  const pre = document.createElement('pre');
  const code = error && error.code ? error.code : 'page.start_failed';
  const message = error && error.message ? error.message : String(error);
  pre.textContent = `${code}: ${message}`;
  document.body.appendChild(pre);
}

// The ops are registered before the demo starts, so the client never finds them missing;
// they wait for the start to settle and reject with its error when it failed.
let opRunning = false;
const demoReady = main();

registerNmcGpu({
  run: async (args) => {
    const demo = await demoReady;
    if (opRunning) throw new Error('run: an op run is already in progress');
    opRunning = true;
    try {
      return await runMeasured(demo, canvas, args);
    } finally {
      opRunning = false;
    }
  },
  // Draw one frame and read the canvas in the same task, before it is presented.
  screenshot: async () => {
    const demo = await demoReady;
    demo.frame(performance.now());
    return canvas.toDataURL('image/png').split(',')[1];
  },
});

async function main() {
  let demo;
  try {
    await init();
    demo = await start(canvas, {});
  } catch (error) {
    console.error(error);
    showError(error);
    throw error;
  }

  // The idle loop pauses while an op run drives frames itself, so no frame is doubled.
  let loggedFrameError = false;
  const loop = (now) => {
    if (!opRunning) {
      try {
        demo.frame(now);
      } catch (error) {
        if (!loggedFrameError) console.error(error);
        loggedFrameError = true;
      }
    }
    nextFrame().then(loop);
  };
  nextFrame().then(loop);
  return demo;
}
