// Registers the page ops for nomorechat's GPU client.
//
// window.nmcGpu.ops.run(args) resolves to the S6 page fields. The client itself calls
// the op named on its command line (`render`) with the whole input file, and keeps
// `{ json, files: [{ name, base64 }] }` from the resolved value; it takes no screenshot
// of its own. `render` therefore runs `run(input.args)` and attaches the PNG returned
// by the optional `screenshot()` (base64, taken after run resolved) as `capture.png`.
export function registerNmcGpu({ run, screenshot }) {
  if (typeof run !== 'function') throw new TypeError('registerNmcGpu: run must be a function');
  if (screenshot !== undefined && typeof screenshot !== 'function') {
    throw new TypeError('registerNmcGpu: screenshot must be a function');
  }
  const render = async (input) => {
    const json = await run(input && input.args ? input.args : input);
    const files = screenshot ? [{ name: 'capture.png', base64: await screenshot() }] : [];
    return { json, files };
  };
  window.nmcGpu = { ops: { run, render } };
  window.nmcGpu.ready = true;
}
