const PADDING = 8;
const LINE_HEIGHT = 18;
const FONT = '14px monospace';
const BACKGROUND = '#0b0d12';
const FOREGROUND = '#f2f2f2';

function fail(message) { throw new Error(`sky capture: ${message}`); }

function displayedText(status) {
  if (!status || typeof status !== 'object') fail('status element is required');
  const shown = typeof status.innerText === 'string' && status.innerText.trim().length > 0
    ? status.innerText
    : status.textContent;
  return typeof shown === 'string' ? shown : '';
}

function createOutput(canvas) {
  const document = canvas.ownerDocument ?? (typeof globalThis.document === 'object' ? globalThis.document : null);
  if (!document || typeof document.createElement !== 'function') {
    fail('a document is required to compose the canvas and status region');
  }
  const output = document.createElement('canvas');
  if (!output || typeof output.getContext !== 'function') fail('cannot create the capture canvas');
  return output;
}

function measure(context, text) {
  if (typeof context.measureText === 'function') return context.measureText(text).width;
  return text.length * 8;
}

function wrapLines(context, text, maxWidth) {
  const source = text.replace(/\r\n?/g, '\n').trim();
  if (!source) return [];
  const lines = [];
  for (const paragraph of source.split('\n')) {
    const words = paragraph.split(/\s+/).filter(Boolean);
    if (words.length === 0) { lines.push(''); continue; }
    let line = '';
    for (const word of words) {
      const candidate = line ? `${line} ${word}` : word;
      if (line && measure(context, candidate) > maxWidth) {
        lines.push(line);
        line = word;
      } else {
        line = candidate;
      }
    }
    if (line) lines.push(line);
  }
  return lines;
}

export async function captureCurrent(canvas, status) {
  if (!canvas || typeof canvas !== 'object') fail('canvas is required');
  const width = canvas.width;
  const height = canvas.height;
  if (!Number.isSafeInteger(width) || width <= 0 || !Number.isSafeInteger(height) || height <= 0) {
    fail('canvas must have positive integer pixel dimensions');
  }

  const text = displayedText(status);
  const rect = typeof status.getBoundingClientRect === 'function' ? status.getBoundingClientRect() : null;
  const output = createOutput(canvas);
  const context = output.getContext('2d', { alpha: false, willReadFrequently: true });
  if (!context) fail('2D canvas context is unavailable');

  context.font = FONT;
  const lines = wrapLines(context, text, Math.max(1, width - PADDING * 2));
  const displayedHeight = rect && Number.isFinite(rect.height) && rect.height > 0 ? Math.ceil(rect.height) : 0;
  const statusHeight = lines.length > 0
    ? Math.max(displayedHeight, lines.length * LINE_HEIGHT + PADDING * 2)
    : displayedHeight;

  output.width = width;
  output.height = height + statusHeight;
  context.fillStyle = BACKGROUND;
  context.fillRect(0, 0, output.width, output.height);
  try {
    context.drawImage(canvas, 0, 0, width, height);
  } catch (error) {
    fail(`cannot draw the current GPU canvas: ${error && error.message ? error.message : String(error)}`);
  }
  if (statusHeight > 0) {
    context.fillStyle = BACKGROUND;
    context.fillRect(0, height, output.width, statusHeight);
    context.fillStyle = FOREGROUND;
    context.textBaseline = 'top';
    context.font = FONT;
    lines.forEach((line, index) => context.fillText(line, PADDING, height + PADDING + index * LINE_HEIGHT));
  }
  // A lost 2D context encodes an all-transparent PNG (not even the opaque background survives).
  if (typeof context.isContextLost === 'function' && context.isContextLost()) {
    fail('the capture canvas lost its 2D context (browser GPU process reset); its PNG would be blank');
  }

  const url = typeof output.toDataURL === 'function' ? output.toDataURL('image/png') : '';
  const prefix = 'data:image/png;base64,';
  if (typeof url !== 'string' || !url.startsWith(prefix) || url.length === prefix.length) {
    fail('the composed canvas did not produce a PNG data URL');
  }
  return url.slice(prefix.length);
}
