import assert from 'node:assert/strict';
import test from 'node:test';
import { createGalleryAoControls } from './lib/gallery-ao.js';

class Element extends EventTarget {
  constructor(tag, ownerDocument) {
    super();
    this.tag = tag;
    this.ownerDocument = ownerDocument;
    this.children = [];
    this.style = {};
    this.value = '';
  }

  append(...children) {
    this.children.push(...children);
  }

  setAttribute(name, value) {
    this[name] = value;
  }

  get valueAsNumber() {
    return Number(this.value);
  }
}

function recordingEngine() {
  const calls = [];
  return {
    calls,
    setAo: settings => calls.push(['settings', { ...settings }]),
    setPassEnabled: (name, enabled) => calls.push([name, enabled]),
  };
}

test('gallery AO edits and enable state reach the renderer and survive an engine rebuild', () => {
  const doc = { createElement: tag => new Element(tag, doc) };
  const container = doc.createElement('div');
  const controls = createGalleryAoControls();
  const engine = recordingEngine();
  controls.mount(container);
  const group = container.children[0];
  const [toggle, ...sliders] = group.children
    .filter(element => element.tag === 'label').map(label => label.children.find(element => element.tag === 'input'));
  const defaults = { radius: 0.8, intensity: 0.6, power: 1, slices: 4, steps: 8, indirectDiffuse: true };
  controls.apply(engine);
  controls.apply(engine);
  assert.deepEqual(engine.calls, [['settings', defaults], ['ao.gtao', true]]);

  const settings = { radius: 1.5, intensity: 0.25, power: 2, slices: 6, steps: 12, indirectDiffuse: true };
  for (const slider of sliders) {
    slider.value = settings[slider.id.replace('gallery-ao-', '')];
    slider.dispatchEvent(new Event('input'));
  }
  controls.apply(engine);
  assert.deepEqual(engine.calls.at(-1), ['settings', settings]);

  toggle.checked = false;
  toggle.dispatchEvent(new Event('change'));
  controls.apply(engine);
  assert.deepEqual(engine.calls.at(-1), ['ao.gtao', false]);
  assert.ok(sliders.every(input => input.disabled));

  const rebuilt = recordingEngine();
  controls.apply(rebuilt);
  assert.deepEqual(rebuilt.calls, [['settings', settings], ['ao.gtao', false]]);

  group.children.find(element => element.tag === 'button').dispatchEvent(new Event('click'));
  controls.apply(rebuilt);
  assert.deepEqual(rebuilt.calls.slice(-2), [['settings', defaults], ['ao.gtao', true]]);
  assert.ok(sliders.every(input => !input.disabled));

  const notice = group.children.find(element => element.role === 'status');
  const reset = group.children.find(element => element.tag === 'button');
  assert.ok(notice.hidden);
  controls.apply(rebuilt, true);
  assert.deepEqual(rebuilt.calls.at(-1), ['ao.gtao', false]);
  assert.ok([toggle, reset, ...sliders].every(input => input.disabled));
  assert.equal(notice.hidden, false);
  assert.match(notice.textContent, /reference bypasses artistic AO/);

  controls.apply(rebuilt, false);
  assert.deepEqual(rebuilt.calls.at(-1), ['ao.gtao', true]);
  assert.ok([toggle, reset, ...sliders].every(input => !input.disabled));
  assert.equal(notice.hidden, true);
});
