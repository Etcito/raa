const fields = [
  { key: 'radius', label: 'Radius (m)', min: 0.05, max: 10, step: 0.05, initial: 0.8 },
  { key: 'intensity', label: 'Strength', min: 0, max: 1, step: 0.05, initial: 0.6 },
  { key: 'power', label: 'Power', min: 0.1, max: 8, step: 0.1, initial: 1 },
  { key: 'slices', label: 'Slices', min: 1, max: 8, step: 1, initial: 4 },
  { key: 'steps', label: 'Steps per side', min: 1, max: 16, step: 1, initial: 8 },
];

export function createGalleryAoControls() {
  const defaults = Object.fromEntries(fields.map(field => [field.key, field.initial]));
  let settings = defaults;
  let enabled = true;
  let appliedEngine;
  let appliedSettings;
  let appliedEnabled;

  function mount(container) {
    const doc = container.ownerDocument;
    const group = doc.createElement('fieldset');
    group.id = 'gallery-ao';
    Object.assign(group.style, {
      display: 'flex', flexWrap: 'wrap', gap: '10px 16px', alignItems: 'center',
      flexBasis: '100%', minWidth: '0', border: '1px solid #41516a', borderRadius: '6px',
    });

    const title = doc.createElement('legend');
    title.textContent = 'Ambient occlusion (GTAO)';
    group.append(title);

    const toggle = doc.createElement('input');
    toggle.type = 'checkbox';
    toggle.id = 'gallery-ao-enabled';
    toggle.checked = enabled;
    const toggleLabel = doc.createElement('label');
    toggleLabel.append(toggle, 'Enabled');
    group.append(toggleLabel);

    const inputs = fields.map(({ key, label, min, max, step }) => {
      const control = doc.createElement('input');
      control.id = `gallery-ao-${key}`;
      control.type = 'range';
      control.min = min;
      control.max = max;
      control.step = step;
      control.style.width = '6rem';
      const value = doc.createElement('output');
      value.setAttribute('for', control.id);
      const fieldLabel = doc.createElement('label');
      fieldLabel.append(label, control, value);
      group.append(fieldLabel);
      control.addEventListener('input', () => {
        const next = control.valueAsNumber;
        settings = { ...settings, [key]: next };
        value.value = control.value;
      });
      return { key, control, value };
    });

    function syncInputs() {
      toggle.checked = enabled;
      for (const { key, control, value } of inputs) {
        control.value = settings[key];
        value.value = control.value;
        control.disabled = !enabled;
      }
    }

    toggle.addEventListener('change', () => {
      enabled = toggle.checked;
      syncInputs();
    });
    const reset = doc.createElement('button');
    reset.type = 'button';
    reset.textContent = 'Reset AO';
    reset.addEventListener('click', () => {
      enabled = true;
      settings = defaults;
      syncInputs();
    });
    group.append(reset);
    syncInputs();
    container.append(group);
  }

  function apply(engine) {
    const newEngine = engine !== appliedEngine;
    if (newEngine || appliedSettings !== settings) engine.setAo(settings);
    if (newEngine || enabled !== appliedEnabled) engine.setPassEnabled('ao.gtao', enabled);
    appliedEngine = engine;
    appliedSettings = settings;
    appliedEnabled = enabled;
  }

  return { mount, apply };
}
