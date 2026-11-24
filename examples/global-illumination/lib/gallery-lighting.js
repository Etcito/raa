export const lightingOptions = [
  { id: 'lighting-view', label: 'Lighting', value: 'combined', choices: [
    ['combined', 'Combined'], ['direct', 'Direct only'], ['indirect', 'Indirect only'], ['bounce', 'Exact bounce depth'],
  ] },
  { id: 'bounce-depth', label: 'Ray bounce depth', value: 1, min: 0, max: 12, step: 1 },
  { id: 'indirect-solver', label: 'Volume indirect solver', value: 'probes', choices: [
    ['probes', 'Probes'], ['restir', 'ReSTIR GI'],
  ] },
];

export function mountLightingNotice(container, controls) {
  const notice = container.ownerDocument.createElement('p');
  notice.id = 'gallery-lighting-applicability';
  notice.setAttribute('aria-live', 'polite');
  notice.style.flexBasis = '100%';
  notice.style.margin = '0';
  container.append(notice);
  const view = controls.get('lighting-view');
  const depth = controls.get('bounce-depth');
  const solver = controls.get('indirect-solver');
  for (const control of [view, depth, solver]) control.setAttribute('aria-describedby', notice.id);
  const exact = [...view.options].find(option => option.value === 'bounce');

  let previous;
  return state => {
    const reference = state.enabled['path-reference'];
    const model = state.enabled['model-assisted-path'];
    const selected = state.values['lighting-view'];
    const isolated = selected !== 'combined';
    const maxDepth = state.values['path-bounces']
      + (state.enabled['geometry-reflections'] ? state.values['reflection-bounces'] : 0);
    const selection = [reference, model, selected, state.values['bounce-depth'], maxDepth].join('|');
    if (selection === previous) return;
    previous = selection;
    exact.disabled = !reference;
    view.value = selected;
    depth.disabled = selected !== 'bounce';
    solver.disabled = reference || model;
    for (const id of ['ssr', 'dof', 'bloom']) controls.get(id).disabled = isolated;
    const applicability = reference
      ? 'Depth 0 shows direct lighting, visible emitters, and sky. Each scattering or mirror continuation adds one bounce.'
      : model
        ? 'Model-assisted indirect lighting includes traced and predicted radiance. Exact bounce depth requires Path-traced reference.'
        : 'Volume GI estimates indirect lighting. Exact bounce depth requires Path-traced reference.';
    const budget = selected === 'bounce'
      ? ` Current bounce budget: ${maxDepth}. Depths above this budget are black.`
      : '';
    const paused = reference || model
      ? ' Post effects are paused in isolated views.'
      : ' Reflections and post effects are paused in isolated views.';
    notice.textContent = applicability + budget + (isolated ? paused : '');
  };
}
