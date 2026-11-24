export async function measureMirrorPriority({ engine, set, settle, capture, render, check, compare, camera }) {
  const results = [];
  const snapshot = async id => {
    const sample = await capture(id, true, true);
    const { trace, cache } = sample;
    const primary = [trace.sphericalPrimarySupported, trace.sphericalPrimaryInGrid];
    const reflected = [trace.sphericalDiffuseOffscreenSupported, trace.sphericalDiffuseOffscreenInGrid];
    check([...primary, ...reflected, cache.reflectedCells, cache.reflectedReadyCells].every(Number.isInteger),
      'Mirror priority telemetry unavailable', { trace, cache });
    check(primary[0] <= primary[1] && reflected[0] <= reflected[1] && cache.cells <= 128
      && cache.reflectedCells <= 32 && cache.reflectedReadyCells <= cache.reflectedCells,
    'Mirror-priority telemetry exceeded the shared budget or receiver counts', { trace, cache });
    return { id, primary, reflected, firstHitHash: trace.firstHitHash, firstHitHits: trace.firstHitHits, cache };
  };

  try {
    for (const [index, policy] of ['measure', 'on', 'measure'].entries()) {
      set('gi-grid', 'volume');
      engine.setCamera(camera);
      engine.setMeshTracing({ mirrorPriority: policy });
      set('gi-grid', 'spherical');
      await settle('spherical');
      const measured = await snapshot(`priority-${index}-fixed`);
      const forward = await capture(`priority-${index}-forward`, false);
      check(measured.primary[1] > 0 && measured.reflected[1] > 0 && measured.firstHitHits > 0,
        'Fixed pose was missing primary or reflected coverage', measured);
      if (policy === 'measure') check(measured.cache.reflectedCells === 0,
        'Measurement-only policy reserved cache entries', measured.cache);
      else check(measured.cache.reflectedReadyCells > 0, 'Priority admitted no ready reflected cells', measured.cache);
      results.push({ policy, measured, forward });
    }
    const [a, b, restored] = results;
    check([b, restored].every(row => row.measured.firstHitHash === a.measured.firstHitHash
      && row.measured.firstHitHits === a.measured.firstHitHits), 'Priority changed true first geometry intersections', results.map(row => row.measured));
    const translate = dx => ({ ...camera, eye: [camera.eye[0] + dx, ...camera.eye.slice(1)],
      target: [camera.target[0] + dx, ...camera.target.slice(1)] });
    const motion = [];
    for (const policy of ['measure', 'on']) {
      set('gi-grid', 'volume');
      engine.setCamera(camera);
      engine.setMeshTracing({ mirrorPriority: policy });
      set('gi-grid', 'spherical');
      await settle('spherical');
      const home = await snapshot(`priority-motion-${policy}-home`);
      const walk = [];
      for (let step = 1; step <= 32; step++) {
        engine.setCamera(translate(0.02 * step));
        if (step % 4 === 0) walk.push({ step, ...await snapshot(`priority-motion-${policy}-step-${step}`) });
        else await render();
      }
      const last = walk.at(-1);
      const epochs = walk.map(sample => sample.cache.originEpoch);
      check(epochs.slice(0, 3).every(epoch => epoch === home.cache.originEpoch)
        && epochs.slice(1).every((epoch, i) => epoch >= epochs[i])
        && last.cache.originEpoch > home.cache.originEpoch,
        'Walk did not preserve the initial tolerance and recenter monotonically', epochs);
      engine.setCamera(translate(1.35));
      const recenter = await snapshot(`priority-motion-${policy}-recenter-first`);
      check(recenter.cache.originEpoch > last.cache.originEpoch && recenter.cache.readyFaces < last.cache.readyFaces,
        'Camera recenter mixed incompatible old and new faces', { last, recenter });
      await settle('spherical');
      const warmed = await snapshot(`priority-motion-${policy}-recenter-ready`);
      check(warmed.cache.originEpoch === recenter.cache.originEpoch && warmed.cache.readyCells > 0,
        'Recentered priority cache did not settle', warmed.cache);
      motion.push({ policy, home, walk, recenter, warmed });
    }
    const matchingMotionHits = motion[0].walk.every((sample, i) => sample.firstHitHash === motion[1].walk[i].firstHitHash
      && sample.firstHitHits === motion[1].walk[i].firstHitHits);
    check(matchingMotionHits, 'Priority changed first geometry intersections at matching motion poses');
    return { cap: 128, reservedLimit: 32, fixed: results.map(({ policy, measured }) => ({ policy, ...measured })),
      primaryHdr: compare(a.forward.rgb, b.forward.rgb), motion, matchingMotionHits };
  } finally {
    engine.setMeshTracing({ mirrorPriority: 'off' });
    engine.setCamera(camera);
  }
}
