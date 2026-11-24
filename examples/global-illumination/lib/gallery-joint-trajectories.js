// Each capture is a contiguous real-frame replay with its own room variant,
// camera curve, moving geometry, light events and reference seed range.
const replays = {
  'joint-train-a':       ['train',       false, 16, 5, 11, 0.18, -0.42, -0.96,  0.22, 0.22, 0.77],
  'joint-train-b':       ['train',       true,  18, 7, 12, 1.20,  0.71, -0.25, -0.41, 1.60, 0.41],
  'joint-train-c':       ['train',       false, 16, 6, 10, -0.4, 0.38, -0.80,  0.62, 0.91, 1.33],
  'joint-development':   ['development', true,  19, 8, 13, 0.86, -0.77, 0.21,  0.33, 1.49, 0.36],
  'joint-final-a':       ['test',        true,  22, 6, 15, 1.35, -0.61, 0.54, -0.26, 0.19, 1.51],
  'joint-final-b':       ['test',        false, 21, 9, 16, -0.5, 0.48, -0.13, -0.72, 1.67, 0.18],
};

export function jointTrajectory(id) {
  if (!Object.hasOwn(replays, id)) throw new Error('Unknown joint SR trajectory');
  const [partition, heldout, frames, firstEvent, secondEvent, eyeX, moveX, eyeZ, turn,
    warmAfter, coolAfter] = replays[id];
  const ordinal = Object.keys(replays).indexOf(id);
  const replay = Array.from({ length: frames }, (_, i) => {
    const t = i / (frames - 1);
    const curve = Math.sin(Math.PI * t);
    const panelScale = [i < firstEvent ? 0.85 + ordinal * 0.05 : warmAfter,
      i < secondEvent ? 0.9 - ordinal * 0.06 : coolAfter];
    return {
      timeMs: 2000 + ordinal * 1000 + i * (1000 / 60),
      camera: {
        eye: [eyeX + moveX * t + 0.2 * curve, 1.49 + 0.09 * curve + ordinal * 0.025,
          eyeZ - 0.65 * t],
        target: [turn * (1 - 2 * t) + 0.25 * curve, 1.27 + 0.12 * t, -6.3 + 0.25 * curve],
      },
      sphere: [-1.27 + 0.29 * t + 0.41 * curve, 0.56, -3.2 + 0.12 * ordinal - 0.18 * t],
      reflector: [2.28 - 0.42 * curve, 1.29 + 0.31 * t + 0.07 * ordinal, -1.10 + 0.20 * t],
      panelScale,
      event: i === firstEvent ? 'warm-panel-step' : i === secondEvent ? 'cool-panel-step' : null,
    };
  });
  return { id, partition, heldout, replay, groupId: id, seedBase: 0x1000000 + (ordinal << 20) };
}
