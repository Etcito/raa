// Game render options on RAA: the player-facing graphics settings of an sdfm game (MAMA's
// GameRenderOptions contract: src/core/game/render-options.ts) applied to one engine. Plain values
// in, engine calls out; nothing here reads MAMA. Options the engine cannot apply (an older
// @etcito/raa without the call, or an option with no engine side) are reported, never faked.

/**
 * GTAO sampling effort per AO quality level (MAMA `aoQuality`): level 0 turns the `ao.gtao`
 * pass off; 3 is RAA's default (4 slices x 8 steps).
 */
export const AO_QUALITY_LEVELS = Object.freeze([
  null,
  Object.freeze({ slices: 2, steps: 4 }),
  Object.freeze({ slices: 3, steps: 6 }),
  Object.freeze({ slices: 4, steps: 8 }),
  Object.freeze({ slices: 6, steps: 12 }),
]);

/** Options with no engine side: applied elsewhere (the loader) or not applicable today. */
const NOT_ENGINE = Object.freeze({
  shading: 'applied by the part loader when the meshes upload (field material, vertex colours or baked textures), not by the engine',
  bakeTexelDensity: 'used by the bake when the meshes upload, not by the engine',
  bakeMaxSize: 'used by the bake when the meshes upload, not by the engine',
  frameRate: 'frame pacing belongs to the page loop, not the engine',
  showFps: 'the frame counter belongs to the page overlay, not the engine',
  lodOffset: 'RAA has no mesh LOD selection and the game meshes carry one level',
});

/**
 * Applies `options` to `engine`. `look` says which post passes the current scene look uses
 * (`applyEnvironment` with a post look turns bloom and grading on); an option only turns a pass
 * off, never on against the look. Call it again after every environment change.
 * Returns the keys applied and, for every other key given, why it was not.
 */
export function applyRenderOptions(engine, options, look = { bloom: true, grading: true }) {
  const applied = [];
  const skipped = [];
  const can = (name) => typeof engine?.[name] === 'function';
  const skip = (key, reason) => skipped.push({ key, reason });
  for (const key of Object.keys(options ?? {})) {
    const value = options[key];
    if (value === undefined) continue;
    if (Object.hasOwn(NOT_ENGINE, key)) { skip(key, NOT_ENGINE[key]); continue; }
    switch (key) {
      case 'aoQuality': {
        const level = AO_QUALITY_LEVELS[value];
        if (!Number.isInteger(value) || level === undefined) { skip(key, `aoQuality ${value} is not one of 0..${AO_QUALITY_LEVELS.length - 1}`); break; }
        if (!can('setPassEnabled')) { skip(key, 'the engine has no setPassEnabled'); break; }
        engine.setPassEnabled('ao.gtao', level !== null);
        if (level !== null) {
          if (!can('setAo')) { skip(key, 'the engine has no setAo'); break; }
          // An @etcito/raa before setAo's slices/steps ignores them: the pass then keeps 4 x 8.
          engine.setAo({ slices: level.slices, steps: level.steps });
        }
        applied.push(key);
        break;
      }
      case 'renderScale':
        if (!can('setRenderScale')) { skip(key, 'this @etcito/raa has no setRenderScale; rebuild the engine package'); break; }
        engine.setRenderScale(value);
        applied.push(key);
        break;
      case 'postBloom':
      case 'postGrading': {
        if (!can('setPassEnabled')) { skip(key, 'the engine has no setPassEnabled'); break; }
        const pass = key === 'postBloom' ? 'bloom' : 'grading';
        engine.setPassEnabled(pass, value === true && look?.[pass] !== false);
        applied.push(key);
        break;
      }
      case 'postStrength':
        if (!can('setPostStrength')) { skip(key, 'this @etcito/raa has no setPostStrength; rebuild the engine package'); break; }
        engine.setPostStrength(value);
        applied.push(key);
        break;
      default:
        skip(key, `'${key}' is not a render option this adapter knows`);
    }
  }
  return { applied, skipped };
}
