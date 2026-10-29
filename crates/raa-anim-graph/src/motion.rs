//! Motion sampling (S19): clips and blend spaces through the S17 sampler.
//! `LayerState::phase` and `Crossfade::from_phase` hold a state's progress,
//! fraction included: clip ticks for clips and unsynchronized blend spaces, or
//! a normalized phase in [0, 1] shared by all points of a `sync` space, which
//! weight changes never move. A sample tick is the floor of the progress (of
//! `phase * L` for a sync point of length L); looping states take it modulo
//! each clip length, others clamp it to the length. Per tick the current state
//! advances by `speed` (`speedParam`, else 1) ticks, or by `speed / D` with
//! `D = sum(weight * L)` for sync spaces. Looping progress wraps into
//! [0, period), other progress clamps to [0, period]. The period is the clip
//! length, 1 for sync spaces, and for unsynchronized spaces the lcm of all
//! point lengths (the longest when not looping or above 2^24). A zero period
//! or `D` keeps progress at 0: zero-duration clips always sample tick 0.
use crate::blend_weights::evaluate;
use crate::load::validate::Motion;
use crate::{Compiled, GraphError, GraphState, ParamValue, Params, codes};
use raa_anim::{Clip, LocalPose, Rig};
use raa_det::{MathError, normalize, sign_align};
/// One clip sampled by a state: supplied clip index, blend weight, S17 tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Contribution { pub clip: usize, pub weight: f32, pub tick: u32 }
fn number(params: &Params, index: usize) -> Result<f32, GraphError> {
    match params.value(index) {
        Some(ParamValue::Float(v)) => Ok(v),
        Some(ParamValue::Int(v)) => Ok(v as f32),
        _ => Err(GraphError::new(codes::PARAM, format!("/params/{index}"), "expected a numeric param")),
    }
}
fn math(e: MathError) -> GraphError { GraphError::new(codes::BLEND_SPACE, "/motion", e.to_string()) }
fn length(clips: &[Clip], clip: usize) -> Result<u32, GraphError> {
    let missing = || GraphError::new(codes::CLIP_UNKNOWN, "/clips", "clip index out of range");
    clips.get(clip).map(|c| c.length_ticks).ok_or_else(missing)
}
/// Progress period in ticks: the lcm for looping states while exact in f32.
fn cycle(lengths: &[u32], looping: bool) -> u32 {
    let longest = lengths.iter().copied().max().unwrap_or(0);
    let mut lcm: u64 = 1;
    for &l in lengths.iter().filter(|&&l| l > 0) {
        let (mut a, mut b) = (lcm, l as u64);
        while b != 0 { (a, b) = (b, a % b); }
        lcm = lcm / a * l as u64;
        if lcm > 1 << 24 { return longest; }
    }
    if looping && longest > 0 { lcm as u32 } else { longest }
}
/// Wrap (looping) or clamp progress into the period; non-finite becomes 0.
fn wrap(x: f32, period: f32, looping: bool) -> f32 {
    if !(period > 0.0) { return 0.0; }
    if !looping { return if x > period { period } else if x > 0.0 { x } else { 0.0 }; }
    let mut r = x - period * ((x / period) as i64 as f32);
    if r < 0.0 { r += period; }
    if r >= period { r -= period; }
    if r >= 0.0 && r < period { r } else { 0.0 }
}
/// Weighted local TRS. Translation and scale sum in contribution order;
/// rotations sign-align to the first contribution, sum, then normalize with
/// sqrt. Zero weights are skipped and a sole contribution is returned as is.
pub fn blend_local(parts: &[(f32, Vec<LocalPose>)]) -> Result<Vec<LocalPose>, GraphError> {
    let parts: Vec<_> = parts.iter().filter(|(w, _)| *w != 0.0).collect();
    let Some(((_, first), rest)) = parts.split_first() else {
        return Err(GraphError::new(codes::BLEND_SPACE, "/motion", "no weighted contribution"));
    };
    if rest.is_empty() { return Ok(first.clone()); }
    let mut out = Vec::with_capacity(first.len());
    for bone in 0..first.len() {
        let mut pose = LocalPose { t: [0.0; 3], r: [0.0; 4], s: [0.0; 3] };
        for (w, local) in &parts {
            let p = local.get(bone).ok_or_else(|| math(MathError::NotFinite))?;
            let r = sign_align(first[bone].r, p.r).map_err(math)?;
            for c in 0..3 { pose.t[c] += *w * p.t[c]; pose.s[c] += *w * p.s[c]; }
            for c in 0..4 { pose.r[c] += *w * r[c]; }
        }
        pose.r = normalize(pose.r).map_err(math)?;
        out.push(pose);
    }
    Ok(out)
}
impl Compiled {
    /// Clips, weights and integer ticks that `state` of `machine` samples at `phase`.
    pub fn contributions(&self, clips: &[Clip], machine: usize, state: usize, phase: f32, params: &Params)
        -> Result<Vec<Contribution>, GraphError> {
        let looping = self.graph.machines[machine].states[state].looping;
        let address = |clip: usize, weight: f32, ticks: f32| -> Result<Contribution, GraphError> {
            let (l, k) = (length(clips, clip)?, ticks as u32);
            let tick = if !looping { k.min(l) } else if l == 0 { 0 } else { k % l };
            Ok(Contribution { clip, weight, tick })
        };
        match self.references.machines[machine].states[state].motion {
            Motion::Clip(clip) => Ok(vec![address(clip, 1.0, phase)?]),
            Motion::BlendSpace(space) => {
                let sync = self.graph.blend_spaces[space].sync;
                let refs = &self.references.blends[space];
                let at = refs.params.iter().map(|&p| number(params, p)).collect::<Result<Vec<_>, _>>()?;
                let weights = evaluate(&self.graph.blend_spaces[space], self.triangulations[space].as_ref(), &at)?;
                weights.iter().map(|w| {
                    let clip = refs.clips[w.point];
                    let ticks = if sync { phase * length(clips, clip)? as f32 } else { phase };
                    address(clip, w.weight, ticks)
                }).collect()
            }
        }
    }
    /// Blended local TRS of `state` at `phase`, through `Clip::local_pose_at`.
    pub fn sample_state(&self, rig: &Rig, clips: &[Clip], machine: usize, state: usize, phase: f32,
        params: &Params) -> Result<Vec<LocalPose>, GraphError> {
        let mut parts = Vec::new();
        for c in self.contributions(clips, machine, state, phase, params)? {
            parts.push((c.weight, clips[c.clip].local_pose_at(rig, c.tick)?));
        }
        blend_local(&parts)
    }
    /// Local TRS of `layer`: its current state, crossfaded from the frozen
    /// source at `Crossfade::weight` (weight 0 is the source, 1 the target).
    pub fn sample_layer(&self, rig: &Rig, clips: &[Clip], state: &GraphState, layer: usize, params: &Params)
        -> Result<Vec<LocalPose>, GraphError> {
        let l = &state.layers[layer];
        let to = self.sample_state(rig, clips, l.machine, l.state, l.phase, params)?;
        let Some(fade) = l.fade else { return Ok(to); };
        let from = self.sample_state(rig, clips, l.machine, fade.from, fade.from_phase, params)?;
        let u = fade.weight();
        blend_local(&[(1.0 - u, from), (u, to)])
    }
    /// Progress of `state` one tick after `phase`, per the module rules.
    pub fn next_phase(&self, clips: &[Clip], machine: usize, state: usize, phase: f32, params: &Params)
        -> Result<f32, GraphError> {
        let looping = self.graph.machines[machine].states[state].looping;
        let refs = &self.references.machines[machine].states[state];
        let speed = refs.speed_param.map_or(Ok(1.0), |p| number(params, p))?;
        let (period, step) = match refs.motion {
            Motion::Clip(clip) => (cycle(&[length(clips, clip)?], looping) as f32, speed),
            Motion::BlendSpace(space) if self.graph.blend_spaces[space].sync => {
                let mut d = 0.0;
                for c in self.contributions(clips, machine, state, phase, params)? {
                    d += c.weight * length(clips, c.clip)? as f32;
                }
                if d > 0.0 { (1.0, speed / d) } else { (0.0, 0.0) }
            }
            Motion::BlendSpace(space) => {
                let lengths = self.references.blends[space].clips.iter()
                    .map(|&c| length(clips, c)).collect::<Result<Vec<_>, _>>()?;
                (cycle(&lengths, looping) as f32, speed)
            }
        };
        Ok(wrap(phase + step, period, looping))
    }
    /// Advance every layer's current-state phase; fading sources stay frozen.
    pub fn advance_motion(&self, clips: &[Clip], state: &mut GraphState, params: &Params) -> Result<(), GraphError> {
        for layer in &mut state.layers {
            layer.phase = self.next_phase(clips, layer.machine, layer.state, layer.phase, params)?;
        }
        Ok(())
    }
    /// One tick: transitions, per-layer sampling at the current phases, then
    /// phase advance. A new state therefore samples its tick 0 first.
    pub fn step_motion(&self, rig: &Rig, clips: &[Clip], state: &mut GraphState, params: &Params)
        -> Result<Vec<Vec<LocalPose>>, GraphError> {
        self.advance_transitions(state, params);
        let poses = (0..state.layers.len())
            .map(|layer| self.sample_layer(rig, clips, state, layer, params))
            .collect::<Result<Vec<_>, _>>()?;
        self.advance_motion(clips, state, params)?;
        Ok(poses)
    }
}
