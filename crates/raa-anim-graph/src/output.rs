//! S19 tick outputs: root-motion deltas and ordered clip events.
//! [`Compiled::step_outputs`] runs transitions, advances every layer not entered
//! this tick, then composes layers, so `LayerState::phase` holds the progress
//! last sampled: a tick sampling clip tick k reports the S17 delta U(k) - U(k')
//! and the events in (k', k] since the previous sample k'. A state entered this
//! tick (or on the first) samples tick 0 with zero delta and its tick-0 events.
//! Drive a state with this or `step_motion`, not both: they differ by one advance.
//! Deltas use the graph `rootMotion` bone's translation track in each clip:
//! looping states cross seams through the unwrapped P(L) without accumulating
//! laps in f32, others clamp to [0, L], zero-length clips never move. Deltas, not
//! positions, blend: by blend weight, by crossfade weight (the frozen source moves
//! nothing), then per layer as in `compose_layers` where the mask holds the bone,
//! so weight changes never jump. Selected axes return to rig rest in the pose,
//! other components keep their samples. Rotation is not extracted: `r` is identity.
//! Events come once per machine from its first weighted layer, from the dominant
//! contribution (first on ties) of its current state, in playback direction.
//! Looping states match ticks modulo L and deliver each at most once per tick.
//! Order: layer, then crossing offset, then clip file order.
use crate::load::validate::Motion;
use crate::{Compiled, GraphError, GraphState, LayerState, ParamValue, Params, codes};
use raa_anim::{Channel, Clip, LocalPose, Rig, RootMotion};
use raa_det::lerp;

pub const IDENTITY: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
/// Extracted root motion of one tick, in the root bone's parent space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootDelta { pub t: [f32; 3], pub r: [f32; 4] }
/// A clip event crossed this tick: layer index, supplied clip index, clip tick, name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphEvent { pub layer: usize, pub clip: usize, pub tick: u32, pub name: String }
#[derive(Debug, Clone, PartialEq)]
pub struct TickOut { pub pose: Vec<LocalPose>, pub root_delta: RootDelta, pub events: Vec<GraphEvent> }
/// One contribution's movement: start address and signed clip ticks entered.
#[derive(Debug, Clone, Copy)]
struct Span { clip: usize, weight: f32, from: u32, ticks: i64, looping: bool }
fn bad(message: &str) -> GraphError { GraphError::new(codes::BLEND_SPACE, "/rootMotion", message) }
/// Floor of finite progress, bounded so tick arithmetic cannot overflow.
fn floor(x: f32) -> i64 {
    let t = if x.is_finite() { x.clamp(-1e15, 1e15) as i64 } else { 0 };
    if (t as f32) > x { t - 1 } else { t }
}
/// Selected translation P(tick) on [0, L]; a looping clip's P(L) is not wrapped.
fn translation(clip: &Clip, track: usize, tick: u32) -> Result<[f32; 3], GraphError> {
    let v = if clip.looping && tick > 0 && tick == clip.length_ticks {
        let once = Clip { name: String::new(), rig: String::new(), length_ticks: tick, looping: false,
            root_motion: RootMotion::None, tracks: vec![clip.tracks[track].clone()], events: Vec::new() };
        once.sample_track(0, tick)?
    } else { clip.sample_track(track, tick)? };
    Ok(if v.len() == 3 { [v[0], v[1], v[2]] } else { [0.0; 3] })
}
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn add(a: [f32; 3], b: [f32; 3], w: f32) -> [f32; 3] { [a[0] + b[0] * w, a[1] + b[1] * w, a[2] + b[2] * w] }
/// `U(from + ticks) - U(from)` as adjacent local differences around the seams.
fn motion(clip: &Clip, track: usize, span: Span) -> Result<[f32; 3], GraphError> {
    let (l, a, n) = (clip.length_ticks as i64, span.from as i64, span.ticks);
    let p = |t: i64| translation(clip, track, t as u32);
    if n == 0 || l == 0 { return Ok([0.0; 3]); }
    if !span.looping { return Ok(sub(p(a + n)?, p(a)?)); }
    let (q, r) = ((a + n).div_euclid(l), (a + n).rem_euclid(l));
    let laps = |k: i64| Ok::<_, GraphError>(if k == 0 { [0.0; 3] } else { add([0.0; 3], sub(p(l)?, p(0)?), k as f32) });
    let out = if q == 0 { sub(p(r)?, p(a)?) }
        else if q > 0 { add(add(sub(p(l)?, p(a)?), laps(q - 1)?, 1.0), sub(p(r)?, p(0)?), 1.0) }
        else { add(add(sub(p(r)?, p(l)?), laps(q + 1)?, 1.0), sub(p(0)?, p(a)?), 1.0) };
    if out.iter().all(|v| v.is_finite()) { Ok(out) } else { Err(bad("root delta is not finite")) }
}
/// First playback offset at which addressed tick `at` is entered, if within the span.
fn crossing(at: i64, span: Span, l: i64) -> Option<i64> {
    let (a, n) = (span.from as i64, span.ticks);
    let off = match (span.looping && l > 0, n > 0) {
        (true, true) => (at - a - 1).rem_euclid(l) + 1,
        (true, false) => (a - at - 1).rem_euclid(l) + 1,
        (false, true) => at - a,
        (false, false) => a - at,
    };
    (n != 0 && off >= 1 && off <= n.abs()).then_some(off)
}
/// Events of the dominant span (first on equal weight), in crossing then file order.
fn events(clips: &[Clip], layer: usize, spans: &[Span], fresh: bool, out: &mut Vec<GraphEvent>) {
    let best = spans.iter().fold(None, |b: Option<&Span>, s| if b.is_some_and(|b| b.weight >= s.weight) { b } else { Some(s) });
    let Some(&span) = best else { return; };
    let (clip, l) = (&clips[span.clip], clips[span.clip].length_ticks as i64);
    let mut hits: Vec<_> = clip.events.iter().filter_map(|e| {
        let at = if span.looping && l > 0 { e.tick as i64 % l } else { e.tick as i64 };
        if fresh { (at == span.from as i64).then_some((0, e)) } else { crossing(at, span, l).map(|o| (o, e)) }
    }).collect();
    hits.sort_by_key(|h| h.0);
    out.extend(hits.into_iter().map(|(_, e)| GraphEvent { layer, clip: span.clip, tick: e.tick, name: e.name.clone() }));
}
impl Compiled {
    /// Current-state contributions with clip ticks entered since `from` (`None`: entered now).
    fn spans(&self, clips: &[Clip], layer: &LayerState, from: Option<f32>, params: &Params)
        -> Result<Vec<Span>, GraphError> {
        let (m, s) = (layer.machine, layer.state);
        let looping = self.graph.machines[m].states[s].looping;
        let now = self.contributions(clips, m, s, layer.phase, params)?;
        let Some(from) = from else {
            return Ok(now.iter().map(|c| Span { clip: c.clip, weight: c.weight, from: c.tick, ticks: 0, looping }).collect());
        };
        let before = self.contributions(clips, m, s, from, params)?;
        let refs = &self.references.machines[m].states[s];
        let speed = match refs.speed_param.map(|p| params.value(p)) {
            None => 1.0, Some(Some(ParamValue::Float(v))) => v, Some(Some(ParamValue::Int(v))) => v as f32,
            _ => return Err(GraphError::new(codes::PARAM, "/params", "expected a numeric param")),
        };
        let sync = matches!(refs.motion, Motion::BlendSpace(b) if self.graph.blend_spaces[b].sync);
        let d = before.iter().fold(0.0, |d, c| d + c.weight * clips[c.clip].length_ticks as f32);
        let step = if !sync { speed } else if d > 0.0 { speed / d } else { 0.0 };
        Ok(before.iter().zip(&now).map(|(b, c)| {
            let l = clips[c.clip].length_ticks as i64;
            let ticks = if !looping { c.tick as i64 - b.tick as i64 } else if l == 0 { 0 } else {
                // Unwrapped tick count from progress, snapped to the sampled addresses.
                let scale = if sync { l as f32 } else { 1.0 };
                let est = floor((from + step) * scale) - floor(from * scale);
                let off = (c.tick as i64 - b.tick as i64 - est).rem_euclid(l);
                if 2 * off >= l { est + off - l } else { est + off }
            };
            Span { clip: c.clip, weight: c.weight, from: b.tick, ticks, looping }
        }).collect())
    }
    /// One tick: transitions, advance, layer composition, root extraction, events.
    pub fn step_outputs(&self, rig: &Rig, clips: &[Clip], state: &mut GraphState, params: &Params)
        -> Result<TickOut, GraphError> {
        let first = state.tick == 0;
        self.advance_transitions(state, params);
        let mut froms = Vec::with_capacity(state.layers.len());
        for layer in &mut state.layers {
            let from = (!first && layer.clock != 0).then_some(layer.phase);
            if from.is_some() { layer.phase = self.next_phase(clips, layer.machine, layer.state, layer.phase, params)?; }
            froms.push(from);
        }
        let mut pose = self.compose_layers(rig, clips, state, params)?;
        let (root, mut delta, mut out) = (self.references.root_bone, [0.0f32; 3], Vec::new());
        let mut seen = vec![false; self.graph.machines.len()];
        for (index, layer) in self.graph.layers.iter().enumerate() {
            let weight = layer.weight.rt; if weight == 0.0 { continue; }
            let cursor = &state.layers[index];
            let spans = self.spans(clips, cursor, froms[index], params)?;
            if self.references.layers[index].bones.contains(&root) {
                let (u, mut moved) = (cursor.fade.map_or(1.0, |f| f.weight()), [0.0f32; 3]);
                for span in &spans {
                    let clip = &clips[span.clip];
                    let found = clip.tracks.iter()
                        .position(|t| t.bone == rig.bones[root].name && t.channel == Channel::Translation);
                    if let Some(track) = found { moved = add(moved, motion(clip, track, *span)?, span.weight * u); }
                }
                for axis in 0..3 {
                    delta[axis] = if layer.additive { delta[axis] + moved[axis] * weight }
                        else { lerp(delta[axis], moved[axis], weight).map_err(|e| bad(&e.to_string()))? };
                }
            }
            if !std::mem::replace(&mut seen[cursor.machine], true) {
                events(clips, index, &spans, froms[index].is_none(), &mut out);
            }
        }
        let (rest, mode) = (rig.local_rest_pose()?, &self.graph.root_motion.mode);
        for axis in 0..3 {
            let selected = *mode == RootMotion::Xyz || (*mode == RootMotion::Xz && axis != 1);
            if selected { pose[root].t[axis] = rest[root].t[axis]; } else { delta[axis] = 0.0; }
        }
        if !delta.iter().all(|v| v.is_finite()) { return Err(bad("root delta is not finite")); }
        Ok(TickOut { pose, root_delta: RootDelta { t: delta, r: IDENTITY }, events: out })
    }
}
