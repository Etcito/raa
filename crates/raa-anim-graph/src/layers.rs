//! S19 layer composition. Layers apply in document order, each restricted to
//! its compiled mask: the listed bones and their descendants. Composition starts
//! from the rig rest local pose, so a bone no layer selects keeps its rest value
//! and a masked-out bone keeps whatever earlier layers left on it.
//!
//! A non-additive layer blends the current pose toward its sampled pose by the
//! layer weight: translation and scale component-wise, rotation with the
//! deterministic shortest-path nlerp. An additive layer adds its delta from the
//! motion's first-key reference: the same motion sampled at progress zero, where
//! S17 clamping yields each track's first key (a blend space blends its points'
//! first keys). During a crossfade the sampled pose and the reference blend with
//! the same weights, keeping each state's own first-key offset. Rotation deltas
//! are `sample * conjugate(reference)`, weighted by an nlerp from identity and
//! pre-multiplied in the parent frame. All arithmetic
//! uses the raa-det source-ordered helpers.
use crate::load::Compiled;
use crate::{GraphError, GraphState, Params, codes};
use raa_anim::{Clip, LocalPose, Rig};
use raa_det::{MathError, lerp, nlerp, normalize, quat_mul};

const IDENTITY: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

fn math(layer: usize, e: MathError) -> GraphError {
    GraphError::new(codes::BLEND_SPACE, format!("/layers/{layer}"), e.to_string())
}

fn finite(layer: usize, value: f32) -> Result<f32, GraphError> {
    if value.is_finite() { Ok(value) } else { Err(math(layer, MathError::NotFinite)) }
}

fn conjugate(q: [f32; 4]) -> [f32; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}

/// `current + u * (target - current)` per component, in source order.
fn blend3(layer: usize, current: &mut [f32; 3], target: [f32; 3], u: f32) -> Result<(), GraphError> {
    for c in 0..3 {
        current[c] = lerp(current[c], target[c], u).map_err(|e| math(layer, e))?;
    }
    Ok(())
}

/// `current + u * (target - reference)` per component, in source order.
fn add3(layer: usize, current: &mut [f32; 3], target: [f32; 3], reference: [f32; 3], u: f32)
    -> Result<(), GraphError> {
    for c in 0..3 {
        let delta = (target[c] - reference[c]) * u;
        current[c] = finite(layer, current[c] + delta)?;
    }
    Ok(())
}

impl Compiled {
    /// Pose of a layer's motion at its first-key reference: sampled at progress
    /// zero, where S17 clamps to each track's first key, and blended across the
    /// same crossfade as [`Compiled::sample_layer`].
    pub fn reference_layer(&self, rig: &Rig, clips: &[Clip], state: &GraphState, layer: usize,
        params: &Params) -> Result<Vec<LocalPose>, GraphError> {
        let l = &state.layers[layer];
        let to = self.sample_state(rig, clips, l.machine, l.state, 0.0, params)?;
        let Some(fade) = l.fade else { return Ok(to); };
        let from = self.sample_state(rig, clips, l.machine, fade.from, 0.0, params)?;
        let u = fade.weight();
        crate::blend_local(&[(1.0 - u, from), (u, to)])
    }

    /// Compose every layer in document order over the rig rest local pose.
    pub fn compose_layers(&self, rig: &Rig, clips: &[Clip], state: &GraphState, params: &Params)
        -> Result<Vec<LocalPose>, GraphError> {
        if state.layers.len() != self.graph.layers.len() {
            return Err(GraphError::new(codes::SCHEMA, "/state/layers", "layer state count mismatch"));
        }
        let mut pose = rig.local_rest_pose().map_err(GraphError::from)?;
        for (index, layer) in self.graph.layers.iter().enumerate() {
            let weight = layer.weight.rt;
            if weight == 0.0 { continue; }
            let sample = self.sample_layer(rig, clips, state, index, params)?;
            let reference = layer.additive
                .then(|| self.reference_layer(rig, clips, state, index, params))
                .transpose()?;
            if sample.len() != pose.len() || reference.as_ref().is_some_and(|r| r.len() != pose.len()) {
                return Err(GraphError::new(codes::MASK, format!("/layers/{index}"), "pose length mismatch"));
            }
            for &bone in &self.references.layers[index].bones {
                let Some(target) = sample.get(bone) else {
                    return Err(GraphError::new(codes::MASK, format!("/layers/{index}"), "bone index out of range"));
                };
                let current = &mut pose[bone];
                match &reference {
                    None => {
                        blend3(index, &mut current.t, target.t, weight)?;
                        blend3(index, &mut current.s, target.s, weight)?;
                        current.r = nlerp(current.r, target.r, weight).map_err(|e| math(index, e))?;
                    }
                    Some(reference) => {
                        let reference = reference[bone];
                        add3(index, &mut current.t, target.t, reference.t, weight)?;
                        add3(index, &mut current.s, target.s, reference.s, weight)?;
                        let delta = quat_mul(target.r, conjugate(reference.r)).map_err(|e| math(index, e))?;
                        let delta = nlerp(IDENTITY, delta, weight).map_err(|e| math(index, e))?;
                        current.r = normalize(quat_mul(delta, current.r).map_err(|e| math(index, e))?)
                            .map_err(|e| math(index, e))?;
                    }
                }
            }
        }
        Ok(pose)
    }
}
