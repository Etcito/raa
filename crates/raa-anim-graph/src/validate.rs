//! Stable references parallel to document arrays; no sorting or rewriting of declarations.
use crate::{Graph, GraphError, GRAPH_CAPS, WILDCARD, codes};
use raa_anim::{Clip, Rig};
type Out<T> = Result<T, GraphError>;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Motion { Clip(usize), BlendSpace(usize) }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateRefs { pub motion: Motion, pub speed_param: Option<usize> }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransitionRefs { pub from: Option<usize>, pub to: usize, pub params: Vec<usize> }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineRefs { pub entry: usize, pub states: Vec<StateRefs>, pub transitions: Vec<TransitionRefs> }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlendRefs { pub params: Vec<usize>, pub clips: Vec<usize> }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerRefs { pub machine: usize, pub bones: Vec<usize> }
/// Array positions mirror the document. Clip indices address the supplied clip
/// slice; bone indices address the supplied rig. Keep those inputs in that order.
/// Transition parameter indices parallel `when`; `from: None` denotes `*`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct References {
    pub blends: Vec<BlendRefs>, pub machines: Vec<MachineRefs>,
    pub layers: Vec<LayerRefs>, pub root_bone: usize,
}
fn bad(code: &'static str, path: &str, message: &str) -> GraphError { GraphError::new(code, path, message) }
fn index(names: &[&str], name: &str, code: &'static str, path: &str) -> Out<usize> {
    let mut hits = names.iter().enumerate().filter(|(_, n)| **n == name);
    match (hits.next(), hits.next()) {
        (Some((i, _)), None) => Ok(i),
        _ => Err(bad(code, path, "unknown or ambiguous reference")),
    }
}
fn unique(names: &[&str], code: &'static str, path: &str) -> Out<()> {
    for (i, name) in names.iter().enumerate() {
        if names[..i].contains(name) { return Err(bad(code, &format!("{path}/{i}/name"), "duplicate name")); }
    }
    Ok(())
}
/// Called only after strict decoding has checked types, counts and declaration references.
pub(super) fn resolve(g: &Graph, rig: &Rig, clips: &[Clip]) -> Out<References> {
    // Binding failures retain the loader's document-root refusal path.
    if g.rig != rig.name { return Err(bad(codes::SCHEMA, "", "rig mismatch")); }
    let bones: Vec<_> = rig.bones.iter().map(|b| b.name.as_str()).collect();
    unique(&bones, codes::MASK, "/rig/bones")?;
    let cn: Vec<_> = clips.iter().map(|c| c.name.as_str()).collect();
    unique(&cn, codes::CLIP_UNKNOWN, "/clips")?;
    for (i, clip) in clips.iter().enumerate() {
        let path = format!("/clips/{i}");
        if clip.rig != rig.name { return Err(bad(codes::CLIP_UNKNOWN, &path, "clip rig mismatch")); }
        for track in &clip.tracks { index(&bones, &track.bone, codes::CLIP_UNKNOWN, &path)?; }
    }
    let pn: Vec<_> = g.params.iter().map(|p| p.name.as_str()).collect();
    let bn: Vec<_> = g.blend_spaces.iter().map(|b| b.name.as_str()).collect();
    let mn: Vec<_> = g.machines.iter().map(|m| m.name.as_str()).collect();
    let mut blends = Vec::new();
    for (i, b) in g.blend_spaces.iter().enumerate() {
        let path = format!("/blendSpaces/{i}");
        if cn.contains(&b.name.as_str()) { return Err(bad(codes::BLEND_SPACE, &path, "ambiguous motion name")); }
        let mut refs = BlendRefs { params: Vec::new(), clips: Vec::new() };
        for p in &b.params { refs.params.push(index(&pn, p, codes::BLEND_SPACE, &path)?); }
        for (j, point) in b.points.iter().enumerate() {
            let pp = format!("{path}/points/{j}");
            refs.clips.push(index(&cn, &point.clip, codes::CLIP_UNKNOWN, &format!("{pp}/clip"))?);
            if b.points[..j].iter().any(|p| p.at.iter().zip(&point.at).all(|(a, b)| a.rt == b.rt)) {
                return Err(bad(codes::BLEND_SPACE, &pp, "duplicate runtime coordinates"));
            }
        }
        if b.params.len() == 2 {
            let mut area = false;
            for j in 1..b.points.len() {
                for k in j + 1..b.points.len() {
                    let (a, c, d) = (&b.points[0].at, &b.points[j].at, &b.points[k].at);
                    let cross = (c[0].rt - a[0].rt) * (d[1].rt - a[1].rt) - (c[1].rt - a[1].rt) * (d[0].rt - a[0].rt);
                    if !cross.is_finite() { return Err(bad(codes::BLEND_SPACE, &path, "geometry overflows")); }
                    area |= cross != 0.0;
                }
            }
            if !area { return Err(bad(codes::BLEND_SPACE, &path, "2D points must span an area")); }
        } else if b.points.iter().any(|p| b.points.iter().any(|q| !(p.at[0].rt - q.at[0].rt).is_finite())) {
            return Err(bad(codes::BLEND_SPACE, &path, "geometry overflows"));
        }
        blends.push(refs);
    }
    let mut machines = Vec::new();
    for (i, m) in g.machines.iter().enumerate() {
        let path = format!("/machines/{i}");
        let sn: Vec<_> = m.states.iter().map(|s| s.name.as_str()).collect();
        let mut refs = MachineRefs { entry: index(&sn, &m.entry, codes::STATE_UNKNOWN, &path)?, states: Vec::new(), transitions: Vec::new() };
        for (j, s) in m.states.iter().enumerate() {
            let path = format!("{path}/states/{j}/motion");
            let motion = match bn.iter().position(|n| *n == s.motion) {
                Some(i) => Motion::BlendSpace(i),
                None => Motion::Clip(index(&cn, &s.motion, codes::CLIP_UNKNOWN, &path)?),
            };
            let speed_param = s.speed_param.as_ref().map(|p| index(&pn, p, codes::PARAM, &path)).transpose()?;
            refs.states.push(StateRefs { motion, speed_param });
        }
        for t in &m.transitions {
            let from = if t.from == WILDCARD { None } else { Some(index(&sn, &t.from, codes::STATE_UNKNOWN, &path)?) };
            let to = index(&sn, &t.to, codes::STATE_UNKNOWN, &path)?;
            let params = t.when.iter().map(|c| index(&pn, &c.param, codes::PARAM, &path)).collect::<Out<_>>()?;
            refs.transitions.push(TransitionRefs { from, to, params });
        }
        machines.push(refs);
    }
    unique(&g.layers.iter().map(|l| l.name.as_str()).collect::<Vec<_>>(), codes::SCHEMA, "/layers")?;
    let mut layers = Vec::new();
    for (i, l) in g.layers.iter().enumerate() {
        let path = format!("/layers/{i}/mask");
        let mut selected = vec![l.mask.is_none(); bones.len()];
        if let Some(mask) = &l.mask {
            for (j, name) in mask.iter().enumerate() { selected[index(&bones, name, codes::MASK, &format!("{path}/{j}"))?] = true; }
        }
        for (j, bone) in rig.bones.iter().enumerate() {
            if let Some(parent) = bone.parent {
                if parent >= j { return Err(bad(codes::MASK, &path, "invalid rig parent order")); }
                selected[j] |= selected[parent];
            }
        }
        let bones: Vec<_> = selected.iter().enumerate().filter_map(|(j, &yes)| yes.then_some(j)).collect();
        GraphError::check_cap(bones.len(), GRAPH_CAPS.max_mask_bones, &path, "expanded mask")?;
        layers.push(LayerRefs { machine: index(&mn, &l.machine, codes::STATE_UNKNOWN, &path)?, bones });
    }
    let root_bone = index(&bones, &g.root_motion.bone, codes::MASK, "/rootMotion/bone")?;
    Ok(References { blends, machines, layers, root_bone })
}
