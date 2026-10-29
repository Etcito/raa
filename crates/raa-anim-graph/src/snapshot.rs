//! RGS2: magic, u32 binding byte length, binding bytes, declaration-ordered
//! parameter words, then the RGS1 cursor payload. Integers and f32 bits are LE.
//! The binding contains canonical S19 UTF-8 followed by resolved asset indices.
//! Restore requires the same rig/clip assets; snapshots do not embed asset data.
//! Tick, local clocks and last sampled phases also encode event progress: no
//! separate event cursor exists. Readers build a candidate before replacing state.
use crate::{Compiled, GraphError, GraphState, ParamType, ParamValue, codes};
use crate::load::validate::Motion;

fn bad() -> GraphError { GraphError::new(codes::SCHEMA, "/state", "invalid or incompatible snapshot") }
fn word(out: &mut Vec<u8>, v: u32) { out.extend_from_slice(&v.to_le_bytes()); }
fn take<'a>(bytes: &mut &'a [u8], count: usize) -> Result<&'a [u8], GraphError> {
    let (head, tail) = bytes.split_at_checked(count).ok_or_else(bad)?;
    *bytes = tail;
    Ok(head)
}
fn next(bytes: &mut &[u8]) -> Result<u32, GraphError> {
    let b = take(bytes, 4)?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}
pub(crate) fn binding(c: &Compiled) -> Vec<u8> {
    let mut out = c.graph.canonical().into_bytes();
    word(&mut out, c.references.root_bone as u32);
    for l in &c.references.layers {
        word(&mut out, l.machine as u32);
        word(&mut out, l.bones.len() as u32);
        for &bone in &l.bones { word(&mut out, bone as u32); }
    }
    for m in &c.references.machines {
        for s in &m.states {
            let (tag, index) = match s.motion { Motion::Clip(i) => (0, i), Motion::BlendSpace(i) => (1, i) };
            word(&mut out, tag); word(&mut out, index as u32);
        }
    }
    for b in &c.references.blends { for &clip in &b.clips { word(&mut out, clip as u32); } }
    out
}
impl GraphState {
    /// Append a complete RGS2 snapshot. Does not clear or overwrite `out`.
    pub fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(b"RGS2");
        word(out, self.binding.len() as u32);
        out.extend_from_slice(&self.binding);
        for (_, value) in self.params.iter() {
            word(out, match value {
                ParamValue::Float(v) => v.to_bits(), ParamValue::Int(v) => v as u32,
                ParamValue::Bool(v) | ParamValue::Trigger(v) => v as u32,
            });
        }
        self.write_payload(out);
    }
    /// Checked, allocation-bounded decode; old versions and trailing data are refused.
    pub fn read(compiled: &Compiled, mut bytes: &[u8]) -> Result<Self, GraphError> {
        if take(&mut bytes, 4)? != b"RGS2" { return Err(bad()); }
        let count = next(&mut bytes)? as usize;
        if take(&mut bytes, count)? != binding(compiled) { return Err(bad()); }
        let mut params = crate::Params::new(&compiled.graph);
        for (i, p) in compiled.graph.params.iter().enumerate() {
            let v = next(&mut bytes)?;
            let value = match p.ty {
                ParamType::Float => ParamValue::Float(f32::from_bits(v)),
                ParamType::Int => ParamValue::Int(v as i32),
                ParamType::Bool | ParamType::Trigger if v > 1 => return Err(bad()),
                ParamType::Bool => ParamValue::Bool(v != 0),
                ParamType::Trigger => { if v != 0 { params.fire_at(i)?; } continue; }
            };
            params.set_at(i, value)?;
        }
        let mut state = Self::read_payload(compiled, bytes)?;
        state.params = params;
        Ok(state)
    }
    /// Replace live state only after the entire snapshot passes validation.
    pub fn restore(&mut self, compiled: &Compiled, bytes: &[u8]) -> Result<(), GraphError> {
        *self = Self::read(compiled, bytes)?;
        Ok(())
    }
}
