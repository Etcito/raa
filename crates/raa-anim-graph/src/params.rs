//! Typed runtime parameters (S19): declaration-ordered values, defaults, and
//! pending triggers. Values keep the declared `float`/`int`/`bool` type.
use crate::error::{GraphError, codes};
use crate::model::{Graph, Param, ParamType};
/// A runtime value; its variant must match the declared parameter type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamValue {
    Float(f32),
    Int(i32),
    Bool(bool),
    Trigger(bool),
}
#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    decls: Vec<(String, ParamType)>,
    values: Vec<ParamValue>,
}
fn bad(index: usize, message: &str) -> GraphError {
    GraphError::new(codes::PARAM, format!("/params/{index}"), message)
}
fn unknown() -> GraphError {
    GraphError::new(codes::PARAM, "/params", "unknown parameter")
}
impl Params {
    /// Default values for every declared parameter; triggers start unpending.
    pub fn new(graph: &Graph) -> Self {
        let params: &[Param] = &graph.params;
        let values = params.iter().map(|p| match p.ty {
            ParamType::Float => ParamValue::Float(p.default.rt),
            ParamType::Int => ParamValue::Int(p.default.rt as i32),
            ParamType::Bool => ParamValue::Bool(p.default.rt != 0.0),
            ParamType::Trigger => ParamValue::Trigger(false),
        }).collect();
        Self { decls: params.iter().map(|p| (p.name.clone(), p.ty)).collect(), values }
    }
    pub fn len(&self) -> usize { self.values.len() }
    pub fn index_of(&self, name: &str) -> Option<usize> { self.decls.iter().position(|d| d.0 == name) }
    pub fn ty(&self, index: usize) -> Option<ParamType> { self.decls.get(index).map(|d| d.1) }
    pub fn value(&self, index: usize) -> Option<ParamValue> { self.values.get(index).copied() }
    /// Declaration-ordered `(name, value)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, ParamValue)> {
        self.decls.iter().map(|d| d.0.as_str()).zip(self.values.iter().copied())
    }
    pub fn pending(&self, index: usize) -> bool {
        matches!(self.value(index), Some(ParamValue::Trigger(true)))
    }
    /// Set a declared value by name; the variant must match its type.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<(), GraphError> {
        self.set_at(self.index_of(name).ok_or_else(unknown)?, value)
    }
    pub fn set_at(&mut self, index: usize, value: ParamValue) -> Result<(), GraphError> {
        let ty = self.ty(index).ok_or_else(|| bad(index, "parameter index out of range"))?;
        if ty == ParamType::Trigger { return Err(bad(index, "triggers are fired, not set")); }
        let matches = matches!((ty, value),
            (ParamType::Float, ParamValue::Float(_)) | (ParamType::Int, ParamValue::Int(_))
                | (ParamType::Bool, ParamValue::Bool(_)));
        if !matches { return Err(bad(index, "value does not match the parameter type")); }
        if let ParamValue::Float(value) = value {
            if !value.is_finite() { return Err(bad(index, "float value is not finite")); }
        }
        self.values[index] = value;
        Ok(())
    }
    /// Make a trigger pending by name; only triggers can be fired.
    pub fn fire(&mut self, name: &str) -> Result<(), GraphError> {
        self.fire_at(self.index_of(name).ok_or_else(unknown)?)
    }
    pub fn fire_at(&mut self, index: usize) -> Result<(), GraphError> {
        match self.ty(index) {
            Some(ParamType::Trigger) => { self.values[index] = ParamValue::Trigger(true); Ok(()) }
            Some(_) => Err(bad(index, "only triggers can be fired")),
            None => Err(bad(index, "parameter index out of range")),
        }
    }
    /// Consume a pending trigger, returning whether it was pending.
    pub fn consume(&mut self, index: usize) -> bool {
        match self.values.get_mut(index) {
            Some(ParamValue::Trigger(pending)) => { let was = *pending; *pending = false; was }
            _ => false,
        }
    }
}
