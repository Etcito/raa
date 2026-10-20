//! `GeometryProvider` trait and the `GeometryRegistry` that dispatches by kind.

use crate::{GeomError, MeshData, Params, Quality};

/// A generator of closed meshes for one geometry kind.
pub trait GeometryProvider {
    /// Stable kind name used for registration and dispatch.
    fn kind(&self) -> &str;

    /// Parameter names the provider reads, in its documented order.
    fn param_names(&self) -> &[&'static str];

    /// True when every generated mesh is convex.
    fn is_convex(&self) -> bool;

    /// Triangle count `generate` would emit for the same inputs, without generating.
    /// Validates `params` like `generate`.
    fn triangle_count(&self, params: &Params, quality: Quality) -> Result<u32, GeomError>;

    /// Writes the mesh into `out`.
    ///
    /// Contract: clears `out` first; deterministic (same inputs give byte-identical
    /// output); validates parameters (`geom.missing_param`, or `geom.invalid_param`
    /// naming the param and the rule); emits a closed mesh with outward winding (S2).
    fn generate(
        &self,
        params: &Params,
        quality: Quality,
        out: &mut MeshData,
    ) -> Result<(), GeomError>;
}

/// Providers stored in registration order, looked up by kind.
#[derive(Default)]
pub struct GeometryRegistry {
    providers: Vec<Box<dyn GeometryProvider>>,
}

impl GeometryRegistry {
    pub fn new() -> GeometryRegistry {
        GeometryRegistry::default()
    }

    /// Adds `p`; a kind that is already registered gives `geom.duplicate_kind`.
    pub fn register(&mut self, p: Box<dyn GeometryProvider>) -> Result<(), GeomError> {
        if self.get(p.kind()).is_some() {
            return Err(GeomError::DuplicateKind(p.kind().to_string()));
        }
        self.providers.push(p);
        Ok(())
    }

    pub fn get(&self, kind: &str) -> Option<&dyn GeometryProvider> {
        self.providers
            .iter()
            .find(|p| p.kind() == kind)
            .map(|p| p.as_ref())
    }

    /// Registered kinds in registration order.
    pub fn kinds(&self) -> Vec<&str> {
        self.providers.iter().map(|p| p.kind()).collect()
    }

    /// Dispatches to the provider of `kind`; an unregistered kind gives `geom.unknown_kind`.
    pub fn generate(
        &self,
        kind: &str,
        params: &Params,
        quality: Quality,
        out: &mut MeshData,
    ) -> Result<(), GeomError> {
        match self.get(kind) {
            Some(p) => p.generate(params, quality, out),
            None => Err(GeomError::UnknownKind(kind.to_string())),
        }
    }
}
