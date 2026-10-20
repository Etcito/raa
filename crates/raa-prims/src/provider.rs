//! `PrimProvider`: one `GeometryProvider` per primitive kind, and `register_all`.

use crate::{PrimKind, PrimParams};
use raa_geom::{GeomError, GeometryProvider, GeometryRegistry, MeshData, Params, Quality};

/// Geometry provider for one primitive kind; `kind` and `param_names` are the S3 names.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PrimProvider(pub PrimKind);

impl GeometryProvider for PrimProvider {
    fn kind(&self) -> &str {
        self.0.as_str()
    }

    fn param_names(&self) -> &[&'static str] {
        self.0.param_names()
    }

    fn is_convex(&self) -> bool {
        self.0.is_convex()
    }

    fn triangle_count(&self, params: &Params, quality: Quality) -> Result<u32, GeomError> {
        // from_params builds the variant of self.0, so the params always match the kind.
        let p = PrimParams::from_params(self.0, params)?;
        Ok(crate::triangle_count(&p, quality))
    }

    fn generate(
        &self,
        params: &Params,
        quality: Quality,
        out: &mut MeshData,
    ) -> Result<(), GeomError> {
        let p = PrimParams::from_params(self.0, params)?;
        crate::generate(&p, quality, out)
    }
}

/// Registers the 8 kinds in `PrimKind::ALL` order; a kind already present gives
/// `geom.duplicate_kind` (the kinds before it stay registered).
pub fn register_all(reg: &mut GeometryRegistry) -> Result<(), GeomError> {
    PrimKind::ALL
        .into_iter()
        .try_for_each(|k| reg.register(Box::new(PrimProvider(k))))
}
