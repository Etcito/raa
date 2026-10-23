//! [`SurfaceLibrary`]: surface materials addressed by dense [`MaterialId`]s.

use raa_core::{MaterialId, RaaError};

use crate::material::{SURFACE_MATERIAL_GPU_SIZE, SurfaceMaterial};

/// Surface materials addressed by dense [`MaterialId`]s from 0, with a dirty flag for GPU
/// upload (the behaviour of `raa_material::MaterialLibrary`).
#[derive(Debug, Clone, Default)]
pub struct SurfaceLibrary {
    materials: Vec<SurfaceMaterial>,
    dirty: bool,
}

impl SurfaceLibrary {
    /// An empty, clean library.
    pub fn new() -> Self {
        Self::default()
    }

    /// Validates `m` and appends it; ids are dense from 0. Marks the library dirty.
    pub fn add(&mut self, m: SurfaceMaterial) -> Result<MaterialId, RaaError> {
        m.validate()?;
        let id = u32::try_from(self.materials.len()).map_err(|_| RaaError::FrameGraph {
            code: "material.invalid",
            message: "surface library is full (u32 ids)".to_string(),
        })?;
        self.materials.push(m);
        self.dirty = true;
        Ok(MaterialId(id))
    }

    /// Validates `m` and replaces material `id`. Marks the library dirty.
    /// An id never returned by [`add`](Self::add) gives `material.unknown_id`.
    pub fn set(&mut self, id: MaterialId, m: SurfaceMaterial) -> Result<(), RaaError> {
        let len = self.materials.len();
        let slot = self
            .materials
            .get_mut(id.0 as usize)
            .ok_or_else(|| RaaError::FrameGraph {
                code: "material.unknown_id",
                message: format!("material id {} is unknown (library has {len})", id.0),
            })?;
        m.validate()?;
        *slot = m;
        self.dirty = true;
        Ok(())
    }

    /// The material `id`, if it exists.
    pub fn get(&self, id: MaterialId) -> Option<&SurfaceMaterial> {
        self.materials.get(id.0 as usize)
    }

    /// Number of materials.
    pub fn len(&self) -> usize {
        self.materials.len()
    }

    /// True when the library holds no material.
    pub fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }

    /// True when a material was added or set since the last [`write_gpu_all`](Self::write_gpu_all).
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Clears `out`, appends 256 bytes per material in id order and clears the dirty flag.
    /// Reuses the capacity of `out`.
    pub fn write_gpu_all(&mut self, out: &mut Vec<u8>) {
        out.clear();
        out.reserve(self.materials.len() * SURFACE_MATERIAL_GPU_SIZE);
        for m in &self.materials {
            m.write_gpu(out);
        }
        self.dirty = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_dense_and_dirty_tracks_writes() {
        let mut lib = SurfaceLibrary::new();
        assert!(lib.is_empty() && !lib.is_dirty());
        assert_eq!(lib.add(SurfaceMaterial::default()).unwrap(), MaterialId(0));
        let red = SurfaceMaterial {
            base_color: [1.0, 0.0, 0.0],
            ..SurfaceMaterial::default()
        };
        assert_eq!(lib.add(red.clone()).unwrap(), MaterialId(1));
        assert_eq!(lib.len(), 2);
        assert!(lib.is_dirty());
        let mut out = vec![7u8; 3];
        lib.write_gpu_all(&mut out);
        assert_eq!(out.len(), 2 * SURFACE_MATERIAL_GPU_SIZE);
        assert!(!lib.is_dirty());
        lib.set(MaterialId(0), red.clone()).unwrap();
        assert!(lib.is_dirty());
        assert_eq!(lib.get(MaterialId(0)), Some(&red));
        assert_eq!(lib.get(MaterialId(2)), None);
    }

    #[test]
    fn set_refuses_unknown_id_and_invalid_material() {
        let mut lib = SurfaceLibrary::new();
        let err = lib
            .set(MaterialId(0), SurfaceMaterial::default())
            .unwrap_err();
        assert_eq!(err.code(), "material.unknown_id");
        assert!(!err.message().is_empty());
        lib.add(SurfaceMaterial::default()).unwrap();
        lib.write_gpu_all(&mut Vec::new());
        let bad = SurfaceMaterial {
            ior: 0.5,
            ..SurfaceMaterial::default()
        };
        assert_eq!(
            lib.set(MaterialId(0), bad.clone()).unwrap_err().code(),
            "material.invalid"
        );
        assert_eq!(lib.add(bad).unwrap_err().code(), "material.invalid");
        assert_eq!(lib.len(), 1);
        assert!(!lib.is_dirty());
    }
}
