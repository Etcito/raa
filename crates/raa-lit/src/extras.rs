//! Per-instance extras (S10 `InstanceExtraGpu`), keyed by the scene instance tag.

/// Size in bytes of one `InstanceExtraGpu` record.
pub const INSTANCE_EXTRA_GPU_SIZE: usize = 128;

/// Per-instance shading and deformation inputs (S10).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct InstanceExtra {
    /// rgb multiplies the base colour, a multiplies opacity.
    pub tint: [f32; 4],
    /// Local squash axis xyz (unit) and amount.
    pub squash: [f32; 4],
    /// Local jiggle offset xyz (m) and height (m).
    pub jiggle: [f32; 4],
    /// `MaterialInput.instance_params`.
    pub params: [[f32; 4]; 4],
}

impl Default for InstanceExtra {
    fn default() -> Self {
        Self {
            tint: [1.0; 4],
            squash: [0.0, 1.0, 0.0, 0.0],
            jiggle: [0.0; 4],
            params: [[0.0; 4]; 4],
        }
    }
}

impl InstanceExtra {
    /// Appends the 128-byte `InstanceExtraGpu` record to `out`.
    pub fn write_gpu(&self, out: &mut Vec<u8>) {
        let vecs = [self.tint, self.squash, self.jiggle]
            .into_iter()
            .chain(self.params)
            .chain([[0.0; 4]]);
        for v in vecs {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
}
