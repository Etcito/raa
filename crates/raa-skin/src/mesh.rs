//! `SkinMesh`: S18 joints and weights validation plus little-endian upload packing.

use raa_mesh::AttributedMesh;

use crate::SkinError;

/// Joint slots per vertex (`Uint16Array (4n)` / `Float32Array (4n)`, S18).
pub const MAX_INFLUENCES: usize = 4;
/// Largest joint index; S18 joints follow S16 bone order, capped at MAX_BONES.
pub const MAX_JOINTS: usize = 256;
/// Maximum `|sum(weights) - 1|` accepted for one vertex (S18 WEIGHT_SUM_EPS).
pub const WEIGHT_SUM_EPS: f32 = 1e-4;

/// Rest geometry plus its validated S18 skin attributes; the rest mesh is unchanged.
#[derive(Clone, Debug)]
pub struct SkinMesh {
    pub rest: AttributedMesh,
    /// Four joints per vertex, in vertex order.
    pub joints: Vec<u16>,
    /// Four weights per vertex, in vertex order.
    pub weights: Vec<f32>,
}

impl SkinMesh {
    /// Validates the flat JS arrays against `rest`: both or neither, four entries
    /// per vertex, finite non-negative weights summing to 1 within
    /// [`WEIGHT_SUM_EPS`], and joints below [`MAX_JOINTS`].
    pub fn new(
        rest: AttributedMesh,
        joints: &[u16],
        weights: &[f32],
    ) -> Result<SkinMesh, SkinError> {
        if joints.is_empty() != weights.is_empty() {
            return Err(SkinError::new(
                "skin.length_mismatch",
                format!(
                    "joints and weights come both or neither (joints {}, weights {})",
                    joints.len(),
                    weights.len()
                ),
            ));
        }
        let vertices = rest.vertex_count();
        for (name, len) in [("joints", joints.len()), ("weights", weights.len())] {
            if len != vertices * MAX_INFLUENCES {
                return Err(SkinError::new(
                    "skin.length_mismatch",
                    format!(
                        "{name} length {len} does not match {MAX_INFLUENCES} x {vertices} vertices"
                    ),
                ));
            }
        }
        for (v, w) in weights.chunks_exact(MAX_INFLUENCES).enumerate() {
            let mut sum = 0.0_f64;
            for (k, &x) in w.iter().enumerate() {
                if !x.is_finite() || x < 0.0 {
                    return Err(SkinError::new(
                        "skin.weights_sum",
                        format!("weights[{v}][{k}] = {x} is not a finite non-negative weight"),
                    ));
                }
                sum += f64::from(x);
            }
            if (sum - 1.0).abs() > f64::from(WEIGHT_SUM_EPS) {
                return Err(SkinError::new(
                    "skin.weights_sum",
                    format!("weights[{v}] sum to {sum}, outside 1 +/- {WEIGHT_SUM_EPS}"),
                ));
            }
        }
        for (v, js) in joints.chunks_exact(MAX_INFLUENCES).enumerate() {
            for (k, &j) in js.iter().enumerate() {
                if usize::from(j) >= MAX_JOINTS {
                    return Err(SkinError::new(
                        "skin.joint_range",
                        format!(
                            "joints[{v}][{k}] = {j} is outside 0..{MAX_JOINTS} (S16 MAX_BONES)"
                        ),
                    ));
                }
            }
        }
        Ok(SkinMesh {
            rest,
            joints: joints.to_vec(),
            weights: weights.to_vec(),
        })
    }

    /// Checks this skin against a rig with `joint_count` joints in S16 bone order:
    /// a referenced joint past the rig is `skin.joint_range`; a rig over the S16
    /// cap is `skin.cap_joints`.
    pub fn validate_joint_count(&self, joint_count: usize) -> Result<(), SkinError> {
        if joint_count > MAX_JOINTS {
            return Err(SkinError::new(
                "skin.cap_joints",
                format!("rig has {joint_count} joints; the S18 cap is {MAX_JOINTS}"),
            ));
        }
        for (i, &j) in self.joints.iter().enumerate() {
            if usize::from(j) >= joint_count {
                let (v, k) = (i / MAX_INFLUENCES, i % MAX_INFLUENCES);
                return Err(SkinError::new(
                    "skin.joint_range",
                    format!("joints[{v}][{k}] = {j} is past the rig's {joint_count} joints"),
                ));
            }
        }
        Ok(())
    }

    /// Appends joints as two little-endian u32 per vertex (`u16x4` as `u32x2`, 8 B).
    pub fn write_joint_bytes(&self, out: &mut Vec<u8>) {
        out.reserve(self.joints.len() * 2);
        for j in self.joints.chunks_exact(MAX_INFLUENCES) {
            let lo = u32::from(j[0]) | (u32::from(j[1]) << 16);
            let hi = u32::from(j[2]) | (u32::from(j[3]) << 16);
            out.extend_from_slice(&lo.to_le_bytes());
            out.extend_from_slice(&hi.to_le_bytes());
        }
    }

    /// Appends weights as little-endian f32x4 per vertex (16 B).
    pub fn write_weight_bytes(&self, out: &mut Vec<u8>) {
        out.reserve(self.weights.len() * 4);
        for &x in &self.weights {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
}
