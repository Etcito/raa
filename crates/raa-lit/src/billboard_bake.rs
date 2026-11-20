//! Billboard bake inputs and framing (M6): the eight orthographic source views, the
//! common source-root framing and the four-vertex proxy quad.
//!
//! [`BillboardBakeInput`] names the submitted draw groups (scene instance tags) and the
//! source-root object frame. [`LitRenderer::bake_billboard`](crate::LitRenderer::bake_billboard)
//! renders every submitted wood and leaf draw in that root frame into the private targets
//! of [`BillboardTargets`](crate::BillboardTargets), publishes the three atlas images and
//! returns them with the framing used. No CPU texel is read back.
//!
//! The direction order is `0..8`, each turned 45 degrees around +Y from +Z: direction 0
//! looks from +Z towards the center, direction 2 from +X, direction 4 from -Z and
//! direction 6 from -X. Every view uses the same square orthographic frame
//! ([`billboard_framing`]), so the eight tiles share one scale and one center.
//!
//! [`billboard_proxy`] builds the display quad in Rust: four vertices, two triangles,
//! centered at the framing center and facing +Z.

use raa_core::RaaError;
use raa_geom::MeshData;
use raa_math::{Aabb, Vec3};
use raa_scene::Camera;

/// Number of baked directions and tiles.
pub const BILLBOARD_DIRECTIONS: usize = 8;

/// The inputs of one billboard bake: the scene tags of the complete submitted draw
/// groups, the source-root object frame their normals are expressed in and the bake
/// resolution (64, 128, 256 or 512).
#[derive(Clone, Debug, PartialEq)]
pub struct BillboardBakeInput {
    pub tags: Vec<u32>,
    pub root_world: raa_math::Mat4,
    pub resolution: u32,
}

/// The published result of one billboard bake: the three atlas ids, the source-root
/// framing and the eight tile rectangles of the layout (direction rank order).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BillboardBakeResult {
    pub diffuse: u32,
    pub normal: u32,
    pub opacity: u32,
    pub center: [f32; 3],
    pub size: [f32; 2],
    pub tiles: [[u32; 4]; 8],
    pub extent: [u32; 2],
}

/// The common square framing of the eight views, in the source-root object frame.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BillboardFraming {
    /// Center of the submitted bounds.
    pub center: [f32; 3],
    /// Square billboard size `[width, height]` in root units.
    pub size: [f32; 2],
}

fn billboard_error(code: &'static str, message: String) -> RaaError {
    RaaError::FrameGraph { code, message }
}

fn invalid(message: String) -> RaaError {
    billboard_error("billboard.invalid", message)
}

/// The common framing of `bounds`: its center and the largest side as a square size, so
/// every direction shares one orthographic scale. `None` for an empty, non-finite or
/// zero-sized box.
pub fn billboard_framing(bounds: &Aabb) -> Option<BillboardFraming> {
    let values = [
        bounds.min.x,
        bounds.min.y,
        bounds.min.z,
        bounds.max.x,
        bounds.max.y,
        bounds.max.z,
    ];
    if !values.iter().all(|v| v.is_finite()) {
        return None;
    }
    let extent = bounds.extent();
    if extent.x < 0.0 || extent.y < 0.0 || extent.z < 0.0 {
        return None;
    }
    let side = extent.x.max(extent.y).max(extent.z);
    if side <= 0.0 {
        return None;
    }
    Some(BillboardFraming {
        center: bounds.center().to_array(),
        size: [side, side],
    })
}

/// The orthographic camera of `direction` (0..8) around +Y, looking at `framing.center`.
/// Direction 0 is the +Z view; the frame is square (`aspect` 1, `ortho_height` the framing
/// size), near 0 and far twice the camera distance.
pub fn billboard_camera(framing: BillboardFraming, direction: u8) -> Result<Camera, RaaError> {
    if usize::from(direction) >= BILLBOARD_DIRECTIONS {
        return Err(invalid(format!(
            "billboard direction {direction} must be within 0..=7"
        )));
    }
    if !framing.center.iter().all(|v| v.is_finite())
        || !framing.size.iter().all(|v| v.is_finite())
        || framing.size[0] <= 0.0
        || framing.size[1] <= 0.0
    {
        return Err(invalid(
            "billboard framing must be finite and larger than zero".into(),
        ));
    }
    let frame = framing.size[0].max(framing.size[1]);
    let angle = f32::from(direction) * std::f32::consts::FRAC_PI_4;
    let center = Vec3::from_array(framing.center);
    let eye = center + Vec3::new(angle.sin(), 0.0, angle.cos()) * frame;
    Ok(Camera {
        eye,
        target: center,
        up: Vec3::Y,
        fov_y_rad: 60.0_f32.to_radians(),
        near: 0.0,
        far: 2.0 * frame,
        aspect: 1.0,
        ortho_height: Some(frame),
    })
}

/// One display quad (four vertices, six indices) centered at `center` with `size`
/// `[width, height]`, facing +Z. Refuses `billboard.invalid` for a non-finite center,
/// a non-positive side or a mesh that fails [`MeshData::validate`].
pub fn billboard_proxy(center: [f32; 3], size: [f32; 2]) -> Result<MeshData, RaaError> {
    if !center.iter().all(|v| v.is_finite())
        || !size.iter().all(|v| v.is_finite())
        || size[0] <= 0.0
        || size[1] <= 0.0
    {
        return Err(invalid(
            "billboard proxy needs a finite center and a positive size".into(),
        ));
    }
    let [cx, cy, cz] = center;
    let (w, h) = (size[0] * 0.5, size[1] * 0.5);
    let mesh = MeshData {
        positions: vec![
            [cx - w, cy - h, cz],
            [cx + w, cy - h, cz],
            [cx + w, cy + h, cz],
            [cx - w, cy + h, cz],
        ],
        normals: vec![[0.0, 0.0, 1.0]; 4],
        indices: vec![0, 1, 2, 0, 2, 3],
    };
    mesh.validate()
        .map_err(|e| invalid(format!("billboard proxy is invalid: {e}")))?;
    Ok(mesh)
}
