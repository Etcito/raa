//! The 768-byte S10 `FrameGpu` uniform (group 0 binding 0) of the lit pipeline.

use raa_core::RaaError;
use raa_math::Mat4;
use raa_scene::Scene;

pub use raa_shader::FRAME_GPU_SIZE;

/// Default sun angular radius in radians (`sun_dir.w`).
pub const SUN_ANGULAR_RADIUS_DEFAULT: f32 = 0.0047;

/// TONEMAP_DEFAULT as written in `FrameGpu.exposure`: exposure 0 EV (scale 2^0), mapper
/// agx (0), look none (0). The tonemap pass reads its own control, not these values.
const EXPOSURE_DEFAULT: [f32; 4] = [0.0, 1.0, 0.0, 0.0];

/// The per-frame values of `FrameGpu` that do not come from the scene.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FrameParams {
    /// Render size in pixels (zero counts as 1).
    pub viewport: (u32, u32),
    /// Tonemap encodes sRGB (the canvas format is not `*Srgb`).
    pub encode_srgb: bool,
    /// MSAA sample count of the forward pass.
    pub msaa: u32,
    /// Logical feature ids 0-31 (flags.z). Prefer [`FrameParams::set_feature`].
    pub feature_bits: u32,
    /// Logical feature ids 32-63 (flags.w). Prefer [`FrameParams::set_feature`].
    pub feature_bits_hi: u32,
    /// Frame index (written as f32).
    pub frame_index: u64,
    /// Time in seconds (`camera_pos.w`).
    pub time_s: f32,
    /// Time since the previous frame in seconds.
    pub dt_s: f32,
    /// Sun angular radius in radians.
    pub sun_angular_radius: f32,
    /// Current and previous camera jitter in NDC xy; defaults to zero.
    pub camera_jitter_ndc: [f32; 2],
    pub previous_jitter_ndc: [f32; 2],
    pub previous_view_proj: Mat4,
    /// Light info: active count, tile columns, rows and stride in words.
    pub active_lights: u32,
    pub tile_columns: u32,
    pub tile_rows: u32,
    pub tile_stride: u32,
    /// Temporal state: parity, accumulation enabled and history valid.
    pub frame_parity: u32,
    pub accumulation_enabled: bool,
    pub history_valid: bool,
}

impl Default for FrameParams {
    fn default() -> Self {
        FrameParams {
            viewport: (1, 1),
            encode_srgb: false,
            msaa: 4,
            feature_bits: 0,
            feature_bits_hi: 0,
            frame_index: 0,
            time_s: 0.0,
            dt_s: 0.0,
            sun_angular_radius: SUN_ANGULAR_RADIUS_DEFAULT,
            camera_jitter_ndc: [0.0; 2],
            previous_jitter_ndc: [0.0; 2],
            previous_view_proj: Mat4::default(),
            active_lights: 0,
            tile_columns: 0,
            tile_rows: 0,
            tile_stride: 0,
            frame_parity: 0,
            accumulation_enabled: false,
            history_valid: false,
        }
    }
}

impl FrameParams {
    /// Sets a logical feature using shared addressing; ids >= 64 are ignored.
    pub fn set_feature(&mut self, id: u32, enabled: bool) {
        let (slot, bit) = match raa_shader::feature_address(id) {
            Some((2, bit)) => (&mut self.feature_bits, bit),
            Some((3, bit)) => (&mut self.feature_bits_hi, bit),
            _ => return,
        };
        if enabled {
            *slot |= 1 << bit;
        } else {
            *slot &= !(1 << bit);
        }
    }
}

/// Writes `values` as consecutive f32 LE into `out`.
fn put_f32s(out: &mut [u8], values: &[f32]) {
    for (chunk, v) in out.chunks_exact_mut(4).zip(values) {
        chunk.copy_from_slice(&v.to_le_bytes());
    }
}

fn put_u32s(out: &mut [u8], values: &[u32]) {
    for (chunk, v) in out.chunks_exact_mut(4).zip(values) {
        chunk.copy_from_slice(&v.to_le_bytes());
    }
}

fn jittered_projection(proj: Mat4, jitter: [f32; 2]) -> Mat4 {
    if jitter == [0.0, 0.0] {
        return proj;
    }
    let mut j = Mat4::IDENTITY;
    j.cols[3] = [jitter[0], jitter[1], 0.0, 1.0];
    j * proj
}

/// Writes the S10 `FrameGpu` from the scene camera, light and ambient and `p`: view_proj
/// @0, view @64, proj @128, inv_proj @192, inv_view @256, camera_pos @320 (w = time),
/// sun_dir @336 (normalised travel direction, w = angular radius), sun_color @352 (rgb x
/// intensity), ambient @368 (rgb x intensity), viewport @384 (w, h, 1/w, 1/h), clip @400
/// (near, far, frame index, dt), flags @416 (encode sRGB, msaa, feature bits 0-31,
/// feature bits 32-63), exposure @432 (TONEMAP_DEFAULT), reserved @448..512 zero,
/// previous_view_proj @512, unjittered_view_proj @576, camera_jitter @640 (current NDC
/// xy, previous NDC xy), light_info @656 (active count, tile columns, tile rows, tile
/// stride in words), temporal @672 (parity, accumulation enabled, history valid, 0) and
/// reserved @688..768 zero.
/// Handoff: `pub const fn feature_address(id: u32) -> Option<(usize, u32)>` from
/// `crates/raa-shader/src/gpu_constants.rs` (landed first) maps `FrameParams` feature
/// ids into flags shared with generated WGSL. Its re-exported `FRAME_GPU_SIZE` sizes
/// this writer; `view_proj`/`inv_proj` carry jitter, `view`/`proj` stay unjittered.
///
/// Refuses `render.invalid_camera` when the view or projection matrix is not invertible
/// (for example eye == target); `out` is then left unchanged.
pub fn write_frame_gpu(
    out: &mut [u8; FRAME_GPU_SIZE],
    scene: &Scene,
    p: &FrameParams,
) -> Result<(), RaaError> {
    let cam = &scene.camera;
    let view = cam.view();
    let proj = cam.projection();
    let jittered_proj = jittered_projection(proj, p.camera_jitter_ndc);
    let (Some(inv_view), Some(inv_proj)) = (view.inverse(), jittered_proj.inverse()) else {
        return Err(RaaError::FrameGraph {
            code: "render.invalid_camera",
            message: format!(
                "camera view or projection is not invertible (eye {:?}, target {:?}, fov {}, near {}, far {})",
                cam.eye, cam.target, cam.fov_y_rad, cam.near, cam.far
            ),
        });
    };
    let mats: [Mat4; 5] = [jittered_proj * view, view, proj, inv_proj, inv_view];
    for (chunk, m) in out[0..320].chunks_exact_mut(64).zip(&mats) {
        m.write_le_bytes(chunk);
    }
    let eye = cam.eye;
    put_f32s(&mut out[320..336], &[eye.x, eye.y, eye.z, p.time_s]);
    let d = scene.light.direction.normalize_or_zero();
    put_f32s(&mut out[336..352], &[d.x, d.y, d.z, p.sun_angular_radius]);
    let (c, i) = (scene.light.color, scene.light.intensity);
    put_f32s(&mut out[352..368], &[c[0] * i, c[1] * i, c[2] * i, 0.0]);
    let (c, i) = (scene.ambient.color, scene.ambient.intensity);
    put_f32s(&mut out[368..384], &[c[0] * i, c[1] * i, c[2] * i, 0.0]);
    let (w, h) = (p.viewport.0.max(1) as f32, p.viewport.1.max(1) as f32);
    put_f32s(&mut out[384..400], &[w, h, 1.0 / w, 1.0 / h]);
    put_f32s(
        &mut out[400..416],
        &[cam.near, cam.far, p.frame_index as f32, p.dt_s],
    );
    put_u32s(
        &mut out[416..432],
        &[
            u32::from(p.encode_srgb),
            p.msaa,
            p.feature_bits,
            p.feature_bits_hi,
        ],
    );
    put_f32s(&mut out[432..448], &EXPOSURE_DEFAULT);
    out[448..512].fill(0);
    p.previous_view_proj.write_le_bytes(&mut out[512..576]);
    (proj * view).write_le_bytes(&mut out[576..640]);
    put_f32s(
        &mut out[640..656],
        [p.camera_jitter_ndc, p.previous_jitter_ndc].as_flattened(),
    );
    put_u32s(
        &mut out[656..672],
        &[p.active_lights, p.tile_columns, p.tile_rows, p.tile_stride],
    );
    put_u32s(
        &mut out[672..688],
        &[
            p.frame_parity,
            u32::from(p.accumulation_enabled),
            u32::from(p.history_valid),
            0,
        ],
    );
    out[688..FRAME_GPU_SIZE].fill(0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_math::Vec3;

    fn f32_at(b: &[u8], off: usize) -> f32 {
        f32::from_le_bytes(b[off..off + 4].try_into().unwrap())
    }

    fn u32_at(b: &[u8], off: usize) -> u32 {
        u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
    }

    fn mat_bytes(m: Mat4) -> [u8; 64] {
        let mut b = [0u8; 64];
        m.write_le_bytes(&mut b);
        b
    }

    #[test]
    fn frame_gpu_offsets_follow_s10() {
        let scene = Scene::new();
        let p = FrameParams {
            viewport: (1280, 720),
            encode_srgb: true,
            msaa: 4,
            feature_bits: 0b101,
            frame_index: 7,
            time_s: 1.5,
            dt_s: 0.016,
            ..FrameParams::default()
        };
        let mut out = [0xAAu8; FRAME_GPU_SIZE];
        write_frame_gpu(&mut out, &scene, &p).unwrap();
        let cam = &scene.camera;
        assert_eq!(&out[0..64], &mat_bytes(cam.view_proj()));
        assert_eq!(&out[64..128], &mat_bytes(cam.view()));
        assert_eq!(&out[128..192], &mat_bytes(cam.projection()));
        assert_eq!(
            &out[192..256],
            &mat_bytes(cam.projection().inverse().unwrap())
        );
        assert_eq!(&out[256..320], &mat_bytes(cam.view().inverse().unwrap()));
        assert_eq!(f32_at(&out, 320), cam.eye.x);
        assert_eq!(f32_at(&out, 332), 1.5);
        let d = scene.light.direction.normalize_or_zero();
        assert_eq!(f32_at(&out, 336), d.x);
        assert_eq!(f32_at(&out, 348), SUN_ANGULAR_RADIUS_DEFAULT);
        let i = scene.light.intensity;
        assert_eq!(f32_at(&out, 352), scene.light.color[0] * i);
        let a = scene.ambient;
        assert_eq!(f32_at(&out, 368), a.color[0] * a.intensity);
        assert_eq!(f32_at(&out, 384), 1280.0);
        assert_eq!(f32_at(&out, 392), 1.0 / 1280.0);
        assert_eq!(f32_at(&out, 396), 1.0 / 720.0);
        assert_eq!(f32_at(&out, 400), cam.near);
        assert_eq!(f32_at(&out, 404), cam.far);
        assert_eq!(f32_at(&out, 408), 7.0);
        assert_eq!(f32_at(&out, 412), 0.016);
        assert_eq!(
            [416, 420, 424, 428].map(|o| u32_at(&out, o)),
            [1, 4, 0b101, 0]
        );
        assert_eq!(
            [432, 436, 440, 444].map(|o| f32_at(&out, o)),
            [0.0, 1.0, 0.0, 0.0]
        );
        assert!(out[448..512].iter().all(|&b| b == 0));
        assert_eq!(&out[512..576], &[0u8; 64]);
        assert_eq!(&out[576..640], &mat_bytes(cam.view_proj()));
        assert_eq!([640, 644, 648, 652].map(|o| f32_at(&out, o)), [0.0; 4]);
        assert_eq!([656, 660, 664, 668].map(|o| u32_at(&out, o)), [0; 4]);
        assert_eq!([672, 676, 680, 684].map(|o| u32_at(&out, o)), [0; 4]);
        assert!(out[688..768].iter().all(|&b| b == 0));
        assert_eq!(FRAME_GPU_SIZE, 768);
        let mut prefix = Vec::from(&out[..320]); // Matrices checked above.
        let eye = cam.eye;
        for lane in [
            [eye.x, eye.y, eye.z, 1.5],
            [d.x, d.y, d.z, SUN_ANGULAR_RADIUS_DEFAULT],
            [
                scene.light.color[0] * i,
                scene.light.color[1] * i,
                scene.light.color[2] * i,
                0.0,
            ],
            [
                a.color[0] * a.intensity,
                a.color[1] * a.intensity,
                a.color[2] * a.intensity,
                0.0,
            ],
            [1280.0, 720.0, 1.0 / 1280.0, 1.0 / 720.0],
            [cam.near, cam.far, 7.0, 0.016],
        ] {
            prefix.extend(lane.into_iter().flat_map(f32::to_le_bytes));
        }
        prefix.extend([1u32, 4, 0b101, 0].into_iter().flat_map(u32::to_le_bytes));
        prefix.extend(
            [0.0f32, 1.0, 0.0, 0.0]
                .into_iter()
                .flat_map(f32::to_le_bytes),
        );
        prefix.resize(512, 0);
        assert_eq!(&out[..512], prefix);
    }

    #[test]
    fn frame_gpu_extended_fields() {
        let scene = Scene::new();
        let prev = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let p = FrameParams {
            viewport: (640, 360),
            camera_jitter_ndc: [0.25, -0.125],
            previous_jitter_ndc: [0.5, -0.25],
            previous_view_proj: prev,
            active_lights: 17,
            tile_columns: 40,
            tile_rows: 23,
            tile_stride: 128,
            frame_parity: 1,
            accumulation_enabled: true,
            history_valid: true,
            ..FrameParams::default()
        };
        let mut out = [0u8; FRAME_GPU_SIZE];
        write_frame_gpu(&mut out, &scene, &p).unwrap();
        let cam = &scene.camera;
        let jittered = jittered_projection(cam.projection(), p.camera_jitter_ndc);
        assert_eq!(&out[0..64], &mat_bytes(jittered * cam.view()));
        assert_eq!(&out[192..256], &mat_bytes(jittered.inverse().unwrap()));
        assert_eq!(&out[512..576], &mat_bytes(prev));
        assert_eq!(
            [640, 644, 648, 652].map(|o| f32_at(&out, o)),
            [0.25, -0.125, 0.5, -0.25]
        );
        assert_eq!(
            [656, 660, 664, 668].map(|o| u32_at(&out, o)),
            [17, 40, 23, 128]
        );
        assert_eq!([672, 676, 680, 684].map(|o| u32_at(&out, o)), [1, 1, 1, 0]);
    }

    #[test]
    fn degenerate_camera_is_refused() {
        let mut scene = Scene::new();
        scene.camera.eye = Vec3::ZERO;
        scene.camera.target = Vec3::ZERO;
        let mut out = [0u8; FRAME_GPU_SIZE];
        let err = write_frame_gpu(&mut out, &scene, &FrameParams::default()).unwrap_err();
        assert_eq!(err.code(), "render.invalid_camera");
        assert!(!err.message().is_empty());
    }
}
