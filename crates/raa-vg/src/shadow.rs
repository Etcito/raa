//! The cascade share of the virtualized-geometry passes (S32, feature id 36): the four
//! orthographic [`PassView`]s that draw virtualized casters into the `shadow.csm` cascade
//! layers ([`cascade_views`]), the per-cascade cull work, depth buffer and bind groups
//! ([`CascadeWork`]) and the recording of the hardware depth draw, the software depth pass and
//! the `shadow_merge` merge ([`record_cascades`]).
//!
//! The cascade views are orthographic views over the existing cascade matrices; their
//! receiver planes are the camera frustum slice that cascade covers, so only casters that can
//! shadow a visible receiver are drawn. [`CascadeWork`] builds the two bind groups from the
//! depth path of `raster.rs`; cascades have no visibility buffer, so the raster `bindings`
//! constructor is never called for them.
//!
//! Adapted solstice parts name their source in their module docs; see README.md.

use raa_core::GpuContext;
use raa_math::Mat4;
use raa_scene::Camera;
use raa_shadow::CascadeTargets;

use crate::culling::{CASCADE_BASE, CullPhase, CullPipelines, CullWork, PassView, ViewSlots};
use crate::raster::{DEPTH_STAMPS, DepthBuffer, RasterBindings, RasterTarget, Rasterizer};
use crate::scene::VgScene;
use crate::settings::VgSettings;

/// The cascade count and the size of the `CascadeWork` arrays (S32: cascades 0..3).
const CASCADES: usize = 4;

/// The six planes of the camera frustum slice between `near` and `far`, in the order left,
/// right, bottom, top, near, far, so [`PassView::shadowing`] can keep the ones a caster in
/// front of the light can shadow.
fn slice_planes(camera: &Camera, near: f32, far: f32) -> [raa_math::Vec4; 6] {
    let projection = Mat4::perspective_rh_zo(camera.fov_y_rad, camera.aspect, near, far);
    PassView::frustum_planes(projection * camera.view())
}

/// The four orthographic cascade views of this frame's `shadow.csm` cascades, or four `None`
/// when `settings.shadows` is false. Each view draws into the cascade layer at
/// [`CASCADE_BASE`] plus its index and:
/// - uses the cascade matrix `targets.view_proj[i]`, the camera eye as its centre, the
///   cascade resolution as its viewport and `0.5 * resolution * |view_proj[0][0]|` pixels per
///   unit;
/// - keeps only the receiver planes of the camera frustum slice between the previous split
///   (or `camera.near`) and split `i`, facing the sun;
/// - is seen from the camera eye, with the main projection's LOD scale and the slice's near
///   distance, so detail the camera cannot see is left out;
/// - sends small clusters to the software rasteriser exactly when `settings.software_raster`
///   is on (`allow_software` 1 else 0).
///
/// The caller passes a camera whose `aspect` was set with `Camera::set_viewport`; the viewport
/// height of the eye LOD scale is the cascade resolution, the only pixel size this function is
/// given, and the projection is the camera's own at that aspect.
///
/// Handoff: `cascade_views`, `CascadeWork` and `record_cascades`
/// (crates/raa-vg/src/shadow.rs) are exposed to card 'raa-vg: cascade integration into
/// vg.cull and vg.raster, feature id 36', which takes `CascadeTargets` from
/// `ShadowHandle::cascade_targets`, writes these views to the `CASCADE_BASE` slots with the
/// stamps from `CascadeWork::next_stamps`, records `CascadeWork::rebind`,
/// `CascadeWork::record_cull` and `record_cascades` in its `vg.cull` and `vg.raster` passes,
/// and reads `CascadeWork::works`, `CascadeWork::resolution` and
/// `CascadeWork::allocated_bytes` for its stats. State/keys: the cascade `PassView` slots of
/// [`ViewSlots`], the cascade split distances of [`CascadeTargets::splits`], the cascade
/// layers of [`CascadeTargets::layer_views`] and the sun direction of
/// [`CascadeTargets::sun_dir`]. It builds on the upstream handoff at
/// `raster.rs:634 depth_bindings` (crates/raa-vg/src/raster.rs).
pub fn cascade_views(
    targets: &CascadeTargets,
    camera: &Camera,
    settings: &VgSettings,
) -> [Option<PassView>; 4] {
    let mut views = [None; 4];
    if !settings.shadows {
        return views;
    }
    let projection = camera.projection();
    let light = -targets.sun_dir;
    let resolution = targets.resolution as f32;
    let eye_lod_scale = 0.5 * resolution * projection.cols[1][1];
    let allow_software = u32::from(settings.software_raster);
    let count = (targets.count as usize).min(CASCADES);
    for (index, view) in views.iter_mut().enumerate().take(count) {
        let near = if index == 0 {
            camera.near
        } else {
            targets.splits[index - 1]
        };
        let far = targets.splits[index];
        let receiver = slice_planes(camera, near, far);
        let view_proj = targets.view_proj[index];
        let mut cascade = PassView::orthographic(
            view_proj,
            camera.eye,
            [resolution, resolution],
            0.5 * resolution * view_proj.cols[0][0].abs(),
            settings.error_pixels,
        )
        .shadowing(&receiver, light)
        .seen_from(camera.eye, light, eye_lod_scale, near);
        cascade.allow_software = allow_software;
        *view = Some(cascade);
    }
    views
}

/// The per-cascade GPU work of the virtualized shadow casters: four [`CullWork`]s, one
/// [`DepthBuffer`] at the cascade resolution, the two group-2 bind groups per cascade and the
/// stamp counter of the stamped depth words.
pub struct CascadeWork {
    works: [CullWork; 4],
    depth: DepthBuffer,
    depth_bindings: [Option<RasterBindings>; 4],
    merge_bindings: [Option<RasterBindings>; 4],
    next_stamp: u32,
    pending_clear: bool,
}

impl CascadeWork {
    /// Four cull works of `capacity` queue and visible entries and one depth buffer at
    /// `resolution`. The cull pipelines are needed to build the works; `raster` is accepted so
    /// the caller creates the whole cascade resource set in one call.
    pub fn new(
        gpu: &GpuContext,
        cull: &CullPipelines,
        _raster: &Rasterizer,
        resolution: u32,
        capacity: u64,
    ) -> CascadeWork {
        CascadeWork {
            works: std::array::from_fn(|index| {
                CullWork::new(
                    gpu,
                    cull,
                    &format!("raa-vg cascade {index}"),
                    capacity,
                    capacity,
                )
            }),
            depth: DepthBuffer::new(gpu, resolution),
            depth_bindings: [None, None, None, None],
            merge_bindings: [None, None, None, None],
            next_stamp: 1,
            pending_clear: false,
        }
    }

    pub fn resolution(&self) -> u32 {
        self.depth.resolution()
    }

    pub fn works(&self) -> &[CullWork; 4] {
        &self.works
    }

    /// Four consecutive stamps in 1..=255 for the next recording. When the four would pass
    /// the last stamp, the round starts again at 1 and the next [`record_cascades`] clears the
    /// depth buffer before drawing, so no word of an older round survives.
    pub fn next_stamps(&mut self) -> [u32; 4] {
        let wrapped = self.next_stamp + (CASCADES as u32 - 1) > DEPTH_STAMPS;
        let start = if wrapped {
            self.pending_clear = true;
            1
        } else {
            self.next_stamp
        };
        let stamps = std::array::from_fn(|index| start + index as u32);
        self.next_stamp = start + CASCADES as u32;
        stamps
    }

    /// Rebuilds the two group-2 bind groups of every cascade at slot `CASCADE_BASE + i`:
    /// `depth_bindings` for the hardware cascade draw and the software depth pass, and
    /// `merge_bindings` for the `shadow_merge` fullscreen draw. Cascades have no visibility
    /// buffer, so the raster `bindings` constructor is never called for them.
    pub fn rebind(
        &mut self,
        gpu: &GpuContext,
        _cull: &CullPipelines,
        raster: &Rasterizer,
        slots: &ViewSlots,
        scene: &VgScene,
    ) {
        for index in 0..CASCADES {
            let slot = CASCADE_BASE + index as u32;
            self.depth_bindings[index] = Some(raster.depth_bindings(
                gpu,
                slots,
                slot,
                scene,
                &self.works[index],
                &self.depth,
            ));
            self.merge_bindings[index] = Some(raster.merge_bindings(gpu, slots, slot, &self.depth));
        }
    }

    /// Runs the first culling phase for every cascade whose view is `Some`, at slot
    /// `CASCADE_BASE + i`. Cascades have no occlusion history, so there is one phase.
    pub fn record_cull(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        cull: &CullPipelines,
        scene: &VgScene,
        scene_group: &wgpu::BindGroup,
        views: &[Option<PassView>; 4],
    ) {
        for (index, view) in views.iter().enumerate() {
            if view.is_some() {
                cull.record(
                    encoder,
                    scene,
                    scene_group,
                    CASCADE_BASE + index as u32,
                    &self.works[index],
                    CullPhase::First,
                );
            }
        }
    }

    /// The four cull works and the depth buffer.
    pub fn allocated_bytes(&self) -> u64 {
        self.works
            .iter()
            .map(CullWork::allocated_bytes)
            .sum::<u64>()
            + self.depth.allocated_bytes()
    }
}

/// Records the cascade depth work of this frame into the existing `shadow.csm` layers: the
/// pending depth-buffer clear first (a stamp round wrapped), then for each `Some` view the
/// hardware depth draw with LOAD, the software depth pass under the same bind group and the
/// `shadow_merge` merge.
pub fn record_cascades(
    encoder: &mut wgpu::CommandEncoder,
    raster: &Rasterizer,
    work: &mut CascadeWork,
    targets: &CascadeTargets,
    views: &[Option<PassView>; 4],
    stamps: [u32; 4],
) {
    if work.pending_clear {
        work.depth.clear(encoder);
        work.pending_clear = false;
    }
    for (index, view) in views.iter().enumerate() {
        if view.is_none() {
            continue;
        }
        let (Some(depth_bindings), Some(merge_bindings)) = (
            work.depth_bindings[index].as_ref(),
            work.merge_bindings[index].as_ref(),
        ) else {
            continue;
        };
        let layer = &targets.layer_views[index];
        raster.record(
            encoder,
            depth_bindings,
            &work.works[index],
            RasterTarget::Cascade {
                layer,
                depth: &work.depth,
                stamp: stamps[index],
            },
        );
        raster.record_software_depth(encoder, depth_bindings, &work.works[index]);
        raster.merge_depth(encoder, merge_bindings, &work.works[index], layer);
    }
}
