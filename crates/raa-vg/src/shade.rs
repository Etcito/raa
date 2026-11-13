//! The `vg.shade` forward contributor and the `VgShadeGpu` slot buffer of RAA's virtualized
//! geometry (S12 order 450, S32 group 2). One fullscreen `VgShade` draw per material function
//! of the frame state shades the surface the resolve wrote, while the slots select the function
//! and the debug view with a dynamic offset.
//!
//! [`VgShadeSlots`] is the 16-byte record buffer of S32, grown in pages of 32 slots with the
//! device's uniform offset alignment as the stride (revised S32). The gbuffer share owns its own
//! instance of it; the contributor owns one while the path is active and drops it on release.

// The feature share that installs the contributor lands after this card; until then the
// contributor is reached only through its trait and this module's tests.
#![allow(dead_code)]

use raa_core::{GpuContext, RaaError, wgpu};
use raa_lit::{ForwardContributor, LitHandle, MaterialFnId, PipelineSpec};
use raa_pipeline::{HDR_FORMAT, PrepareCtx};
use raa_shader::{PassKind, VariantKey};

use crate::frame_passes::SharedVg;
use crate::resolve::{surface_bind_group, surface_layout};

/// Bytes of one `VgShadeGpu` record: the material function id, the debug view code and two zero
/// words (S32).
pub const VG_SHADE_GPU_SIZE: u64 = 16;

/// Slots in one page of [`VgShadeSlots`]: the buffer grows by this many records at a time.
const PAGE_SLOTS: usize = 32;

/// The `VgShadeGpu` slots of one material-function list: slot `i` holds the 16-byte record
/// `(material function id, debug view code, 0, 0)` at byte offset `i * stride` (S32, revised).
///
/// Handoff: `raa-vg: vg.gbuffer merge into depth and G-buffer` takes
/// `VgShadeSlots::new(gpu: &GpuContext) -> VgShadeSlots`,
/// `VgShadeSlots::write(&mut self, gpu: &GpuContext, entries: &[(MaterialFnId, u32)]) -> ()`,
/// `VgShadeSlots::offset(&self, i: usize) -> u32`, `VgShadeSlots::buffer(&self) -> &wgpu::Buffer`,
/// `VgShadeSlots::generation(&self) -> u64` and `VgShadeSlots::allocated_bytes(&self) -> u64`
/// exported by crates/raa-vg/src/shade.rs; it attaches by owning its own `VgShadeSlots` inside
/// `VgGbufferPass`, writing `(VgFrameState.material_fns[i], VgFrameState.settings.debug_view.code())`
/// in that pass's `prepare`, and binding the group-2 set of `surface_bind_group` with
/// `&[slots.offset(i)]` in its `record`. State/keys: `VgFrameState::material_fns` (the function
/// per slot) and `VgFrameState::settings.debug_view` (the per-slot code).
pub struct VgShadeSlots {
    buffer: wgpu::Buffer,
    stride: u32,
    capacity: usize,
    generation: u64,
    entries: Vec<(MaterialFnId, u32)>,
}

impl VgShadeSlots {
    /// One page of 32 slots, with the device's uniform offset alignment as the slot stride.
    pub fn new(gpu: &GpuContext) -> VgShadeSlots {
        let stride = gpu
            .device()
            .limits()
            .min_uniform_buffer_offset_alignment
            .max(VG_SHADE_GPU_SIZE as u32);
        VgShadeSlots {
            buffer: create_buffer(gpu, PAGE_SLOTS, stride),
            stride,
            capacity: PAGE_SLOTS,
            generation: 0,
            entries: Vec::new(),
        }
    }

    /// The byte distance between two slots: the device's `min_uniform_buffer_offset_alignment`,
    /// at least [`VG_SHADE_GPU_SIZE`].
    pub fn stride(&self) -> u32 {
        self.stride
    }

    /// The byte offset of slot `i` in [`buffer`](Self::buffer), the dynamic offset of the draw.
    pub fn offset(&self, i: usize) -> u32 {
        i as u32 * self.stride
    }

    /// The uniform buffer of the slots, one 16-byte record per slot stride.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Bumped whenever a growth recreated the buffer, so a holder rebuilds its bind group then.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The GPU memory of the slot buffer.
    pub fn allocated_bytes(&self) -> u64 {
        self.buffer.size()
    }

    /// Writes `entries` as the slot records, growing by pages of 32 (recreating the buffer and
    /// bumping the generation). A call with the entries of the previous write writes nothing.
    pub fn write(&mut self, gpu: &GpuContext, entries: &[(MaterialFnId, u32)]) {
        if self.entries.as_slice() == entries {
            return;
        }
        let capacity = entries.len().max(1).div_ceil(PAGE_SLOTS) * PAGE_SLOTS;
        if capacity != self.capacity {
            self.capacity = capacity;
            self.buffer = create_buffer(gpu, capacity, self.stride);
            self.generation += 1;
        }
        for (i, &(material_fn, debug)) in entries.iter().enumerate() {
            let mut record = [0u8; VG_SHADE_GPU_SIZE as usize];
            record[..4].copy_from_slice(&material_fn.0.to_le_bytes());
            record[4..8].copy_from_slice(&debug.to_le_bytes());
            gpu.queue()
                .write_buffer(&self.buffer, u64::from(self.offset(i)), &record);
        }
        self.entries.clear();
        self.entries.extend_from_slice(entries);
    }
}

/// A uniform buffer of `capacity` slots with the given stride.
fn create_buffer(gpu: &GpuContext, capacity: usize, stride: u32) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("raa-vg shade"),
        size: capacity as u64 * u64::from(stride),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// The `vg.shade` forward contributor (S12 order 450): one fullscreen `VgShade` draw per
/// material function of the frame state, into the forward pass after the opaque draws.
///
/// Handoff: `raa-vg: VgFeature::install and VgHandle (settings, instances, active state,
/// prepare_frame)` takes `VgShadeContributor::new(state: SharedVg) -> VgShadeContributor`
/// exported by crates/raa-vg/src/shade.rs; it attaches by calling
/// `LitRenderer::add_contributor(Box::new(VgShadeContributor::new(state)))` beside the five vg
/// passes, keeping the entry disabled until `VgFrameState::active`. State/keys: the `SharedVg`
/// (`Rc<RefCell<VgFrameState>>`), reading `VgFrameState::material_fns`,
/// `VgFrameState::settings.debug_view` and `VgFrameState::resources` (its `VgSurface`); the
/// contributor owns its `VgShadeSlots` and its group-2 bind group.
pub(crate) struct VgShadeContributor {
    state: SharedVg,
    layout: Option<wgpu::BindGroupLayout>,
    pipelines: Vec<wgpu::RenderPipeline>,
    pipeline_fns: Vec<MaterialFnId>,
    pipeline_msaa: u32,
    entries: Vec<(MaterialFnId, u32)>,
    slots: Option<VgShadeSlots>,
    group: Option<wgpu::BindGroup>,
    group_generation: u64,
    surface: Option<wgpu::Texture>,
}

impl VgShadeContributor {
    pub(crate) const NAME: &'static str = "vg.shade";
    pub(crate) const ORDER: u32 = 450;

    pub(crate) fn new(state: SharedVg) -> VgShadeContributor {
        VgShadeContributor {
            state,
            layout: None,
            pipelines: Vec::new(),
            pipeline_fns: Vec::new(),
            pipeline_msaa: 0,
            entries: Vec::new(),
            slots: None,
            group: None,
            group_generation: 0,
            surface: None,
        }
    }
}

impl ForwardContributor for VgShadeContributor {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn order(&self) -> u32 {
        Self::ORDER
    }

    /// While the path is active with resources, writes the slots for `material_fns` (the debug
    /// view code comes from the settings) and fetches one `VgShade` pipeline per function with
    /// the group-2 surface layout. Inactive or without resources it drops its slots and bind
    /// group, so no vg resource exists while the path is off.
    fn prepare(&mut self, lit: &LitHandle, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let state = self.state.borrow_mut();
        if !state.active || state.resources.is_none() {
            self.pipelines.clear();
            self.pipeline_fns.clear();
            self.pipeline_msaa = 0;
            self.entries.clear();
            self.slots = None;
            self.group = None;
            self.surface = None;
            return Ok(());
        }
        let debug = state.settings.debug_view.code();
        self.entries.clear();
        self.entries
            .extend(state.material_fns.iter().map(|&f| (f, debug)));
        let slots = self.slots.get_or_insert_with(|| VgShadeSlots::new(ctx.gpu));
        slots.write(ctx.gpu, &self.entries);

        let msaa = lit.msaa();
        let rebuild = self.pipeline_msaa != msaa
            || self.pipeline_fns.as_slice() != state.material_fns.as_slice();
        if rebuild {
            let layout = self.layout.get_or_insert_with(|| surface_layout(ctx.gpu));
            let spec = PipelineSpec {
                color_targets: vec![Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                depth: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                cull: None,
                alpha_to_coverage: false,
                extra_layout: Some(layout.clone()),
            };
            self.pipelines.clear();
            for &material_fn in &state.material_fns {
                let key = VariantKey {
                    material_fn,
                    pass: PassKind::VgShade,
                    msaa,
                    flags: 0,
                };
                self.pipelines.push(lit.pipeline_for(ctx.gpu, key, &spec)?);
            }
            self.pipeline_fns.clear();
            self.pipeline_fns.extend_from_slice(&state.material_fns);
            self.pipeline_msaa = msaa;
        }

        let Some(resources) = state.resources.as_ref() else {
            return Ok(());
        };
        let surface = resources.surface.ids().texture();
        let generation = self.slots.as_ref().map_or(0, VgShadeSlots::generation);
        let stale = self.group.is_none()
            || self.group_generation != generation
            || self.surface.as_ref() != Some(surface);
        if stale {
            let layout = self.layout.get_or_insert_with(|| surface_layout(ctx.gpu));
            let Some(slots) = self.slots.as_ref() else {
                return Ok(());
            };
            self.group = Some(surface_bind_group(
                ctx.gpu,
                layout,
                &resources.surface,
                slots.buffer(),
            ));
            self.group_generation = generation;
            self.surface = Some(surface.clone());
        }
        Ok(())
    }

    /// Sets each function's pipeline, binds the group-2 surface with the slot's dynamic offset
    /// (groups 0 and 1 are bound by the forward pass) and draws its fullscreen triangle. Records
    /// nothing while inactive or without resources; pushes `vg.shade` onto `recorded` when it
    /// draws, borrowing the frame state mutably for that push only.
    fn record(&self, pass: &mut wgpu::RenderPass<'_>, _lit: &LitHandle) -> Result<(), RaaError> {
        let ready = {
            let state = self.state.borrow();
            state.active && state.resources.is_some()
        };
        if !ready {
            return Ok(());
        }
        let (Some(group), Some(slots)) = (self.group.as_ref(), self.slots.as_ref()) else {
            return Ok(());
        };
        for (i, pipeline) in self.pipelines.iter().enumerate() {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(2, group, &[slots.offset(i)]);
            pass.draw(0..3, 0..1);
        }
        if !self.pipelines.is_empty() {
            self.state.borrow_mut().recorded.push(Self::NAME);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use raa_core::testing::{skip, software_gpu};

    use super::*;

    /// A new slot buffer holds one page of 32 records at the device stride; 33 entries grow it
    /// to 64 slots and bump the generation.
    #[test]
    fn slots_grow_by_pages_and_bump_the_generation() {
        let test = "slots_grow_by_pages_and_bump_the_generation";
        let Some(gpu) = software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let mut slots = VgShadeSlots::new(&gpu);
        let stride = slots.stride();
        assert!(stride >= VG_SHADE_GPU_SIZE as u32);
        assert_eq!(stride % 16, 0);
        assert_eq!(slots.offset(1), stride);
        assert_eq!(slots.buffer().size(), 32 * u64::from(stride));
        let generation = slots.generation();
        let mut entries: Vec<(MaterialFnId, u32)> = (0..32).map(|i| (MaterialFnId(i), 0)).collect();
        slots.write(&gpu, &entries);
        assert_eq!(slots.buffer().size(), 32 * u64::from(stride));
        assert_eq!(slots.generation(), generation);
        entries.push((MaterialFnId(32), 0));
        slots.write(&gpu, &entries);
        assert_eq!(slots.buffer().size(), 64 * u64::from(stride));
        assert!(slots.generation() > generation);
        assert_eq!(slots.allocated_bytes(), 64 * u64::from(stride));
    }
}
