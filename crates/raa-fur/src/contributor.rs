//! The `fur.shells` forward contributor (S12, order 500): every Fur draw again as N shells
//! with the `FurShell` variant of its material function, after the renderer's opaque base.
//!
//! Group 2 is the `RaaFurShellGpu` uniform of `raa_shader::PASS_FURSHELL_WGSL`
//! (`@group(2) @binding(0)`, 16 B: shell index u32 + 3 reserved u32): one uniform buffer of
//! [`FUR_SHELLS_MAX`] slots at the device's min uniform offset alignment (slot i holds
//! index i), written once at install and bound with a dynamic offset per shell. `prepare`
//! resolves the pipeline and shell count of every Fur draw into a list that keeps its
//! capacity, so a warm frame allocates nothing; `record` only binds and draws.

use std::num::NonZeroU64;

use raa_core::{GpuContext, MaterialId, RaaError};
use raa_lit::{
    DrawClass, FLAG_DOUBLE_SIDED, FLAG_UNLIT, FLAG_VERTEX_FN, ForwardContributor, LitHandle,
    LitRenderer, PipelineSpec,
};
use raa_pipeline::{HDR_FORMAT, PrepareCtx};
use raa_shader::{PassKind, VariantKey};

/// S12 name of the contributor (also the Fur class handler).
pub const FUR_NAME: &str = "fur.shells";
/// S12 order of the contributor inside the forward pass.
pub const FUR_ORDER: u32 = 500;
/// FUR value: shells of a fur material whose description leaves the count at its default.
pub const FUR_SHELLS_DEFAULT: u32 = 32;
/// Most shells per draw (slots of the shell uniform buffer).
pub const FUR_SHELLS_MAX: u32 = 64;
/// Size of `RaaFurShellGpu` (index u32 + 3 reserved u32).
pub const SHELL_UNIFORM_SIZE: u64 = 16;

/// Material flags that select a FurShell shader variant.
const VARIANT_FLAGS: u32 = FLAG_DOUBLE_SIDED | FLAG_VERTEX_FN | FLAG_UNLIT;

/// Shells drawn for a material with `fur.w = fur_w`: clamp(round(fur_w), 1, 64). A
/// non-finite value (refused by `SurfaceMaterial::validate`) gives 1.
pub fn shell_count(fur_w: f32) -> u32 {
    // `as u32` saturates and maps NaN to 0, which `max(1)` lifts back into range.
    (fur_w.round().clamp(1.0, FUR_SHELLS_MAX as f32) as u32).max(1)
}

/// Distance in bytes between two shell slots: [`SHELL_UNIFORM_SIZE`] rounded up to the
/// device's `min_uniform_buffer_offset_alignment` (0 is treated as 1).
pub fn shell_slot_stride(min_uniform_offset_alignment: u32) -> u64 {
    let a = u64::from(min_uniform_offset_alignment.max(1));
    SHELL_UNIFORM_SIZE.div_ceil(a) * a
}

/// Writes the bytes of the shell uniform buffer into `out` (cleared first):
/// [`FUR_SHELLS_MAX`] slots of `stride` bytes (at least [`SHELL_UNIFORM_SIZE`]), slot i
/// holding `index = i` little-endian and zeros elsewhere.
pub fn write_shell_slots(stride: u64, out: &mut Vec<u8>) {
    let stride = stride.max(SHELL_UNIFORM_SIZE) as usize;
    out.clear();
    out.resize(stride * FUR_SHELLS_MAX as usize, 0);
    for (i, slot) in (0..FUR_SHELLS_MAX).zip(out.chunks_exact_mut(stride)) {
        slot[..4].copy_from_slice(&i.to_le_bytes());
    }
}

/// The fixed-function state of every shell pipeline: one HDR_FORMAT target without
/// blending, Depth32Float Less with depth write, no culling, alpha-to-coverage when
/// `msaa > 1`, group 2 = `layout`.
fn shell_spec(layout: Option<wgpu::BindGroupLayout>, msaa: u32) -> PipelineSpec {
    PipelineSpec {
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
        alpha_to_coverage: msaa > 1,
        extra_layout: layout,
    }
}

/// `material.unknown_id` for a Fur draw whose material is not in the library.
fn unknown_material(id: MaterialId) -> RaaError {
    RaaError::FrameGraph {
        code: "material.unknown_id",
        message: format!(
            "{FUR_NAME}: fur draw uses material id {} which is not in the library",
            id.0
        ),
    }
}

/// What `record` draws for one Fur draw, resolved by `prepare`.
struct ShellDraw {
    pipeline: wgpu::RenderPipeline,
    shells: u32,
}

/// Draws the fur shells of every Fur draw; see the module docs.
pub struct FurContributor {
    /// Kept alive beside the bind group that reads it.
    _buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Dynamic offset step between shell slots.
    stride: u32,
    spec: PipelineSpec,
    /// One entry per Fur draw of the current frame, in draw order.
    plan: Vec<ShellDraw>,
}

impl FurContributor {
    /// Creates the shell uniform buffer (written once), its group-2 layout and bind group,
    /// adds the contributor `fur.shells` (order 500) to `lit` and registers it as the Fur
    /// class handler. Refusals (nothing registered): `pipeline.duplicate_pass` (already
    /// installed), `pipeline.busy`.
    pub fn install(gpu: &GpuContext, lit: &mut LitRenderer) -> Result<(), RaaError> {
        let c = FurContributor::new(gpu, lit.handle().msaa());
        lit.add_contributor(Box::new(c))?;
        lit.register_class_handler(DrawClass::Fur, FUR_NAME);
        Ok(())
    }

    fn new(gpu: &GpuContext, msaa: u32) -> FurContributor {
        let device = gpu.device();
        let stride = shell_slot_stride(gpu.limits().min_uniform_buffer_offset_alignment);
        let mut bytes = Vec::new();
        write_shell_slots(stride, &mut bytes);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa.fur.shells"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue().write_buffer(&buffer, 0, &bytes);
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa.fur.shells.layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: NonZeroU64::new(SHELL_UNIFORM_SIZE),
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa.fur.shells"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: NonZeroU64::new(SHELL_UNIFORM_SIZE),
                }),
            }],
        });
        FurContributor {
            _buffer: buffer,
            bind_group,
            // The buffer is 64 x stride bytes, so the stride fits a u32 offset.
            stride: u32::try_from(stride).unwrap_or(u32::MAX),
            spec: shell_spec(Some(layout), msaa),
            plan: Vec::new(),
        }
    }

    /// Fills `plan` with the pipeline and shell count of every Fur draw of `lit`.
    fn plan_draws(&mut self, lit: &LitHandle, gpu: &GpuContext) -> Result<(), RaaError> {
        let msaa = lit.msaa();
        self.spec.alpha_to_coverage = msaa > 1;
        let draws = lit.draws(DrawClass::Fur);
        for d in draws.iter() {
            let record = lit
                .material_record(d.material)
                .ok_or_else(|| unknown_material(d.material))?;
            let key = VariantKey {
                material_fn: d.material_fn,
                pass: PassKind::FurShell,
                msaa,
                flags: record.flags & VARIANT_FLAGS,
            };
            let pipeline = lit.pipeline_for(gpu, key, &self.spec)?;
            self.plan.push(ShellDraw {
                pipeline,
                shells: shell_count(record.fur[3]),
            });
        }
        Ok(())
    }
}

impl ForwardContributor for FurContributor {
    fn name(&self) -> &str {
        FUR_NAME
    }

    fn order(&self) -> u32 {
        FUR_ORDER
    }

    /// Resolves each Fur draw's FurShell pipeline (from the lit pipeline cache) and shell
    /// count. Refusals (nothing drawn this frame): `material.unknown_id` for a draw whose
    /// material record is missing, the `LitHandle::pipeline_for` codes.
    fn prepare(&mut self, lit: &LitHandle, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        self.plan.clear();
        let planned = self.plan_draws(lit, ctx.gpu);
        if planned.is_err() {
            self.plan.clear();
        }
        planned
    }

    /// Draws shell i = 0..N of every Fur draw with group 2 at offset i x stride; groups 0
    /// and 1 are bound by the forward pass.
    fn record(&self, pass: &mut wgpu::RenderPass<'_>, lit: &LitHandle) -> Result<(), RaaError> {
        let draws = lit.draws(DrawClass::Fur);
        if draws.len() != self.plan.len() {
            return Err(RaaError::Resource(format!(
                "{FUR_NAME}: record saw {} fur draws but prepare planned {}",
                draws.len(),
                self.plan.len()
            )));
        }
        for (d, plan) in draws.iter().zip(&self.plan) {
            if d.instance_count == 0 {
                continue;
            }
            let count = lit.bind_mesh(pass, d.mesh)?;
            if count == 0 {
                continue;
            }
            pass.set_pipeline(&plan.pipeline);
            let instances = d.first_instance..d.first_instance.saturating_add(d.instance_count);
            for i in 0..plan.shells {
                pass.set_bind_group(2, &self.bind_group, &[i.saturating_mul(self.stride)]);
                lit.draw_surface_instances(pass, 0..count, instances.clone());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_count_rounds_and_clamps_to_1_64() {
        assert_eq!(shell_count(32.0), FUR_SHELLS_DEFAULT);
        assert_eq!(shell_count(12.4), 12);
        assert_eq!(shell_count(12.5), 13);
        assert_eq!(shell_count(0.0), 1);
        assert_eq!(shell_count(-5.0), 1);
        assert_eq!(shell_count(64.4), 64);
        assert_eq!(shell_count(1000.0), 64);
        assert_eq!(shell_count(f32::NAN), 1);
        assert_eq!(shell_count(f32::INFINITY), 64);
    }

    #[test]
    fn slot_stride_follows_the_uniform_offset_alignment() {
        assert_eq!(shell_slot_stride(256), 256);
        assert_eq!(shell_slot_stride(64), 64);
        assert_eq!(shell_slot_stride(16), 16);
        assert_eq!(shell_slot_stride(4), 16);
        assert_eq!(shell_slot_stride(0), 16);
    }

    #[test]
    fn shell_slots_hold_their_index_at_each_stride() {
        let mut out = vec![0xAA; 3];
        write_shell_slots(256, &mut out);
        assert_eq!(out.len(), 256 * 64);
        for i in 0..FUR_SHELLS_MAX as usize {
            let slot = &out[i * 256..(i + 1) * 256];
            assert_eq!(&slot[..4], &(i as u32).to_le_bytes());
            assert!(slot[4..].iter().all(|b| *b == 0), "slot {i} padding");
        }
        write_shell_slots(1, &mut out);
        assert_eq!(out.len(), 16 * 64);
        assert_eq!(&out[16 * 63..16 * 63 + 4], &63u32.to_le_bytes());
    }

    #[test]
    fn shell_spec_matches_the_fur_pipeline_state() {
        for (msaa, a2c) in [(1, false), (4, true)] {
            let spec = shell_spec(None, msaa);
            assert_eq!(spec.alpha_to_coverage, a2c);
            assert_eq!(spec.cull, None);
            assert_eq!(spec.color_targets.len(), 1);
            let target = spec.color_targets[0].as_ref().expect("colour target");
            assert_eq!(target.format, HDR_FORMAT);
            assert!(target.blend.is_none());
            let depth = spec.depth.as_ref().expect("depth state");
            assert_eq!(depth.format, wgpu::TextureFormat::Depth32Float);
            assert_eq!(depth.depth_compare, Some(wgpu::CompareFunction::Less));
            assert_eq!(depth.depth_write_enabled, Some(true));
        }
    }

    #[test]
    fn name_order_and_variant_flags_follow_the_contract() {
        assert_eq!(FUR_NAME, "fur.shells");
        assert_eq!(FUR_ORDER, 500);
        assert_eq!(VARIANT_FLAGS, 8 | 32 | 64);
        assert_eq!(
            unknown_material(MaterialId(7)).code(),
            "material.unknown_id"
        );
    }
}
