//! Surface-sized pipeline targets (S12): depth, normal, material and the HDR A/B chain.

use std::cell::Cell;

use raa_core::GpuContext;

use crate::HDR_FORMAT;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const MATERIAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

const GBUFFER_USAGE: wgpu::TextureUsages =
    wgpu::TextureUsages::RENDER_ATTACHMENT.union(wgpu::TextureUsages::TEXTURE_BINDING);
const HDR_USAGE: wgpu::TextureUsages = GBUFFER_USAGE
    .union(wgpu::TextureUsages::STORAGE_BINDING)
    .union(wgpu::TextureUsages::COPY_SRC)
    .union(wgpu::TextureUsages::COPY_DST);

/// Index of the chain texture written by the forward pass after `reset` (A).
const RESET_DST: u8 = 0;

/// Index of the destination texture after one swap.
fn swapped(dst: u8) -> u8 {
    dst ^ 1
}

/// Index of the source texture: the one that is not the destination.
fn src_of(dst: u8) -> usize {
    usize::from(dst ^ 1)
}

/// One texture with its default full view.
struct Target {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Target {
    fn new(
        gpu: &GpuContext,
        label: &str,
        size: (u32, u32),
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> Target {
        let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Target { texture, view }
    }
}

/// HDR ping-pong pair A/B in HDR_FORMAT. A pass reads `src()`, writes `dst()`, then
/// calls `swap()`; `reset()` at frame start makes A the destination of the forward pass.
pub struct HdrChain {
    textures: [Target; 2],
    /// Index (0 = A, 1 = B) of the current destination texture.
    dst: Cell<u8>,
}

impl HdrChain {
    fn new(gpu: &GpuContext, size: (u32, u32)) -> HdrChain {
        HdrChain {
            textures: [
                Target::new(gpu, "raa-pipeline.hdr.a", size, HDR_FORMAT, HDR_USAGE),
                Target::new(gpu, "raa-pipeline.hdr.b", size, HDR_FORMAT, HDR_USAGE),
            ],
            dst: Cell::new(RESET_DST),
        }
    }

    fn src_target(&self) -> &Target {
        // invariant: src_of returns 0 or 1
        &self.textures[src_of(self.dst.get())]
    }

    fn dst_target(&self) -> &Target {
        // invariant: the index is only ever RESET_DST or swapped(), i.e. 0 or 1
        &self.textures[usize::from(self.dst.get())]
    }

    /// View of the texture the next pass reads.
    pub fn src(&self) -> &wgpu::TextureView {
        &self.src_target().view
    }

    /// View of the texture the next pass writes.
    pub fn dst(&self) -> &wgpu::TextureView {
        &self.dst_target().view
    }

    /// Texture the next pass reads.
    pub fn src_texture(&self) -> &wgpu::Texture {
        &self.src_target().texture
    }

    /// Texture the next pass writes.
    pub fn dst_texture(&self) -> &wgpu::Texture {
        &self.dst_target().texture
    }

    /// Exchanges source and destination (after a pass wrote `dst()`).
    pub fn swap(&self) {
        self.dst.set(swapped(self.dst.get()));
    }

    /// Makes A the destination (frame start: forward writes A, then after a swap src = A).
    pub fn reset(&self) {
        self.dst.set(RESET_DST);
    }
}

/// Surface-sized pipeline targets (S12 formats and usages), recreated when the size changes.
pub struct Targets {
    size: (u32, u32),
    depth: Target,
    normal: Target,
    material: Target,
    hdr: HdrChain,
}

impl Targets {
    /// Creates every target at `size` (zero dimensions clamp to 1). The pipeline creates
    /// them at its first prepare; feature passes only read them.
    pub fn new(gpu: &GpuContext, size: (u32, u32)) -> Targets {
        let size = (size.0.max(1), size.1.max(1));
        Targets {
            size,
            depth: Target::new(gpu, "raa-pipeline.depth", size, DEPTH_FORMAT, GBUFFER_USAGE),
            normal: Target::new(
                gpu,
                "raa-pipeline.normal",
                size,
                NORMAL_FORMAT,
                GBUFFER_USAGE,
            ),
            material: Target::new(
                gpu,
                "raa-pipeline.material",
                size,
                MATERIAL_FORMAT,
                GBUFFER_USAGE,
            ),
            hdr: HdrChain::new(gpu, size),
        }
    }

    /// Recreates every target when `size` (clamped to at least 1) differs from the current
    /// size; returns true when it did.
    pub fn ensure(&mut self, gpu: &GpuContext, size: (u32, u32)) -> bool {
        let size = (size.0.max(1), size.1.max(1));
        if size == self.size {
            return false;
        }
        *self = Targets::new(gpu, size);
        true
    }

    /// Current size in pixels (width, height).
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Depth32Float single-sample depth written by the prepass.
    pub fn depth(&self) -> &wgpu::TextureView {
        &self.depth.view
    }

    /// Rgba16Float view-space normal xyz and roughness.
    pub fn normal(&self) -> &wgpu::TextureView {
        &self.normal.view
    }

    /// Rgba8Unorm F0 rgb (sRGB-encoded) and metallic.
    pub fn material(&self) -> &wgpu::TextureView {
        &self.material.view
    }

    /// Texture behind `depth()`.
    pub fn depth_texture(&self) -> &wgpu::Texture {
        &self.depth.texture
    }

    /// Texture behind `normal()`.
    pub fn normal_texture(&self) -> &wgpu::Texture {
        &self.normal.texture
    }

    /// Texture behind `material()`.
    pub fn material_texture(&self) -> &wgpu::Texture {
        &self.material.texture
    }

    /// The HDR A/B chain.
    pub fn hdr(&self) -> &HdrChain {
        &self.hdr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_makes_a_the_destination() {
        assert_eq!(RESET_DST, 0);
        assert_eq!(src_of(RESET_DST), 1);
    }

    #[test]
    fn swap_after_reset_makes_a_the_source() {
        let dst = swapped(RESET_DST);
        assert_eq!(dst, 1);
        assert_eq!(src_of(dst), 0);
    }

    #[test]
    fn two_swaps_restore_the_destination() {
        assert_eq!(swapped(swapped(RESET_DST)), RESET_DST);
        assert_eq!(swapped(swapped(1)), 1);
    }

    #[test]
    fn source_and_destination_always_differ() {
        let mut dst = RESET_DST;
        for _ in 0..5 {
            assert!(dst <= 1);
            assert_ne!(usize::from(dst), src_of(dst));
            dst = swapped(dst);
        }
    }

    #[test]
    fn reset_after_odd_swaps_returns_to_a() {
        let cell = Cell::new(RESET_DST);
        for _ in 0..3 {
            cell.set(swapped(cell.get()));
        }
        assert_eq!(cell.get(), 1);
        cell.set(RESET_DST);
        assert_eq!(src_of(cell.get()), 1);
    }
}
