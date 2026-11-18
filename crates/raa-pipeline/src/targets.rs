//! Surface-sized pipeline targets (S12): depth, normal, material, motion (Rgba16Float: UV
//! displacement and history reject) and the HDR A/B chain.

use std::cell::{Cell, OnceCell};

use raa_core::GpuContext;

use crate::HDR_FORMAT;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const MATERIAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const MOTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

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
    /// Registry of views bound since the last `Targets::ensure`.
    bound_views: [OnceCell<wgpu::TextureView>; BOUND_SLOTS],
    /// Registry slots of the bound (src, dst) views.
    bound: Cell<Option<(u8, u8)>>,
}

/// Number of views one chain can hold bound per frame.
const BOUND_SLOTS: usize = 32;

/// Slot of a value equal to `wanted`, else the first empty slot, else None.
fn slot_for<T: PartialEq>(held: &[Option<&T>], wanted: &T) -> Option<usize> {
    held.iter()
        .position(|h| *h == Some(wanted))
        .or_else(|| held.iter().position(Option::is_none))
}

impl HdrChain {
    fn new(gpu: &GpuContext, size: (u32, u32)) -> HdrChain {
        HdrChain {
            textures: [
                Target::new(gpu, "raa-pipeline.hdr.a", size, HDR_FORMAT, HDR_USAGE),
                Target::new(gpu, "raa-pipeline.hdr.b", size, HDR_FORMAT, HDR_USAGE),
            ],
            dst: Cell::new(RESET_DST),
            bound_views: std::array::from_fn(|_| OnceCell::new()),
            bound: Cell::new(None),
        }
    }

    /// Empties the view registry and unbinds.
    fn clear_bound(&mut self) {
        for slot in &mut self.bound_views {
            slot.take();
        }
        self.bound.set(None);
    }

    fn slot_of(&self, view: &wgpu::TextureView) -> Option<u8> {
        let held: [Option<&wgpu::TextureView>; BOUND_SLOTS] =
            std::array::from_fn(|i| self.bound_views[i].get());
        let slot = slot_for(&held, view)?;
        // A fresh slot is empty, so the set cannot fail.
        let _ = self.bound_views[slot].set(view.clone());
        u8::try_from(slot).ok()
    }

    fn bound_view(&self, pick: impl Fn((u8, u8)) -> u8) -> Option<&wgpu::TextureView> {
        let slot = pick(self.bound.get()?);
        self.bound_views.get(usize::from(slot))?.get()
    }

    /// Binds `src` and `dst` as the chain pair: `src()`, `dst()` and the texture
    /// methods return them and `swap` does nothing until `unbind` or `reset`. When
    /// the 32-slot registry has no room the chain stays unbound.
    ///
    /// Handoff: the downstream card 'Wiring 3/5: colour-chain graph frame, resource pool
    /// and order from links' calls `HdrChain::bind`, `HdrChain::unbind` and
    /// `HdrChain::is_bound` from its `wiring` graph frame (before_record binds,
    /// after_record unbinds); the fog, oit, transmission, particles, gi, ssr, dof, shafts,
    /// bloom, grading, cel and taa chain-binding cards ('Chain binding in raa-fog: fog pass
    /// bind groups for a bound HDR chain', 'Chain binding in raa-oit: transparent.oit
    /// resolve for a bound HDR chain', 'Chain binding in raa-transmission: opaque.copy
    /// pass-through for a bound HDR chain', 'Chain binding in raa-particles:
    /// particles.blend for a bound HDR chain', 'Chain binding in raa-gi: gi.history
    /// pass-through and debug view for a bound HDR chain', 'raa-ssr: bound HDR chain and
    /// SsrFeature::install_instance', 'raa-dof: bound HDR chain and
    /// DofFeature::install_instance', 'raa-shafts: bound HDR chain and
    /// ShaftsFeature::install_instance', 'raa-bloom: bound HDR chain and
    /// BloomFeature::install_instance', 'raa-grading: bound HDR chain and
    /// GradingFeature::install_instance', 'raa-cel: bound HDR chain and
    /// CelFeature::install_instance', 'raa-taa: bound HDR chain and
    /// TaaFeature::install_instance') call `HdrChain::is_bound`. Today bind stores the
    /// views in `HdrChain.bound_views` and the slot pair in `HdrChain.bound`.
    pub fn bind(&self, src: &wgpu::TextureView, dst: &wgpu::TextureView) {
        let pair = self.slot_of(src).zip(self.slot_of(dst));
        self.bound.set(pair);
    }

    /// Returns to the A/B pair with the destination it had before `bind`.
    pub fn unbind(&self) {
        self.bound.set(None);
    }

    /// True between `bind` and `unbind` (or `reset`).
    pub fn is_bound(&self) -> bool {
        self.bound.get().is_some()
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
        self.bound_view(|p| p.0)
            .unwrap_or_else(|| &self.src_target().view)
    }

    /// View of the texture the next pass writes.
    pub fn dst(&self) -> &wgpu::TextureView {
        self.bound_view(|p| p.1)
            .unwrap_or_else(|| &self.dst_target().view)
    }

    /// Texture the next pass reads.
    pub fn src_texture(&self) -> &wgpu::Texture {
        self.bound_view(|p| p.0)
            .map_or_else(|| &self.src_target().texture, |v| v.texture())
    }

    /// Texture the next pass writes.
    pub fn dst_texture(&self) -> &wgpu::Texture {
        self.bound_view(|p| p.1)
            .map_or_else(|| &self.dst_target().texture, |v| v.texture())
    }

    /// Exchanges source and destination (after a pass wrote `dst()`).
    pub fn swap(&self) {
        if !self.is_bound() {
            self.dst.set(swapped(self.dst.get()));
        }
    }

    /// Makes A the destination (frame start: forward writes A, then after a swap src = A).
    pub fn reset(&self) {
        self.dst.set(RESET_DST);
        self.unbind();
    }
}

/// Surface-sized pipeline targets (S12 formats and usages), recreated when the size changes.
pub struct Targets {
    size: (u32, u32),
    depth: Target,
    normal: Target,
    material: Target,
    motion: Target,
    hdr: HdrChain,
    output: Option<Target>,
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
            motion: Target::new(
                gpu,
                "raa-pipeline.motion",
                size,
                MOTION_FORMAT,
                GBUFFER_USAGE,
            ),
            hdr: HdrChain::new(gpu, size),
            output: None,
        }
    }

    /// Recreates every target when `size` (clamped to at least 1) differs from the current
    /// size; returns true when it did.
    pub fn ensure(&mut self, gpu: &GpuContext, size: (u32, u32)) -> bool {
        let size = (size.0.max(1), size.1.max(1));
        self.hdr.clear_bound();
        if size == self.size {
            return false;
        }
        *self = Targets::new(gpu, size);
        true
    }

    /// Display-space target for tone mapping and overlays; absent on the direct path.
    pub(crate) fn ensure_output(&mut self, gpu: &GpuContext, format: wgpu::TextureFormat) {
        if self
            .output
            .as_ref()
            .is_none_or(|t| t.texture.format() != format)
        {
            self.output = Some(Target::new(
                gpu,
                "raa.views.output",
                self.size,
                format,
                GBUFFER_USAGE,
            ));
        }
    }

    pub(crate) fn output(&self) -> Option<&wgpu::TextureView> {
        self.output.as_ref().map(|t| &t.view)
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

    /// Rgba16Float signed UV motion xy with the history reject flag in z, written by the
    /// prepass (location 2).
    pub fn motion(&self) -> &wgpu::TextureView {
        &self.motion.view
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

    /// Texture behind `motion()`.
    pub fn motion_texture(&self) -> &wgpu::Texture {
        &self.motion.texture
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
    fn motion_uses_the_s12_format_and_attachment_usage() {
        assert_eq!(MOTION_FORMAT, wgpu::TextureFormat::Rgba16Float);
        assert!(GBUFFER_USAGE.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
        assert!(GBUFFER_USAGE.contains(wgpu::TextureUsages::TEXTURE_BINDING));
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

    #[test]
    fn slot_for_reuses_then_fills_then_fails() {
        assert_eq!(slot_for(&[Some(&1), Some(&2), None], &2), Some(1));
        assert_eq!(slot_for(&[Some(&1), None, None], &5), Some(1));
        assert_eq!(slot_for(&[Some(&1), Some(&2)], &5), None);
    }

    fn extra(gpu: &GpuContext) -> wgpu::TextureView {
        Target::new(gpu, "t", (8, 8), HDR_FORMAT, HDR_USAGE).view
    }

    #[test]
    fn bind_unbind_reset_and_registry_limit() {
        use raa_core::testing::{skip, software_gpu};
        let name = "bind_unbind_reset_and_registry_limit";
        let Some(gpu) = software_gpu() else {
            skip(name, "software_gpu returned None");
            return;
        };
        let mut targets = Targets::new(&gpu, (8, 8));
        let (a, b) = (extra(&gpu), extra(&gpu));
        let hdr = targets.hdr();
        hdr.swap();
        let (old_src, old_dst) = (hdr.src().clone(), hdr.dst().clone());
        hdr.bind(&a, &b);
        assert!(hdr.is_bound());
        assert!(hdr.src() == &a && hdr.dst() == &b);
        assert!(hdr.src_texture() == a.texture() && hdr.dst_texture() == b.texture());
        hdr.swap();
        assert!(hdr.src() == &a && hdr.dst() == &b);
        hdr.unbind();
        assert!(!hdr.is_bound());
        assert!(hdr.src() == &old_src && hdr.dst() == &old_dst);
        hdr.bind(&a, &b);
        hdr.reset();
        assert!(!hdr.is_bound());
        assert!(!targets.ensure(&gpu, (8, 8)));
        let hdr = targets.hdr();
        let views: Vec<_> = (0..33).map(|_| extra(&gpu)).collect();
        for v in &views[..32] {
            hdr.bind(v, v);
            assert!(hdr.is_bound());
        }
        hdr.bind(&views[32], &views[32]);
        assert!(!hdr.is_bound());
        assert!(!targets.ensure(&gpu, (8, 8)));
        let hdr = targets.hdr();
        hdr.bind(&views[32], &views[32]);
        assert!(hdr.is_bound() && hdr.src() == &views[32]);
    }
}
