//! Material textures: one rgba8unorm atlas, its sampler and the rect of every texture,
//! bound in group 0 (bindings 3, 4, 5) and sampled by the prelude's `raa_texture(id, uv)`.
//!
//! Textures are packed on shelves (rows as tall as their tallest texture, reused once every
//! texture of the row is removed). The atlas texture is created on the first add: until
//! then group 0 binds a 1 x 1 placeholder, so engines without textures keep their memory.
//! Rect `id` of the storage buffer is (x, y, width, height) / [`ATLAS_SIZE`]; the shader
//! clamps the sampled point half a texel inside it, so linear filtering never reads a
//! neighbour. Id 0 is never allocated: its rect stays zero and samples (0, 0, 0, 0).
//!
//! Mip chain (S48): the atlas has [`ATLAS_MIP_LEVELS`] (5) levels. Every texture reserves
//! `round_up(side, ATLAS_ALIGN)` (16) texels per side, so its mip texels (origin `x >> k`)
//! belong to it alone; the alignment lowers the capacity by up to 15 texels per side per
//! texture. Memory: 64 MiB for level 0, about 85 MiB with the chain. The first add creates
//! the 5-level atlas; with `TextureOptions::mips` each add uploads levels
//! 1..`texture_levels(w, h)`, each made from the previous level by `downsample_rgba8`.
//!
//! Rect encoding (binding 5, 16 B): `(x, y, w, h) / ATLAS_SIZE`; a negative `w` marks a
//! texture added with `mips: false`, sampled at level 0 only; the shader uses `abs(w)`.
//! Id 0 and removed ids stay all zero.

use raa_core::{GpuContext, RaaError};

/// Width and height of the atlas texture in texels.
pub const ATLAS_SIZE: u32 = 4096;
/// Number of texture ids (rects), id 0 included and never allocated.
pub const MAX_TEXTURES: u32 = 1024;
/// Size in bytes of one rect of the storage buffer (`vec4<f32>`).
pub const TEXTURE_RECT_SIZE: usize = 16;
/// Format of the atlas texture.
pub const ATLAS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Mip levels of the atlas texture (levels 0..=4).
pub const ATLAS_MIP_LEVELS: u32 = 5;
/// Texel alignment of rect origins and reserved extents.
pub const ATLAS_ALIGN: u32 = 16;

/// Mip levels of a `width` x `height` texture: `min(5, 1 + floor(log2(min side)))`; a zero
/// side returns 1.
pub fn texture_levels(width: u32, height: u32) -> u32 {
    let m = width.min(height);
    if m == 0 {
        return 1;
    }
    (32 - m.leading_zeros()).min(ATLAS_MIP_LEVELS)
}

/// `size` halved `level` times, each step `(s + 1) / 2`.
pub fn level_size(size: u32, level: u32) -> u32 {
    (0..level).fold(size, |s, _| s.div_ceil(2))
}

/// Footprint stretch covered by the prelude's 4 taps along the major axis (`RAA_ATLAS_MAX_ANISO`).
pub const ATLAS_MAX_ANISO: f32 = 4.0;

/// CPU twin of the prelude's lod: `clamp(0.5 * log2(max(rho2, 1e-12)), 0, levels - 1)`
/// with `rho2 = max(minor2, major2 / 16)`, minor2/major2 the smaller/larger squared length of
/// `duv_dx * size` and `duv_dy * size` (the minor axis picks the level, 4 taps cover the major).
pub fn texture_lod(size_texels: [f32; 2], duv_dx: [f32; 2], duv_dy: [f32; 2], levels: u32) -> f32 {
    let len2 = |d: [f32; 2]| {
        let (x, y) = (d[0] * size_texels[0], d[1] * size_texels[1]);
        x * x + y * y
    };
    let (a, b) = (len2(duv_dx), len2(duv_dy));
    let rho2 = a.min(b).max(a.max(b) / (ATLAS_MAX_ANISO * ATLAS_MAX_ANISO));
    let top = levels.max(1) - 1;
    (0.5 * rho2.max(1e-12).log2()).clamp(0.0, top as f32)
}

/// CPU twin of the prelude's inset: `0.5 * exp2(ceil(lod)) / atlas_size`.
pub fn level_inset(lod: f32, atlas_size: f32) -> f32 {
    0.5 * lod.ceil().exp2() / atlas_size
}

/// Options of an added texture.
///
/// Handoff: "raa-lit: 5-level atlas upload, mipmap sampler and LitRenderer::add_texture_with"
/// uses `TextureOptions`, `texture_levels(width: u32, height: u32) -> u32`,
/// `level_size(size: u32, level: u32) -> u32`, `ATLAS_MIP_LEVELS` and
/// `AtlasRect::gpu_bytes_with(&self, mips: bool) -> [u8; 16]` from this file to create the
/// 5-level atlas and upload levels 1..texture_levels-1; today they are defined here and
/// nothing in this file calls them yet. "raa-js m22: addTexture mips/srgb options, docs and
/// typings" builds `TextureOptions` through the raa_lit root re-export.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TextureOptions {
    /// Generate mip levels (false: level 0 only).
    pub mips: bool,
    /// Average colour in linear light when downsampling.
    pub srgb: bool,
}

impl Default for TextureOptions {
    fn default() -> Self {
        TextureOptions {
            mips: true,
            srgb: false,
        }
    }
}

fn round_up_align(side: u32) -> u32 {
    side.div_ceil(ATLAS_ALIGN).saturating_mul(ATLAS_ALIGN)
}

/// Texel data of [`TextureAtlas::add`]: rgba8 bytes, or rgba floats clamped to [0, 1] and
/// quantized to 8 bits (`round(x * 255)`, NaN as 0). Rows top first, 4 values per texel.
#[derive(Copy, Clone, Debug)]
pub enum TexelData<'a> {
    Rgba8(&'a [u8]),
    Float(&'a [f32]),
}

impl TexelData<'_> {
    fn len(&self) -> usize {
        match self {
            TexelData::Rgba8(d) => d.len(),
            TexelData::Float(d) => d.len(),
        }
    }

    fn to_rgba8(self) -> Vec<u8> {
        match self {
            TexelData::Rgba8(d) => d.to_vec(),
            TexelData::Float(d) => d.iter().map(|&x| quantize(x)).collect(),
        }
    }
}

/// `x` clamped to [0, 1] and quantized to 8 bits; NaN is 0.
pub fn quantize(x: f32) -> u8 {
    if x.is_nan() {
        0
    } else {
        (x.clamp(0.0, 1.0) * 255.0).round() as u8
    }
}

fn texture_error(code: &'static str, message: String) -> RaaError {
    RaaError::FrameGraph { code, message }
}

/// A packed rectangle of the atlas, in texels.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AtlasRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl AtlasRect {
    /// The storage-buffer rect: (x, y, width, height) / [`ATLAS_SIZE`].
    pub fn gpu_bytes(&self) -> [u8; TEXTURE_RECT_SIZE] {
        self.gpu_bytes_with(true)
    }

    /// Like [`AtlasRect::gpu_bytes`], with the width negated when `mips` is false.
    pub fn gpu_bytes_with(&self, mips: bool) -> [u8; TEXTURE_RECT_SIZE] {
        let s = ATLAS_SIZE as f32;
        let mut v = [self.x, self.y, self.width, self.height].map(|t| t as f32 / s);
        if !mips {
            v[2] = -v[2];
        }
        let mut out = [0; TEXTURE_RECT_SIZE];
        for (i, x) in v.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes());
        }
        out
    }
}

#[derive(Clone, Debug)]
struct Shelf {
    y: u32,
    height: u32,
    /// Next free x.
    x: u32,
    /// Live textures on the shelf.
    live: u32,
}

/// Shelf packing of `size` x `size` texels with ids `1..max_ids`.
#[derive(Clone, Debug)]
pub struct ShelfPacker {
    size: u32,
    shelves: Vec<Shelf>,
    rects: Vec<Option<AtlasRect>>,
}

impl ShelfPacker {
    pub fn new(size: u32, max_ids: u32) -> Self {
        ShelfPacker {
            size,
            shelves: Vec::new(),
            rects: vec![None; max_ids as usize],
        }
    }

    /// The rect of texture `id`, if it is live.
    pub fn rect(&self, id: u32) -> Option<AtlasRect> {
        self.rects.get(id as usize).copied().flatten()
    }

    /// Live textures.
    pub fn len(&self) -> usize {
        self.rects.iter().filter(|r| r.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Packs `width` x `height` and returns (id, rect): the lowest free id, on the
    /// shortest shelf it fits, else on a new shelf. Refuses `texture.invalid` (a zero or
    /// oversized side) and `texture.atlas_full` (no free id or no room); nothing changes.
    pub fn alloc(&mut self, width: u32, height: u32) -> Result<(u32, AtlasRect), RaaError> {
        if width == 0 || height == 0 || width > self.size || height > self.size {
            return Err(texture_error(
                "texture.invalid",
                format!(
                    "texture size {width} x {height} must be within 1..={} on each side",
                    self.size
                ),
            ));
        }
        let Some(id) = (1..self.rects.len()).find(|&i| self.rects[i].is_none()) else {
            return Err(texture_error(
                "texture.atlas_full",
                format!("all {} texture ids are in use", self.rects.len() - 1),
            ));
        };
        let size = self.size;
        let (rw, rh) = (round_up_align(width), round_up_align(height));
        let shelf = self
            .shelves
            .iter_mut()
            .filter(|s| s.height >= rh && s.x.saturating_add(rw) <= size)
            .min_by_key(|s| s.height);
        let shelf = match shelf {
            Some(s) => s,
            None => {
                let top = self.shelves.last().map_or(0, |s| s.y + s.height);
                if top.saturating_add(rh) > size {
                    return Err(texture_error(
                        "texture.atlas_full",
                        format!(
                            "no room for {width} x {height} in the {size} x {size} atlas \
                             ({} textures live)",
                            self.len()
                        ),
                    ));
                }
                self.shelves.push(Shelf {
                    y: top,
                    height: rh,
                    x: 0,
                    live: 0,
                });
                self.shelves.last_mut().expect("just pushed")
            }
        };
        let rect = AtlasRect {
            x: shelf.x,
            y: shelf.y,
            width,
            height,
        };
        shelf.x += rw;
        shelf.live += 1;
        self.rects[id] = Some(rect);
        Ok((id as u32, rect))
    }

    /// Frees texture `id`; returns whether it was live (an unknown id changes nothing). A
    /// shelf whose textures are all freed is reused from x 0; empty top shelves give
    /// their height back.
    pub fn free(&mut self, id: u32) -> bool {
        let Some(rect) = self.rects.get_mut(id as usize).and_then(Option::take) else {
            return false;
        };
        if let Some(s) = self.shelves.iter_mut().find(|s| s.y == rect.y) {
            s.live -= 1;
            if s.live == 0 {
                s.x = 0;
            }
        }
        while self.shelves.last().is_some_and(|s| s.live == 0) {
            self.shelves.pop();
        }
        true
    }
}

/// The atlas texture (or its 1 x 1 placeholder before the first add), the sampler, the
/// rect buffer and the packing.
pub(crate) struct TextureAtlas {
    packer: ShelfPacker,
    view: wgpu::TextureView,
    /// Whether `view` is the full atlas (false: the placeholder).
    allocated: bool,
    texture: wgpu::Texture,
    sampler: wgpu::Sampler,
    rects: wgpu::Buffer,
}

fn create_texture(gpu: &GpuContext, size: u32, levels: u32) -> wgpu::Texture {
    gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("raa.lit.texture_atlas"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: ATLAS_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

impl TextureAtlas {
    pub(crate) fn new(gpu: &GpuContext) -> Self {
        let texture = create_texture(gpu, 1, 1);
        let sampler = gpu.device().create_sampler(&wgpu::SamplerDescriptor {
            label: Some("raa.lit.texture_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        // Zero-initialised: every rect is (0, 0, 0, 0) until its texture is added.
        let rects = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa.lit.texture_rects"),
            size: u64::from(MAX_TEXTURES) * TEXTURE_RECT_SIZE as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        TextureAtlas {
            packer: ShelfPacker::new(ATLAS_SIZE, MAX_TEXTURES),
            view: texture.create_view(&Default::default()),
            allocated: false,
            texture,
            sampler,
            rects,
        }
    }

    /// The group-0 entries 3 (atlas view), 4 (sampler) and 5 (rects).
    pub(crate) fn bind_entries(&self) -> [wgpu::BindGroupEntry<'_>; 3] {
        [
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&self.view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(&self.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: self.rects.as_entire_binding(),
            },
        ]
    }

    pub(crate) fn packer(&self) -> &ShelfPacker {
        &self.packer
    }

    /// Validates, packs and uploads a `width` x `height` texture and returns its id. The
    /// boolean is true when the atlas texture was created by this call (group 0 must be
    /// bound again). Refusals (nothing changes): `texture.invalid` (a zero or oversized
    /// side, `data` length not width * height * 4), `texture.atlas_full`.
    /// Level 0 is always uploaded; with `options.mips` levels 1..`texture_levels(w, h)` are
    /// made from the previous level by `downsample_rgba8(.., options.srgb)` and written at
    /// `(x >> k, y >> k)`. The rect is written with `gpu_bytes_with(options.mips)`.
    pub(crate) fn add(
        &mut self,
        gpu: &GpuContext,
        width: u32,
        height: u32,
        data: TexelData<'_>,
        options: TextureOptions,
    ) -> Result<(u32, bool), RaaError> {
        let want = u64::from(width) * u64::from(height) * 4;
        if data.len() as u64 != want {
            return Err(texture_error(
                "texture.invalid",
                format!(
                    "data length {} must be width * height * 4 = {want} ({width} x {height} rgba)",
                    data.len()
                ),
            ));
        }
        let (id, rect) = self.packer.alloc(width, height)?;
        let created = !self.allocated;
        if created {
            self.texture = create_texture(gpu, ATLAS_SIZE, ATLAS_MIP_LEVELS);
            self.view = self.texture.create_view(&Default::default());
            self.allocated = true;
        }
        let queue = gpu.queue();
        let mut level = data.to_rgba8();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: rect.x,
                    y: rect.y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &level,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        if options.mips {
            let (mut pw, mut ph) = (width, height);
            for k in 1..texture_levels(width, height) {
                let (next, _, _) = crate::mipgen::downsample_rgba8(&level, pw, ph, options.srgb);
                let (lw, lh) = (level_size(width, k), level_size(height, k));
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.texture,
                        mip_level: k,
                        origin: wgpu::Origin3d {
                            x: rect.x >> k,
                            y: rect.y >> k,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &next,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(lw * 4),
                        rows_per_image: Some(lh),
                    },
                    wgpu::Extent3d {
                        width: lw,
                        height: lh,
                        depth_or_array_layers: 1,
                    },
                );
                (level, pw, ph) = (next, lw, lh);
            }
        }
        let offset = u64::from(id) * TEXTURE_RECT_SIZE as u64;
        queue.write_buffer(&self.rects, offset, &rect.gpu_bytes_with(options.mips));
        Ok((id, created))
    }

    /// Whether the full atlas texture exists (false: group 0 binds the 1 x 1 placeholder).
    /// A caller of [`TextureAtlas::add_gpu_image`] compares it before and after the call
    /// to know when group 0 must be bound again.
    #[allow(
        dead_code,
        reason = "read by the billboard bake owner in the next share"
    )]
    pub(crate) fn is_allocated(&self) -> bool {
        self.allocated
    }

    /// Packs an `extent[0]` x `extent[1]` image and copies it from mip zero of `source`
    /// (origin 0, 0) into the atlas on the GPU, without CPU texels or readback. Returns its
    /// id. Refusals (nothing changes): `texture.invalid` (`source` not a single-sample 2D
    /// rgba8unorm `COPY_SRC` texture, a zero extent side or an extent larger than
    /// `source`), then the [`ShelfPacker::alloc`] refusals. The atlas texture is created
    /// like the first [`TextureAtlas::add`] (see [`TextureAtlas::is_allocated`]). Only
    /// level 0 is written; the rect is written with `gpu_bytes_with(false)` (no mips). The
    /// copy is submitted on `gpu.queue()` before the call returns.
    pub(crate) fn add_gpu_image(
        &mut self,
        gpu: &GpuContext,
        source: &wgpu::Texture,
        extent: [u32; 2],
    ) -> Result<u32, RaaError> {
        let [width, height] = extent;
        if source.format() != ATLAS_FORMAT
            || !source.usage().contains(wgpu::TextureUsages::COPY_SRC)
            || source.dimension() != wgpu::TextureDimension::D2
            || source.sample_count() != 1
        {
            return Err(texture_error(
                "texture.invalid",
                format!(
                    "the source texture must be a single-sample 2D {ATLAS_FORMAT:?} texture \
                     with COPY_SRC (got {:?}, {:?}, {} samples)",
                    source.format(),
                    source.usage(),
                    source.sample_count()
                ),
            ));
        }
        if width == 0 || height == 0 || width > source.width() || height > source.height() {
            return Err(texture_error(
                "texture.invalid",
                format!(
                    "image size {width} x {height} must be at least 1 and within the {} x {} \
                     source texture",
                    source.width(),
                    source.height()
                ),
            ));
        }
        let (id, rect) = self.packer.alloc(width, height)?;
        if !self.allocated {
            self.texture = create_texture(gpu, ATLAS_SIZE, ATLAS_MIP_LEVELS);
            self.view = self.texture.create_view(&Default::default());
            self.allocated = true;
        }
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("raa.lit.texture_atlas.gpu_image"),
            });
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: source,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: rect.x,
                    y: rect.y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let offset = u64::from(id) * TEXTURE_RECT_SIZE as u64;
        gpu.queue()
            .write_buffer(&self.rects, offset, &rect.gpu_bytes_with(false));
        gpu.queue().submit([encoder.finish()]);
        Ok(id)
    }

    /// Removes texture `id` (its rect samples zero again); an unknown or already removed
    /// id does nothing. Returns whether it was live.
    pub(crate) fn remove(&mut self, gpu: &GpuContext, id: u32) -> bool {
        if !self.packer.free(id) {
            return false;
        }
        let offset = u64::from(id) * TEXTURE_RECT_SIZE as u64;
        gpu.queue()
            .write_buffer(&self.rects, offset, &[0; TEXTURE_RECT_SIZE]);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_clamps_and_rounds() {
        assert_eq!(
            [
                -1.0,
                0.0,
                0.5,
                1.0,
                2.0,
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY
            ]
            .map(quantize),
            [0, 0, 128, 255, 255, 0, 255, 0]
        );
        assert_eq!(quantize(0.2), 51);
    }

    #[test]
    fn shelves_pack_without_overlap_and_ids_start_at_1() {
        let mut p = ShelfPacker::new(64, 16);
        let (a, ra) = p.alloc(32, 16).unwrap();
        let (b, rb) = p.alloc(32, 8).unwrap();
        let (c, rc) = p.alloc(16, 16).unwrap();
        assert_eq!((a, b, c), (1, 2, 3));
        assert_eq!(
            ra,
            AtlasRect {
                x: 0,
                y: 0,
                width: 32,
                height: 16
            }
        );
        assert_eq!(
            rb,
            AtlasRect {
                x: 32,
                y: 0,
                width: 32,
                height: 8
            }
        );
        assert_eq!(
            rc,
            AtlasRect {
                x: 0,
                y: 16,
                width: 16,
                height: 16
            }
        );
        assert_eq!(p.rect(0), None);
        assert_eq!(p.len(), 3);
    }

    #[test]
    fn full_atlas_and_bad_sizes_refuse_without_change() {
        let mut p = ShelfPacker::new(32, 4);
        assert_eq!(p.alloc(0, 4).unwrap_err().code(), "texture.invalid");
        assert_eq!(p.alloc(33, 4).unwrap_err().code(), "texture.invalid");
        p.alloc(32, 32).unwrap();
        let full = p.alloc(1, 1).unwrap_err();
        assert_eq!(full.code(), "texture.atlas_full");
        assert!(full.message().contains("no room"), "{}", full.message());
        assert_eq!(p.len(), 1);
        // Out of ids: 3 usable ids (id 0 is reserved).
        let mut q = ShelfPacker::new(32, 4);
        for _ in 0..3 {
            q.alloc(1, 1).unwrap();
        }
        assert_eq!(q.alloc(1, 1).unwrap_err().code(), "texture.atlas_full");
    }

    #[test]
    fn free_is_idempotent_and_reclaims_empty_shelves() {
        let mut p = ShelfPacker::new(32, 8);
        let (a, _) = p.alloc(32, 32).unwrap();
        assert!(p.free(a));
        assert!(!p.free(a));
        assert!(!p.free(0) && !p.free(99));
        let (b, rb) = p.alloc(16, 16).unwrap();
        assert_eq!((b, rb.x, rb.y), (1, 0, 0), "ids and space are reused");
        let (c, rc) = p.alloc(16, 8).unwrap();
        assert_eq!((c, rc.x, rc.y), (2, 16, 0));
        assert!(p.free(b));
        // The shelf still holds c: its space comes back only when it empties.
        let (d, rd) = p.alloc(16, 16).unwrap();
        assert_eq!((d, rd.y), (1, 16));
        assert!(p.free(c) && p.free(d));
        assert!(p.is_empty());
        assert_eq!(p.alloc(32, 32).unwrap().1.y, 0);
    }

    #[test]
    fn rect_bytes_are_normalised() {
        let r = AtlasRect {
            x: 1024,
            y: 2048,
            width: 512,
            height: 4096,
        };
        let b = r.gpu_bytes();
        let f = |i: usize| f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!([f(0), f(1), f(2), f(3)], [0.25, 0.5, 0.125, 1.0]);
    }
}
