//! Implementations the lit handle needs to draw M7 foliage (contract `m7-lit-binding`): the
//! mode-and-layout keyed pipeline cache, the complete-layout accounting of groups 0, 1 and 2
//! and the fragment-only group-2 identity. [`crate::handle::LitHandle`] owns the additive
//! entry points `set_foliage`, `foliage_mode`, `pipeline_for_foliage` and `bind_foliage`;
//! this module holds their data structures so the handle stays small.
//!
//! Upstream anchors: `foliage_layout_entries()` (`crates/raa-lit/src/foliage_resources.rs:196`)
//! declares the exact fragment-only group 2, `FoliageResources::bind_group`
//! (`crates/raa-lit/src/foliage_resources.rs:392`) is the immutable generation group bound per
//! draw, and `Composer::compose_foliage` (`crates/raa-shader/src/composer.rs:179`) composes
//! the prepass and forward modules once `mode` is known.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use raa_core::RaaError;
use raa_shader::VariantKey;

use crate::handle::{PipelineSpec, frame_layout_entries};

/// The WebGPU fragment sampled-texture limit of the composed foliage layout.
pub const MAX_FOLIAGE_FRAGMENT_TEXTURES: u32 = 16;
/// The WebGPU fragment storage-buffer limit of the composed foliage layout.
pub const MAX_FOLIAGE_FRAGMENT_STORAGE_BUFFERS: u32 = 8;
/// Fragment sampled textures of the S44 opaque group 1 (`opaque_layout_entries` without the
/// opaque colour and depth 9/10): `crates/raa-pipeline/src/lighting.rs:607`.
pub const OPAQUE_GROUP1_FRAGMENT_TEXTURES: u32 = 10;
/// Fragment storage buffers of the S44 opaque group 1 (none: it holds samplers, textures and
/// uniforms only): `crates/raa-pipeline/src/lighting.rs:607`.
pub const OPAQUE_GROUP1_FRAGMENT_STORAGE_BUFFERS: u32 = 0;

fn invalid(message: impl Into<String>) -> RaaError {
    RaaError::FrameGraph {
        code: "foliage.invalid",
        message: message.into(),
    }
}

/// The fragment-visible draw-count of a binding set: sampled textures and storage buffers.
/// Samplers, uniforms and entries whose visibility excludes `FRAGMENT` are not counted.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct FoliageLayoutCounts {
    pub textures: u32,
    pub storage_buffers: u32,
}

impl FoliageLayoutCounts {
    /// Counts the fragment-visible sampled textures and storage buffers of `entries`.
    pub fn of(entries: &[wgpu::BindGroupLayoutEntry]) -> Self {
        let mut counts = Self::default();
        for entry in entries {
            if !entry.visibility.contains(wgpu::ShaderStages::FRAGMENT) {
                continue;
            }
            match &entry.ty {
                wgpu::BindingType::Texture { .. } => counts.textures += 1,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { .. },
                    ..
                } => counts.storage_buffers += 1,
                _ => {}
            }
        }
        counts
    }

    /// Adds `other` into `self`.
    pub fn add(&mut self, other: Self) {
        self.textures += other.textures;
        self.storage_buffers += other.storage_buffers;
    }

    /// The complete groups 0, 1 and 2 of a foliage pipeline: the S10 frame layout
    /// (`frame_layout_entries`), the S44 opaque group 1 and the foliage group 2
    /// (`foliage_layout_entries`).
    pub fn complete() -> Self {
        let mut counts = Self::of(&frame_layout_entries());
        counts.add(Self::of(&crate::foliage_resources::foliage_layout_entries()));
        counts.textures += OPAQUE_GROUP1_FRAGMENT_TEXTURES;
        counts.storage_buffers += OPAQUE_GROUP1_FRAGMENT_STORAGE_BUFFERS;
        counts
    }

    /// Refuses `foliage.invalid` when a complete composed layout exceeds the fragment limits.
    pub fn validate(self) -> Result<(), RaaError> {
        if self.textures > MAX_FOLIAGE_FRAGMENT_TEXTURES {
            return Err(invalid(format!(
                "foliage layout uses {} fragment sampled textures (limit {MAX_FOLIAGE_FRAGMENT_TEXTURES})",
                self.textures
            )));
        }
        if self.storage_buffers > MAX_FOLIAGE_FRAGMENT_STORAGE_BUFFERS {
            return Err(invalid(format!(
                "foliage layout uses {} fragment storage buffers (limit {MAX_FOLIAGE_FRAGMENT_STORAGE_BUFFERS})",
                self.storage_buffers
            )));
        }
        Ok(())
    }
}

/// Checks the complete groups 0+1+2 before a foliage pipeline is allocated.
pub fn validate_foliage_layouts() -> Result<FoliageLayoutCounts, RaaError> {
    let counts = FoliageLayoutCounts::complete();
    counts.validate()?;
    Ok(counts)
}

/// The identity of a concrete group layout: the hash of its actual entries. Two handles
/// created from equal entries on one device share the pooled layout, so the identity names
/// the layout the pipeline and the generation bind group use.
pub fn layout_identity(entries: &[wgpu::BindGroupLayoutEntry]) -> u64 {
    let mut hasher = DefaultHasher::new();
    entries.hash(&mut hasher);
    hasher.finish()
}

/// One cache slot of a foliage pipeline: the mode rank, the actual group-2 layout identity,
/// the shader variant and the fixed-function spec hash.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FoliagePipelineKey {
    pub mode: u32,
    pub layout: u64,
    pub variant: VariantKey,
    pub spec: u64,
}

impl FoliagePipelineKey {
    /// The key of `variant` and `spec_hash` for `mode` and the `entries` of the actual
    /// group-2 layout.
    pub fn new(
        mode: u32,
        entries: &[wgpu::BindGroupLayoutEntry],
        variant: VariantKey,
        spec_hash: u64,
    ) -> Self {
        Self::from_parts(mode, layout_identity(entries), variant, spec_hash)
    }

    /// The key from an already computed layout identity.
    pub fn from_parts(mode: u32, layout: u64, variant: VariantKey, spec_hash: u64) -> Self {
        FoliagePipelineKey {
            mode,
            layout,
            variant,
            spec: spec_hash,
        }
    }
}

/// Pipelines by [`FoliagePipelineKey`], with the same slot-reset and exact-spec rules as the
/// S10 `PipelineCache`: a lookup with another slots hash empties the cache, and two specs
/// with one hash never share a value.
#[derive(Debug)]
pub struct FoliagePipelineCache<T> {
    slots: u64,
    entries: HashMap<FoliagePipelineKey, (PipelineSpec, T)>,
}

impl<T> Default for FoliagePipelineCache<T> {
    fn default() -> Self {
        FoliagePipelineCache {
            slots: 0,
            entries: HashMap::new(),
        }
    }
}

impl<T: Clone> FoliagePipelineCache<T> {
    /// An empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Drops every value (the slot providers changed).
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// The value cached for `key` at `slots`, else `build()` stored and returned. A failed
    /// build stores nothing.
    pub fn get_or_build<E>(
        &mut self,
        key: FoliagePipelineKey,
        spec: &PipelineSpec,
        slots: u64,
        build: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        if slots != self.slots {
            self.entries.clear();
            self.slots = slots;
        }
        if let Some((stored, value)) = self.entries.get(&key)
            && stored == spec
        {
            return Ok(value.clone());
        }
        let value = build()?;
        self.entries.insert(key, (spec.clone(), value.clone()));
        Ok(value)
    }
}
