//! Native foliage resources (M7): the group-2 bind group layout and its immutable
//! generation, plus the CPU records the bake, reference and optimizer kernels share.
//!
//! [`FoliageParams`] is the validated description of one tree generation: the source-root
//! volume bounds, the native voxel resolution (16 or 32), the impostor or distilled mode
//! rank, translucency, cutoff, camera count, texture resolution and the generation number.
//! [`foliage_layout_entries`] declares the fragment-only group 2 of a foliage pipeline:
//! a 64-byte uniform, the Rgba8Unorm D3 volume, a filtering clamp sampler, the
//! Rgba8Unorm albedo/coverage D2 array, the Rgba16Float normal/depth D2 array and a
//! 128-record uniform view table. No 2D atlas stands in for the native volume.
//!
//! [`FoliageResources::new`] validates the parameters and allocates the generation-owned
//! textures, buffers and bind group. [`FoliageResources::initialize`] copies the packed
//! layer arrays and the view table, and uploads the x-fastest volume payload before the
//! generation is published. [`SourceTriangle`] / [`FoliageSource`] are the retained
//! source records of `m7-source`; [`MicroTriangle`], [`FoliageCompute`],
//! [`FoliageReferences`] and [`PackedFoliageLayers`] are the concrete compute and
//! reference records used by later kernels. None of these declarations reads another
//! module of the milestone.

use raa_core::{GpuContext, RaaError};

/// Bytes of one [`FoliageParams`] uniform record.
pub const FOLIAGE_PARAMS_SIZE: usize = 64;
/// Records of the group-2 uniform view table.
pub const FOLIAGE_VIEW_RECORDS: usize = 128;
/// Bytes of one view-table record: a column-major mat4x4 and near/far/zero/zero.
pub const FOLIAGE_VIEW_RECORD_SIZE: usize = 80;
/// Bytes of the whole group-2 uniform view table.
pub const FOLIAGE_VIEW_TABLE_SIZE: usize = FOLIAGE_VIEW_RECORDS * FOLIAGE_VIEW_RECORD_SIZE;

/// Impostor mode rank `spom`.
pub const FOLIAGE_MODE_SPOM: u32 = 0;
/// Impostor mode rank `pom`.
pub const FOLIAGE_MODE_POM: u32 = 1;
/// Impostor mode rank `offset`.
pub const FOLIAGE_MODE_OFFSET: u32 = 2;
/// Impostor mode rank `shell`.
pub const FOLIAGE_MODE_SHELL: u32 = 3;
/// Distilled mode rank `stochastic`.
pub const FOLIAGE_MODE_STOCHASTIC: u32 = 4;
/// Distilled mode rank `differentiable`.
pub const FOLIAGE_MODE_DIFFERENTIABLE: u32 = 5;

fn invalid(message: String) -> RaaError {
    RaaError::FrameGraph {
        code: "foliage.invalid",
        message,
    }
}

fn finite3(v: [f32; 3]) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn finite16(m: [f32; 16]) -> bool {
    m.iter().all(|x| x.is_finite())
}

/// The validated description of one native foliage generation.
///
/// `min` and `max` bound the tree in source-root metres; `resolution` is the native
/// volume extent (16 or 32); `mode` is the rank of [`FOLIAGE_MODE_SPOM`] to
/// [`FOLIAGE_MODE_DIFFERENTIABLE`]; `translucency` and `cutoff` lie in `0..=1`;
/// `view_count` lies in `1..=128`; `texture_resolution` is one of 64, 128, 256 or 512.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FoliageParams {
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub resolution: u32,
    pub mode: u32,
    pub translucency: f32,
    pub generation: u32,
    pub cutoff: f32,
    pub view_count: u32,
    pub texture_resolution: u32,
}

impl FoliageParams {
    /// Whether `mode` is one of the four impostor modes (`spom` to `shell`).
    pub fn is_impostor(&self) -> bool {
        self.mode <= FOLIAGE_MODE_SHELL
    }

    /// Refuses `foliage.invalid` for a non-finite or degenerate bound, an unsupported
    /// resolution or texture resolution, a mode above `differentiable`, an out-of-range
    /// translucency, cutoff or view count. The bounds must be strictly increasing on every
    /// axis.
    pub fn validate(&self) -> Result<(), RaaError> {
        if !finite3(self.min) || !finite3(self.max) {
            return Err(invalid("foliage bounds must be finite".into()));
        }
        for axis in 0..3 {
            if self.min[axis] >= self.max[axis] {
                return Err(invalid(format!(
                    "foliage bounds are degenerate on axis {axis}: {} >= {}",
                    self.min[axis], self.max[axis]
                )));
            }
        }
        if self.resolution != 16 && self.resolution != 32 {
            return Err(invalid(format!(
                "foliage resolution {} must be 16 or 32",
                self.resolution
            )));
        }
        if self.mode > FOLIAGE_MODE_DIFFERENTIABLE {
            return Err(invalid(format!(
                "foliage mode {} must be within 0..=5",
                self.mode
            )));
        }
        if !(0.0..=1.0).contains(&self.translucency) {
            return Err(invalid(format!(
                "foliage translucency {} must be within 0..=1",
                self.translucency
            )));
        }
        if !(0.0..=1.0).contains(&self.cutoff) {
            return Err(invalid(format!(
                "foliage cutoff {} must be within 0..=1",
                self.cutoff
            )));
        }
        if !(1..=128).contains(&self.view_count) {
            return Err(invalid(format!(
                "foliage view count {} must be within 1..=128",
                self.view_count
            )));
        }
        if !matches!(self.texture_resolution, 64 | 128 | 256 | 512) {
            return Err(invalid(format!(
                "foliage texture resolution {} must be 64, 128, 256 or 512",
                self.texture_resolution
            )));
        }
        Ok(())
    }

    /// The 64-byte group-2 params record: `min.xyz@0`, `resolution@12`, `max.xyz@16`,
    /// `mode@28`, `translucency@32`, `generation@36`, `cutoff@40`, `view_count@44`,
    /// `texture_resolution@48` and zero at 52, 56 and 60.
    pub fn to_bytes(&self) -> [u8; FOLIAGE_PARAMS_SIZE] {
        let mut out = [0u8; FOLIAGE_PARAMS_SIZE];
        let f = |b: &mut [u8; FOLIAGE_PARAMS_SIZE], o: usize, v: f32| {
            b[o..o + 4].copy_from_slice(&v.to_le_bytes());
        };
        let u = |b: &mut [u8; FOLIAGE_PARAMS_SIZE], o: usize, v: u32| {
            b[o..o + 4].copy_from_slice(&v.to_le_bytes());
        };
        for (i, v) in self.min.iter().enumerate() {
            f(&mut out, 4 * i, *v);
        }
        u(&mut out, 12, self.resolution);
        for (i, v) in self.max.iter().enumerate() {
            f(&mut out, 16 + 4 * i, *v);
        }
        u(&mut out, 28, self.mode);
        f(&mut out, 32, self.translucency);
        u(&mut out, 36, self.generation);
        f(&mut out, 40, self.cutoff);
        u(&mut out, 44, self.view_count);
        u(&mut out, 48, self.texture_resolution);
        out
    }

    /// The 80-byte view-table record of one camera: its column-major root-to-clip
    /// matrix at 0 and `near, far, 0, 0` at 64.
    pub fn view_record(matrix: [f32; 16], near: f32, far: f32) -> [u8; FOLIAGE_VIEW_RECORD_SIZE] {
        let mut out = [0u8; FOLIAGE_VIEW_RECORD_SIZE];
        for (i, v) in matrix.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
        }
        out[64..68].copy_from_slice(&near.to_le_bytes());
        out[68..72].copy_from_slice(&far.to_le_bytes());
        out
    }
}

/// The fragment-only group 2 of a foliage pipeline.
///
/// Handoff: 'Bind foliage resources in lit pipelines' (card #9133) calls
/// `foliage_layout_entries()` and attaches the returned entries as group 2 next to the
/// existing opaque group 1; the consumer retains the immutable resource generations in
/// its MaterialId-keyed associations, with mode and layout identity in its pipeline keys.
/// Today this declares binding 0 (64-byte uniform params),
/// binding 1 (filterable `texture_3d<f32>` Rgba8Unorm), binding 2 (filtering clamp
/// sampler), binding 3 (filterable Rgba8Unorm D2 array), binding 4 (filterable Rgba16Float
/// D2 array) and binding 5 (uniform view table, 128 records of 80 bytes), all visible to
/// the fragment stage only.
///
/// LIFECYCLE: preparation allocates generation-owned bake/compute resources; successful
/// GPU completion publishes one immutable FoliageResources generation; group detachment
/// precedes asset release, submitted work retains resources until completion, and Engine
/// disposal releases remaining generations.
pub fn foliage_layout_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    use wgpu::TextureSampleType as S;
    use wgpu::TextureViewDimension as D;
    let fragment = wgpu::ShaderStages::FRAGMENT;
    let filterable = S::Float { filterable: true };
    let uniform = |binding: u32, size: usize| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: fragment,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: std::num::NonZeroU64::new(size as u64),
        },
        count: None,
    };
    let texture = |binding: u32, dimension: D| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: fragment,
        ty: wgpu::BindingType::Texture {
            sample_type: filterable,
            view_dimension: dimension,
            multisampled: false,
        },
        count: None,
    };
    vec![
        uniform(0, FOLIAGE_PARAMS_SIZE),
        texture(1, D::D3),
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: fragment,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        texture(3, D::D2Array),
        texture(4, D::D2Array),
        uniform(5, FOLIAGE_VIEW_TABLE_SIZE),
    ]
}

fn texture_3d(gpu: &GpuContext, label: &str, resolution: u32) -> wgpu::Texture {
    gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: resolution,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn texture_array(
    gpu: &GpuContext,
    label: &str,
    params: &FoliageParams,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: params.texture_resolution,
            height: params.texture_resolution,
            depth_or_array_layers: params.view_count,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// One immutable, generation-owned foliage resource set: the validated parameters, the
/// native Rgba8Unorm volume, the neutral-or-packed albedo/coverage and normal/depth
/// arrays, the 128-record view table and the group-2 bind group.
///
/// A generation never changes after [`FoliageResources::initialize`]; the owner detaches
/// the bind group before releasing the textures.
pub struct FoliageResources {
    params: FoliageParams,
    volume: wgpu::Texture,
    albedo_coverage: wgpu::Texture,
    normal_depth: wgpu::Texture,
    views: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    initialized: bool,
}

impl FoliageResources {
    /// Handoff: 'Publish completed foliage assets in the renderer' (card #9135) calls
    /// `FoliageResources::new(gpu: &GpuContext, params: FoliageParams)` and publishes the
    /// returned generation to its MaterialId-keyed associations only after GPU completion,
    /// keeping mode and layout identity in its pipeline keys. Today this validates the
    /// parameters, allocates the generation-owned group-2 resources (native Rgba8Unorm D3
    /// volume, two D2 arrays with one mip, the 128-record view buffer and the filtering
    /// clamp sampler), writes the params record and builds the bind group. The arrays start as valid neutral allocations; impostor generations
    /// need [`FoliageResources::initialize`] with packed layers, distilled generations
    /// accept `None`.
    pub fn new(gpu: &GpuContext, params: FoliageParams) -> Result<Self, RaaError> {
        params.validate()?;
        gpu.ensure_live()?;
        let device = gpu.device();
        let volume = texture_3d(gpu, "raa.lit.foliage.volume", params.resolution);
        let albedo_coverage = texture_array(
            gpu,
            "raa.lit.foliage.albedo_coverage",
            &params,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let normal_depth = texture_array(
            gpu,
            "raa.lit.foliage.normal_depth",
            &params,
            wgpu::TextureFormat::Rgba16Float,
        );
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa.lit.foliage.params"),
            size: FOLIAGE_PARAMS_SIZE as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let views = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa.lit.foliage.views"),
            size: FOLIAGE_VIEW_TABLE_SIZE as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("raa.lit.foliage.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        gpu.queue()
            .write_buffer(&params_buffer, 0, &params.to_bytes());
        let entries = foliage_layout_entries();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa.lit.foliage.group2"),
            entries: &entries,
        });
        let volume_view = volume.create_view(&Default::default());
        let albedo_view = albedo_coverage.create_view(&Default::default());
        let normal_view = normal_depth.create_view(&Default::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa.lit.foliage.group2"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&volume_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: views.as_entire_binding(),
                },
            ],
        });
        Ok(FoliageResources {
            params,
            volume,
            albedo_coverage,
            normal_depth,
            views,
            bind_group,
            initialized: false,
        })
    }

    /// The group-2 bind group of this generation.
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// The native Rgba8Unorm volume texture (`resolution` cubed, one mip).
    pub fn volume(&self) -> &wgpu::Texture {
        &self.volume
    }

    /// The validated parameters this generation was allocated from.
    pub fn params(&self) -> FoliageParams {
        self.params
    }

    fn validate_layers(&self, layers: &PackedFoliageLayers) -> Result<(), RaaError> {
        let p = &self.params;
        for (name, texture, format) in [
            (
                "albedo_coverage",
                &layers.albedo_coverage,
                wgpu::TextureFormat::Rgba8Unorm,
            ),
            (
                "normal_depth",
                &layers.normal_depth,
                wgpu::TextureFormat::Rgba16Float,
            ),
        ] {
            if texture.format() != format {
                return Err(invalid(format!(
                    "foliage {name} layers must use {format:?}"
                )));
            }
            if texture.dimension() != wgpu::TextureDimension::D2
                || texture.width() != p.texture_resolution
                || texture.height() != p.texture_resolution
                || texture.depth_or_array_layers() != p.view_count
                || texture.mip_level_count() != 1
            {
                return Err(invalid(format!(
                    "foliage {name} layers must be {}x{} with {} layer(s) and one mip",
                    p.texture_resolution, p.texture_resolution, p.view_count
                )));
            }
        }
        if layers.views.len() != p.view_count as usize {
            return Err(invalid(format!(
                "foliage view table has {} record(s), expected {}",
                layers.views.len(),
                p.view_count
            )));
        }
        for (matrix, near_far) in &layers.views {
            if !finite16(*matrix) || !near_far.iter().all(|v| v.is_finite()) {
                return Err(invalid(
                    "foliage view matrices and near/far values must be finite".into(),
                ));
            }
        }
        Ok(())
    }

    /// Copies the packed layer arrays into bindings 3 and 4, uploads the ordered view table
    /// to binding 5 and writes the x-fastest native volume payload to binding 1 before
    /// returning the submission index of the copy. Every payload is validated first: the
    /// initialization may run exactly once, impostor modes require `Some(layers)`,
    /// distilled modes accept `None` and keep the neutral array allocations, and `volume`
    /// must hold `4 * resolution^3` bytes. A repeated call returns `foliage.invalid`.
    pub fn initialize(
        &mut self,
        gpu: &GpuContext,
        layers: Option<&PackedFoliageLayers>,
        volume: &[u8],
    ) -> Result<wgpu::SubmissionIndex, RaaError> {
        if self.initialized {
            return Err(invalid("foliage resources are already initialized".into()));
        }
        if self.params.is_impostor() && layers.is_none() {
            return Err(invalid(
                "impostor foliage initialization needs packed layers".into(),
            ));
        }
        if let Some(layers) = layers {
            self.validate_layers(layers)?;
        }
        let resolution = self.params.resolution as usize;
        let want = 4 * resolution * resolution * resolution;
        if volume.len() != want {
            return Err(invalid(format!(
                "foliage volume payload has {} bytes, expected {want}",
                volume.len()
            )));
        }
        gpu.ensure_live()?;
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("raa.lit.foliage.initialize"),
            });
        if let Some(layers) = layers {
            let extent = wgpu::Extent3d {
                width: self.params.texture_resolution,
                height: self.params.texture_resolution,
                depth_or_array_layers: self.params.view_count,
            };
            encoder.copy_texture_to_texture(
                layers.albedo_coverage.as_image_copy(),
                self.albedo_coverage.as_image_copy(),
                extent,
            );
            encoder.copy_texture_to_texture(
                layers.normal_depth.as_image_copy(),
                self.normal_depth.as_image_copy(),
                extent,
            );
            let mut table = Vec::with_capacity(layers.views.len() * FOLIAGE_VIEW_RECORD_SIZE);
            for (matrix, near_far) in &layers.views {
                let record = FoliageParams::view_record(*matrix, near_far[0], near_far[1]);
                table.extend_from_slice(&record);
            }
            gpu.queue().write_buffer(&self.views, 0, &table);
        }
        let resolution = self.params.resolution;
        gpu.queue().write_texture(
            self.volume.as_image_copy(),
            volume,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * resolution),
                rows_per_image: Some(resolution),
            },
            wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: resolution,
            },
        );
        let index = gpu.queue().submit([encoder.finish()]);
        self.initialized = true;
        Ok(index)
    }
}

/// One retained source triangle in source-root coordinates (`m7-source`).
///
/// Positions and normals are source-root values, `uv` preserves the source UVs,
/// `material` is the existing source surface material, `source_tag` is the submitted
/// instance tag and `foliage` is the procedural mesh-source classification (never a
/// colour-derived value).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SourceTriangle {
    pub positions: [[f32; 3]; 3],
    pub normals: [[f32; 3]; 3],
    pub uv: [[f32; 2]; 3],
    pub material: u32,
    pub source_tag: u32,
    pub foliage: bool,
}

/// The retained source geometry of one tree (`m7-source`).
///
/// `triangles` are in input draw order followed by ascending source triangle index;
/// `min` and `max` bound them in source-root metres (zero for an empty list),
/// `root_world` is the source-root object frame and `generation` is the submitted
/// generation number.
#[derive(Clone, Debug, PartialEq)]
pub struct FoliageSource {
    pub triangles: Vec<SourceTriangle>,
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub root_world: raa_math::Mat4,
    pub generation: u32,
}

/// One optimizer micro-triangle in source-root metres (`m7-compute`).
///
/// `normal` is independent of the positions and unit length; `opacity` starts at 1.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MicroTriangle {
    pub p0: [f32; 3],
    pub p1: [f32; 3],
    pub p2: [f32; 3],
    pub normal: [f32; 3],
    pub opacity: f32,
}

/// The rendered native reference views of one generation (`m7-reference-views`).
///
/// Every texture is a D2 array with exactly `views` layers; each tuple is the
/// column-major root-to-clip matrix followed by near/far metres.
pub struct FoliageReferences {
    pub normal_coverage: wgpu::Texture,
    pub depth: wgpu::Texture,
    pub albedo_coverage: wgpu::Texture,
    pub views: Vec<([f32; 16], [f32; 2])>,
    pub submission: wgpu::SubmissionIndex,
}

impl FoliageReferences {
    /// The signed source-root normal XYZ and normalized depth W reference array.
    pub fn normal_coverage(&self) -> &wgpu::Texture {
        &self.normal_coverage
    }

    /// The standard-Z depth reference array.
    pub fn depth(&self) -> &wgpu::Texture {
        &self.depth
    }

    /// The unlit linear RGB albedo and coverage A reference array.
    pub fn albedo_coverage(&self) -> &wgpu::Texture {
        &self.albedo_coverage
    }
}

/// The reference views converted to source-root packed layers (`m7-reference-views`).
///
/// `albedo_coverage` is an Rgba8Unorm D2 array, `normal_depth` an Rgba16Float D2 array;
/// both have `view_count` layers of `texture_resolution` squared texels and `views` holds
/// the ordered camera table.
pub struct PackedFoliageLayers {
    pub albedo_coverage: wgpu::Texture,
    pub normal_depth: wgpu::Texture,
    pub views: Vec<([f32; 16], [f32; 2])>,
    pub submission: wgpu::SubmissionIndex,
}

/// The GPU buffers and reference views of one differentiable optimizer generation
/// (`m7-compute`). The optimizer card allocates and finalizes it; the kernel modules of
/// the same crate read these fields.
#[allow(
    dead_code,
    reason = "allocated and consumed by the differentiable optimizer card"
)]
pub struct FoliageCompute {
    pub(crate) triangles: wgpu::Buffer,
    pub(crate) gradients: wgpu::Buffer,
    pub(crate) bins: wgpu::Buffer,
    pub(crate) pixels: wgpu::Buffer,
    pub(crate) references: FoliageReferences,
    pub(crate) params: wgpu::Buffer,
    pub(crate) count: u32,
    pub(crate) views: u32,
    pub(crate) resolution: u32,
    pub(crate) generation: u32,
}
