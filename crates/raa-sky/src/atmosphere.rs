//! Physical atmosphere data and the private S10 `PhysicalSkyGpu` uniform.
//!
//! The constants, exponential density profile, phase functions, ray/sphere
//! intersection and LUT parameterizations are adapted from Solstice revision
//! `7e465b599c13da165948ebaa44df824c36c3ae07`:
//! `crates/solstice-render/shaders/common/atmosphere.wgsl:4-102`. The viewer
//! metres-to-kilometres conversion is the one at
//! `crates/solstice-render/src/sky.rs:104-108`. Adapted source retains the
//! Solstice MIT notice and `Copyright (c) 2026 SamG-Coder`.
//!
//! All atmospheric distances are kilometres. `PhysicalSkyGpu` is the sixteen-lane
//! S10 group-2 binding-0 record; [`PhysicalSkyGpu::to_bytes`] is its only
//! serialization boundary and always writes explicit little-endian words.
//!
//! [`AtmosphereResources`] owns the Rgba16Float transmittance 256x64,
//! multiple-scattering 32x32 and per-view sky-view 256x144 LUTs
//! (`crates/solstice-render/src/sky.rs:8-11,35-95`), the immutable sampler and
//! group-2 layouts, and the compute pipelines that fill them. Static tables are
//! invalidated only by atmosphere parameters; a view's sky view follows its own
//! height and light snapshot. Prepared resources stay pending until the LUT work
//! is recorded and submitted; allocation alone claims no readiness.
/*
MIT License

Copyright (c) 2026 SamG-Coder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

use std::num::NonZeroU64;

use raa_core::{GpuContext, RaaError};

use crate::shaders::{ATMOSPHERE_LUT_ENTRIES, atmosphere_luts_module_wgsl};

/// Ground radius at sea level, in kilometres.
pub(crate) const GROUND_RADIUS_KM: f32 = 6360.0;
/// Top of the atmosphere, in kilometres.
pub(crate) const TOP_RADIUS_KM: f32 = 6460.0;

/// Rayleigh scattering coefficients, per kilometre.
pub(crate) const RAYLEIGH_SCATTERING: [f32; 3] = [5.802e-3, 13.558e-3, 33.1e-3];
/// Rayleigh density scale height, in kilometres.
pub(crate) const RAYLEIGH_SCALE_HEIGHT_KM: f32 = 8.0;

/// Mie scattering coefficient, per kilometre (before `mieMultiplier`).
pub(crate) const MIE_SCATTERING: f32 = 14.0e-3;
/// Mie absorption coefficient, per kilometre.
pub(crate) const MIE_ABSORPTION: f32 = 4.40e-3;
/// Mie density scale height, in kilometres.
pub(crate) const MIE_SCALE_HEIGHT_KM: f32 = 1.6;
/// Cornette-Shanks asymmetry parameter.
pub(crate) const MIE_G: f32 = 0.8;

/// Ozone absorption coefficients, per kilometre.
pub(crate) const OZONE_ABSORPTION: [f32; 3] = [0.650e-3, 1.881e-3, 0.085e-3];

/// The one metres-to-kilometres viewer conversion of the sky feature.
///
/// Solstice `sky.rs:104-108`: a viewer height at or below one metre counts as
/// one metre; the result is the distance from the planet centre in kilometres.
pub(crate) fn atmospheric_radius_km(height_m: f32) -> f32 {
    GROUND_RADIUS_KM + height_m.max(1.0) * 0.001
}

/// The S10 private physical-sky uniform (group 2, binding 0, 256 bytes).
///
/// Exactly sixteen 16-byte lanes. `viewer` is the world position in metres with
/// the already-converted viewer radius in kilometres in `w`; directions are
/// unit vectors toward the body/light. See the architecture S10 table for every
/// component meaning.
#[repr(C)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PhysicalSkyGpu {
    /// @0: viewer world xyz metres, viewer radius kilometres.
    pub(crate) viewer: [f32; 4],
    /// @16: direction toward the sun, moon lit fraction.
    pub(crate) sun: [f32; 4],
    /// @32: direction toward the moon, moon radiance.
    pub(crate) moon: [f32; 4],
    /// @48: celestial pole xyz, star intensity.
    pub(crate) pole: [f32; 4],
    /// @64: equinox xyz, night glow.
    pub(crate) equinox: [f32; 4],
    /// @80: direction toward the dominant light, raw intensity.
    pub(crate) dominant: [f32; 4],
    /// @96: daylight linear rgb, baseline intensity.
    pub(crate) daylight: [f32; 4],
    /// @112: sun radius, sun multiplier, moon radius, pixel angle.
    pub(crate) discs: [f32; 4],
    /// @128: ground albedo rgb, Mie multiplier.
    pub(crate) ground_albedo: [f32; 4],
    /// @144: Rayleigh coefficients rgb, scale height kilometres.
    pub(crate) rayleigh: [f32; 4],
    /// @160: Mie scattering, absorption, scale height kilometres, g.
    pub(crate) mie: [f32; 4],
    /// @176: ozone coefficients rgb, rain.
    pub(crate) ozone: [f32; 4],
    /// @192: ground radius kilometres, top radius kilometres, animation seconds, 0.
    pub(crate) planet: [f32; 4],
    /// @208: physical, cloud screen valid, cloud panorama valid, 0.
    pub(crate) flags: [u32; 4],
    /// @224: reserved words, serialized as zeros.
    pub(crate) reserved: [[f32; 4]; 2],
}

impl Default for PhysicalSkyGpu {
    /// Source defaults: physical atmosphere on, white daylight, sea-level viewer,
    /// clear-day ground albedo and the Solstice atmosphere constants.
    fn default() -> Self {
        Self {
            viewer: [0.0, 0.0, 0.0, atmospheric_radius_km(1.0)],
            sun: [0.0, 1.0, 0.0, 1.0],
            moon: [0.0, -1.0, 0.0, 0.0],
            pole: [0.0, 1.0, 0.0, 1.0],
            equinox: [1.0, 0.0, 0.0, 1.2e-4],
            dominant: [0.0, 1.0, 0.0, 1.0],
            daylight: [1.0, 1.0, 1.0, 1.0],
            discs: [0.003, 8.0, 0.0075, 0.0],
            ground_albedo: [0.3, 0.3, 0.3, 1.0],
            rayleigh: [
                RAYLEIGH_SCATTERING[0],
                RAYLEIGH_SCATTERING[1],
                RAYLEIGH_SCATTERING[2],
                RAYLEIGH_SCALE_HEIGHT_KM,
            ],
            mie: [MIE_SCATTERING, MIE_ABSORPTION, MIE_SCALE_HEIGHT_KM, MIE_G],
            ozone: [
                OZONE_ABSORPTION[0],
                OZONE_ABSORPTION[1],
                OZONE_ABSORPTION[2],
                0.0,
            ],
            planet: [GROUND_RADIUS_KM, TOP_RADIUS_KM, 0.0, 0.0],
            flags: [1, 0, 0, 0],
            reserved: [[0.0; 4]; 2],
        }
    }
}

impl PhysicalSkyGpu {
    /// Size in bytes of the S10 uniform.
    pub(crate) const SIZE: usize = 256;

    /// Little-endian S10 bytes: thirteen `vec4<f32>` lanes at @0..208, `uvec4<u32>`
    /// flags at @208 and the two reserved zero lanes at @224/@240.
    pub(crate) fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        let lanes = [
            self.viewer,
            self.sun,
            self.moon,
            self.pole,
            self.equinox,
            self.dominant,
            self.daylight,
            self.discs,
            self.ground_albedo,
            self.rayleigh,
            self.mie,
            self.ozone,
            self.planet,
        ];
        for (lane, values) in lanes.into_iter().enumerate() {
            put_f32x4(&mut out, lane * 16, values);
        }
        put_u32x4(&mut out, 208, self.flags);
        // S10 lanes 14 and 15 are reserved padding: the declared record keeps them,
        // but the ABI always uploads zeros so placeholder content cannot leak.
        debug_assert_eq!(self.reserved, [[0.0; 4]; 2]);
        out
    }
}

fn put_f32x4(out: &mut [u8; PhysicalSkyGpu::SIZE], offset: usize, values: [f32; 4]) {
    for (i, value) in values.into_iter().enumerate() {
        let at = offset + i * 4;
        out[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }
}

fn put_u32x4(out: &mut [u8; PhysicalSkyGpu::SIZE], offset: usize, values: [u32; 4]) {
    for (i, value) in values.into_iter().enumerate() {
        let at = offset + i * 4;
        out[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }
}

/// Texel format of the three atmosphere LUTs (`sky.rs:8`).
const LUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Transmittance LUT dimensions (`sky.rs:9`).
const TRANSMITTANCE_SIZE: (u32, u32) = (256, 64);
/// Multiple-scattering LUT dimensions (`sky.rs:10`).
const MULTI_SCATTER_SIZE: (u32, u32) = (32, 32);
/// Per-view sky-view LUT dimensions (`sky.rs:11`).
const SKY_VIEW_SIZE: (u32, u32) = (256, 144);
/// Prepared atmosphere views, matching the engine's four view slots.
const MAX_VIEWS: usize = 4;

fn invalid(message: impl Into<String>) -> RaaError {
    RaaError::FrameGraph {
        code: "sky.invalid",
        message: message.into(),
    }
}

fn validate_view(view: u32) -> Result<usize, RaaError> {
    if (view as usize) < MAX_VIEWS {
        Ok(view as usize)
    } else {
        Err(invalid(format!(
            "atmosphere view {view} is outside 0..{MAX_VIEWS}"
        )))
    }
}

fn validate_view_size(size: (u32, u32)) -> Result<(), RaaError> {
    if size.0 == 0 || size.1 == 0 {
        return Err(invalid(format!(
            "atmosphere view size must be nonzero, got {}x{}",
            size.0, size.1
        )));
    }
    Ok(())
}

/// Refuses a non-finite f32 lane before any view slot is created or updated.
fn validate_parameters(parameters: &PhysicalSkyGpu) -> Result<(), RaaError> {
    let lanes: [(&str, &[f32; 4]); 15] = [
        ("viewer", &parameters.viewer),
        ("sun", &parameters.sun),
        ("moon", &parameters.moon),
        ("pole", &parameters.pole),
        ("equinox", &parameters.equinox),
        ("dominant", &parameters.dominant),
        ("daylight", &parameters.daylight),
        ("discs", &parameters.discs),
        ("ground_albedo", &parameters.ground_albedo),
        ("rayleigh", &parameters.rayleigh),
        ("mie", &parameters.mie),
        ("ozone", &parameters.ozone),
        ("planet", &parameters.planet),
        ("reserved0", &parameters.reserved[0]),
        ("reserved1", &parameters.reserved[1]),
    ];
    for (name, lane) in lanes {
        for (index, value) in lane.iter().enumerate() {
            if !value.is_finite() {
                return Err(invalid(format!(
                    "atmosphere `{name}[{index}]` must be finite, got {value}"
                )));
            }
        }
    }
    Ok(())
}

/// The lanes that feed the static transmittance and multiple-scattering tables.
#[derive(Clone, Copy, Debug, PartialEq)]
struct StaticParams {
    ground_albedo: [f32; 4],
    rayleigh: [f32; 4],
    mie: [f32; 4],
    ozone: [f32; 4],
    ground_radius_km: f32,
    top_radius_km: f32,
}

impl From<&PhysicalSkyGpu> for StaticParams {
    fn from(parameters: &PhysicalSkyGpu) -> Self {
        StaticParams {
            ground_albedo: parameters.ground_albedo,
            rayleigh: parameters.rayleigh,
            mie: parameters.mie,
            ozone: parameters.ozone,
            ground_radius_km: parameters.planet[0],
            top_radius_km: parameters.planet[1],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Prepare {
    /// The view had no resources; its sky view and uniform were allocated.
    Created,
    /// The view existed and its size or parameters changed.
    Updated,
    /// The view existed and nothing that affects its resources changed.
    Unchanged,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Prepared {
    outcome: Prepare,
    static_changed: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct ViewSlot {
    parameters: PhysicalSkyGpu,
    size: (u32, u32),
    sky_view_pending: bool,
}

/// Device-free view lifecycle: validation, allocation decisions and pending table work.
#[derive(Clone, Debug)]
struct Lifecycle {
    static_params: Option<StaticParams>,
    static_tables_pending: bool,
    views: [Option<ViewSlot>; MAX_VIEWS],
}

impl Lifecycle {
    fn new() -> Self {
        Lifecycle {
            static_params: None,
            static_tables_pending: true,
            views: std::array::from_fn(|_| None),
        }
    }

    fn prepare(
        &mut self,
        view: u32,
        size: (u32, u32),
        parameters: &PhysicalSkyGpu,
    ) -> Result<Prepared, RaaError> {
        let index = validate_view(view)?;
        validate_view_size(size)?;
        validate_parameters(parameters)?;
        let static_changed = self.static_params != Some(StaticParams::from(parameters));
        if static_changed {
            self.static_params = Some(StaticParams::from(parameters));
            for slot in self.views.iter_mut().flatten() {
                slot.sky_view_pending = true;
            }
        }
        self.static_tables_pending |= static_changed;
        match &mut self.views[index] {
            Some(slot) => {
                let changed = slot.parameters != *parameters || slot.size != size;
                slot.sky_view_pending |= changed;
                if changed {
                    slot.parameters = parameters.clone();
                    slot.size = size;
                }
                Ok(Prepared {
                    outcome: if changed {
                        Prepare::Updated
                    } else {
                        Prepare::Unchanged
                    },
                    static_changed,
                })
            }
            None => {
                self.views[index] = Some(ViewSlot {
                    parameters: parameters.clone(),
                    size,
                    sky_view_pending: true,
                });
                Ok(Prepared {
                    outcome: Prepare::Created,
                    static_changed,
                })
            }
        }
    }

    fn remove(&mut self, view: u32) {
        if let Some(slot) = self.views.get_mut(view as usize) {
            *slot = None;
        }
    }
}

struct ViewGpu {
    sky_view: wgpu::TextureView,
    uniform: wgpu::Buffer,
    /// Bound by the LUT record step of card 7429, which reads `_bind_group`.
    _bind_group: wgpu::BindGroup,
}

/// The shared atmosphere LUTs and the per-view resources that fill them.
///
/// The static transmittance and multiple-scattering targets are created once; the
/// static uniform is rewritten only when an atmosphere parameter lane changes,
/// which also marks every prepared view's sky view pending. A view's 256x144
/// target, uniform and bind group are allocated on its first prepare and released
/// by [`AtmosphereResources::remove_view`]. Preparing an unchanged view performs
/// no GPU allocation or upload.
pub(crate) struct AtmosphereResources {
    sampler: wgpu::Sampler,
    layout: wgpu::BindGroupLayout,
    /// Dispatched by the LUT record step of card 7429, which reads `_pipelines`.
    _pipelines: [wgpu::ComputePipeline; 3],
    transmittance: wgpu::TextureView,
    multiple_scattering: wgpu::TextureView,
    static_uniform: wgpu::Buffer,
    /// The two static-table bind groups recorded by card 7429.
    _static_groups: [wgpu::BindGroup; 2],
    views: [Option<ViewGpu>; MAX_VIEWS],
    lifecycle: Lifecycle,
}

impl AtmosphereResources {
    /// Creates the two static LUT targets, the placeholder dummy, the sampler, the
    /// group-2 layout and the three LUT compute pipelines.
    pub(crate) fn new(gpu: &GpuContext) -> Result<Self, RaaError> {
        gpu.ensure_live()?;
        let device = gpu.device();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa-sky.atmosphere"),
            entries: &layout_entries(),
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("raa-sky.atmosphere"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let transmittance = texture_view(
            gpu,
            lut_descriptor("raa-sky.atmosphere.transmittance", TRANSMITTANCE_SIZE),
        );
        let multiple_scattering = texture_view(
            gpu,
            lut_descriptor("raa-sky.atmosphere.multiple_scattering", MULTI_SCATTER_SIZE),
        );
        // The first static build binds this 1x1 dummy for the table it does not read.
        let placeholder = texture_view(gpu, dummy_descriptor("raa-sky.atmosphere.placeholder"));
        let static_uniform = uniform_buffer(gpu, "raa-sky.atmosphere.static");
        gpu.queue()
            .write_buffer(&static_uniform, 0, &PhysicalSkyGpu::default().to_bytes());
        let pipelines = lut_pipelines(gpu, &layout);
        let static_groups = [
            lut_bind_group(
                gpu,
                &layout,
                &static_uniform,
                &placeholder,
                &placeholder,
                &sampler,
                &transmittance,
            ),
            lut_bind_group(
                gpu,
                &layout,
                &static_uniform,
                &transmittance,
                &placeholder,
                &sampler,
                &multiple_scattering,
            ),
        ];
        Ok(AtmosphereResources {
            sampler,
            layout,
            _pipelines: pipelines,
            transmittance,
            multiple_scattering,
            static_uniform,
            _static_groups: static_groups,
            views: std::array::from_fn(|_| None),
            lifecycle: Lifecycle::new(),
        })
    }

    /// Validates and prepares one view: allocates its sky view on first use and
    /// uploads the serializer's bytes for a changed size or parameter snapshot.
    ///
    /// Invalid view indices, zero sizes and non-finite parameters refuse with
    /// `sky.invalid` before any slot, texture, buffer or upload is touched.
    pub(crate) fn prepare_view(
        &mut self,
        gpu: &GpuContext,
        view: u32,
        size: (u32, u32),
        parameters: &PhysicalSkyGpu,
    ) -> Result<(), RaaError> {
        let prepared = self.lifecycle.prepare(view, size, parameters)?;
        if prepared.static_changed {
            gpu.queue()
                .write_buffer(&self.static_uniform, 0, &parameters.to_bytes());
        }
        let index = view as usize;
        match prepared.outcome {
            Prepare::Created => {
                let sky_view = texture_view(
                    gpu,
                    lut_descriptor("raa-sky.atmosphere.sky_view", SKY_VIEW_SIZE),
                );
                let uniform = uniform_buffer(gpu, "raa-sky.atmosphere.view");
                gpu.queue()
                    .write_buffer(&uniform, 0, &parameters.to_bytes());
                let bind_group = lut_bind_group(
                    gpu,
                    &self.layout,
                    &uniform,
                    &self.transmittance,
                    &self.multiple_scattering,
                    &self.sampler,
                    &sky_view,
                );
                self.views[index] = Some(ViewGpu {
                    sky_view,
                    uniform,
                    _bind_group: bind_group,
                });
            }
            Prepare::Updated => {
                if let Some(slot) = &self.views[index] {
                    gpu.queue()
                        .write_buffer(&slot.uniform, 0, &parameters.to_bytes());
                }
            }
            Prepare::Unchanged => {}
        }
        Ok(())
    }

    /// The static 256x64 transmittance table.
    pub(crate) fn transmittance(&self) -> &wgpu::TextureView {
        &self.transmittance
    }

    /// The static 32x32 multiple-scattering table.
    pub(crate) fn multiple_scattering(&self) -> &wgpu::TextureView {
        &self.multiple_scattering
    }

    /// A prepared view's 256x144 sky view, or `None` when it was removed or never prepared.
    pub(crate) fn sky_view(&self, view: u32) -> Option<&wgpu::TextureView> {
        self.views.get(view as usize)?.as_ref().map(|v| &v.sky_view)
    }

    /// The one linear filtering sampler shared by every LUT consumer.
    pub(crate) fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    /// Releases a view's sky view, uniform and bind group; out-of-range indices are a no-op.
    pub(crate) fn remove_view(&mut self, view: u32) {
        if let Some(slot) = self.views.get_mut(view as usize) {
            *slot = None;
        }
        self.lifecycle.remove(view);
    }
}

/// Compile-time contract of the opaque atmosphere resource set.
///
/// Cards 7429 (LUT recording), 7430 (Milky Way bake) and 7434 (sky passes) call
/// exactly these items. Pinning each signature here keeps the private resource
/// state reachable for the crate's `dead_code` lint until the first consumer
/// lands, and refuses an accidental interface drift in between. Remove this
/// block once a real consumer exists.
const _: () = {
    type New = fn(&GpuContext) -> Result<AtmosphereResources, RaaError>;
    type Prepare = fn(
        &mut AtmosphereResources,
        &GpuContext,
        u32,
        (u32, u32),
        &PhysicalSkyGpu,
    ) -> Result<(), RaaError>;
    type ViewRef = for<'a> fn(&'a AtmosphereResources) -> &'a wgpu::TextureView;
    type SkyView = for<'a> fn(&'a AtmosphereResources, u32) -> Option<&'a wgpu::TextureView>;
    type SamplerRef = for<'a> fn(&'a AtmosphereResources) -> &'a wgpu::Sampler;
    type Remove = fn(&mut AtmosphereResources, u32);
    let _: (New, Prepare, ViewRef, ViewRef, SkyView, SamplerRef, Remove) = (
        AtmosphereResources::new,
        AtmosphereResources::prepare_view,
        AtmosphereResources::transmittance,
        AtmosphereResources::multiple_scattering,
        AtmosphereResources::sky_view,
        AtmosphereResources::sampler,
        AtmosphereResources::remove_view,
    );
};

/// The explicit group-2 compute layout: 0 uniform, 1 transmittance, 2 multiple
/// scattering, 3 filtering sampler and 4 the Rgba16Float storage output.
fn layout_entries() -> [wgpu::BindGroupLayoutEntry; 5] {
    let texture = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(PhysicalSkyGpu::SIZE as u64),
            },
            count: None,
        },
        texture(1),
        texture(2),
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: LUT_FORMAT,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        },
    ]
}

/// Groups 0 and 1 stay empty; the resources live explicitly in shader group 2.
fn lut_pipelines(gpu: &GpuContext, layout: &wgpu::BindGroupLayout) -> [wgpu::ComputePipeline; 3] {
    let device = gpu.device();
    let empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("raa-sky.atmosphere.empty"),
        entries: &[],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("raa-sky.atmosphere.lut"),
        bind_group_layouts: &[Some(&empty), Some(&empty), Some(layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("raa-sky.atmosphere.lut"),
        source: wgpu::ShaderSource::Wgsl(atmosphere_luts_module_wgsl().into()),
    });
    ATMOSPHERE_LUT_ENTRIES.map(|entry| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    })
}

fn lut_bind_group(
    gpu: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    transmittance: &wgpu::TextureView,
    multiple_scattering: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    output: &wgpu::TextureView,
) -> wgpu::BindGroup {
    gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("raa-sky.atmosphere"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(transmittance),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(multiple_scattering),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(output),
            },
        ],
    })
}

fn lut_descriptor(label: &'static str, size: (u32, u32)) -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: LUT_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    }
}

/// The 1x1 zero-filled input dummy (wgpu zero-initializes a new texture).
fn dummy_descriptor(label: &'static str) -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: LUT_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    }
}

fn texture_view(
    gpu: &GpuContext,
    descriptor: wgpu::TextureDescriptor<'static>,
) -> wgpu::TextureView {
    gpu.device()
        .create_texture(&descriptor)
        .create_view(&wgpu::TextureViewDescriptor::default())
}

fn uniform_buffer(gpu: &GpuContext, label: &str) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: PhysicalSkyGpu::SIZE as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_at(bytes: &[u8; 256], offset: usize) -> f32 {
        f32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("four bytes"))
    }

    fn u32_at(bytes: &[u8; 256], offset: usize) -> u32 {
        u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("four bytes"))
    }

    #[test]
    fn radius_boundaries_follow_the_solstice_conversion() {
        assert_eq!(atmospheric_radius_km(-100.0), 6360.001);
        assert_eq!(atmospheric_radius_km(0.0), 6360.001);
        assert_eq!(atmospheric_radius_km(1.0), 6360.001);
        assert!((atmospheric_radius_km(1000.0) - 6361.0).abs() < 1e-6);
        assert!((atmospheric_radius_km(10_000.0) - 6370.0).abs() < 1e-4);
        // NaN is ignored by `f32::max`, so it takes the one-metre floor.
        assert_eq!(atmospheric_radius_km(f32::NAN), 6360.001);
        assert!(atmospheric_radius_km(f32::INFINITY).is_infinite());
    }

    #[test]
    fn lanes_serialize_at_the_s10_offsets_in_little_endian() {
        let gpu = PhysicalSkyGpu {
            viewer: [11.0, 12.0, 13.0, 6360.5],
            sun: [21.0, 22.0, 23.0, 0.25],
            moon: [31.0, 32.0, 33.0, 0.5],
            pole: [41.0, 42.0, 43.0, 0.75],
            equinox: [51.0, 52.0, 53.0, 1.5],
            dominant: [61.0, 62.0, 63.0, 2.5],
            daylight: [71.0, 72.0, 73.0, 3.5],
            discs: [81.0, 82.0, 83.0, 84.0],
            ground_albedo: [91.0, 92.0, 93.0, 94.0],
            rayleigh: [101.0, 102.0, 103.0, 104.0],
            mie: [111.0, 112.0, 113.0, 114.0],
            ozone: [121.0, 122.0, 123.0, 124.0],
            planet: [131.0, 132.0, 133.0, 134.0],
            flags: [1, 2, 3, 4],
            reserved: [[0.0; 4]; 2],
        };
        let bytes = gpu.to_bytes();
        assert_eq!(bytes.len(), 256);
        assert_eq!(&bytes[0..4], &11.0f32.to_le_bytes());
        for (offset, lane) in [
            (0, gpu.viewer),
            (16, gpu.sun),
            (32, gpu.moon),
            (48, gpu.pole),
            (64, gpu.equinox),
            (80, gpu.dominant),
            (96, gpu.daylight),
            (112, gpu.discs),
            (128, gpu.ground_albedo),
            (144, gpu.rayleigh),
            (160, gpu.mie),
            (176, gpu.ozone),
            (192, gpu.planet),
        ] {
            for (i, value) in lane.into_iter().enumerate() {
                assert_eq!(f32_at(&bytes, offset + i * 4).to_bits(), value.to_bits());
            }
        }
        for (i, value) in gpu.flags.into_iter().enumerate() {
            assert_eq!(u32_at(&bytes, 208 + i * 4), value);
        }
        assert!(bytes[224..256].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn default_uses_source_constants_and_zero_reserved_words() {
        let gpu = PhysicalSkyGpu::default();
        assert_eq!(gpu.viewer[3], 6360.001);
        assert_eq!(gpu.rayleigh[0..3], RAYLEIGH_SCATTERING);
        assert_eq!(gpu.rayleigh[3], RAYLEIGH_SCALE_HEIGHT_KM);
        assert_eq!(
            gpu.mie,
            [MIE_SCATTERING, MIE_ABSORPTION, MIE_SCALE_HEIGHT_KM, MIE_G]
        );
        assert_eq!(gpu.ozone[0..3], OZONE_ABSORPTION);
        assert_eq!(gpu.planet, [GROUND_RADIUS_KM, TOP_RADIUS_KM, 0.0, 0.0]);
        assert_eq!(gpu.reserved, [[0.0; 4]; 2]);
        let bytes = gpu.to_bytes();
        assert_eq!(f32_at(&bytes, 224), 0.0);
        assert_eq!(f32_at(&bytes, 240), 0.0);
        assert_eq!(f32_at(&bytes, 252), 0.0);
    }

    fn assert_lut(descriptor: &wgpu::TextureDescriptor<'_>, size: (u32, u32)) {
        assert_eq!(descriptor.format, LUT_FORMAT);
        assert_eq!(
            (descriptor.size.width, descriptor.size.height),
            size,
            "unexpected LUT dimensions"
        );
        assert_eq!(descriptor.size.depth_or_array_layers, 1);
        assert_eq!(descriptor.mip_level_count, 1);
        assert!(
            descriptor
                .usage
                .contains(wgpu::TextureUsages::TEXTURE_BINDING)
        );
        assert!(
            descriptor
                .usage
                .contains(wgpu::TextureUsages::STORAGE_BINDING)
        );
        assert!(descriptor.usage.contains(wgpu::TextureUsages::COPY_DST));
    }

    #[test]
    fn lut_descriptors_match_the_solstice_targets() {
        assert_lut(&lut_descriptor("t", TRANSMITTANCE_SIZE), (256, 64));
        assert_lut(&lut_descriptor("m", MULTI_SCATTER_SIZE), (32, 32));
        assert_lut(&lut_descriptor("s", SKY_VIEW_SIZE), (256, 144));
        let dummy = dummy_descriptor("dummy");
        assert_eq!((dummy.size.width, dummy.size.height), (1, 1));
        assert_eq!(dummy.format, LUT_FORMAT);
        assert!(dummy.usage.contains(wgpu::TextureUsages::TEXTURE_BINDING));
        assert!(
            !dummy.usage.contains(wgpu::TextureUsages::STORAGE_BINDING),
            "the input dummy is never a storage output"
        );
    }

    #[test]
    fn prepare_creates_updates_and_invalidates_per_view_slots() {
        let mut lifecycle = Lifecycle::new();
        let parameters = PhysicalSkyGpu::default();
        let first = lifecycle
            .prepare(0, (1280, 720), &parameters)
            .expect("view 0");
        assert_eq!(first.outcome, Prepare::Created);
        assert!(
            first.static_changed,
            "the first prepare builds the static tables"
        );
        assert!(lifecycle.views[0].as_ref().expect("slot").sky_view_pending);

        let again = lifecycle
            .prepare(0, (1280, 720), &parameters)
            .expect("view 0");
        assert_eq!(again.outcome, Prepare::Unchanged);
        assert!(!again.static_changed);

        let mut sun = parameters.clone();
        sun.sun = [1.0, 0.0, 0.0, 1.0];
        let updated = lifecycle.prepare(0, (1280, 720), &sun).expect("view 0");
        assert_eq!(updated.outcome, Prepare::Updated, "light snapshot changed");
        assert!(!updated.static_changed);

        let resized = lifecycle.prepare(0, (640, 360), &sun).expect("view 0");
        assert_eq!(
            resized.outcome,
            Prepare::Updated,
            "size invalidates the view"
        );
        assert_eq!(
            lifecycle
                .prepare(1, (800, 600), &sun)
                .expect("view 1")
                .outcome,
            Prepare::Created
        );

        lifecycle.remove(0);
        assert!(lifecycle.views[0].is_none());
        let recreated = lifecycle.prepare(0, (640, 360), &sun).expect("view 0");
        assert_eq!(
            recreated.outcome,
            Prepare::Created,
            "a removed view is reallocated"
        );
        lifecycle.remove(99);
        assert!(
            lifecycle.views[1].is_some(),
            "out-of-range removal is a no-op"
        );
    }

    #[test]
    fn static_parameters_invalidate_every_prepared_view() {
        let mut lifecycle = Lifecycle::new();
        let parameters = PhysicalSkyGpu::default();
        lifecycle
            .prepare(0, (1280, 720), &parameters)
            .expect("view 0");
        lifecycle
            .prepare(1, (800, 600), &parameters)
            .expect("view 1");
        lifecycle.views[0].as_mut().expect("slot").sky_view_pending = false;
        lifecycle.views[1].as_mut().expect("slot").sky_view_pending = false;

        let mut albedo = parameters.clone();
        albedo.ground_albedo[0] = 0.9;
        let changed = lifecycle.prepare(1, (800, 600), &albedo).expect("view 1");
        assert!(
            changed.static_changed,
            "ground albedo feeds both static tables"
        );
        assert_eq!(changed.outcome, Prepare::Updated);
        assert!(lifecycle.views[0].as_ref().expect("slot").sky_view_pending);
        assert!(lifecycle.views[1].as_ref().expect("slot").sky_view_pending);

        lifecycle.views[0].as_mut().expect("slot").sky_view_pending = false;
        lifecycle.views[1].as_mut().expect("slot").sky_view_pending = false;
        let mut sun = albedo.clone();
        sun.sun = [0.0, 0.5, 0.0, 1.0];
        let dynamic = lifecycle.prepare(1, (800, 600), &sun).expect("view 1");
        assert!(!dynamic.static_changed);
        assert!(!lifecycle.views[0].as_ref().expect("slot").sky_view_pending);
        assert!(lifecycle.views[1].as_ref().expect("slot").sky_view_pending);
    }

    #[test]
    fn invalid_prepare_is_atomic_and_reports_sky_invalid() {
        let mut lifecycle = Lifecycle::new();
        let valid = PhysicalSkyGpu::default();
        lifecycle.prepare(0, (1280, 720), &valid).expect("view 0");
        let before = lifecycle.views.clone();
        let mut nan = valid.clone();
        nan.viewer[0] = f32::NAN;
        let mut infinite = valid.clone();
        infinite.mie[3] = f32::INFINITY;
        for (view, size, parameters) in [
            (MAX_VIEWS as u32, (1280, 720), valid.clone()),
            (0, (0, 720), valid.clone()),
            (0, (1280, 0), valid.clone()),
            (0, (1280, 720), nan),
            (0, (1280, 720), infinite),
        ] {
            let error = lifecycle.prepare(view, size, &parameters).unwrap_err();
            assert_eq!(error.code(), "sky.invalid");
            assert_eq!(lifecycle.views, before, "invalid input mutates nothing");
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn resources_allocate_the_source_descriptors_and_release_views() {
        let Some(gpu) = raa_core::testing::software_gpu() else {
            raa_core::testing::skip(
                "resources_allocate_the_source_descriptors_and_release_views",
                "software_gpu returned None; see the software_gpu line above",
            );
            return;
        };
        let mut resources = AtmosphereResources::new(&gpu).expect("atmosphere resources");
        let transmittance = resources.transmittance().texture();
        assert_eq!(
            (transmittance.width(), transmittance.height()),
            TRANSMITTANCE_SIZE
        );
        assert_eq!(transmittance.format(), LUT_FORMAT);
        let multiple = resources.multiple_scattering().texture();
        assert_eq!((multiple.width(), multiple.height()), MULTI_SCATTER_SIZE);
        assert!(resources.sky_view(0).is_none());

        let parameters = PhysicalSkyGpu::default();
        resources
            .prepare_view(&gpu, 0, (1280, 720), &parameters)
            .expect("view 0");
        let sky = resources.sky_view(0).expect("prepared view").texture();
        assert_eq!((sky.width(), sky.height()), SKY_VIEW_SIZE);
        assert_eq!(sky.format(), LUT_FORMAT);
        resources
            .prepare_view(&gpu, 3, (800, 600), &parameters)
            .expect("view 3");
        assert!(resources.sky_view(3).is_some());
        assert!(
            resources.sky_view(4).is_none(),
            "out-of-range lookup is None"
        );

        resources.remove_view(0);
        assert!(resources.sky_view(0).is_none());
        resources.remove_view(99);
        assert!(
            resources.sky_view(3).is_some(),
            "removing another index keeps view 3"
        );
    }
}
