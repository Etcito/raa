use raa_core::{GpuContext, RaaError};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use wgpu::util::DeviceExt;

pub(crate) const WGSL: &str = include_str!("prediction_model.wgsl");
const WEIGHT_BYTES: usize = 1984;
const CHANNELS: [&str; 14] = [
    "logWarpR",
    "logWarpG",
    "logWarpB",
    "logFreshR",
    "logFreshG",
    "logFreshB",
    "warpValid",
    "freshValid",
    "targetDepthZ",
    "motionU",
    "motionV",
    "lightingChanged",
    "sourceAgeOver8",
    "motionReject",
];
const TENSORS: [(&str, &[usize], usize); 8] = [
    ("stem.weight", &[12, 14], 0),
    ("stem.bias", &[12], 672),
    ("spatial.weight", &[12, 3, 3], 720),
    ("spatial.bias", &[12], 1152),
    ("mix.weight", &[12, 12], 1200),
    ("mix.bias", &[12], 1776),
    ("head.weight", &[3, 12], 1824),
    ("head.bias", &[3], 1968),
];

pub(crate) struct GeneratedModel {
    weights: wgpu::Buffer,
    stem: wgpu::ComputePipeline,
    decode: wgpu::ComputePipeline,
    pub(crate) digest: String,
    pub(crate) kind: String,
}

pub(crate) struct GeneratedInputs<'a> {
    pub warped: &'a wgpu::TextureView,
    pub refreshed: &'a wgpu::TextureView,
    pub resolved: &'a wgpu::TextureView,
    pub depth: &'a wgpu::TextureView,
    pub motion: &'a wgpu::TextureView,
    pub lighting: &'a wgpu::TextureView,
    pub source_age: u64,
}

impl GeneratedModel {
    pub(crate) fn new(gpu: &GpuContext, manifest: &str, bytes: &[u8]) -> Result<Self, RaaError> {
        let invalid = |message: &str| RaaError::FrameGraph {
            code: "graph.generated_model_package",
            message: message.into(),
        };
        let m: Value = serde_json::from_str(manifest)
            .map_err(|_| invalid("expected a JSON model manifest"))?;
        let digest = format!("{:x}", Sha256::digest(bytes));
        let kind = m["kind"]
            .as_str()
            .filter(|kind| !kind.is_empty())
            .ok_or_else(|| invalid("the model provenance is missing"))?;
        if m["format"] != "raa.generated-frame-model/1"
            || m["modelId"] != "generated-frame-v1"
            || m["precision"] != "fp32"
            || m["tensorLayout"] != "NHWC"
            || m["tensorAlignment"] != 16
            || m["cadence"] != "admitted predictive presentation only"
            || m["colorDomain"] != "nonnegative scene-linear HDR at fixed exposure 1"
            || m["featureChannels"] != json!(CHANNELS)
            || m["trainedHeads"] != json!(["frame-generation"])
            || m["weightsByteLength"] != WEIGHT_BYTES
            || bytes.len() != WEIGHT_BYTES
            || m["weightsSha256"] != digest
        {
            return Err(invalid(
                "the model ABI, status, feature order, size or digest differs",
            ));
        }
        let tensors = m["tensors"]
            .as_array()
            .filter(|tensors| tensors.len() == TENSORS.len())
            .ok_or_else(|| invalid("expected all tensors in ABI order"))?;
        let mut end = 0;
        for (tensor, (name, shape, offset)) in tensors.iter().zip(TENSORS) {
            let length = 4 * shape.iter().product::<usize>();
            if tensor["name"] != name
                || tensor["shape"] != json!(shape)
                || tensor["offset"] != offset
                || tensor["dtype"] != "f32le"
                || tensor["byteLength"] != length
                || bytes[end..offset].iter().any(|&byte| byte != 0)
                || bytes[offset..offset + length].chunks_exact(4).any(|word| {
                    let value = f32::from_le_bytes(word.try_into().expect("FP32 word"));
                    !value.is_finite() || value.abs() > 4096.0
                })
            {
                return Err(invalid(
                    "the tensor layout, zero padding or finite bounds differ",
                ));
            }
            end = offset + length;
        }
        if bytes[end..].iter().any(|&byte| byte != 0) {
            return Err(invalid("the model has nonzero trailing padding"));
        }

        use wgpu::{
            BindingType as T, BufferBindingType as B, TextureSampleType as S,
            TextureViewDimension as D,
        };
        let device = gpu.device();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prediction.learned"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let buffer = |ty| T::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let sampled = |sample_type| T::Texture {
            sample_type,
            view_dimension: D::D2,
            multisampled: false,
        };
        let float = sampled(S::Float { filterable: false });
        let weights = buffer(B::Storage { read_only: true });
        let features = buffer(B::Storage { read_only: false });
        let make = |name, bindings: &[(u32, T)]| {
            let entries = bindings
                .iter()
                .map(|&(binding, ty)| wgpu::BindGroupLayoutEntry {
                    binding,
                    ty,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    count: None,
                })
                .collect::<Vec<_>>();
            let group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(name),
                entries: &entries,
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(name),
                bind_group_layouts: &[Some(&group)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Ok(Self {
            weights: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("prediction.learned.weights"),
                usage: wgpu::BufferUsages::STORAGE,
                contents: bytes,
            }),
            stem: make(
                "generated_stem",
                &[
                    (0, weights),
                    (1, features),
                    (2, float),
                    (3, float),
                    (4, float),
                    (5, float),
                    (6, sampled(S::Uint)),
                    (7, buffer(B::Uniform)),
                ],
            ),
            decode: make(
                "generated_residual",
                &[
                    (0, weights),
                    (1, features),
                    (2, float),
                    (3, float),
                    (8, float),
                    (
                        9,
                        T::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: D::D2,
                        },
                    ),
                ],
            ),
            digest,
            kind: kind.to_owned(),
        })
    }

    pub(crate) fn weights(&self) -> &wgpu::Buffer {
        &self.weights
    }

    pub(crate) fn record(
        &self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        inputs: GeneratedInputs<'_>,
    ) -> Option<wgpu::TextureView> {
        use wgpu::{BindingResource as R, BufferUsages as B, TextureUsages as T};
        let size = inputs.resolved.texture().size();
        let device = gpu.device();
        let bytes = u64::from(size.width) * u64::from(size.height) * 48;
        let limits = device.limits();
        if bytes > u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size)
            || bytes > 128 * 1024 * 1024
            || inputs.source_age > 8
        {
            return None;
        }
        let stem = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prediction.learned.features"),
            size: bytes,
            usage: B::STORAGE,
            mapped_at_creation: false,
        });
        let age = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("prediction.learned.age"),
            usage: B::UNIFORM,
            contents: &(inputs.source_age as f32 / 8.0).to_le_bytes(),
        });
        let output = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("prediction.learned.hdr"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: T::STORAGE_BINDING | T::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let run = |encoder: &mut wgpu::CommandEncoder,
                   pipeline: &wgpu::ComputePipeline,
                   bindings: &[(u32, R<'_>)]| {
            let entries = bindings
                .iter()
                .map(|(binding, resource)| wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: resource.clone(),
                })
                .collect::<Vec<_>>();
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("prediction.learned"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prediction.learned"),
                ..Default::default()
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(size.width.div_ceil(8), size.height.div_ceil(8), 1);
        };
        let w = || self.weights.as_entire_binding();
        let s = || stem.as_entire_binding();
        run(
            encoder,
            &self.stem,
            &[
                (0, w()),
                (1, s()),
                (2, R::TextureView(inputs.warped)),
                (3, R::TextureView(inputs.refreshed)),
                (4, R::TextureView(inputs.depth)),
                (5, R::TextureView(inputs.motion)),
                (6, R::TextureView(inputs.lighting)),
                (7, age.as_entire_binding()),
            ],
        );
        run(
            encoder,
            &self.decode,
            &[
                (0, w()),
                (1, s()),
                (2, R::TextureView(inputs.warped)),
                (3, R::TextureView(inputs.refreshed)),
                (8, R::TextureView(inputs.resolved)),
                (9, R::TextureView(&output)),
            ],
        );
        Some(output)
    }
}
