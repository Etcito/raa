//! The optional per-vertex colour stream of the forward voxelizer (driver A), on a software GPU
//! (SKIP without an adapter): a red and a green triangle with vertex colours reach the volume as
//! red and green albedo, and a draw without a stream gives the albedo it gave before
//! (base colour x instance tint, unchanged).
//!
//! The rig dispatches the real module `voxelize_forward_module_wgsl` with the real group-2
//! entries `forward_group2_entries` on an 8^3 level of 1 m voxels at the origin, with one
//! instance at the identity, a material of base (0.6, 0.4, 0.2) (153, 102, 51 in unorm8) and a
//! tint of 1. The red triangle lies in the plane z = 2.5 and the green one in z = 5.5, each
//! covering x and y in [0.5, 4.5] m, so the voxels at z = 2 are the red triangle's and those at
//! z = 5 the green one's. Voxel word 0 is `pack4x8unorm(albedo, 1)`: the alpha byte 255 is the
//! occupancy flag. The driver B kernel has no CPU-checkable rig of its own; its module and group
//! are pinned by `tests/virtual_variants.rs`.

use raa_core::GpuContext;
use raa_core::testing::{skip, software_gpu};
use raa_gi::layouts::{GiForwardWorkGpu, GiVoxelizeGpu};
use raa_gi::parts::virtual_driver::{virtual_group2_entries, voxelize_virtual_module_wgsl};
use raa_gi::shaders::{forward_group2_entries, voxelize_forward_module_wgsl};
use raa_gi::vertex_colors::{WHITE_TEXEL, pack_colors};

const RESOLUTION: u32 = 8;
const BASE: [f32; 3] = [0.6, 0.4, 0.2];

fn buffer(gpu: &GpuContext, usage: wgpu::BufferUsages, bytes: &[u8]) -> wgpu::Buffer {
    let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("vertex_colors"),
        size: bytes.len().max(4) as u64,
        usage: usage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    gpu.queue().write_buffer(&buffer, 0, bytes);
    buffer
}

fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn words(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Voxelizes the two triangles and returns word 0 of every voxel (`x + 8 y + 64 z`).
fn voxelize(gpu: &GpuContext, colors: &[u32]) -> Vec<u32> {
    let device = gpu.device();
    let storage = wgpu::BufferUsages::STORAGE;
    // Stream 0: position xyz, normal xyz per vertex; the normals face +Z.
    let mut stream0 = Vec::new();
    for (z, xy) in [(2.5, 0.5), (5.5, 0.5)] {
        for (dx, dy) in [(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)] {
            stream0.extend([xy + dx, xy + dy, z, 0.0, 0.0, 1.0]);
        }
    }
    let identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let mut instance = floats(&identity.concat());
    instance.extend(floats(&[0.0; 12]));
    instance.extend(words(&[0, 0, 0, 0]));
    let mut extras = floats(&[1.0, 1.0, 1.0, 1.0]);
    extras.resize(128, 0);
    let mut material = floats(&[BASE[0], BASE[1], BASE[2], 1.0]);
    material.resize(256, 0);

    let voxelize = GiVoxelizeGpu {
        origin: [0.0; 3],
        voxel_size: 1.0,
        resolution: RESOLUTION,
        level_offset: 0,
    };
    let work = GiForwardWorkGpu {
        slot: 0,
        first_index: 0,
        triangle_count: 2,
        level: 0,
    };
    let mut work_bytes = work.to_bytes().to_vec();
    work_bytes.resize(256, 0);
    let voxel_words = (RESOLUTION * RESOLUTION * RESOLUTION * 3) as usize;
    let voxels = buffer(gpu, storage, &words(&vec![0; voxel_words]));
    let resources = [
        buffer(gpu, wgpu::BufferUsages::UNIFORM, &voxelize.to_bytes()),
        buffer(gpu, wgpu::BufferUsages::UNIFORM, &work_bytes),
        buffer(gpu, storage, &instance),
        buffer(gpu, storage, &extras),
        buffer(gpu, storage, &material),
        buffer(gpu, storage, &floats(&stream0)),
        buffer(gpu, storage, &words(&[0, 1, 2, 3, 4, 5])),
        voxels.clone(),
        buffer(gpu, storage, &words(colors)),
    ];
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("vertex_colors.group2"),
        entries: &forward_group2_entries(),
    });
    let entries: Vec<wgpu::BindGroupEntry> = resources
        .iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: if binding == 1 {
                wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(GiForwardWorkGpu::SIZE as u64),
                })
            } else {
                buffer.as_entire_binding()
            },
        })
        .collect();
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vertex_colors.group2"),
        layout: &layout,
        entries: &entries,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("vertex_colors"),
        bind_group_layouts: &[None, None, Some(&layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vertex_colors"),
        source: wgpu::ShaderSource::Wgsl(voxelize_forward_module_wgsl().into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("vertex_colors"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("raa_gi_voxelize_forward"),
        compilation_options: Default::default(),
        cache: None,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vertex_colors.readback"),
        size: (voxel_words * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(2, &group, &[0]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&voxels, 0, &staging, 0, (voxel_words * 4) as u64);
    gpu.queue().submit([encoder.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
    let data = staging
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback")
        .to_vec();
    data.chunks_exact(12)
        .map(|voxel| u32::from_le_bytes([voxel[0], voxel[1], voxel[2], voxel[3]]))
        .collect()
}

/// The occupied voxels (alpha 255) of plane `z` as unorm8 rgb.
fn plane(voxels: &[u32], z: usize) -> Vec<[u32; 3]> {
    voxels[z * 64..(z + 1) * 64]
        .iter()
        .filter(|word| **word >> 24 == 255)
        .map(|word| [word & 255, (word >> 8) & 255, (word >> 16) & 255])
        .collect()
}

#[test]
fn red_and_green_vertex_colours_reach_the_volume_albedo() {
    let test = "red_and_green_vertex_colours_reach_the_volume_albedo";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let red = [1.0, 0.0, 0.0, 1.0];
    let green = [0.0, 1.0, 0.0, 1.0];
    let stream: Vec<f32> = [red, red, red, green, green, green].concat();
    let colors = pack_colors("vertexColors", &stream).expect("valid colours");
    let voxels = voxelize(&gpu, &colors);
    let red_plane = plane(&voxels, 2);
    let green_plane = plane(&voxels, 5);
    assert!(!red_plane.is_empty(), "the red triangle left no voxel");
    assert!(!green_plane.is_empty(), "the green triangle left no voxel");
    // base (0.6, 0.4, 0.2) x red (1, 0, 0) = (153, 0, 0); x green (0, 1, 0) = (0, 102, 0).
    for albedo in &red_plane {
        assert_eq!(*albedo, [153, 0, 0], "red triangle voxel");
    }
    for albedo in &green_plane {
        assert_eq!(*albedo, [0, 102, 0], "green triangle voxel");
    }
    // Neither is the white of an unmodulated base colour.
    assert!(
        red_plane
            .iter()
            .chain(&green_plane)
            .all(|a| *a != [153, 102, 51])
    );
}

#[test]
fn a_draw_without_colours_keeps_its_albedo() {
    let test = "a_draw_without_colours_keeps_its_albedo";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    // The driver binds one white word for a mesh without a stream; an explicit all-white stream
    // gives the same volume.
    let dummy = voxelize(&gpu, &[WHITE_TEXEL]);
    let explicit = voxelize(&gpu, &[WHITE_TEXEL; 6]);
    assert_eq!(dummy, explicit);
    for z in [2, 5] {
        let occupied = plane(&dummy, z);
        assert!(!occupied.is_empty(), "plane {z} is empty");
        for albedo in occupied {
            assert_eq!(albedo, [153, 102, 51], "base x tint, plane {z}");
        }
    }
}

#[test]
fn the_virtual_kernel_builds_with_the_colour_texture_group() {
    let test = "the_virtual_kernel_builds_with_the_colour_texture_group";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let device = gpu.device();
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("vertex_colors.virtual"),
        entries: &virtual_group2_entries(),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("vertex_colors.virtual"),
        bind_group_layouts: &[None, None, Some(&layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vertex_colors.virtual"),
        source: wgpu::ShaderSource::Wgsl(voxelize_virtual_module_wgsl().into()),
    });
    let _pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("vertex_colors.virtual"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("raa_gi_voxelize_virtual"),
        compilation_options: Default::default(),
        cache: None,
    });
    let error = raa_core::testing::block_on(scope.pop());
    assert!(error.is_none(), "pipeline validation: {error:?}");
}
