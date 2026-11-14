//! Twin test of the S38 `gi-records`: every record's naga struct span and member offsets in its
//! WGSL module equal the Rust SIZE and the contract offsets, and `to_bytes` writes each field
//! (distinct values) at that offset with zero padding. The pattern of
//! `raa-vg/tests/cull_variants.rs` `cull_control_args_have_the_s32_offsets`.
//!
//! Consumes the `to_bytes` methods of `layouts.rs:30 GiFrameGpu` (its `Handoff:` doc comment).

#![cfg(not(target_arch = "wasm32"))]

use raa_core::wgpu::naga;
use raa_gi::layouts::{
    GI_FRAME_SIZE, GI_WORK_STRIDE, GiFloodGpu, GiForwardWorkGpu, GiFrameGpu, GiLightGpu,
    GiVolumeParamsGpu, GiVoxelizeGpu, GiWriteGpu,
};
use raa_gi::shaders::{
    flood_module_wgsl, volume_write_module_wgsl, voxel_light_module_wgsl,
    voxelize_forward_module_wgsl,
};
use raa_math::Mat4;
use raa_shader::PRELUDE_WGSL;

/// One expected member: WGSL name, byte offset and the little-endian words stored there.
type Member = (&'static str, usize, Vec<u32>);

fn f(v: f32) -> u32 {
    v.to_bits()
}

/// Checks the naga struct `name` of `wgsl` against `members` and `bytes` (the whole record).
fn check(wgsl: &str, name: &str, size: usize, members: &[Member], bytes: &[u8]) {
    let module = naga::front::wgsl::parse_str(wgsl).expect("the module parses");
    let (_, ty) = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("{name} is declared"));
    let naga::TypeInner::Struct {
        members: fields,
        span,
    } = &ty.inner
    else {
        panic!("{name} is a struct");
    };
    assert_eq!(*span as usize, size, "{name} span");
    assert_eq!(bytes.len(), size, "{name} to_bytes length");
    let found: Vec<(&str, u32)> = fields
        .iter()
        .map(|m| (m.name.as_deref().unwrap_or(""), m.offset))
        .collect();
    let want: Vec<(&str, u32)> = members.iter().map(|(n, o, _)| (*n, *o as u32)).collect();
    assert_eq!(found, want, "{name} member offsets");
    let mut covered = 0;
    for (member, offset, words) in members {
        for (i, word) in words.iter().enumerate() {
            let at = offset + 4 * i;
            let read = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
            assert_eq!(read, *word, "{name}.{member} word {i} at {at}");
            covered += 4;
        }
    }
    assert_eq!(covered, size, "{name} members cover every byte");
}

fn mat(base: f32) -> Mat4 {
    let mut m = Mat4::default();
    for c in 0..4 {
        for r in 0..4 {
            m.cols[c][r] = base + (c * 4 + r) as f32;
        }
    }
    m
}

fn mat_words(m: &Mat4) -> Vec<u32> {
    m.to_cols_array().map(f32::to_bits).to_vec()
}

#[test]
fn frame_record_matches_its_wgsl_struct() {
    let frame = GiFrameGpu {
        view_proj: mat(1.0),
        inv_view_proj: mat(101.0),
        inv_proj: mat(201.0),
        inv_view: mat(301.0),
        previous_view_proj: mat(401.0),
        camera_position: [1.5, 2.5, 3.5, 0.0],
        sun_direction: [4.5, 5.5, 6.5, 0.0],
        sun_color: [7.5, 8.5, 9.5, 0.0],
        viewport: [1280.0, 720.0, 1.0 / 1280.0, 1.0 / 720.0],
        counts: [3, 77, 2, 5],
        settings: [0.75, 0.25, 0.1, 500.0],
        flags: [1, 0, 1, 1],
    };
    assert_eq!(GI_FRAME_SIZE, 432);
    assert_eq!(GI_WORK_STRIDE, 256);
    assert_eq!(GiFrameGpu::SIZE, 432);
    let members: Vec<Member> = vec![
        ("view_proj", 0, mat_words(&frame.view_proj)),
        ("inv_view_proj", 64, mat_words(&frame.inv_view_proj)),
        ("inv_proj", 128, mat_words(&frame.inv_proj)),
        ("inv_view", 192, mat_words(&frame.inv_view)),
        (
            "previous_view_proj",
            256,
            mat_words(&frame.previous_view_proj),
        ),
        (
            "camera_position",
            320,
            frame.camera_position.map(f).to_vec(),
        ),
        ("sun_direction", 336, frame.sun_direction.map(f).to_vec()),
        ("sun_color", 352, frame.sun_color.map(f).to_vec()),
        ("viewport", 368, frame.viewport.map(f).to_vec()),
        ("counts", 384, frame.counts.to_vec()),
        ("settings", 400, frame.settings.map(f).to_vec()),
        ("flags", 416, frame.flags.to_vec()),
    ];
    // Column-major: column 1 row 2 of view_proj is the seventh float.
    let bytes = frame.to_bytes();
    assert_eq!(f32::from_le_bytes(bytes[24..28].try_into().unwrap()), 7.0);
    let wgsl = voxel_light_module_wgsl();
    check(&wgsl, "GiFrameGpu", 432, &members, &bytes);
}

#[test]
fn voxelize_record_matches_its_wgsl_struct() {
    let r = GiVoxelizeGpu {
        origin: [-1.5, 2.5, 3.5],
        voxel_size: 0.4,
        resolution: 96,
        level_offset: 884_736,
    };
    let members: Vec<Member> = vec![
        ("origin", 0, r.origin.map(f).to_vec()),
        ("voxel_size", 12, vec![f(0.4)]),
        ("resolution", 16, vec![96]),
        ("level_offset", 20, vec![884_736]),
        ("_pad0", 24, vec![0]),
        ("_pad1", 28, vec![0]),
    ];
    let wgsl = voxelize_forward_module_wgsl();
    check(
        &wgsl,
        "GiVoxelizeGpu",
        GiVoxelizeGpu::SIZE,
        &members,
        &r.to_bytes(),
    );
}

#[test]
fn flood_record_matches_its_wgsl_struct() {
    let r = GiFloodGpu {
        step: [-48, 0, 24],
        resolution: 96,
        level_offset: 7,
    };
    let members: Vec<Member> = vec![
        ("step", 0, r.step.map(|v| v as u32).to_vec()),
        ("resolution", 12, vec![96]),
        ("level_offset", 16, vec![7]),
        ("_pad0", 20, vec![0]),
        ("_pad1", 24, vec![0]),
        ("_pad2", 28, vec![0]),
    ];
    check(
        &flood_module_wgsl(),
        "GiFloodGpu",
        GiFloodGpu::SIZE,
        &members,
        &r.to_bytes(),
    );
}

#[test]
fn light_record_matches_its_wgsl_struct() {
    let r = GiLightGpu {
        wrap_origin: [-5, 6, -7],
        level: 3,
        updates: 41,
        resolution: 64,
    };
    let members: Vec<Member> = vec![
        ("wrap_origin", 0, r.wrap_origin.map(|v| v as u32).to_vec()),
        ("level", 12, vec![3]),
        ("updates", 16, vec![41]),
        ("resolution", 20, vec![64]),
        ("_pad0", 24, vec![0]),
        ("_pad1", 28, vec![0]),
    ];
    let wgsl = voxel_light_module_wgsl();
    check(
        &wgsl,
        "GiLightGpu",
        GiLightGpu::SIZE,
        &members,
        &r.to_bytes(),
    );
}

#[test]
fn write_record_matches_its_wgsl_struct() {
    let r = GiWriteGpu {
        wrap_origin: [8, -9, 10],
        level: 2,
        resolution: 96,
        voxel_size: 0.8,
    };
    let members: Vec<Member> = vec![
        ("wrap_origin", 0, r.wrap_origin.map(|v| v as u32).to_vec()),
        ("level", 12, vec![2]),
        ("resolution", 16, vec![96]),
        ("voxel_size", 20, vec![f(0.8)]),
        ("_pad0", 24, vec![0]),
        ("_pad1", 28, vec![0]),
    ];
    let wgsl = volume_write_module_wgsl();
    check(
        &wgsl,
        "GiWriteGpu",
        GiWriteGpu::SIZE,
        &members,
        &r.to_bytes(),
    );
}

#[test]
fn forward_work_record_matches_its_wgsl_struct() {
    let r = GiForwardWorkGpu {
        slot: 11,
        first_index: 300,
        triangle_count: 4096,
        level: 4,
    };
    let members: Vec<Member> = vec![
        ("slot", 0, vec![11]),
        ("first_index", 4, vec![300]),
        ("triangle_count", 8, vec![4096]),
        ("level", 12, vec![4]),
        ("_pad0", 16, vec![0; 4]),
        ("_pad1", 32, vec![0; 4]),
    ];
    let wgsl = voxelize_forward_module_wgsl();
    check(&wgsl, "GiForwardWorkGpu", 48, &members, &r.to_bytes());
}

#[test]
fn volume_params_match_the_prelude_struct() {
    let mut levels = [[0.0f32; 4]; 5];
    for (i, level) in levels.iter_mut().take(3).enumerate() {
        let i = i as f32;
        *level = [-10.0 * (i + 1.0), 2.0 + i, 3.0 + i, 0.2 * (i + 1.0)];
    }
    let r = GiVolumeParamsGpu {
        levels,
        resolution: 96.0,
        intensity: 1.25,
        level_count: 3,
        flags: 5,
    };
    let members: Vec<Member> = vec![
        (
            "levels",
            0,
            levels.iter().flatten().map(|v| f(*v)).collect(),
        ),
        ("resolution", 80, vec![f(96.0)]),
        ("intensity", 84, vec![f(1.25)]),
        ("level_count", 88, vec![3]),
        ("flags", 92, vec![5]),
    ];
    // The prelude only declares the type; a probe global makes naga emit it.
    let wgsl =
        format!("{PRELUDE_WGSL}\n@group(3) @binding(0) var<uniform> probe: RaaGiVolumeParams;");
    check(
        &wgsl,
        "RaaGiVolumeParams",
        GiVolumeParamsGpu::SIZE,
        &members,
        &r.to_bytes(),
    );
}
