//! CPU numeric tests for the area temporal blend and ReSTIR confidence clamp.
//! Expected values below are derived from the documented formulas, not from the functions under test.

use raa_gi::parts::area_shadow::cpu::{TEMPORAL_ALPHA, temporal_blend};
use raa_gi::parts::area_shadow::restir_cpu::clamp_confidence;

#[test]
fn temporal_blend_uses_history_only_when_valid() {
    // TEMPORAL_ALPHA is 0.1, so a usable history of 1 blended toward current 0 is
    // (1 - TEMPORAL_ALPHA) * 1 + TEMPORAL_ALPHA * 0 = 0.9.
    assert_eq!(temporal_blend(1.0, 0.0, true), 0.9);

    // An invalid history takes the current value directly.
    assert_eq!(temporal_blend(1.0, 0.0, false), 0.0);

    // Three successive usable frames at current 0 multiply the retained history by 0.9 each time.
    let after_three: f32 = (1.0 - TEMPORAL_ALPHA).powi(3);
    assert!((after_three - 0.729).abs() < 1e-6);
    let mut value = 1.0;
    for _ in 0..3 {
        value = temporal_blend(value, 0.0, true);
    }
    assert!((value - after_three).abs() < 1e-6);
}

#[test]
fn clamp_confidence_caps_previous_count_by_current_count() {
    // Formula: max(min(previous, cap * current), 0). With (50, 1, 20),
    // cap * current = 20, min(50, 20) = 20, and max(20, 0) = 20.
    assert_eq!(clamp_confidence(50.0, 1.0, 20.0), 20.0);
}

#[test]
fn changed_slot_rejects_only_its_own_packed_history() {
    use raa_gi::parts::area_shadow::cpu::slot_rejection_bits;

    let previous = [1, 2, 3, 4, 5, 6, 7, 8];
    let mut current = previous;
    current.swap(0, 7);
    assert_eq!(
        std::array::from_fn::<_, 4, _>(|group| slot_rejection_bits(&previous, &current, group)),
        [2, 0, 0, 4]
    );
}

#[test]
fn weighted_pair_resolve_preserves_black_umbra_and_empty_slot() {
    use raa_gi::parts::area_shadow::cpu::ratio_visibility;

    assert_eq!(ratio_visibility(0.0, 0.5), 0.0);
    assert_eq!(ratio_visibility(0.0, 0.0), 1.0);

    // The same filter weights are applied to each component before the division.
    let [dark, bright] = [[0.0, 0.5], [0.125, 0.125]];
    let [visible, total] = std::array::from_fn(|i| (dark[i] + 3.0 * bright[i]) / 4.0);
    assert_eq!(ratio_visibility(visible, total), 3.0 / 7.0);
}

#[test]
fn staged_area_filters_validate_as_wgsl() {
    use raa_gi::parts::area_shadow::passes::{spatial_module_wgsl, temporal_module_wgsl};

    for source in [temporal_module_wgsl(), spatial_module_wgsl()] {
        raa_shader::Composer::validate(&source).expect("area filter WGSL");
    }
}

#[test]
#[cfg_attr(not(target_os = "windows"), ignore = "Windows GPU regression")]
fn same_receiver_samples_compare_anti_lag_current_only_and_legacy() {
    use raa_core::testing::block_on;
    use raa_core::{GpuContext, GpuOptions};
    use raa_gi::parts::area_shadow::passes::temporal_module_wgsl;

    let gpu = block_on(GpuContext::headless(&GpuOptions {
        allow_software: false,
        ..GpuOptions::default()
    }))
    .expect("Windows hardware GPU");
    let device = gpu.device();
    let mut source = temporal_module_wgsl();
    // Fixed weighted samples from a stationary receiver as a blocker enters its light path.
    // The other slot stays unchanged. All three branches consume the same raw pairs.
    source.push_str(
        r#"
        @group(0) @binding(9) var<storage, read_write> results: array<vec2<f32>, 4>;
        @compute @workgroup_size(1)
        fn compare_filters() {
            let previous = vec2<f32>(0.5, 0.5);
            let current = vec2<f32>(0.0, 0.25);
            let evidence = vec2<f32>(0.0, 1.5);
            results[0] = raa_area_blend_pair(previous, current, evidence, true, true);
            results[1] = raa_area_blend_pair(previous, current, evidence, false, false);
            results[2] = raa_area_blend_pair(previous, current, evidence, true, false);
            results[3] = raa_area_blend_pair(previous, previous, previous, true, true);
        }
    "#,
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("area-shadow-anti-lag-regression"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("compare_filters"),
        compilation_options: Default::default(),
        cache: None,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 32,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 32,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 9,
            resource: output.as_entire_binding(),
        }],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 32);
    gpu.queue().submit([encoder.finish()]);
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |result| {
        result.expect("map comparison")
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("read comparison");
    let data = slice.get_mapped_range().expect("mapped comparison");
    let results: Vec<f32> = data
        .chunks_exact(4)
        .map(|lane| f32::from_le_bytes(lane.try_into().unwrap()))
        .collect();
    for (actual, expected) in results
        .iter()
        .zip([0.0, 0.25, 0.0, 0.25, 0.45, 0.475, 0.5, 0.5])
    {
        assert!((actual - expected).abs() < 1e-6, "observed {results:?}");
    }
}
