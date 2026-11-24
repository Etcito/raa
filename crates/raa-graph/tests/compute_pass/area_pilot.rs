use raa_core::{DeviceRequest, GpuContext, GpuOptions, testing::block_on};
use raa_graph::learned::{AreaPilot, AreaPilotPackage, AreaPilotRect, CHANNELS};
use raa_loader::json::{JsonValue, parse};
use std::time::Instant;

fn gpu() -> GpuContext {
    assert!(cfg!(windows), "area pilot GPU checks require Windows");
    block_on(GpuContext::headless_with(
        &GpuOptions {
            allow_software: false,
            ..GpuOptions::default()
        },
        &DeviceRequest {
            optional_features: wgpu::Features::TIMESTAMP_QUERY,
            adapter_limits: true,
        },
    ))
    .expect("area pilot requires a WebGPU adapter")
}

fn buffer(gpu: &GpuContext, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("area pilot measurement"),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn read(gpu: &GpuContext, staging: &wgpu::Buffer) -> Vec<u8> {
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    rx.recv().unwrap().unwrap();
    let bytes = slice.get_mapped_range().unwrap().to_vec();
    staging.unmap();
    bytes
}

fn numbers(fields: &[(String, JsonValue)], key: &str) -> Vec<f32> {
    let JsonValue::Array(items) = &fields.iter().find(|(k, _)| k == key).unwrap().1 else {
        panic!("missing golden array {key}")
    };
    items
        .iter()
        .map(|v| match v {
            JsonValue::Number(n) => *n as f32,
            _ => panic!("invalid golden number"),
        })
        .collect()
}

fn compare(gpu: &GpuContext, pilot: &AreaPilot, input: &[f32], expected: &[f32]) -> (f32, usize) {
    pilot.upload(input).unwrap();
    let staging = buffer(
        gpu,
        pilot.memory().io,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );
    let mut encoder = gpu.device().create_command_encoder(&Default::default());
    pilot.record_real_frame(&mut encoder, None);
    encoder.copy_buffer_to_buffer(pilot.output(), 0, &staging, 0, pilot.memory().io);
    gpu.queue().submit([encoder.finish()]);
    let actual = read(gpu, &staging);
    assert_eq!(actual.len(), 4 * expected.len());
    let mut max_error = 0.0_f32;
    let mut outside_tolerance = 0;
    for (bytes, reference) in actual.chunks_exact(4).zip(expected) {
        let value = f32::from_le_bytes(bytes.try_into().unwrap());
        let error = (value - reference).abs();
        max_error = max_error.max(if error.is_finite() {
            error
        } else {
            f32::INFINITY
        });
        if !error.is_finite() || error > 1e-5 + 1e-5 * reference.abs() {
            outside_tolerance += 1;
        }
    }
    (max_error, outside_tolerance)
}

fn rect(fields: &[(String, JsonValue)]) -> AreaPilotRect {
    let pair = |key| {
        let a = numbers(fields, key);
        [a[0] as u32, a[1] as u32]
    };
    AreaPilotRect {
        size: pair("size"),
        origin: pair("origin"),
        extent: pair("extent"),
    }
}

#[test]
#[cfg_attr(not(windows), ignore = "Windows hardware only")]
fn golden_tensors_on_existing_device() {
    let gpu = gpu();
    let vectors = include_str!("../../src/exec_ext/compute/learned/golden.json");
    let JsonValue::Array(cases) = parse(vectors).unwrap() else {
        panic!("invalid vectors")
    };
    assert_eq!(cases.len(), 5);
    let mut max_error = 0.0_f32;
    for case in cases {
        let JsonValue::Object(fields) = case else {
            panic!("invalid case")
        };
        let rect = rect(&fields);
        let pilot = block_on(AreaPilot::prepare(&gpu, rect, AreaPilotPackage::embedded())).unwrap();
        let mut slot = raa_graph::learned::AreaPilotSlot::default();
        slot.stage(pilot);
        assert!(slot.active().is_none());
        let pilot = slot.begin_real_frame().unwrap();
        let table = raa_pipeline::ResourceTable::default();
        pilot.publish(&table, "area");
        assert_eq!(
            table.buffer("area", "weights").unwrap().1,
            pilot.memory().weights
        );
        assert_eq!(table.buffer("area", "output").unwrap().0, *pilot.output());
        let input = numbers(&fields, "input");
        let expected = numbers(&fields, "output");
        assert!(pilot.upload(&input[1..]).is_err());
        let (error, outside) = compare(&gpu, pilot, &input, &expected);
        max_error = max_error.max(error);
        assert_eq!(
            outside, 0,
            "rect {:?}, max absolute error {error}",
            rect.size
        );
    }
    eprintln!("adapter={:?} max_abs_error={max_error}", gpu.summary());
}

/// RAA_PILOT_FIT_DIR points at the directory written by train-area-pilot.py.
#[test]
#[ignore = "requires Windows hardware and a fitted capture package"]
fn fitted_capture_matches_trainer() {
    let root = std::path::PathBuf::from(std::env::var_os("RAA_PILOT_FIT_DIR").unwrap());
    let manifest = std::fs::read_to_string(root.join("manifest.json")).unwrap();
    let weights = std::fs::read(root.join("weights.fp32le")).unwrap();
    let vectors = std::fs::read_to_string(root.join("trainer-vectors.json")).unwrap();
    let JsonValue::Array(cases) = parse(&vectors).unwrap() else {
        panic!("invalid vectors")
    };
    assert!(!cases.is_empty());
    let gpu = gpu();
    eprintln!("fitted adapter={:?}", gpu.summary());
    let mut failed = 0;
    for case in cases {
        let JsonValue::Object(fields) = case else {
            panic!("invalid case")
        };
        let name = &fields.iter().find(|(key, _)| key == "name").unwrap().1;
        let package = block_on(AreaPilotPackage::load(async {
            Ok((manifest.clone(), weights.clone()))
        }))
        .unwrap();
        let pilot = block_on(AreaPilot::prepare(&gpu, rect(&fields), package)).unwrap();
        let (max_error, outside) = compare(
            &gpu,
            &pilot,
            &numbers(&fields, "input"),
            &numbers(&fields, "output"),
        );
        eprintln!("frame={name:?} max_abs_error={max_error} lanes_outside_tolerance={outside}");
        failed += outside;
    }
    assert_eq!(failed, 0);
}

fn p95(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[(values.len() * 95).div_ceil(100) - 1]
}

/// Opt in with cargo test -p raa-graph --test compute_pass benchmark_fp32 -- --ignored --nocapture
#[test]
#[ignore = "960x540 adapter timing benchmark"]
fn benchmark_fp32() {
    let gpu = gpu();
    let device = gpu.device();
    let features = gpu.adapter().features();
    let limits = device.limits();
    let format = gpu
        .adapter()
        .get_texture_format_features(wgpu::TextureFormat::R32Float);
    eprintln!(
        "adapter={:?} hardware={} adapter_f16={} timestamp={} r32float={:?} storage_limit={} max_buffer={} max_texture_2d={}",
        gpu.summary(),
        gpu.summary().is_hardware(),
        features.contains(wgpu::Features::SHADER_F16),
        gpu.features().contains(wgpu::Features::TIMESTAMP_QUERY),
        format.allowed_usages,
        limits.max_storage_buffer_binding_size,
        limits.max_buffer_size,
        limits.max_texture_dimension_2d
    );
    let rect = AreaPilotRect::full(960, 540);
    let start = Instant::now();
    let pilot = block_on(AreaPilot::prepare(&gpu, rect, AreaPilotPackage::embedded())).unwrap();
    gpu.queue().submit([]);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let cold_ms = start.elapsed().as_secs_f64() * 1000.0;
    let samples: Vec<f32> =
        [0.5, 1.0, 0.25, 1.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0, 0.0].repeat(960 * 540);
    let memory = pilot.memory();
    assert_eq!(samples.len() * 4, memory.io as usize);
    let timers = gpu
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY)
        .then(|| {
            let query = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("area pilot elapsed"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            });
            let resolve = buffer(
                &gpu,
                16,
                wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            );
            let staging = buffer(
                &gpu,
                16,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            );
            (query, resolve, staging)
        });
    let n: usize = std::env::var("RAA_PILOT_SAMPLES")
        .ok()
        .map(|v| v.parse().unwrap())
        .unwrap_or(20);
    assert!(n > 0);
    eprintln!(
        "960x540 FP32 cold_prepare_ms={cold_ms:.3} resident_bytes={} explicit_gpu_buffer_peak_bytes={} host_input_bytes={} host_pack_bytes={} queue_upload_bytes/frame={} cold_upload_bytes={} channels={CHANNELS}",
        memory.resident_bytes(),
        memory.resident_bytes() + if timers.is_some() { 32 } else { 0 },
        memory.io,
        memory.io,
        memory.io,
        memory.weights + 24
    );
    for (name, dispatches) in [
        ("single", 6),
        ("four-independent", 24),
        ("four-shared-identical-input", 9),
    ] {
        let mut elapsed = Vec::new();
        let mut timestamp = Vec::new();
        for iteration in 0..=n {
            let start = Instant::now();
            pilot.upload(&samples).unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            let query = timers.as_ref().map(|(q, _, _)| q);
            if dispatches == 6 {
                pilot.record_real_frame(&mut encoder, query);
            } else {
                pilot.record_four_light_benchmark(&mut encoder, dispatches == 9, query);
            }
            if let Some((q, resolve, staging)) = &timers {
                encoder.resolve_query_set(q, 0..2, resolve, 0);
                encoder.copy_buffer_to_buffer(resolve, 0, staging, 0, 16);
            }
            gpu.queue().submit([encoder.finish()]);
            if let Some((_, _, staging)) = &timers {
                let bytes = read(&gpu, staging);
                let ticks = |i| u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
                if iteration > 0 {
                    timestamp.push(
                        (ticks(8) - ticks(0)) as f64
                            * f64::from(gpu.queue().get_timestamp_period())
                            / 1e6,
                    );
                }
            } else {
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            }
            if iteration > 0 {
                elapsed.push(start.elapsed().as_secs_f64() * 1000.0);
            }
        }
        eprintln!(
            "{name} n={n} dispatches={dispatches} wall_p95_ms={:.3} gpu_p95_ms={:?}",
            p95(&mut elapsed),
            (!timestamp.is_empty()).then(|| p95(&mut timestamp))
        );
    }
}
