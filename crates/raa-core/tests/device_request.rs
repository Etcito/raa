//! DeviceRequest: optional features are intersected with the adapter's, adapter limits
//! are requested on demand. Skips with the SKIP line when no software adapter is available.

#![cfg(not(target_arch = "wasm32"))]

use raa_core::testing::{block_on, skip};
use raa_core::{DeviceRequest, GpuContext, GpuOptions};

#[test]
fn device_request_intersects_adapter_features() {
    let requested = wgpu::Features::FLOAT32_FILTERABLE | wgpu::Features::RG11B10UFLOAT_RENDERABLE;
    let req = DeviceRequest {
        optional_features: requested,
        adapter_limits: true,
    };
    let opts = GpuOptions {
        allow_software: true,
        ..GpuOptions::default()
    };
    let gpu = match block_on(GpuContext::headless_with(&opts, &req)) {
        Ok(gpu) => gpu,
        Err(e) => {
            skip("device_request_intersects_adapter_features", &e.to_string());
            return;
        }
    };
    let adapter_features = gpu.adapter().features();
    let enabled = gpu.features();
    // Only the requested optional features the adapter has may be enabled on top of core.
    assert!(
        (enabled & requested)
            .difference(adapter_features)
            .is_empty(),
        "enabled {enabled:?} not within adapter {adapter_features:?}"
    );
    assert_eq!(enabled & requested, requested & adapter_features);
    assert!(
        enabled.difference(requested & adapter_features).is_empty(),
        "unrequested features enabled: {:?}",
        enabled.difference(requested & adapter_features)
    );
    assert_eq!(gpu.limits(), gpu.adapter().limits());
    eprintln!(
        "device_request: adapter {} ({}), enabled {enabled:?}",
        gpu.summary().device,
        gpu.summary().backend
    );
    // Same as tests/gpu_readback.rs: tearing down an instance that enumerated Mesa's D3D12 GL
    // adapter (WSL) segfaults inside the driver; leak the context at the end of the test.
    std::mem::forget(gpu);
}
