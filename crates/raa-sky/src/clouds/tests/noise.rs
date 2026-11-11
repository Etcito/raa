//! Readiness of the one-time cloud noise generation.

use super::super::noise::{CloudNoise, NoiseGate};

#[test]
fn generation_stays_pending_until_its_submission_is_acknowledged() {
    let mut gate = NoiseGate::default();
    assert!(gate.record(3, 0));
    assert!(!gate.generated());
    assert!(
        gate.record(4, 0),
        "an unacknowledged generation records again"
    );
    gate.acknowledge(0, 4);
    assert!(gate.generated());
}

#[test]
fn an_unrelated_acknowledgement_does_not_make_noise_ready() {
    let mut gate = NoiseGate::default();
    gate.acknowledge(0, 3);
    assert!(gate.record(3, 0));
    gate.acknowledge(1, 3);
    gate.acknowledge(0, 2);
    assert!(!gate.generated());
    assert!(gate.record(4, 0));
    gate.acknowledge(0, 3);
    assert!(!gate.generated(), "a superseded invocation does not count");
}

#[test]
fn acknowledged_generation_is_never_recorded_again() {
    let mut gate = NoiseGate::default();
    assert!(gate.record(3, 1));
    gate.acknowledge(1, 3);
    assert!(gate.generated());
    for (invocation, view) in [(3, 1), (4, 0), (5, 2)] {
        assert!(!gate.record(invocation, view));
    }
    assert!(gate.generated());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn noise_generation_records_without_validation_errors() {
    let Some(gpu) = raa_core::testing::software_gpu() else {
        raa_core::testing::skip(
            "noise_generation_records_without_validation_errors",
            "software_gpu returned None; see the software_gpu line above",
        );
        return;
    };
    // The generation passes read no other input: noise is procedural and the
    // coverage passes read only the weather map generated earlier in the encoder.
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let mut noise = CloudNoise::new(&gpu).expect("cloud noise");
    let mut encoder = gpu.device().create_command_encoder(&Default::default());
    noise.record(&mut encoder, 1, 0);
    gpu.queue().submit([encoder.finish()]);
    noise.acknowledge(0, 1);
    assert!(noise.generated());
    let error = raa_core::testing::block_on(errors.pop());
    assert!(error.is_none(), "validation error: {error:?}");
}
