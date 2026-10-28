//! Lit-owned adaptation of pass entry points; shared S10 stream 1 stays unchanged.
use raa_shader::PassKind;

pub(crate) const POSED: u32 = 1 << 31;

/// Adds auxiliary tangent xyz to the selected pass, preserving stream-1 handedness.
/// Rest variants use the unmodified entry point.
pub fn entry_points(pass: PassKind, posed: bool) -> String {
    let source = match pass {
        PassKind::Prepass => raa_shader::PASS_PREPASS_WGSL,
        PassKind::Forward => raa_shader::PASS_FORWARD_WGSL,
        PassKind::Shadow => raa_shader::PASS_SHADOW_WGSL,
        PassKind::Transparent => raa_shader::PASS_TRANSPARENT_WGSL,
        PassKind::FurShell => raa_shader::PASS_FURSHELL_WGSL,
    };
    if !posed {
        return source.into();
    }
    source
        .replace(
            "    v1: RaaVertex1,",
            "    v1: RaaVertex1,\n    @location(13) raa_posed_tangent: vec3<f32>,",
        )
        .replace("v1.tangent.xyz", "raa_posed_tangent")
}


/// Match culling bounds to the selected geometry; unknown posed bounds stay visible.
pub fn instance_bounds(
    rest: raa_math::Aabb,
    posed: bool,
    joints: Option<&[f32]>,
) -> Option<raa_math::Aabb> {
    if posed {
        joints.and_then(|joints| crate::skin_bounds::posed_bounds(rest, joints))
    } else {
        Some(rest)
    }
}
