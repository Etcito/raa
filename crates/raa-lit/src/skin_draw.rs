//! Lit-owned adaptation of pass entry points for instanced skinning (S18).
//!
//! A posed variant skins in the vertex stage, by the formula of the skin compute kernel:
//! the blended matrix `sum(w_k * palette[base + j_k])` transforms the rest position (with
//! translation), the rest normal and the rest tangent xyz (upper 3x3, normalized once,
//! collapsed directions become zero). Per vertex, the joints (`@location(13)`, u16x4),
//! weights (`@location(14)`) and rest tangent (`@location(11)`, handedness in w) come
//! from the mesh's 64-byte rest skin records bound in vertex slot 3; per instance, the
//! palette base comes from the S1 record bytes 120..124 (`@location(15)`); the palette
//! itself is group 3, binding 0. Rest variants use the unmodified entry point.
use raa_shader::PassKind;

pub(crate) const POSED: u32 = 1 << 31;

/// Group of the joint palette in posed variants.
pub const PALETTE_GROUP: u32 = 3;

const POSED_DECLS: &str = "
// Instanced skinning (raa-lit skin_draw): joint palette of every posed instance.
@group(3) @binding(0) var<storage, read> raa_skin_palette: array<mat4x4<f32>>;

fn raa_skin_direction(v: vec3<f32>) -> vec3<f32> {
    let length = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if length > 0.0 { return v / length; }
    return vec3<f32>(0.0);
}
";

const POSED_PARAMS: &str = "    v1: RaaVertex1,
    @location(13) raa_skin_joints: vec4<u32>,
    @location(14) raa_skin_weights: vec4<f32>,
    @location(15) raa_skin_base: u32,";

const POSED_PROLOGUE: &str = "
    var raa_skin_m = mat4x4<f32>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    for (var k = 0u; k < 4u; k++) {
        let raa_joint = raa_skin_base + raa_skin_joints[k];
        let raa_w = raa_skin_weights[k];
        for (var c = 0u; c < 4u; c++) { raa_skin_m[c] += raa_w * raa_skin_palette[raa_joint][c]; }
    }
    let raa_p = raa_rest0.position;
    let raa_n = raa_rest0.normal;
    let raa_t = v1.tangent.xyz;
    var v0 = raa_rest0;
    v0.position = raa_skin_m[0].xyz * raa_p.x + raa_skin_m[1].xyz * raa_p.y
        + raa_skin_m[2].xyz * raa_p.z + raa_skin_m[3].xyz;
    v0.normal = raa_skin_direction(raa_skin_m[0].xyz * raa_n.x + raa_skin_m[1].xyz * raa_n.y
        + raa_skin_m[2].xyz * raa_n.z);
    let raa_posed_tangent = raa_skin_direction(raa_skin_m[0].xyz * raa_t.x
        + raa_skin_m[1].xyz * raa_t.y + raa_skin_m[2].xyz * raa_t.z);
";

/// The entry points of `pass`; `posed` selects the instanced skinning variant, which keeps
/// the stream-1 handedness (`v1.tangent.w`), uv and colour.
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
    let mut out = source
        .replace("    v0: RaaVertex0,", "    raa_rest0: RaaVertex0,")
        .replace("    v1: RaaVertex1,", POSED_PARAMS)
        .replace("v1.tangent.xyz", "raa_posed_tangent");
    // The prologue goes at the top of vs_main's body (after `) -> ... {`).
    if let Some(start) = out.find("fn vs_main(")
        && let Some(rel) = out[start..].find("{\n")
    {
        let at = start + rel + 2;
        out.insert_str(at, POSED_PROLOGUE);
    }
    format!("{POSED_DECLS}{out}")
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
