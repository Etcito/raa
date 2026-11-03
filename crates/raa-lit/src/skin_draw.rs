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
    posed_entry_points(source).unwrap_or_else(|| source.into())
}

/// The instanced skinning variant of pass entry points `source` as the composer emitted
/// them (after any composer rewrite such as tint mode `graph`), or `None` when `source`
/// has no `vs_main` taking `v0: RaaVertex0` and `v1: RaaVertex1` to skin.
pub fn posed_entry_points(source: &str) -> Option<String> {
    if !source.contains("    v0: RaaVertex0,") || !source.contains("    v1: RaaVertex1,") {
        return None;
    }
    let mut out = source
        .replace("    v0: RaaVertex0,", "    raa_rest0: RaaVertex0,")
        .replace("    v1: RaaVertex1,", POSED_PARAMS)
        .replace("v1.tangent.xyz", "raa_posed_tangent");
    // The prologue goes at the top of vs_main's body (after `) -> ... {`).
    let start = out.find("fn vs_main(")?;
    let rel = out[start..].find("{\n")?;
    out.insert_str(start + rel + 2, POSED_PROLOGUE);
    Some(format!("{POSED_DECLS}{out}"))
}

/// Marker the composer writes before the pass entry points (its last section).
const PASS_SECTION: &str = "\n// raa section: pass ";

/// The posed variant of a whole composed module: the instanced skinning rewrite applied to
/// its pass section, whatever the composer changed in it (tint mode `graph` rewrites the
/// tint call there). `None` when the module has no pass section the rewrite applies to;
/// the caller refuses, since the rest pipeline with posed bindings draws the rest pose.
pub fn posed_module(wgsl: &str) -> Option<String> {
    let at = wgsl.rfind(PASS_SECTION)?;
    let (head, pass) = wgsl.split_at(at);
    Some(format!("{head}{}", posed_entry_points(pass)?))
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
