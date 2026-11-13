//! S32 virtualized-geometry surface declarations (raa-shader-vg): the group-2 outputs of
//! `vg.resolve` and the `MaterialInput` of a virtualized pixel. [`VG_SURFACE_WGSL`] declares
//! the five group-2 bindings (`raa_vg_ids`, `raa_vg_normal`, `raa_vg_uv_motion`,
//! `raa_vg_albedo`, `raa_vg_shade: VgShadeGpu`) and `raa_vg_material_input`, which reads the
//! surface with `textureLoad` only (never sampling or derivatives) and rebuilds the S10
//! `MaterialInput` so an unchanged material function can shade it. Both VG pass kinds
//! compose this module after [`DEPTH_WGSL`](crate::DEPTH_WGSL).

/// S32 group-2 surface declarations: `VgShadeGpu` (16 B), `RaaVgSurface`,
/// `raa_vg_default_tangent`, the five group-2 bindings and
/// `fn raa_vg_material_input(frag: vec4<f32>) -> RaaVgSurface`. A pixel with no virtualized
/// surface or another material function reports `covered = false` with a zero input and
/// reads no record. Depends on [`PRELUDE_WGSL`](crate::PRELUDE_WGSL) for `FrameGpu`,
/// `MaterialInput`, `raa_instance`, `raa_extras`, `raa_materials` and `RAA_FLAG_DOUBLE_SIDED`.
pub const VG_SURFACE_WGSL: &str = r#"// RAA VG surface (S32): group-2 outputs of vg.resolve as a MaterialInput.
struct VgShadeGpu {
    material_fn: u32,
    debug_view: u32,
    pad0: u32,
    pad1: u32,
}

struct RaaVgSurface {
    covered: bool,
    input: MaterialInput,
    slot: u32,
    depth: f32,
    motion: vec2<f32>,
    debug: u32,
}

@group(2) @binding(0) var raa_vg_ids: texture_2d<u32>;
@group(2) @binding(1) var raa_vg_normal: texture_2d<f32>;
@group(2) @binding(2) var raa_vg_uv_motion: texture_2d<f32>;
@group(2) @binding(3) var raa_vg_albedo: texture_2d<f32>;
@group(2) @binding(4) var<uniform> raa_vg_shade: VgShadeGpu;

// Duff et al. 2017 ("Building an Orthonormal Basis, Revisited"), w = 1; a zero or
// non-finite normal reads as +Z (raa-mesh `default_tangent`).
fn raa_vg_default_tangent(n: vec3<f32>) -> vec4<f32> {
    let raa_vg_len = length(n);
    var raa_vg_n = vec3<f32>(0.0, 0.0, 1.0);
    if (raa_vg_len > 0.0 && raa_vg_len < 1.0e30) {
        raa_vg_n = n / raa_vg_len;
    }
    let raa_vg_sign = select(-1.0, 1.0, raa_vg_n.z >= 0.0);
    let raa_vg_a = -1.0 / (raa_vg_sign + raa_vg_n.z);
    let raa_vg_b = raa_vg_n.x * raa_vg_n.y * raa_vg_a;
    return vec4<f32>(
        1.0 + raa_vg_sign * raa_vg_n.x * raa_vg_n.x * raa_vg_a,
        raa_vg_sign * raa_vg_b,
        -raa_vg_sign * raa_vg_n.x,
        1.0,
    );
}

fn raa_vg_material_input(frag: vec4<f32>) -> RaaVgSurface {
    var raa_vg_out: RaaVgSurface;
    let raa_vg_pixel = vec2<i32>(frag.xy);
    let raa_vg_ids_word = textureLoad(raa_vg_ids, raa_vg_pixel, 0);
    if (raa_vg_ids_word.x == 0u || raa_vg_ids_word.y != raa_vg_shade.material_fn) {
        var raa_vg_zero: MaterialInput;
        raa_vg_out.covered = false;
        raa_vg_out.input = raa_vg_zero;
        raa_vg_out.slot = 0u;
        raa_vg_out.depth = 1.0;
        raa_vg_out.motion = vec2<f32>(0.0);
        raa_vg_out.debug = 0u;
        return raa_vg_out;
    }
    let raa_vg_slot = raa_vg_ids_word.x - 1u;
    let raa_vg_depth = bitcast<f32>(raa_vg_ids_word.w);
    let raa_vg_uv_motion_word = textureLoad(raa_vg_uv_motion, raa_vg_pixel, 0);
    let raa_vg_normal_word = textureLoad(raa_vg_normal, raa_vg_pixel, 0);
    let raa_vg_ndc = vec2<f32>(
        frag.x * raa_frame.viewport.z * 2.0 - 1.0,
        1.0 - frag.y * raa_frame.viewport.w * 2.0,
    );
    let raa_vg_view = raa_frame.inv_proj * vec4<f32>(raa_vg_ndc, raa_vg_depth, 1.0);
    let raa_vg_position =
        (raa_frame.inv_view * vec4<f32>(raa_vg_view.xyz / raa_vg_view.w, 1.0)).xyz;
    let raa_vg_inst = raa_instance(raa_vg_slot);
    let raa_vg_record = raa_materials[raa_vg_inst.material_index];
    // The normal matrix holds the inverse-transpose columns, so transposing it inverts the
    // model's upper 3x3.
    let raa_vg_basis = mat3x3<f32>(
        raa_vg_inst.normal0.xyz,
        raa_vg_inst.normal1.xyz,
        raa_vg_inst.normal2.xyz,
    );
    let raa_vg_local_position =
        transpose(raa_vg_basis) * (raa_vg_position - raa_vg_inst.model3.xyz);
    var raa_vg_normal = normalize(raa_vg_normal_word.xyz);
    let raa_vg_front_facing = raa_vg_normal_word.w > 0.0;
    if (!raa_vg_front_facing && (raa_vg_record.header.z & RAA_FLAG_DOUBLE_SIDED) != 0u) {
        raa_vg_normal = -raa_vg_normal;
    }
    var raa_vg_input: MaterialInput;
    raa_vg_input.position = raa_vg_position;
    raa_vg_input.local_position = raa_vg_local_position;
    raa_vg_input.normal = raa_vg_normal;
    raa_vg_input.tangent = raa_vg_default_tangent(raa_vg_normal);
    raa_vg_input.uv = raa_vg_uv_motion_word.xy;
    raa_vg_input.color = vec4<f32>(1.0);
    raa_vg_input.view_dir = normalize(raa_frame.camera_pos.xyz - raa_vg_position);
    raa_vg_input.time = raa_frame.camera_pos.w;
    raa_vg_input.front_facing = raa_vg_front_facing;
    raa_vg_input.frag_coord = frag;
    raa_vg_input.instance_params = raa_extras[raa_vg_slot].params;
    raa_vg_input.record = raa_vg_record;
    raa_vg_out.covered = true;
    raa_vg_out.input = raa_vg_input;
    raa_vg_out.slot = raa_vg_slot;
    raa_vg_out.depth = raa_vg_depth;
    raa_vg_out.motion = raa_vg_uv_motion_word.zw;
    raa_vg_out.debug = raa_vg_ids_word.z;
    return raa_vg_out;
}
"#;

#[cfg(test)]
mod tests {
    use crate::{
        CLOUD_SHADOW_WGSL, Composer, DEPTH_WGSL, LIB_WGSL, MaterialFnId, MaterialFnRegistry,
        PRELUDE_WGSL, VG_SURFACE_WGSL,
    };

    const PROBE: &str = "
@fragment
fn probe(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let raa_vg_surface = raa_vg_material_input(frag);
    let o = material_main(raa_vg_surface.input);
    return vec4<f32>(o.base_color * select(0.0, 1.0, raa_vg_surface.covered), o.opacity);
}";

    fn module() -> String {
        let reg = MaterialFnRegistry::new();
        let standard = &reg.get(MaterialFnId::STANDARD).expect("raa.standard").wgsl;
        format!(
            "{PRELUDE_WGSL}\n{LIB_WGSL}\n{CLOUD_SHADOW_WGSL}\n{standard}\n\
             {DEPTH_WGSL}\n{VG_SURFACE_WGSL}\n{PROBE}"
        )
    }

    #[test]
    fn surface_declarations_validate_with_the_standard_material() {
        if let Err(e) = Composer::validate(&module()) {
            panic!("{}: {}", e.code, e.message);
        }
    }

    #[test]
    fn group_two_holds_the_four_textures_and_the_shade_record() {
        let source = module();
        let m = naga::front::wgsl::parse_str(&source).expect("parse");
        let global = |name: &str| {
            m.global_variables
                .iter()
                .find(|(_, v)| v.name.as_deref() == Some(name))
                .map(|(_, v)| v)
                .unwrap_or_else(|| panic!("{name} declared"))
        };
        let sampled = |kind: naga::ScalarKind| naga::TypeInner::Image {
            dim: naga::ImageDimension::D2,
            arrayed: false,
            class: naga::ImageClass::Sampled { kind, multi: false },
        };
        let mut seen = Vec::new();
        for (index, (name, kind)) in [
            ("raa_vg_ids", naga::ScalarKind::Uint),
            ("raa_vg_normal", naga::ScalarKind::Float),
            ("raa_vg_uv_motion", naga::ScalarKind::Float),
            ("raa_vg_albedo", naga::ScalarKind::Float),
        ]
        .into_iter()
        .enumerate()
        {
            let var = global(name);
            assert_eq!(var.space, naga::AddressSpace::Handle);
            assert_eq!(
                var.binding,
                Some(naga::ResourceBinding {
                    group: 2,
                    binding: index as u32,
                }),
                "{name}"
            );
            assert_eq!(m.types[var.ty].inner, sampled(kind), "{name}");
            seen.push(index as u32);
        }
        let shade = global("raa_vg_shade");
        assert_eq!(shade.space, naga::AddressSpace::Uniform);
        assert_eq!(
            shade.binding,
            Some(naga::ResourceBinding {
                group: 2,
                binding: 4,
            })
        );
        let (handle, ty) = m
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("VgShadeGpu"))
            .expect("VgShadeGpu");
        assert_eq!(shade.ty, handle);
        let naga::TypeInner::Struct { members, span } = &ty.inner else {
            panic!("VgShadeGpu is a struct");
        };
        assert_eq!(*span, 16);
        assert_eq!(
            members
                .iter()
                .map(|m| m.name.as_deref())
                .collect::<Vec<_>>(),
            [
                Some("material_fn"),
                Some("debug_view"),
                Some("pad0"),
                Some("pad1")
            ]
        );
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(m.to_ctx()).expect("layout");
        assert_eq!(layouter[handle].size, 16);
        seen.push(4);
        let mut all: Vec<u32> = m
            .global_variables
            .iter()
            .filter_map(|(_, v)| v.binding.as_ref())
            .filter(|b| b.group == 2)
            .map(|b| b.binding)
            .collect();
        all.sort_unstable();
        assert_eq!(all, seen);
    }
}
