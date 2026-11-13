// RAA virtualized-geometry shade pass (S32, contributor of forward): one fullscreen draw
// per material function over the vg.resolve surface. It evaluates the S11 opaque final
// colour on the rebuilt MaterialInput and writes it with the resolved standard-Z depth, so
// forward content that follows is depth-tested against it; a pixel of another material
// function or without a virtualized surface is discarded after every slot call.

struct RaaVgShadeTargets {
    // S11 HDR colour.
    @location(0) color: vec4<f32>,
    // The resolve output's standard-Z depth.
    @builtin(frag_depth) frag_depth: f32,
}

// Fullscreen triangle without a vertex buffer: (-1, -1), (3, -1), (-1, 3) at z 0.
@vertex
fn vs_main(@builtin(vertex_index) raa_vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((raa_vertex_index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(raa_vertex_index & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

// material_main followed by the instance tint (S10). mat_instance_tint and
// mat_instance_tint_set are set first so material graph code can read them (S23).
fn raa_pass_material(in: MaterialInput, instance: u32) -> MaterialOutput {
    let x = raa_extras[instance];
    raa_set_instance_tint(x);
    return raa_apply_tint(material_main(in), x);
}

// Flat model or UNLIT flag: the fragment is base + emissive (wgsl/pass_forward.wgsl).
fn raa_pass_unlit(record: SurfaceMaterialGpu) -> bool {
    return record.header.x == RAA_MODEL_FLAT || (record.header.z & RAA_FLAG_UNLIT) != 0u;
}

// Normal-incidence reflectance: dielectric from the ior (0.04 at ior 1.5), base colour for
// metals (wgsl/pass_forward.wgsl).
fn raa_pass_f0(s: MaterialOutput) -> vec3<f32> {
    let ior = clamp(s.ior, 1.0, 3.0);
    let r = (ior - 1.0) / (ior + 1.0);
    return mix(vec3<f32>(r * r), s.base_color, clamp(s.metallic, 0.0, 1.0));
}

fn raa_pass_shade_ctx(s: MaterialOutput, in: MaterialInput, ao: f32, shadow: f32) -> ShadeCtx {
    var c: ShadeCtx;
    c.material_flags = in.record.header.z;
    c.position = in.position;
    c.view_dir = in.view_dir;
    c.frag = in.frag_coord;
    c.n_dot_v = max(dot(s.normal, in.view_dir), 1e-4);
    c.ao = ao;
    c.shadow = shadow;
    c.f0 = raa_pass_f0(s);
    return c;
}

// S11 final colour: emissive, sun, ambient/environment, indirect and the shared tiled local
// lights. A debug view replaces the base colour by the resolved albedo (S32). Every slot and
// helper runs before the unlit select and the discard, so the calls stay in uniform control
// flow; the Composer attaches LOCAL_LIGHTING_WGSL ahead of this pass.
@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> RaaVgShadeTargets {
    let raa_vg_surface = raa_vg_material_input(frag);
    let in = raa_vg_surface.input;
    var s = raa_pass_material(in, raa_vg_surface.slot);
    if (raa_vg_shade.debug_view != 0u) {
        s.base_color = raa_srgb_to_linear(textureLoad(raa_vg_albedo, vec2<i32>(frag.xy), 0).rgb);
    }
    let shadow = raa_slot_shadow(in.position, in.normal, in.frag_coord);
    let ao = raa_slot_ao(in.frag_coord);
    let c = raa_pass_shade_ctx(s, in, ao, shadow);
    let lit = s.emissive
        + raa_slot_direct(s, c, -raa_frame.sun_dir.xyz,
            raa_directional_radiance(raa_frame.sun_color.rgb) * shadow)
        + raa_slot_ambient(s, c) * min(s.ao, ao)
        + raa_slot_indirect(s, c)
        + raa_local_lighting(s, c);
    let color = select(lit, s.base_color + s.emissive, raa_pass_unlit(in.record));
    var out: RaaVgShadeTargets;
    out.color = vec4<f32>(color, 1.0);
    out.frag_depth = raa_vg_surface.depth;
    if (!raa_vg_surface.covered) {
        discard;
    }
    return out;
}
