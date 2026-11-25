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
        s.base_color = textureLoad(raa_vg_albedo, vec2<i32>(frag.xy), 0).rgb;
    }
    var out: RaaVgShadeTargets;
    out.color = raa_opaque_lighting(in, s);
    out.frag_depth = raa_vg_surface.depth;
    if (!raa_vg_surface.covered) {
        discard;
    }
    return out;
}
