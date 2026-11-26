// lightingOutput "layers": the opaque lighting as demodulated layers (opaque_layers.wgsl),
// without the sun shadow, screen AO or probe GI lookups. Location 0 is the HDR chain.
struct RaaForwardLayers {
    @location(0) d0: vec4<f32>,
    @location(1) kd: vec4<f32>,
    @location(2) s0: vec4<f32>,
    @location(3) em: vec4<f32>,
}

@fragment
fn fs_main(v: RaaVaryings, @builtin(front_facing) front_facing: bool) -> RaaForwardLayers {
    let in = raa_pass_material_input(v, front_facing);
    let s = raa_pass_material(in, v.instance_index);
    let l = raa_opaque_layers(in, s);
    var out: RaaForwardLayers;
    out.d0 = vec4<f32>(l.d0, 1.0);
    out.kd = vec4<f32>(l.kd, 1.0);
    out.s0 = vec4<f32>(l.s0, l.sky);
    out.em = vec4<f32>(l.em, 1.0);
    return out;
}
