// lightingOutput "layers-signals": the layers of "layers" plus the signals s (location 0
// alpha), a (location 1 alpha), e and w (location 4) and the residual of the recomposition
// against raa_opaque_lighting (location 5). Location 0 is the HDR chain.
struct RaaForwardLayerSignals {
    @location(0) d0: vec4<f32>,
    @location(1) kd: vec4<f32>,
    @location(2) s0: vec4<f32>,
    @location(3) em: vec4<f32>,
    @location(4) e: vec4<f32>,
    @location(5) residual: vec4<f32>,
}

@fragment
fn fs_main(v: RaaVaryings, @builtin(front_facing) front_facing: bool) -> RaaForwardLayerSignals {
    let in = raa_pass_material_input(v, front_facing);
    let s = raa_pass_material(in, v.instance_index);
    let combined = raa_opaque_lighting(in, s).rgb;
    let l = raa_opaque_layers(in, s);
    let g = raa_opaque_signals(in, s);
    let recomposed = l.d0 * g.shadow + (l.kd * g.e + l.s0) * g.ao + l.em;
    var out: RaaForwardLayerSignals;
    out.d0 = vec4<f32>(l.d0, g.shadow);
    out.kd = vec4<f32>(l.kd, g.ao);
    out.s0 = vec4<f32>(l.s0, 1.0);
    out.em = vec4<f32>(l.em, 1.0);
    out.e = vec4<f32>(g.e, g.w);
    out.residual = vec4<f32>(combined - recomposed, 1.0);
    return out;
}
