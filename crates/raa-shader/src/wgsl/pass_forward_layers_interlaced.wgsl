// lightingOutput "layers-interlaced": the layers of "layers" every frame, and the signals of
// raa_opaque_signals only at the phase pixel of each 2x2 block on an R frame (one pixel in 8
// on average). FrameGpu.temporal.w (raa-pipeline LightingInterlace::word): bit 0 mode, bit 1 R
// frame (real-frame counter n even), bits 2 and 3 the phase (px, py) of lighting frame n / 2,
// cycling (0,0), (1,1), (1,0), (0,1). A fragment samples when bit 1 is set and
// (x & 1, y & 1) == (px, py) for its integer pixel (x, y).
//
// Sample:    color.a = s, layerKd.a = a, layerE = (e, w)  (exactly raa_opaque_signals).
// No sample: color.a = -1, layerKd.a = -1, layerE = (0, 0, 0, -1); the lookups are skipped.
// The branch is non-uniform: everything raa_opaque_signals reaches is derivative-free
// (textureLoad, textureSampleLevel, textureSampleCompareLevel, storage reads).
struct RaaForwardLayersInterlaced {
    @location(0) d0: vec4<f32>,
    @location(1) kd: vec4<f32>,
    @location(2) s0: vec4<f32>,
    @location(3) em: vec4<f32>,
    @location(4) e: vec4<f32>,
}

// Whether the fragment at `frag` (pixel centre) evaluates the signals this frame.
fn raa_interlace_sample(frag: vec2<f32>) -> bool {
    let word = raa_frame.temporal.w;
    let pixel = vec2<u32>(frag) & vec2<u32>(1u);
    let phase = vec2<u32>((word >> 2u) & 1u, (word >> 3u) & 1u);
    return (word & 2u) != 0u && all(pixel == phase);
}

@fragment
fn fs_main(v: RaaVaryings, @builtin(front_facing) front_facing: bool) -> RaaForwardLayersInterlaced {
    let in = raa_pass_material_input(v, front_facing);
    let s = raa_pass_material(in, v.instance_index);
    let l = raa_opaque_layers(in, s);
    var g: RaaOpaqueSignals;
    g.shadow = -1.0;
    g.ao = -1.0;
    g.e = vec3<f32>(0.0);
    g.w = -1.0;
    if raa_interlace_sample(in.frag_coord.xy) {
        g = raa_opaque_signals(in, s);
    }
    var out: RaaForwardLayersInterlaced;
    out.d0 = vec4<f32>(l.d0, g.shadow);
    out.kd = vec4<f32>(l.kd, g.ao);
    out.s0 = vec4<f32>(l.s0, 1.0);
    out.em = vec4<f32>(l.em, 1.0);
    out.e = vec4<f32>(g.e, g.w);
    return out;
}
