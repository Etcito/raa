// lightingOutput "layers-interlaced": the layers of "layers" every frame, and the signals of
// raa_opaque_signals only at the phase pixel of each 2x2 block on an R frame (one pixel in 8
// on average). FrameGpu.temporal.w (raa-pipeline LightingInterlace::word): bit 0 mode, bit 1 R
// frame (real-frame counter n even), bits 2 and 3 the phase (px, py) of lighting frame n / 2,
// cycling (0,0), (1,1), (1,0), (0,1). A fragment samples when bit 1 is set and
// (x & 1, y & 1) == (px, py) for its integer pixel (x, y).
//
// Sample:    color.a = s, layerKd.a = a, layerE = (e, w)  (exactly raa_opaque_signals).
// No sample: color.a = -1, layerKd.a = -1, layerE = (0, 0, 0, -1); the lookups are skipped.
// Signal rates (word bits 6..9, raa-pipeline InterlaceSignals): s at every pixel of an R frame
// (bit 6) and of a G frame (bit 7, cascade maps of the R frame); E at the phase pixel of a G
// frame (bit 8, the G frame's phase is in bits 2..3) or at every pixel of every frame (bit 9).
// AO stays at the phase pixel of R frames.
// The branch is non-uniform: everything raa_opaque_signals reaches is derivative-free
// (textureLoad, textureSampleLevel, textureSampleCompareLevel, storage reads).
struct RaaForwardLayersInterlaced {
    @location(0) d0: vec4<f32>,
    @location(1) kd: vec4<f32>,
    @location(2) s0: vec4<f32>,
    @location(3) em: vec4<f32>,
    @location(4) e: vec4<f32>,
}

// Whether the fragment at `frag` (pixel centre) is the phase pixel of its 2x2 block.
fn raa_interlace_phase_pixel(frag: vec2<f32>) -> bool {
    let word = raa_frame.temporal.w;
    let pixel = vec2<u32>(frag) & vec2<u32>(1u);
    let phase = vec2<u32>((word >> 2u) & 1u, (word >> 3u) & 1u);
    return all(pixel == phase);
}

@fragment
fn fs_main(v: RaaVaryings, @builtin(front_facing) front_facing: bool) -> RaaForwardLayersInterlaced {
    let in = raa_pass_material_input(v, front_facing);
    let s = raa_pass_material(in, v.instance_index);
    let l = raa_opaque_layers(in, s);
    let word = raa_frame.temporal.w;
    let real = (word & 2u) != 0u;
    let phase_pixel = raa_interlace_phase_pixel(in.frag_coord.xy);
    let shadow_all = (real && (word & 64u) != 0u) || (word & 128u) != 0u;
    let probe_all = (word & 512u) != 0u;
    let probe_phase = phase_pixel && (real || (word & 256u) != 0u);
    var g: RaaOpaqueSignals;
    g.shadow = -1.0;
    g.ao = -1.0;
    g.e = vec3<f32>(0.0);
    g.w = -1.0;
    if real && phase_pixel {
        g = raa_opaque_signals(in, s);
    } else {
        if shadow_all {
            g.shadow = raa_opaque_shadow(in, s);
        }
        if probe_all || probe_phase {
            let e = raa_opaque_probe(in, s);
            g.e = e.rgb;
            g.w = e.w;
        }
    }
    var out: RaaForwardLayersInterlaced;
    out.d0 = vec4<f32>(l.d0, g.shadow);
    out.kd = vec4<f32>(l.kd, g.ao);
    out.s0 = vec4<f32>(l.s0, l.sky);
    out.em = vec4<f32>(l.em, 1.0);
    out.e = vec4<f32>(g.e, g.w);
    return out;
}
