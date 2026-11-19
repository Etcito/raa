// ReSTIR GI stage entry points for the stage graph (S58), composed after restir_gi_main.wgsl so
// that `raa_rgi_initial`, `raa_rgi_temporal_candidate`, `raa_rgi_refresh` and the merge and
// boiling filter of `raa_rgi_stage1` are reused unchanged. `raa_gi_restir_gi_initial` writes the
// one-candidate initial reservoir of each pixel, the stage's `reservoirs` output;
// `raa_gi_restir_gi_temporal` merges it with the previous frame's reservoirs and writes both the
// merged reservoir and its raw irradiance. RAA's own code.
//
// Group-2 bindings declared here: 16 the initial reservoirs (read_write; the initial stage's
// output and the temporal stage's input), 17 the Rgba16Float raw irradiance the temporal stage
// writes.

@group(2) @binding(16) var<storage, read_write> raa_rgi_initial_out: array<vec4<u32>>;
@group(2) @binding(17) var raa_rgi_temporal_raw: texture_storage_2d<rgba16float, write>;

fn raa_rgi_initial_load(pixel: vec2<u32>) -> RaaRgiReservoir {
    let i = raa_rgi_index(pixel);
    return raa_rgi_unpack(raa_rgi_initial_out[i], raa_rgi_initial_out[i + 1u]);
}

fn raa_rgi_initial_store(pixel: vec2<u32>, r: RaaRgiReservoir) {
    let i = raa_rgi_index(pixel);
    raa_rgi_initial_out[i] = raa_rgi_pack_a(r);
    raa_rgi_initial_out[i + 1u] = raa_rgi_pack_b(r);
}

// The temporal merge of `raa_rgi_stage1` with `c` the initial stage's reservoir of the pixel: the
// history's M is clamped to m_cap times the current M and both candidates are merged as Ouyang et
// al. 2021, Algorithm 3.
fn raa_rgi_stage1_merge(pixel: vec2<u32>, s: RaaRgiSurface, c: RaaRgiReservoir) -> RaaRgiStage1 {
    let target_c = raa_rgi_target(s.position, s.normal, c.position, c.normal, c.radiance);
    var t = raa_rgi_temporal_candidate(pixel, s);
    if !(t.confidence > 0.0) {
        return RaaRgiStage1(c, target_c);
    }
    t.confidence = raa_restir_clamp_confidence(t.confidence, c.confidence, raa_rgi_params.m_cap);
    let target_t = raa_rgi_target(s.position, s.normal, t.position, t.normal, t.radiance);
    let seed = raa_rgi_seed(pixel, raa_rgi_params.frame_index, RAA_RGI_SALT_TEMPORAL);
    var merge = raa_rgi_merge_start();
    raa_rgi_merge(&merge, c, target_c, raa_restir_random(seed));
    raa_rgi_merge(&merge, t, target_t, raa_restir_random(seed + 1u));
    var out = merge.kept;
    out.confidence = merge.confidence;
    out.weight = raa_restir_contribution_weight(merge.w_sum / max(merge.confidence, 1.0e-6), merge.target_value);
    out.owner_normal = s.normal;
    out.owner_depth = s.depth;
    return RaaRgiStage1(out, merge.target_value);
}

// The raw irradiance of a merged reservoir, as restir_gi_spatial.wgsl resolves it: L cos W, with
// the alpha channel carrying the kept M (capped at m_cap as the spatial stage stores it) plus 256
// for a sky sample (the debug views).
fn raa_rgi_temporal_irradiance(s: RaaRgiSurface, r: RaaRgiReservoir) -> vec4<f32> {
    let d = r.position - s.position;
    let cos_v = max(dot(s.normal, d / max(length(d), 1.0e-6)), 0.0);
    let irradiance = r.radiance * cos_v * r.weight;
    let safe = select(vec3<f32>(0.0), irradiance, all(irradiance == irradiance) && all(irradiance < vec3<f32>(65504.0)));
    let sky = length(d) >= 0.5 * RAA_RGI_SKY_DISTANCE;
    return vec4<f32>(safe, min(r.confidence, raa_rgi_params.m_cap) + select(0.0, 256.0, sky));
}

@compute @workgroup_size(8, 8)
fn raa_gi_restir_gi_initial(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2<u32>(raa_gi_frame.viewport.xy)) {
        return;
    }
    let s = raa_rgi_surface(vec2<i32>(id.xy));
    var r = raa_rgi_empty();
    if s.valid {
        r = raa_rgi_initial(id.xy, s);
    }
    raa_rgi_initial_store(id.xy, r);
}

// Every invocation reaches both barriers; outside the viewport q = 0 and nothing is written.
@compute @workgroup_size(8, 8)
fn raa_gi_restir_gi_temporal(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let inside = all(id.xy < vec2<u32>(raa_gi_frame.viewport.xy));
    var s = RaaRgiSurface(vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0), 0.0, false);
    var result = RaaRgiStage1(raa_rgi_empty(), 0.0);
    if inside {
        s = raa_rgi_surface(vec2<i32>(id.xy));
        if s.valid {
            result = raa_rgi_stage1_merge(id.xy, s, raa_rgi_initial_load(id.xy));
        }
    }
    let q = raa_rgi_finite(result.target_value * result.reservoir.weight);
    raa_rgi_boiling_q[lane] = q;
    workgroupBarrier();
    if lane == 0u {
        var sum = 0.0;
        var count = 0u;
        for (var i = 0u; i < 64u; i = i + 1u) {
            let value = raa_rgi_boiling_q[i];
            if value > 0.0 {
                sum = sum + value;
                count = count + 1u;
            }
        }
        raa_rgi_boiling_mean = sum / f32(max(count, 1u));
    }
    workgroupBarrier();
    if !raa_restir_boiling_keep(q, raa_rgi_boiling_mean, raa_rgi_params.boiling_factor) {
        result.reservoir.weight = 0.0;
    }
    if inside {
        raa_rgi_store(id.xy, result.reservoir);
        textureStore(raa_rgi_temporal_raw, id.xy, raa_rgi_temporal_irradiance(s, result.reservoir));
    }
}
