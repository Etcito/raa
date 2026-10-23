// BRDF LUT (IBL value): 128x128 split-sum terms of GGX with height-correlated Smith
// visibility, by 1024 GGX importance samples on a Hammersley sequence. Texel (x, y) holds
// (A, B) at n_dot_v = (x + 0.5) / 128 and roughness = (y + 0.5) / 128 (texel centres, so
// raa_ibl_dfg samples it at uv = (n_dot_v, roughness)); alpha = roughness^2. A is the scale
// and B the bias of F0, A + B the single-scatter directional albedo that the Fdez-Aguera
// multiscatter compensation of ambient.wgsl divides by.
// Core WebGPU has no Rg16Float storage texture: each texel is written as
// pack2x16float(A, B) into a u32 buffer, then copied into the Rg16Float texture.

const RAA_LUT_SIZE: u32 = 128u;
const RAA_LUT_SAMPLES: u32 = 1024u;
const RAA_LUT_PI: f32 = 3.14159265;

@group(0) @binding(0) var<storage, read_write> raa_lut_out: array<u32>;

// Point i of an n-point Hammersley set: (i / n, radical inverse of i in base 2).
fn raa_lut_hammersley(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(n), f32(reverseBits(i)) * 2.3283064365386963e-10);
}

// GGX-distributed half vector around +z for the uniform point xi.
fn raa_lut_sample_ggx(xi: vec2<f32>, alpha: f32) -> vec3<f32> {
    let a2 = alpha * alpha;
    let phi = 2.0 * RAA_LUT_PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y));
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    return vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

// Height-correlated Smith visibility G2 / (4 n_dot_v n_dot_l) (Heitz 2014).
fn raa_lut_visibility(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let ggx_v = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    let ggx_l = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    return 0.5 / max(ggx_v + ggx_l, 1e-12);
}

// Split-sum (A, B) at (n_dot_v, roughness). With pdf(l) = D n_dot_h / (4 v_dot_h), each
// sample weighs Vis x 4 n_dot_l v_dot_h / n_dot_h, split by the Schlick factor (1 - v_dot_h)^5.
fn raa_lut_integrate(n_dot_v: f32, roughness: f32) -> vec2<f32> {
    let alpha = roughness * roughness;
    let v = vec3<f32>(sqrt(max(1.0 - n_dot_v * n_dot_v, 0.0)), 0.0, n_dot_v);
    var a = 0.0;
    var b = 0.0;
    for (var i = 0u; i < RAA_LUT_SAMPLES; i = i + 1u) {
        let h = raa_lut_sample_ggx(raa_lut_hammersley(i, RAA_LUT_SAMPLES), alpha);
        let v_dot_h = dot(v, h);
        let l = 2.0 * v_dot_h * h - v;
        let n_dot_l = l.z;
        let n_dot_h = h.z;
        if (n_dot_l > 0.0 && v_dot_h > 0.0 && n_dot_h > 0.0) {
            let weight = raa_lut_visibility(n_dot_v, n_dot_l, alpha) * 4.0 * n_dot_l * v_dot_h
                / n_dot_h;
            let fc = pow(1.0 - v_dot_h, 5.0);
            a = a + (1.0 - fc) * weight;
            b = b + fc * weight;
        }
    }
    return vec2<f32>(a, b) / f32(RAA_LUT_SAMPLES);
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= RAA_LUT_SIZE || id.y >= RAA_LUT_SIZE) {
        return;
    }
    let n_dot_v = (f32(id.x) + 0.5) / f32(RAA_LUT_SIZE);
    let roughness = (f32(id.y) + 0.5) / f32(RAA_LUT_SIZE);
    raa_lut_out[id.y * RAA_LUT_SIZE + id.x] = pack2x16float(raa_lut_integrate(n_dot_v, roughness));
}
