// GGX prefilter of the IBL output cube (raa-ibl filter): writes output mip m (1..5) from the
// whole source sky cube (mips 0..5). Standalone compute module (no PRELUDE), own bindings in
// group 0. Workgroup 8x8x1 over the destination texels; global invocation z = face.
// Dispatch (ceil(face_size / 8), ceil(face_size / 8), 6).
//
// Per texel: direction from face and uv (face order +X,-X,+Y,-Y,+Z,-Z, v grows downwards,
// as src/sh.rs), roughness = m / 5, N = V = R (Karis 2013). sample_count (1024) GGX
// importance samples on the Hammersley set read the source cube at the PDF-based source mip
// (Krivanek and Colbert 2007) lod = 0.5 log2(omega_s / omega_p) + 1 with
// omega_s = 1 / (N pdf) and omega_p = 4 pi / (6 x 256^2), clamped to [0, 5]. Samples with
// NdotL <= 0 are skipped; the others are weighted by NdotL.

struct PrefilterParams {
    // Destination mip level m (1..5).
    mip: u32,
    // Perceptual roughness m / 5 (GGX alpha = roughness^2).
    roughness: f32,
    // Destination face size in px (256 >> m).
    face_size: u32,
    // GGX importance samples per texel.
    sample_count: u32,
}

// The source sky cube, all mips (SKY_ENV: 256 px, 6 mips).
@group(0) @binding(0) var src_cube: texture_cube<f32>;
// Linear min/mag/mipmap filter, clamp to edge.
@group(0) @binding(1) var src_sampler: sampler;
// Mip m of the output cube viewed as a 6-layer 2d array (one mip).
@group(0) @binding(2) var dst: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(3) var<uniform> params: PrefilterParams;

const PI: f32 = 3.14159265;
// Source cube (SKY_ENV): face size in px and last mip.
const SOURCE_SIZE: f32 = 256.0;
const SOURCE_MAX_LOD: f32 = 5.0;

// Unit direction through the centre of texel (x, y) of a `size` px face (src/sh.rs).
fn face_direction(face: u32, x: u32, y: u32, size: u32) -> vec3<f32> {
    let s = f32(max(size, 1u));
    let u = 2.0 * (f32(x) + 0.5) / s - 1.0;
    let v = 2.0 * (f32(y) + 0.5) / s - 1.0;
    var d: vec3<f32>;
    switch face {
        case 0u: { d = vec3<f32>(1.0, -v, -u); }
        case 1u: { d = vec3<f32>(-1.0, -v, u); }
        case 2u: { d = vec3<f32>(u, 1.0, v); }
        case 3u: { d = vec3<f32>(u, -1.0, -v); }
        case 4u: { d = vec3<f32>(u, -v, 1.0); }
        default: { d = vec3<f32>(-u, -v, -1.0); }
    }
    return normalize(d);
}

// Point i of the n-point Hammersley set (base-2 radical inverse as the second coordinate).
fn hammersley(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(n), f32(reverseBits(i)) * 2.3283064365386963e-10);
}

// GGX normal distribution for the cosine n_dot_h and alpha a.
fn d_ggx(n_dot_h: f32, a: f32) -> f32 {
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

@compute @workgroup_size(8, 8, 1)
fn prefilter(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = params.face_size;
    if (id.x >= size || id.y >= size || id.z >= 6u) {
        return;
    }
    let n = face_direction(id.z, id.x, id.y, size);
    let roughness = clamp(params.roughness, 0.0, 1.0);
    let a = max(roughness * roughness, 1e-4);
    // Orthonormal frame around n for the tangent-space half vectors.
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(n.z) < 0.999);
    let tx = normalize(cross(up, n));
    let ty = cross(n, tx);
    let count = max(params.sample_count, 1u);
    let omega_p = 4.0 * PI / (6.0 * SOURCE_SIZE * SOURCE_SIZE);
    var sum = vec3<f32>(0.0);
    var weight = 0.0;
    for (var i = 0u; i < count; i = i + 1u) {
        // GGX importance sample of the half vector h.
        let xi = hammersley(i, count);
        let phi = 2.0 * PI * xi.x;
        let cos_t = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
        let sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));
        let h = normalize(tx * (sin_t * cos(phi)) + ty * (sin_t * sin(phi)) + n * cos_t);
        // V = N, so n_dot_h = v_dot_h and L = reflect(-V, h).
        let n_dot_h = clamp(dot(n, h), 0.0, 1.0);
        let l = 2.0 * n_dot_h * h - n;
        let n_dot_l = dot(n, l);
        if (n_dot_l <= 0.0) {
            continue;
        }
        // pdf(L) = D(h) n_dot_h / (4 v_dot_h) = D(h) / 4.
        let pdf = d_ggx(n_dot_h, a) * 0.25;
        let omega_s = 1.0 / max(f32(count) * pdf, 1e-12);
        let lod = clamp(0.5 * log2(omega_s / omega_p) + 1.0, 0.0, SOURCE_MAX_LOD);
        sum = sum + textureSampleLevel(src_cube, src_sampler, l, lod).rgb * n_dot_l;
        weight = weight + n_dot_l;
    }
    let c = sum / max(weight, 1e-6);
    textureStore(dst, vec2<u32>(id.xy), id.z, vec4<f32>(c, 1.0));
}
