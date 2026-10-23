// RAA helper library (S11 LIB).

// pcg3d hash of the cell; each component in [0, 1).
fn raa_hash3(p: vec3<f32>) -> vec3<f32> {
    var v = bitcast<vec3<u32>>(vec3<i32>(floor(p)));
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v = v ^ (v >> vec3<u32>(16u));
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return vec3<f32>(v >> vec3<u32>(8u)) * (1.0 / 16777216.0);
}

fn raa_grad3(cell: vec3<f32>, f: vec3<f32>) -> f32 {
    let g = normalize(raa_hash3(cell) * 2.0 - 1.0 + vec3<f32>(1e-6));
    return dot(g, f);
}

// Gradient noise in about [-1, 1], quintic fade.
fn raa_noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = p - i;
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let n000 = raa_grad3(i, f);
    let n100 = raa_grad3(i + vec3<f32>(1.0, 0.0, 0.0), f - vec3<f32>(1.0, 0.0, 0.0));
    let n010 = raa_grad3(i + vec3<f32>(0.0, 1.0, 0.0), f - vec3<f32>(0.0, 1.0, 0.0));
    let n110 = raa_grad3(i + vec3<f32>(1.0, 1.0, 0.0), f - vec3<f32>(1.0, 1.0, 0.0));
    let n001 = raa_grad3(i + vec3<f32>(0.0, 0.0, 1.0), f - vec3<f32>(0.0, 0.0, 1.0));
    let n101 = raa_grad3(i + vec3<f32>(1.0, 0.0, 1.0), f - vec3<f32>(1.0, 0.0, 1.0));
    let n011 = raa_grad3(i + vec3<f32>(0.0, 1.0, 1.0), f - vec3<f32>(0.0, 1.0, 1.0));
    let n111 = raa_grad3(i + vec3<f32>(1.0, 1.0, 1.0), f - vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(n000, n100, u.x);
    let x10 = mix(n010, n110, u.x);
    let x01 = mix(n001, n101, u.x);
    let x11 = mix(n011, n111, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z) * 1.4;
}

// Fractal sum of gradient noise, normalised to about [-1, 1].
fn raa_fbm3(p: vec3<f32>, octaves: i32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var norm = 0.0;
    var q = p;
    for (var i = 0; i < octaves; i = i + 1) {
        sum = sum + amp * raa_noise3(q);
        norm = norm + amp;
        amp = amp * 0.5;
        q = q * 2.03 + vec3<f32>(17.1, 3.7, 9.2);
    }
    return sum / max(norm, 1e-6);
}

// Cellular noise: (F1, F2) distances to the two nearest feature points.
fn raa_voronoi3(p: vec3<f32>) -> vec2<f32> {
    let i = floor(p);
    let f = p - i;
    var d = vec2<f32>(8.0, 8.0);
    for (var z = -1; z <= 1; z = z + 1) {
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                let o = vec3<f32>(f32(x), f32(y), f32(z));
                let r = o + raa_hash3(i + o) - f;
                let dist = length(r);
                if (dist < d.x) {
                    d = vec2<f32>(dist, d.x);
                } else if (dist < d.y) {
                    d.y = dist;
                }
            }
        }
    }
    return d;
}

// Blend weights of the X, Y and Z projections for a unit normal (sum 1).
fn raa_triplanar_weights(n: vec3<f32>) -> vec3<f32> {
    let w = pow(abs(n), vec3<f32>(4.0));
    return w / max(w.x + w.y + w.z, 1e-6);
}

// Tangent-space normal to world space; tangent.w is the bitangent sign.
fn raa_perturb_normal(n: vec3<f32>, tangent: vec4<f32>, ts_normal: vec3<f32>) -> vec3<f32> {
    let t = normalize(tangent.xyz - n * dot(n, tangent.xyz));
    let b = cross(n, t) * tangent.w;
    return normalize(t * ts_normal.x + b * ts_normal.y + n * ts_normal.z);
}

fn raa_luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn raa_srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}
