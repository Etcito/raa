// Deform slot of provider `squash-jiggle`: squash-and-stretch, then a height-weighted jiggle
// offset. CPU twins: raa_deform::squash_point, squash_normal and jiggle_offset_weight.
// Every helper is prefixed `raa_squash_jiggle_` because the provider name has a hyphen.

// Squash axis a = normalize(x.squash.xyz), falling back to (0, 1, 0) for a zero or
// non-finite axis (as the CPU twin does; WGSL has no isfinite, so bound the length).
fn raa_squash_jiggle_axis(x: InstanceExtraGpu) -> vec3<f32> {
    let a = x.squash.xyz;
    let len = length(a);
    let usable = len > 0.0 && len < 1e38;
    return select(vec3<f32>(0.0, 1.0, 0.0), a / max(len, 1e-20), usable);
}

// p' = (p.a) a (1 + s) + (p - (p.a) a) / sqrt(1 + s); the determinant is 1, so volume is kept.
fn raa_squash_jiggle_point(p: vec3<f32>, a: vec3<f32>, s: f32) -> vec3<f32> {
    let parallel = a * dot(p, a);
    return parallel * (1.0 + s) + (p - parallel) / sqrt(1.0 + s);
}

// Inverse-transpose normal: parallel part over (1 + s), perpendicular part times sqrt(1 + s),
// renormalised (falling back to up when the result vanishes).
fn raa_squash_jiggle_normal(n: vec3<f32>, a: vec3<f32>, s: f32) -> vec3<f32> {
    let parallel = a * dot(n, a);
    let out = parallel / (1.0 + s) + (n - parallel) * sqrt(1.0 + s);
    let len = length(out);
    return select(vec3<f32>(0.0, 1.0, 0.0), out / max(len, 1e-20), len > 0.0);
}

// Contract jiggle weight w(y) = smoothstep(0, 1, clamp(y / h, 0, 1))^2; zero when h <= 0.
fn raa_squash_jiggle_weight(y: f32, h: f32) -> f32 {
    if (!(h > 0.0)) {
        return 0.0;
    }
    let t = clamp(y / h, 0.0, 1.0);
    let s = t * t * (3.0 - 2.0 * t);
    return s * s;
}

// w'(y) = 2 S(t) S'(t) / h with S'(t) = 6 t (1 - t) inside (0, 1), else 0.
fn raa_squash_jiggle_weight_derivative(y: f32, h: f32) -> f32 {
    if (!(h > 0.0)) {
        return 0.0;
    }
    let t = clamp(y / h, 0.0, 1.0);
    let s = t * t * (3.0 - 2.0 * t);
    let inside = t > 0.0 && t < 1.0;
    let d_s = select(0.0, 6.0 * t * (1.0 - t), inside);
    return 2.0 * s * d_s / h;
}

// Squash (axis a, amount s clamped to >= -0.9) then jiggle along `offset` with height `h`:
// p'' = p' + offset w(y), and the normal tilted by the inverse transpose of
// J = I + offset up^T w'.
fn raa_slot_deform(p: vec3<f32>, n: vec3<f32>, x: InstanceExtraGpu, t: f32) -> VertexMaterialOutput {
    let a = raa_squash_jiggle_axis(x);
    let s = max(x.squash.w, -0.9);
    let p1 = raa_squash_jiggle_point(p, a, s);
    let n1 = raa_squash_jiggle_normal(n, a, s);

    let offset = x.jiggle.xyz;
    let h = x.jiggle.w;
    if (!(h > 0.0)) {
        return VertexMaterialOutput(p1, n1);
    }

    let w = raa_squash_jiggle_weight(p1.y, h);
    let dw = raa_squash_jiggle_weight_derivative(p1.y, h);
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let p2 = p1 + offset * w;
    let denom = 1.0 + dw * dot(up, offset);
    let n2 = n1 - up * (dot(offset, n1) * dw / denom);
    return VertexMaterialOutput(p2, normalize(n2));
}
