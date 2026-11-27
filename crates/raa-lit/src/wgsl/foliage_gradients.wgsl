// Counted-bin differentiable foliage gradients (M7). One invocation owns one optimizer
// micro-triangle: it visits the pixels of its counted 16-by-16 tiles in every reference
// view and accumulates the analytic derivatives of the weighted silhouette, the 1-cosine
// normal and the squared-edge term in registers before writing its own gradient record,
// so the reduction needs no float atomic. The shared `foliage_soft_coverage` helper is
// prepended by foliage_gradients.rs from foliage_raster.wgsl and is never repeated here.
struct Params {
    count: u32, views: u32, resolution: u32, iteration: u32,
    silhouette: f32, normal: f32, edge: f32, temperature: f32,
    learning_rate: f32, opacity_target: f32, capacity: u32, generation: u32,
    reserved: vec4<u32>,
}
struct Triangle { p0: vec4<f32>, p1: vec4<f32>, p2: vec4<f32>, normal: vec4<f32>, reserved: vec4<f32> }
struct Gradient { dp0: vec4<f32>, dp1: vec4<f32>, dp2: vec4<f32>, normal_opacity: vec4<f32>, reserved: vec4<f32> }
struct View { matrix: mat4x4<f32>, near_far: vec4<f32> }
struct ViewTable { views: array<View, 128> }
struct Pixel { normal_coverage: vec4<f32>, depth_loss: vec4<f32> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(2) var<storage, read_write> gradients: array<Gradient>;
@group(0) @binding(3) var<storage, read_write> bins: array<u32>;
@group(0) @binding(4) var<storage, read_write> pixels: array<Pixel>;
@group(0) @binding(5) var reference_normal_coverage: texture_2d_array<f32>;
@group(0) @binding(6) var reference_depth: texture_2d_array<f32>;
@group(0) @binding(7) var<uniform> table: ViewTable;

fn cross2(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return a.x * b.y - a.y * b.x;
}

fn screen_position(clip: vec4<f32>, resolution: f32) -> vec2<f32> {
    let ndc = clip.xy / clip.w;
    return vec2<f32>((ndc.x * 0.5 + 0.5) * resolution, (0.5 - ndc.y * 0.5) * resolution);
}

fn barycentric(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, point: vec2<f32>, area: f32) -> vec3<f32> {
    return vec3<f32>(cross2(b - point, c - point), cross2(c - point, a - point), cross2(a - point, b - point)) / area;
}

// The signed inward pixel distance of one triangle edge together with its analytic
// gradients with respect to the start and end screen positions.
struct EdgePartial { distance: f32, start: vec2<f32>, end: vec2<f32> }

fn edge_partial(start: vec2<f32>, end: vec2<f32>, point: vec2<f32>, orient: f32) -> EdgePartial {
    let edge = end - start;
    let inv = max(length(edge), 1e-6);
    let distance = orient * cross2(edge, point - start) / inv;
    var out: EdgePartial;
    out.distance = distance;
    if (length(edge) > 1e-6) {
        let toward = edge / (inv * inv);
        out.start = orient * vec2<f32>(end.y - point.y, point.x - end.x) / inv + distance * toward;
        out.end = orient * vec2<f32>(point.y - start.y, start.x - point.x) / inv - distance * toward;
    } else {
        out.start = vec2<f32>(0.0);
        out.end = vec2<f32>(0.0);
    }
    return out;
}

// The screen-space gradients of the interpolated depth with respect to the three screen
// positions: the barycentric weights depend on every vertex and the depth value on its own.
struct DepthScreenGradient { a: vec2<f32>, b: vec2<f32>, c: vec2<f32> }

fn depth_screen_gradient(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, point: vec2<f32>, area: f32, l: vec3<f32>, z: vec3<f32>) -> DepthScreenGradient {
    let da = vec2<f32>(b.y - c.y, c.x - b.x);
    let db = vec2<f32>(c.y - a.y, a.x - c.x);
    let dc = vec2<f32>(a.y - b.y, b.x - a.x);
    let n1a = vec2<f32>(point.y - c.y, c.x - point.x);
    let n2a = vec2<f32>(b.y - point.y, point.x - b.x);
    let n0b = vec2<f32>(c.y - point.y, point.x - c.x);
    let n2b = vec2<f32>(point.y - a.y, a.x - point.x);
    let n0c = vec2<f32>(point.y - b.y, b.x - point.x);
    let n1c = vec2<f32>(a.y - point.y, point.x - a.x);
    var out: DepthScreenGradient;
    out.a = (z.x * (-l.x * da) + z.y * (n1a - l.y * da) + z.z * (n2a - l.z * da)) / area;
    out.b = (z.x * (n0b - l.x * db) + z.y * (-l.y * db) + z.z * (n2b - l.z * db)) / area;
    out.c = (z.x * (n0c - l.x * dc) + z.y * (n1c - l.y * dc) + z.z * (-l.z * dc)) / area;
    return out;
}

// Converts a screen-position gradient and a depth-value gradient of one vertex into its
// clip-space gradient; world_gradient then applies the column-major view matrix.
fn clip_gradient(screen: vec2<f32>, depth: f32, clip: vec4<f32>, resolution: f32) -> vec4<f32> {
    let half = 0.5 * resolution;
    let w = clip.w;
    let dw = -screen.x * half * clip.x / (w * w) + screen.y * half * clip.y / (w * w) - depth * clip.z / (w * w);
    return vec4<f32>(screen.x * half / w, -screen.y * half / w, depth / w, dw);
}

fn world_gradient(clip: vec4<f32>, matrix: mat4x4<f32>) -> vec3<f32> {
    return vec3<f32>(dot(clip, matrix[0]), dot(clip, matrix[1]), dot(clip, matrix[2]));
}

@compute @workgroup_size(64)
fn gradient_triangles(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= params.count || params.resolution == 0u) { return; }
    let triangle = triangles[index];
    let resolution = f32(params.resolution);
    let side = (params.resolution + 15u) / 16u;
    let temperature = max(params.temperature, 1e-6);
    let depth_temperature = max(temperature / resolution, 1e-6);
    var dp0 = vec3<f32>(0.0);
    var dp1 = vec3<f32>(0.0);
    var dp2 = vec3<f32>(0.0);
    var dnormal = vec3<f32>(0.0);
    var dopacity = 0.0;
    for (var view = 0u; view < params.views; view++) {
        let matrix = table.views[view].matrix;
        let a = matrix * vec4<f32>(triangle.p0.xyz, 1.0);
        let b = matrix * vec4<f32>(triangle.p1.xyz, 1.0);
        let c = matrix * vec4<f32>(triangle.p2.xyz, 1.0);
        if (min(min(a.w, b.w), c.w) <= 0.0) { continue; }
        // The bin producer rejects a triangle wholly outside one homogeneous plane.
        if ((a.x < -a.w && b.x < -b.w && c.x < -c.w) ||
            (a.x > a.w && b.x > b.w && c.x > c.w) ||
            (a.y < -a.w && b.y < -b.w && c.y < -c.w) ||
            (a.y > a.w && b.y > b.w && c.y > c.w) ||
            (a.z < 0.0 && b.z < 0.0 && c.z < 0.0) ||
            (a.z > a.w && b.z > b.w && c.z > c.w)) { continue; }
        let pa = screen_position(a, resolution);
        let pb = screen_position(b, resolution);
        let pc = screen_position(c, resolution);
        let area = cross2(pb - pa, pc - pa);
        if (abs(area) < 1e-8) { continue; }
        let orient = select(-1.0, 1.0, area > 0.0);
        let za = a.z / a.w;
        let zb = b.z / b.w;
        let zc = c.z / c.w;
        // The same conservative tile rectangle the bin producer filled; a triangle that
        // crosses the near plane was binned over the whole viewport, so it scans it too.
        var lo = vec2<u32>(0u);
        var hi = vec2<u32>(side - 1u);
        if (min(min(a.z, b.z), c.z) >= 0.0) {
            lo = vec2<u32>(clamp(floor(min(min(pa, pb), pc) / 16.0), vec2<f32>(0.0), vec2<f32>(f32(side - 1u))));
            hi = vec2<u32>(clamp(floor(max(max(pa, pb), pc) / 16.0), vec2<f32>(0.0), vec2<f32>(f32(side - 1u))));
        }
        for (var ty = lo.y; ty <= hi.y; ty++) {
            for (var tx = lo.x; tx <= hi.x; tx++) {
                for (var py = 0u; py < 16u; py++) {
                    let y = ty * 16u + py;
                    if (y >= params.resolution) { continue; }
                    for (var px = 0u; px < 16u; px++) {
                        let x = tx * 16u + px;
                        if (x >= params.resolution) { continue; }
                        let point = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5);
                        let pixel = pixels[view * params.resolution * params.resolution + y * params.resolution + x];
                        if (pixel.normal_coverage.w <= 0.0) { continue; }
                        let reference = textureLoad(reference_normal_coverage, vec2<i32>(i32(x), i32(y)), i32(view), 0);
                        let reference_z = textureLoad(reference_depth, vec2<i32>(i32(x), i32(y)), i32(view), 0).x;
                        let visible = reference.w > 0.0 && reference_z > 0.0;
                        let e0 = edge_partial(pb, pc, point, orient);
                        let e1 = edge_partial(pc, pa, point, orient);
                        let e2 = edge_partial(pa, pb, point, orient);
                        let s0 = foliage_soft_coverage(e0.distance, temperature);
                        let s1 = foliage_soft_coverage(e1.distance, temperature);
                        let s2 = foliage_soft_coverage(e2.distance, temperature);
                        let k = s0 * s1 * s2;
                        let t = k * triangle.normal.w;
                        if (t <= 0.0) { continue; }
                        let l = barycentric(pa, pb, pc, point, area);
                        let z = l.x * za + l.y * zb + l.z * zc;
                        let v = select(0.0, foliage_soft_coverage((reference_z - z) / depth_temperature, 1.0), visible);
                        // Weighted silhouette; the clamp derivative vanishes once the
                        // published coverage reaches 1.
                        var dl_dt = params.silhouette * (pixel.normal_coverage.w - reference.w)
                            * select(0.0, 1.0, pixel.normal_coverage.w < 1.0);
                        var dl_dv = 0.0;
                        if (visible && params.normal > 0.0) {
                            let reference_length = length(reference.xyz);
                            let predicted = pixel.normal_coverage.xyz;
                            if (reference_length > 1e-8 && length(predicted) > 0.5) {
                                let n_ref = reference.xyz / reference_length;
                                // The normalization derivative needs the length of the
                                // coverage/visibility-weighted normal sum of this pixel's
                                // tile, which the raster stage stored in lane 1 z: one read
                                // instead of a re-sum of the tile per pixel and triangle.
                                let sum_length = pixel.depth_loss.z;
                                if (sum_length > 1e-8) {
                                    let dl_ds = -(params.normal / sum_length) * (n_ref - predicted * dot(n_ref, predicted));
                                    let dw = dot(dl_ds, triangle.normal.xyz);
                                    dl_dt += dw * v;
                                    dl_dv = dw * t;
                                    dnormal += (t * v) * dl_ds;
                                }
                            }
                        }
                        // Chain the sample coverage through the three shared sigmoids:
                        // dK/dd = K * (1 - s) / temperature because K already holds s.
                        let scale = dl_dt * triangle.normal.w * k / temperature;
                        let c0 = scale * (1.0 - s0);
                        let c1 = scale * (1.0 - s1);
                        let c2 = scale * (1.0 - s2);
                        var ga = c1 * e1.end + c2 * e2.start;
                        var gb = c0 * e0.start + c2 * e2.end;
                        var gc = c0 * e0.end + c1 * e1.start;
                        var gza = 0.0;
                        var gzb = 0.0;
                        var gzc = 0.0;
                        if (dl_dv != 0.0) {
                            let dv_dz = dl_dv * (-v * (1.0 - v) / depth_temperature);
                            let dz = depth_screen_gradient(pa, pb, pc, point, area, l, vec3<f32>(za, zb, zc));
                            ga += dv_dz * dz.a;
                            gb += dv_dz * dz.b;
                            gc += dv_dz * dz.c;
                            gza = dv_dz * l.x;
                            gzb = dv_dz * l.y;
                            gzc = dv_dz * l.z;
                        }
                        dp0 += world_gradient(clip_gradient(ga, gza, a, resolution), matrix);
                        dp1 += world_gradient(clip_gradient(gb, gzb, b, resolution), matrix);
                        dp2 += world_gradient(clip_gradient(gc, gzc, c, resolution), matrix);
                        dopacity += dl_dt * k;
                    }
                }
            }
        }
    }
    // Squared-edge regularization, once per triangle; each edge gives 2 * (a - b).
    dp0 += 2.0 * params.edge * (2.0 * triangle.p0.xyz - triangle.p1.xyz - triangle.p2.xyz);
    dp1 += 2.0 * params.edge * (2.0 * triangle.p1.xyz - triangle.p0.xyz - triangle.p2.xyz);
    dp2 += 2.0 * params.edge * (2.0 * triangle.p2.xyz - triangle.p0.xyz - triangle.p1.xyz);
    var record: Gradient;
    record.dp0 = vec4<f32>(dp0, 0.0);
    record.dp1 = vec4<f32>(dp1, 0.0);
    record.dp2 = vec4<f32>(dp2, 0.0);
    record.normal_opacity = vec4<f32>(dnormal, dopacity);
    record.reserved = vec4<f32>(0.0);
    gradients[index] = record;
}
