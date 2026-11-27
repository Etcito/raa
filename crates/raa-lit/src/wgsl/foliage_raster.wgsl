// Counted-bin differentiable foliage raster (M7). One invocation resolves one pixel of one
// reference view from the triangles of its 16-by-16 tile: the three soft edge coverages
// give the silhouette, the captured reference depth soft-orders the overlapping predicted
// surfaces of the normal, and the record holds the predicted world normal and coverage in
// lane 0 with the predicted depth, zero loss, the length of the coverage/visibility-weighted
// normal sum and a zero in lane 1 (the gradient stage reads that length instead of
// re-summing the tile for every pixel of every triangle). A pixel without predicted
// coverage is the background record: zero coverage and a defined zero normal.
struct Params {
    count: u32, views: u32, resolution: u32, iteration: u32,
    silhouette: f32, normal: f32, edge: f32, temperature: f32,
    learning_rate: f32, opacity_target: f32, capacity: u32, generation: u32,
    reserved: vec4<u32>,
}
struct Triangle { p0: vec4<f32>, p1: vec4<f32>, p2: vec4<f32>, normal: vec4<f32>, reserved: vec4<f32> }
struct View { matrix: mat4x4<f32>, near_far: vec4<f32> }
struct ViewTable { views: array<View, 128> }
struct Pixel { normal_coverage: vec4<f32>, depth_loss: vec4<f32> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(3) var<storage, read_write> bins: array<u32>;
@group(0) @binding(4) var<storage, read_write> pixels: array<Pixel>;
@group(0) @binding(5) var normal_coverage: texture_2d_array<f32>;
@group(0) @binding(6) var reference_depth: texture_2d_array<f32>;
@group(0) @binding(7) var<uniform> table: ViewTable;

// raa foliage-raster part: soft coverage
// Analytic soft coverage of one triangle edge in pixels: sigmoid(edge_distance / temperature)
// with a positive distance inside the triangle. At distance 0 the value is 0.5 and its
// derivative with respect to distance is 1 / (4 * temperature); temperature is positive and
// measured in pixels. The gradient stage includes this section instead of transcribing the
// formula independently.
fn foliage_soft_coverage(edge_distance: f32, temperature: f32) -> f32 {
    return 1.0 / (1.0 + exp(-edge_distance / temperature));
}
// raa foliage-raster part: raster
fn cross2(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return a.x * b.y - a.y * b.x;
}
fn screen_position(clip: vec4<f32>, resolution: f32) -> vec2<f32> {
    let ndc = clip.xy / clip.w;
    return vec2<f32>((ndc.x * 0.5 + 0.5) * resolution, (0.5 - ndc.y * 0.5) * resolution);
}

@compute @workgroup_size(64)
fn raster_pixels(@builtin(global_invocation_id) id: vec3<u32>) {
    let resolution = params.resolution;
    if (resolution == 0u || id.x >= resolution * resolution || id.y >= params.views) { return; }
    let coord = vec2<i32>(i32(id.x % resolution), i32(id.x / resolution));
    let layer = i32(id.y);
    let reference = textureLoad(normal_coverage, coord, layer, 0);
    let reference_z = textureLoad(reference_depth, coord, layer, 0).x;
    let side = (resolution + 15u) / 16u;
    let tile = id.y * side * side + (u32(coord.y) / 16u) * side + (u32(coord.x) / 16u);
    let base = tile * (params.capacity + 1u);
    // The producer's count; an overflow signals a retry before any consumer reads the bins,
    // so the clamp only keeps this scan inside the allocated records.
    let count = min(bins[base], params.capacity);
    let point = vec2<f32>(f32(coord.x) + 0.5, f32(coord.y) + 0.5);
    let temperature = params.temperature;
    let depth_temperature = max(temperature / f32(resolution), 1e-6);
    let matrix = table.views[id.y].matrix;
    var coverage_sum = 0.0;
    var weight_sum = 0.0;
    var normal_sum = vec3<f32>(0.0);
    var depth_sum = 0.0;
    for (var i = 0u; i < count; i++) {
        let triangle = triangles[bins[base + 1u + i]];
        let a = matrix * vec4<f32>(triangle.p0.xyz, 1.0);
        let b = matrix * vec4<f32>(triangle.p1.xyz, 1.0);
        let c = matrix * vec4<f32>(triangle.p2.xyz, 1.0);
        if (min(min(a.w, b.w), c.w) <= 0.0) { continue; }
        let pa = screen_position(a, f32(resolution));
        let pb = screen_position(b, f32(resolution));
        let pc = screen_position(c, f32(resolution));
        let area = cross2(pb - pa, pc - pa);
        if (abs(area) < 1e-8) { continue; }
        // Inward-signed pixel distances of the three edges, whatever the screen winding.
        let orient = select(-1.0, 1.0, area > 0.0);
        let d0 = orient * cross2(pc - pb, point - pb) / max(length(pc - pb), 1e-6);
        let d1 = orient * cross2(pa - pc, point - pc) / max(length(pa - pc), 1e-6);
        let d2 = orient * cross2(pb - pa, point - pa) / max(length(pb - pa), 1e-6);
        let coverage = foliage_soft_coverage(d0, temperature) * foliage_soft_coverage(d1, temperature)
            * foliage_soft_coverage(d2, temperature) * triangle.normal.w;
        if (coverage <= 0.0) { continue; }
        let l0 = cross2(pb - point, pc - point) / area;
        let l1 = cross2(pc - point, pa - point) / area;
        let l2 = cross2(pa - point, pb - point) / area;
        let z = l0 * (a.z / a.w) + l1 * (b.z / b.w) + l2 * (c.z / c.w);
        // The captured reference depth soft-orders overlapping predicted surfaces: a surface
        // at the reference is half visible, one a temperature band in front nearly one and
        // one behind nearly zero. A background reference has no surface and no normal target.
        let visibility = select(0.0, foliage_soft_coverage((reference_z - z) / depth_temperature, 1.0), reference.w > 0.0 && reference_z > 0.0);
        coverage_sum += coverage;
        let weight = coverage * visibility;
        weight_sum += weight;
        normal_sum += weight * triangle.normal.xyz;
        depth_sum += weight * z;
    }
    var record: Pixel;
    if (coverage_sum > 0.0) {
        var normal = vec3<f32>(0.0);
        if (length(normal_sum) > 1e-8) { normal = normalize(normal_sum); }
        let depth = select(1.0, depth_sum / max(weight_sum, 1e-8), weight_sum > 1e-8);
        record.normal_coverage = vec4<f32>(normal, clamp(coverage_sum, 0.0, 1.0));
        record.depth_loss = vec4<f32>(depth, 0.0, length(normal_sum), 0.0);
    } else {
        record.normal_coverage = vec4<f32>(0.0);
        record.depth_loss = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    }
    pixels[id.x + id.y * resolution * resolution] = record;
}
