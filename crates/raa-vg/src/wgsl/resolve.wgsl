/*
MIT License

Copyright (c) 2026 SamG-Coder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
// Adapted from solstice@7e465b599c13da165948ebaa44df824c36c3ae07
// crates/solstice-render/shaders/vg/resolve.wgsl:1-283.
// Bindings move to group 2 (contract `vg-resolve-module`). The external include, the
// sea-history bindings, the material texture tiers and their samplers and the emissive
// output are removed: no sampler is left, so the surface is read with loads and written
// with storage textures. The scene twins come from `pass_view.wgsl`.

// The S32 resolve frame record, 320 bytes, the twin of `VgFrameGpu` in vg/resolve.rs:
// view_proj @0, unjittered_view_proj @64, previous_view_proj @128, inv_view_proj @192,
// viewport @256, camera_jitter @272 (current xy, previous zw), camera_pos @288 and
// flags @304 (debug view code, history valid, 0, 0).
struct VgFrame {
    view_proj: mat4x4<f32>,
    unjittered_view_proj: mat4x4<f32>,
    previous_view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    viewport: vec4<f32>,
    camera_jitter: vec4<f32>,
    camera_pos: vec4<f32>,
    flags: vec4<u32>,
}

// The S10 `SurfaceMaterialGpu` record, 256 bytes. Only `base` @0 and `header` @96 are read:
// `header.x` is the model (2 = a registered material function) and `header.y` its function id.
struct VgMaterial {
    base: vec4<f32>,
    emissive: vec4<f32>,
    surface: vec4<f32>,
    sheen: vec4<f32>,
    transmission: vec4<f32>,
    attenuation: vec4<f32>,
    header: vec4<u32>,
    p: array<vec4<f32>, 8>,
    fur: vec4<f32>,
}

@group(2) @binding(0) var<uniform> frame: VgFrame;
@group(2) @binding(1) var<storage, read> vis_depth: array<u32>;
@group(2) @binding(2) var<storage, read> vis_id: array<u32>;
@group(2) @binding(3) var<storage, read> visible: array<VisibleCluster>;
@group(2) @binding(4) var<storage, read> instances: array<VgInstance>;
@group(2) @binding(5) var<storage, read> clusters: array<VgCluster>;
@group(2) @binding(6) var<storage, read> vertices: array<PackedVertex>;
@group(2) @binding(7) var<storage, read> triangles: array<u32>;
@group(2) @binding(8) var<storage, read> materials: array<VgMaterial>;
@group(2) @binding(9) var out_ids: texture_storage_2d<rgba32uint, write>;
@group(2) @binding(10) var out_normal: texture_storage_2d<rgba16float, write>;
@group(2) @binding(11) var out_uv_motion: texture_storage_2d<rgba32float, write>;
@group(2) @binding(12) var out_albedo: texture_storage_2d<rgba32float, write>;
// RAA addition (not in solstice): the per-vertex attributes of every pool vertex, two
// `rgba32uint` texels per vertex, VG_COLOR_WIDTH texels per row (`VgScene::colors`): the
// linear rgba colour as f32 bits, then the shading uv (u bits, v bits, flag, 0). A vertex
// outside the texture (the 1x1 white placeholder) is white and keeps its cluster uv.
@group(2) @binding(13) var vertex_colors: texture_2d<u32>;

// S29 packs optional tangent-plane angle and handedness in the high half of d.
fn authored_tangent(instance: VgInstance, vertex: PackedVertex) -> vec4<f32> {
    let word = vertex.d >> 16u;
    if (word & 0x8000u) == 0u { return vec4<f32>(0.0); }
    let n = decode_normal(vertex);
    let axis = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(n.z) < 0.875);
    let x = normalize(cross(axis, n));
    let y = cross(n, x);
    let angle = f32(word & 0x3fffu) * (6.283185307179586 / 16384.0);
    let t = x * cos(angle) + y * sin(angle);
    return vec4<f32>((instance.model * vec4<f32>(t, 0.0)).xyz, select(1.0, -1.0, (word & 0x4000u) != 0u));
}

// Octahedral direction, handedness, and presence in the existing surface ID word.
fn pack_surface_tangent(tangent: vec4<f32>) -> u32 {
    let magnitude = dot(abs(tangent.xyz), vec3<f32>(1.0));
    if magnitude <= 1.0e-12 { return 0u; }
    let unit = tangent.xyz / magnitude;
    let sign = select(vec2<f32>(-1.0), vec2<f32>(1.0), unit.xy >= vec2<f32>(0.0));
    let oct = select((1.0 - abs(unit.yx)) * sign, unit.xy, unit.z >= 0.0);
    let bits = vec2<u32>(round(clamp(oct * 0.5 + 0.5, vec2<f32>(0.0), vec2<f32>(1.0)) * 32767.0));
    return bits.x | (bits.y << 15u) | select(0u, 0x40000000u, tangent.w < 0.0) | 0x80000000u;
}

// `VG_COLOR_WIDTH` of vg/scene.rs.
const VG_COLOR_WIDTH: u32 = 4096u;

// The attribute texel `slot` (0 colour, 1 uv) of pool vertex `index`, or `fallback` outside
// the texture.
fn vertex_texel(index: u32, slot: u32, fallback: vec4<u32>) -> vec4<u32> {
    let dims = textureDimensions(vertex_colors);
    let flat = index * 2u + slot;
    let texel = vec2<u32>(flat % VG_COLOR_WIDTH, flat / VG_COLOR_WIDTH);
    if (texel.x >= dims.x || texel.y >= dims.y) {
        return fallback;
    }
    return textureLoad(vertex_colors, texel, 0);
}

fn vertex_color(index: u32) -> vec4<f32> {
    return bitcast<vec4<f32>>(vertex_texel(index, 0u, vec4<u32>(0x3f800000u)));
}

// The shading uv in xy and 1 in z when the vertex has one, else z = 0.
fn vertex_uv(index: u32) -> vec3<f32> {
    let t = vertex_texel(index, 1u, vec4<u32>(0u));
    return vec3<f32>(bitcast<f32>(t.x), bitcast<f32>(t.y), select(0.0, 1.0, t.z != 0u));
}

fn interp4(b: vec3<f32>, a0: vec4<f32>, a1: vec4<f32>, a2: vec4<f32>) -> vec4<f32> {
    return a0 * b.x + a1 * b.y + a2 * b.z;
}

// Piecewise sRGB EOTF: a debug colour picked in sRGB written as the linear colour it shows.
fn decode_srgb(c: vec3<f32>) -> vec3<f32> {
    return select(
        pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4)),
        c / 12.92,
        c <= vec3<f32>(0.04045),
    );
}

// `SurfaceMaterialGpu.header.x`: a registered material function.
const MATERIAL_MODEL_FUNCTION: u32 = 2u;

fn pixel_to_ndc(pixel: vec2<f32>, viewport: vec2<f32>) -> vec2<f32> {
    let uv = pixel / viewport;
    return vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}

// Perspective-correct barycentrics of a pixel inside a triangle (solstice resolve.wgsl:80-94).
fn barycentrics(clip: array<vec4<f32>, 3>, ndc: vec2<f32>) -> vec3<f32> {
    let inv_w = vec3<f32>(1.0 / clip[0].w, 1.0 / clip[1].w, 1.0 / clip[2].w);
    let p0 = clip[0].xy * inv_w.x;
    let p1 = clip[1].xy * inv_w.y;
    let p2 = clip[2].xy * inv_w.z;
    let e0 = p1 - p0;
    let e1 = p2 - p0;
    let d = ndc - p0;
    let inv_det = 1.0 / (e0.x * e1.y - e0.y * e1.x);
    let screen_b = (d.x * e1.y - d.y * e1.x) * inv_det;
    let screen_c = (e0.x * d.y - e0.y * d.x) * inv_det;
    let screen = vec3<f32>(1.0 - screen_b - screen_c, screen_b, screen_c);
    let weighted = screen * inv_w;
    return weighted / (weighted.x + weighted.y + weighted.z);
}

fn interp2(b: vec3<f32>, a0: vec2<f32>, a1: vec2<f32>, a2: vec2<f32>) -> vec2<f32> {
    return a0 * b.x + a1 * b.y + a2 * b.z;
}

fn interp3(b: vec3<f32>, a0: vec3<f32>, a1: vec3<f32>, a2: vec3<f32>) -> vec3<f32> {
    return a0 * b.x + a1 * b.y + a2 * b.z;
}

// The signed area of the triangle projected to NDC: positive when counter-clockwise (front).
fn projected_area(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32 {
    let e0 = b - a;
    let e1 = c - a;
    return e0.x * e1.y - e0.y * e1.x;
}

// `hash_color` gives the per-cluster hash colour for debug code 1: the PCG hash of the
// cluster index (solstice resolve.wgsl:131-134). The `frame.flags.x` debug branch of `main`
// maps code 1 to `hash_color`, 2 to `lod_ramp`, 3 to `raster_color` and 4 to
// `occlusion_color`, sRGB-encoded with a = 1.
fn hash_color(value: u32) -> vec3<f32> {
    let h = pcg(value);
    return vec3<f32>(f32(h & 255u), f32((h >> 8u) & 255u), f32((h >> 16u) & 255u)) / 255.0 * 0.8 + 0.2;
}

// Piecewise sRGB OETF of a value in [0, 1].
fn encode_srgb(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    let lo = x * 12.92;
    let hi = 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(hi, lo, x <= vec3<f32>(0.0031308));
}

// `lod_ramp` gives the DAG level as a colour ramp for debug code 2: blue at level 0 through
// green at level 6 to red at level 12 and above, linear between the stops.
fn lod_ramp(level: u32) -> vec3<f32> {
    let t = f32(min(level, 12u));
    let blue = vec3<f32>(0.1, 0.3, 1.0);
    let green = vec3<f32>(0.2, 0.9, 0.3);
    let red = vec3<f32>(1.0, 0.2, 0.1);
    if (t <= 6.0) {
        return mix(blue, green, t / 6.0);
    }
    return mix(green, red, (t - 6.0) / 6.0);
}

// `raster_color` gives the debug colour for code 3: hardware rasterisation is blue and
// software rasterisation is orange.
fn raster_color(cluster_word: u32) -> vec3<f32> {
    return select(
        vec3<f32>(0.15, 0.55, 1.0),
        vec3<f32>(1.0, 0.6, 0.1),
        (cluster_word & VISIBLE_SOFTWARE) != 0u,
    );
}

// `occlusion_color` gives the debug colour for code 4: the first occlusion phase is green
// and a cluster recovered by the second phase is magenta.
fn occlusion_color(cluster_word: u32) -> vec3<f32> {
    return select(
        vec3<f32>(0.25, 0.8, 0.35),
        vec3<f32>(0.95, 0.2, 0.75),
        (cluster_word & VISIBLE_SECOND) != 0u,
    );
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<u32>(frame.viewport.xy);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let pixel = id.xy;
    let index = pixel.y * size.x + pixel.x;
    let key = vis_depth[index];
    if (key == 0u) {
        textureStore(out_ids, pixel, vec4<u32>(0u));
        textureStore(out_normal, pixel, vec4<f32>(0.0));
        textureStore(out_uv_motion, pixel, vec4<f32>(0.0));
        textureStore(out_albedo, pixel, vec4<f32>(0.0));
        return;
    }

    let depth = raa_depth_from_key(key);
    let packed_id = vis_id[index];
    let visible_index = packed_id >> 7u;
    let triangle_index = packed_id & 0x7fu;
    let entry = visible[visible_index];
    let cluster_index = visible_cluster_index(entry);
    let instance = instances[entry.instance];
    let cluster = clusters[cluster_index];
    let material = materials[instance.ids.y];

    let packed = triangles[instance.ranges.w + cluster.triangle_offset + triangle_index];
    let i0 = instance.ranges.z + cluster.vertex_offset + (packed & 0xffu);
    let i1 = instance.ranges.z + cluster.vertex_offset + ((packed >> 8u) & 0xffu);
    let i2 = instance.ranges.z + cluster.vertex_offset + ((packed >> 16u) & 0xffu);
    let v0 = vertices[i0];
    let v1 = vertices[i1];
    let v2 = vertices[i2];

    let local0 = decode_position(instance, v0);
    let local1 = decode_position(instance, v1);
    let local2 = decode_position(instance, v2);
    let world0 = transform_point(instance, local0);
    let world1 = transform_point(instance, local1);
    let world2 = transform_point(instance, local2);
    let clip = array<vec4<f32>, 3>(
        frame.view_proj * vec4<f32>(world0, 1.0),
        frame.view_proj * vec4<f32>(world1, 1.0),
        frame.view_proj * vec4<f32>(world2, 1.0),
    );

    let viewport = frame.viewport.xy;
    let center = vec2<f32>(pixel) + vec2<f32>(0.5);
    let ndc = pixel_to_ndc(center, viewport);
    let b = barycentrics(clip, ndc);

    let ndc0 = clip[0].xy / clip[0].w;
    let ndc1 = clip[1].xy / clip[1].w;
    let ndc2 = clip[2].xy / clip[2].w;
    let face_sign = select(-1.0, 1.0, projected_area(ndc0, ndc1, ndc2) > 0.0);

    let world = interp3(b, world0, world1, world2);
    let normal = normalize(interp3(
        b,
        transform_normal(instance, decode_normal(v0)),
        transform_normal(instance, decode_normal(v1)),
        transform_normal(instance, decode_normal(v2)),
    ));
    // A section with shading attributes hands its material their uv (VgScene::set_cluster_shading).
    var uv = interp2(b, decode_uv(instance, v0), decode_uv(instance, v1), decode_uv(instance, v2));
    let suv0 = vertex_uv(i0);
    let suv1 = vertex_uv(i1);
    let suv2 = vertex_uv(i2);
    if (suv0.z * suv1.z * suv2.z > 0.0) {
        uv = interp2(b, suv0.xy, suv1.xy, suv2.xy);
    }

    // S12 UV motion: current unjittered NDC minus previous, positive X right and Y down.
    var motion = vec2<f32>(0.0);
    if ((instance.ids.z & PREVIOUS_VALID) != 0u && frame.flags.y != 0u) {
        let local = interp3(b, local0, local1, local2);
        let prev_world = transform_point_prev(instance, local);
        let current_clip = frame.unjittered_view_proj * vec4<f32>(world, 1.0);
        let previous_clip = frame.previous_view_proj * vec4<f32>(prev_world, 1.0);
        if (current_clip.w > 0.0 && previous_clip.w > 0.0) {
            let current_ndc = current_clip.xy / current_clip.w;
            let previous_ndc = previous_clip.xy / previous_clip.w - frame.camera_jitter.zw;
            motion = vec2<f32>(
                (current_ndc.x - previous_ndc.x) * 0.5,
                (previous_ndc.y - current_ndc.y) * 0.5,
            );
        }
    }

    let level = min(cluster_level(cluster), 63u);
    var material_fn = 0u;
    if (material.header.x == MATERIAL_MODEL_FUNCTION) {
        material_fn = material.header.y;
    }

    // RAA: with no debug view the albedo holds the interpolated per-vertex colour, linear rgba
    // at f32 and unclamped (white without a stream); `raa_vg_material_input` hands it to the
    // material as `MaterialInput.color`, exactly the forward path's colour varying. A debug
    // view writes its colour (rank table of `DebugView`): 1 cluster colours, 2 the DAG level
    // ramp, 3 hardware versus software, 4 the first phase versus recovered by the second and
    // 5 one colour per instance, picked sRGB-encoded and written linear.
    let debug_view = frame.flags.x;
    let vertex_rgba = interp4(b, vertex_color(i0), vertex_color(i1), vertex_color(i2));
    var albedo = vec3<f32>(0.0);
    if (debug_view == 1u) {
        albedo = hash_color(cluster_index);
    } else if (debug_view == 2u) {
        albedo = encode_srgb(lod_ramp(level));
    } else if (debug_view == 3u) {
        albedo = encode_srgb(raster_color(entry.cluster));
    } else if (debug_view == 4u) {
        albedo = encode_srgb(occlusion_color(entry.cluster));
    } else if (debug_view == 5u) {
        // One hash colour per scene instance (its slot), so neighbouring instances differ.
        albedo = hash_color(instance.ids.x ^ 0x9e3779b9u);
    }
    var albedo_out = vec4<f32>(decode_srgb(albedo), 1.0);
    if (debug_view == 0u) {
        albedo_out = vertex_rgba;
    }

    let t0 = authored_tangent(instance, v0);
    let t1 = authored_tangent(instance, v1);
    let t2 = authored_tangent(instance, v2);
    var tangent_word = 0u;
    if t0.w * t1.w * t2.w != 0.0 {
        tangent_word = pack_surface_tangent(interp4(b, t0, t1, t2));
    }
    textureStore(
        out_ids,
        pixel,
        vec4<u32>(instance.ids.x + 1u, material_fn, tangent_word, bitcast<u32>(depth)),
    );
    textureStore(out_normal, pixel, vec4<f32>(normal, face_sign));
    textureStore(out_uv_motion, pixel, vec4<f32>(uv, motion));
    textureStore(out_albedo, pixel, albedo_out);
}
