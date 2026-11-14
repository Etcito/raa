// S39 depth pyramid: mip 0 copies the S12 depth and every later mip keeps the nearest depth
// of its 2x2 source footprint through DEPTH_WGSL's `raa_depth_nearest`, so the screen trace
// can reject a whole tile in front of its nearest surface. There is no padding: a destination
// texel i covers source 2i..2i+1, and the last destination column or row also takes the extra
// odd source column or row, up to a 3x3 footprint. Levels are max(1, w/2) x max(1, h/2) down
// to 1x1.
//
// Adapted from solstice@7e465b5,
// crates/solstice-render/shaders/gi/depth_pyramid.wgsl:12-36. MIT License
//
// Copyright (c) 2026 SamG-Coder
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Group-2 binding registry of contract 'gi-screen-wgsl' (binding number to resource):
// copy {0 raa_gi_pyr_depth, 2 raa_gi_pyr_dst}; reduce {1 raa_gi_pyr_src, 2 raa_gi_pyr_dst};
// probe_common {0 raa_gi_frame, 1 raa_gi_depth, 2 raa_gi_normal, 7 raa_gi_screen_params}.

@group(2) @binding(0) var raa_gi_pyr_depth: texture_depth_2d;
// A one-mip view of the level being reduced.
@group(2) @binding(1) var raa_gi_pyr_src: texture_2d<f32>;
@group(2) @binding(2) var raa_gi_pyr_dst: texture_storage_2d<r32float, write>;

// Copy the S12 depth into mip 0.
//
// Handoff: card 'raa-gi screen: passes.rs module builders, group-2 layouts and pipelines'
// builds the compute pipelines on `raa_gi_pyramid_copy` and `raa_gi_pyramid_reduce`; today
// they copy the S12 depth and reduce each mip with `raa_depth_nearest` under the group-2
// registry above.
@compute @workgroup_size(8, 8)
fn raa_gi_pyramid_copy(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(raa_gi_pyr_dst)) {
        return;
    }
    let depth = textureLoad(raa_gi_pyr_depth, id.xy, 0);
    textureStore(raa_gi_pyr_dst, id.xy, vec4<f32>(depth, 0.0, 0.0, 0.0));
}

// Reduce one level into the next mip, keeping the nearest depth of the up-to-3x3 footprint.
@compute @workgroup_size(8, 8)
fn raa_gi_pyramid_reduce(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(raa_gi_pyr_dst);
    if any(id.xy >= size) {
        return;
    }
    let source_size = textureDimensions(raa_gi_pyr_src);
    let base = id.xy * 2u;
    var last = base + vec2<u32>(1u);
    if id.x + 1u == size.x {
        last.x = min(base.x + 2u, source_size.x - 1u);
    }
    if id.y + 1u == size.y {
        last.y = min(base.y + 2u, source_size.y - 1u);
    }
    var nearest = RAA_DEPTH_FAR;
    for (var y = base.y; y <= last.y; y++) {
        for (var x = base.x; x <= last.x; x++) {
            let depth = textureLoad(raa_gi_pyr_src, vec2<u32>(x, y), 0).r;
            nearest = raa_depth_nearest(nearest, depth);
        }
    }
    textureStore(raa_gi_pyr_dst, id.xy, vec4<f32>(nearest, 0.0, 0.0, 0.0));
}
