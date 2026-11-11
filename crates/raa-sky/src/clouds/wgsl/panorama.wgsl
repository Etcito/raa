// Cloud panorama of the M2 volumetric cloud backend: the clouds over the whole sky at
// low resolution, laid out like the sky-view table, for the environment cube and IBL.
//
// Adapted from Solstice revision 7e465b599c13da165948ebaa44df824c36c3ae07,
// crates/solstice-render/shaders/clouds/panorama.wgsl:15-29.
// MIT License, Copyright (c) 2026 SamG-Coder.
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
// `clouds/shaders.rs` composes this text after `cloud_volume_wgsl()` and the output
// binding 10 `raa_cloud_output` (192x108 Rgba16Float). RGB is linear HDR cloud radiance,
// A the scalar transmittance; texels below the horizon stay neutral (0, 0, 0, 1).

@compute @workgroup_size(8, 8)
fn raa_cloud_panorama_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(raa_cloud_output);
    if any(id.xy >= size) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let direction = raa_atmosphere_sky_view_direction(uv);
    var result = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if direction.y > -0.02 {
        let ray = normalize(vec3<f32>(direction.x, max(direction.y, 0.002), direction.z));
        let clouds_here = raa_cloud_along(ray, 32, 4, 0.5);
        result = vec4<f32>(clouds_here.radiance, clouds_here.transmittance);
    }
    textureStore(raa_cloud_output, id.xy, result);
}
