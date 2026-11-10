// S10: 64-byte UNIFORM | COPY_DST record, separate from FrameGpu and EnvGpu.
struct WeatherLightingGpu {
    map: vec4<f32>,                      // 0: origin X/Z, extent in metres, cloud enabled
    light: vec4<f32>,                    // 16: unit direction toward dominant light, zero
    atmosphericTransmission: vec4<f32>,  // 32: directional RGB transmission, one
    flags: vec4<u32>,                    // 48: physical enabled, three zeros
}

const RAA_WEATHER_LIGHTING_DEFAULT: WeatherLightingGpu = WeatherLightingGpu(
    vec4<f32>(0.0, 0.0, 1.0, 0.0),
    vec4<f32>(0.0, 1.0, 0.0, 0.0),
    vec4<f32>(1.0, 1.0, 1.0, 1.0),
    vec4<u32>(0u, 0u, 0u, 0u),
);

// Fragment-only shared resources. Binding 14 remains reserved; reuse binding 6's sampler.
// Rgba16Float: layer 0 is the ground map, layer 1 texel (0,0) its mean. Disabled:
// a valid initialized 1x1 two-layer texture of ones, with map.w = 0.
@group(1) @binding(15) var raa_cloud_transmittance: texture_2d_array<f32>;
@group(1) @binding(16) var<uniform> raa_weather_lighting: WeatherLightingGpu;

// CPU writes map/light/flags; atmosphere copies its separate 16-byte compute result
// to offset 32. CPU uploads must preserve that current transmission after the copy.
// Apply once at a dominant directional consumer, before its material light response.
fn raa_directional_radiance(raw: vec3<f32>) -> vec3<f32> {
    if (raa_weather_lighting.flags.x == 0u) {
        return raw;
    }
    return raw * raa_weather_lighting.atmosphericTransmission.rgb;
}
