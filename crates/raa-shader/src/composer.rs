//! Shader composition (S11) and naga validation of composed WGSL.

use naga::valid::{Capabilities, ValidationFlags, Validator};

use crate::error::ShaderError;

/// Builds the WGSL module of a shader variant from the registry and the slot providers.
#[derive(Clone, Debug, Default)]
pub struct Composer {
    _private: (),
}

impl Composer {
    /// A composer without slot providers.
    pub fn new() -> Composer {
        Composer::default()
    }

    /// Parses `wgsl` with the naga WGSL front-end and validates the module with the
    /// capabilities core WebGPU allows. Refuses with `shader.parse` or `shader.validate`;
    /// the message starts with the line:col of the first span when naga gives one.
    pub fn validate(wgsl: &str) -> Result<(), ShaderError> {
        let module = naga::front::wgsl::parse_str(wgsl).map_err(|e| {
            let mut message = e.message().to_string();
            for (_, label) in e.labels() {
                if !label.is_empty() && label != message {
                    message.push_str(&format!("; {label}"));
                }
            }
            for note in e.notes() {
                message.push_str(&format!("; {note}"));
            }
            let span = e.labels().next().map(|(span, _)| span);
            ShaderError::new(ShaderError::PARSE, located(wgsl, span, message))
        })?;
        let mut validator = Validator::new(ValidationFlags::all(), core_webgpu_capabilities());
        validator.validate(&module).map_err(|e| {
            let mut message = e.to_string();
            let mut source = std::error::Error::source(&e);
            while let Some(cause) = source {
                message.push_str(&format!(": {cause}"));
                source = cause.source();
            }
            let mut spans: Vec<String> = Vec::new();
            for (span, label) in e.spans() {
                let label = label.trim_start_matches("naga::ir::");
                spans.push(located(wgsl, Some(*span), label.to_string()));
            }
            if !spans.is_empty() {
                message.push_str(&format!(" (at {})", spans.join("; ")));
            }
            // The narrowest span is the most specific one (an expression inside the
            // function that carries the first span).
            let span = e
                .spans()
                .filter_map(|(span, _)| Some((*span, span.to_range()?.len())))
                .min_by_key(|&(_, len)| len)
                .map(|(span, _)| span);
            ShaderError::new(ShaderError::VALIDATE, located(wgsl, span, message))
        })?;
        Ok(())
    }
}

/// Shader capabilities of a core WebGPU device without optional features: what wgpu
/// derives from compliant downlevel flags (multisampled shading, cube array textures,
/// f16 packing in f32).
fn core_webgpu_capabilities() -> Capabilities {
    Capabilities::MULTISAMPLED_SHADING
        | Capabilities::CUBE_ARRAY_TEXTURES
        | Capabilities::SHADER_FLOAT16_IN_FLOAT32
}

/// `message` prefixed with the 1-based `line:col` of `span` in `source` (column in
/// characters); unchanged when the span is absent or does not fall inside `source`.
fn located(source: &str, span: Option<naga::Span>, message: String) -> String {
    let prefix = span
        .and_then(|s| s.to_range())
        .and_then(|range| source.get(..range.start));
    match prefix {
        Some(prefix) => {
            let line = prefix.matches('\n').count() + 1;
            let line_start = prefix.rfind('\n').map_or(0, |i| i + 1);
            let col = prefix[line_start..].chars().count() + 1;
            format!("{line}:{col}: {message}")
        }
        None => message,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRIVIAL: &str = "@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i) * 0.5;
    return vec4<f32>(x, 0.0, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.5, 0.25, 1.0);
}
";

    #[test]
    fn validate_accepts_trivial_module() {
        assert_eq!(Composer::validate(TRIVIAL), Ok(()));
    }

    #[test]
    fn validate_refuses_syntax_error_with_line_col() {
        let src = "fn f() -> f32 {\n    let x = 1.0\n    return x;\n}\n";
        let e = Composer::validate(src).unwrap_err();
        assert_eq!(e.code(), "shader.parse");
        assert!(e.message.starts_with("3:5: "), "{e}");
    }

    #[test]
    fn validate_refuses_type_error() {
        let src = "fn f() -> f32 {\n    return 1u;\n}\n";
        let e = Composer::validate(src).unwrap_err();
        assert_eq!(e.code(), "shader.validate");
        assert!(e.message.starts_with("2:12: "), "{e}");
        assert!(e.message.contains("1:1: Function [0]"), "{e}");
    }

    #[test]
    fn validate_allows_core_and_refuses_optional_capabilities() {
        let cube_array = "@group(0) @binding(0) var env: texture_cube_array<f32>;
@group(0) @binding(1) var s: sampler;
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return textureSample(env, s, vec3<f32>(0.0, 0.0, 1.0), 0);
}
";
        assert_eq!(Composer::validate(cube_array), Ok(()));
        let f64_src = "fn f() -> f64 {\n    return 1.0lf;\n}\n";
        let e = Composer::validate(f64_src).unwrap_err();
        assert_eq!(e.code(), "shader.validate", "{e}");
    }

    #[test]
    fn located_counts_lines_and_characters() {
        let src = "ab\ncé x";
        let span = naga::Span::new(7, 8);
        assert_eq!(located(src, Some(span), "m".into()), "2:4: m");
        assert_eq!(located(src, None, "m".into()), "m");
        assert_eq!(located(src, Some(naga::Span::new(40, 41)), "m".into()), "m");
    }
}
