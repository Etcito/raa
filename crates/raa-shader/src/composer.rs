//! Shader composition (S11) and naga validation of composed WGSL.

use std::borrow::Cow;

use naga::valid::{Capabilities, ValidationFlags, Validator};

use crate::error::ShaderError;
use crate::ids::{PassKind, SlotId, VariantKey};
use crate::registry::MaterialFnRegistry;
use crate::{
    LIB_WGSL, PASS_FORWARD_WGSL, PASS_FURSHELL_WGSL, PASS_PREPASS_WGSL, PASS_SHADOW_WGSL,
    PASS_TRANSPARENT_WGSL, PRELUDE_WGSL,
};

/// Inserted after the material function when it defines no `material_vertex`.
const PASSTHROUGH_MATERIAL_VERTEX: &str =
    "fn material_vertex(in: VertexMaterialInput) -> VertexMaterialOutput {
    return VertexMaterialOutput(in.local_position, in.normal);
}
";

/// FNV-1a 64-bit offset basis and prime.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Implementation of one slot function (S11). `wgsl` defines exactly
/// [`SlotId::fn_name`] plus private helpers prefixed `raa_<provider>_`.
#[derive(Clone, Debug)]
pub struct SlotProvider {
    pub slot: SlotId,
    /// The highest priority on a slot wins; the default providers use 0.
    pub priority: i32,
    pub provider: &'static str,
    pub wgsl: Cow<'static, str>,
}

/// Builds the WGSL module of a shader variant from the registry and the slot providers.
#[derive(Clone, Debug, Default)]
pub struct Composer {
    providers: Vec<SlotProvider>,
}

impl Composer {
    /// A composer without slot providers.
    pub fn new() -> Composer {
        Composer::default()
    }

    /// Installs a provider. Refuses `shader.slot_conflict` when the slot already has a
    /// provider with the same priority (the composer is then unchanged).
    pub fn add_slot(&mut self, p: SlotProvider) -> Result<(), ShaderError> {
        if let Some(other) = self
            .providers
            .iter()
            .find(|q| q.slot == p.slot && q.priority == p.priority)
        {
            return Err(ShaderError::new(
                ShaderError::SLOT_CONFLICT,
                format!(
                    "slot {:?} (`{}`): providers {:?} and {:?} both have priority {}",
                    p.slot,
                    p.slot.fn_name(),
                    other.provider,
                    p.provider,
                    p.priority
                ),
            ));
        }
        self.providers.push(p);
        Ok(())
    }

    /// The provider of `slot` with the highest priority.
    pub fn winner(&self, slot: SlotId) -> Option<&SlotProvider> {
        // Priorities on one slot are distinct (add_slot), so the maximum is unique.
        self.providers
            .iter()
            .filter(|p| p.slot == slot)
            .max_by_key(|p| p.priority)
    }

    /// FNV-1a (64-bit) over the winner of every slot in [`SlotId::ALL`] order: provider
    /// name, priority and source, each length-prefixed; a slot without provider hashes a
    /// marker byte. Providers that do not win do not change it.
    pub fn slots_hash(&self) -> u64 {
        let mut h = FNV_OFFSET;
        for slot in SlotId::ALL {
            match self.winner(slot) {
                Some(p) => {
                    h = fnv1a(h, &[1]);
                    h = fnv1a(h, &(p.provider.len() as u64).to_le_bytes());
                    h = fnv1a(h, p.provider.as_bytes());
                    h = fnv1a(h, &p.priority.to_le_bytes());
                    h = fnv1a(h, &(p.wgsl.len() as u64).to_le_bytes());
                    h = fnv1a(h, p.wgsl.as_bytes());
                }
                None => h = fnv1a(h, &[0]),
            }
        }
        h
    }

    /// The WGSL module of `key`, in the S11 order: PRELUDE, LIB, the material function, a
    /// passthrough `material_vertex` when the function defines none, the winner of each
    /// slot in [`SlotId::ALL`] order, then the pass entry points preceded by the variant
    /// constants `RAA_MSAA` and `RAA_VARIANT_FLAGS`. Refuses `shader.unknown_fn` when the
    /// registry has no function `key.material_fn` and `shader.slot_missing` when a slot has
    /// no provider. The result is not validated (see [`Composer::validate`]).
    pub fn compose(
        &self,
        reg: &MaterialFnRegistry,
        key: &VariantKey,
    ) -> Result<String, ShaderError> {
        let entry = reg.get(key.material_fn).ok_or_else(|| {
            ShaderError::new(
                ShaderError::UNKNOWN_FN,
                format!(
                    "material function id {} is not registered ({} registered)",
                    key.material_fn.0,
                    reg.len()
                ),
            )
        })?;
        let mut winners = Vec::with_capacity(SlotId::ALL.len());
        for slot in SlotId::ALL {
            let p = self.winner(slot).ok_or_else(|| {
                ShaderError::new(
                    ShaderError::SLOT_MISSING,
                    format!("slot {slot:?} (`{}`) has no provider", slot.fn_name()),
                )
            })?;
            winners.push(p);
        }

        let mut out = String::new();
        out.push_str(&format!(
            "// raa variant: pass {:?}, msaa {}, flags {:#x}, slots {:016x}\n",
            key.pass,
            key.msaa,
            key.flags,
            self.slots_hash()
        ));
        section(&mut out, "prelude", PRELUDE_WGSL);
        section(&mut out, "lib", LIB_WGSL);
        section(&mut out, "material", &entry.wgsl);
        if !entry.has_vertex {
            section(
                &mut out,
                "material_vertex passthrough",
                PASSTHROUGH_MATERIAL_VERTEX,
            );
        }
        for p in winners {
            let title = format!(
                "slot {:?}, provider {:?}, priority {}",
                p.slot, p.provider, p.priority
            );
            section(&mut out, &title, &p.wgsl);
        }
        let pass = format!(
            "const RAA_MSAA: u32 = {}u;\nconst RAA_VARIANT_FLAGS: u32 = {}u;\n\n{}",
            key.msaa,
            key.flags,
            pass_wgsl(key.pass)
        );
        section(&mut out, &format!("pass {:?}", key.pass), &pass);
        Ok(out)
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

/// Entry points of `pass`.
fn pass_wgsl(pass: PassKind) -> &'static str {
    match pass {
        PassKind::Prepass => PASS_PREPASS_WGSL,
        PassKind::Forward => PASS_FORWARD_WGSL,
        PassKind::Shadow => PASS_SHADOW_WGSL,
        PassKind::Transparent => PASS_TRANSPARENT_WGSL,
        PassKind::FurShell => PASS_FURSHELL_WGSL,
    }
}

/// Appends a `// raa section: <title>` comment line, then `body` ending with a newline.
fn section(out: &mut String, title: &str, body: &str) {
    out.push_str("\n// raa section: ");
    out.push_str(title);
    out.push('\n');
    out.push_str(body);
    if !body.ends_with('\n') {
        out.push('\n');
    }
}

fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
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
    fn fnv1a_matches_reference_vectors() {
        assert_eq!(fnv1a(FNV_OFFSET, b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(FNV_OFFSET, b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a(FNV_OFFSET, b"foobar"), 0x8594_4171_f739_67e8);
        assert_eq!(
            fnv1a(fnv1a(FNV_OFFSET, b"foo"), b"bar"),
            0x8594_4171_f739_67e8
        );
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
