//! Registry of material functions (S11): built-in reference functions and user WGSL.

use crate::composer::Composer;
use crate::error::ShaderError;
use crate::ids::{MaterialFnId, PassKind, VariantKey};
use crate::{LIB_WGSL, PRELUDE_WGSL};

/// Name of the built-in function every registry holds as [`MaterialFnId::STANDARD`].
const STANDARD_NAME: &str = "raa.standard";

/// Source of `raa.standard`: every output field from the material record, nothing else.
const STANDARD_WGSL: &str = "fn material_main(in: MaterialInput) -> MaterialOutput {
    return raa_default_output(in);
}
";

/// One registered material function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaterialFnEntry {
    pub name: String,
    pub wgsl: String,
    /// The source defines `material_vertex` (sets VERTEX_FN on its materials).
    pub has_vertex: bool,
    /// Registered by the engine rather than from JS.
    pub builtin: bool,
}

/// Material functions by dense id; `raa.standard` is always id 0.
#[derive(Clone, Debug)]
pub struct MaterialFnRegistry {
    entries: Vec<MaterialFnEntry>,
}

impl Default for MaterialFnRegistry {
    fn default() -> MaterialFnRegistry {
        MaterialFnRegistry::new()
    }
}

impl MaterialFnRegistry {
    /// A registry holding only `raa.standard` as [`MaterialFnId::STANDARD`].
    pub fn new() -> MaterialFnRegistry {
        MaterialFnRegistry {
            entries: vec![MaterialFnEntry {
                name: STANDARD_NAME.to_string(),
                wgsl: STANDARD_WGSL.to_string(),
                has_vertex: false,
                builtin: true,
            }],
        }
    }

    /// Registers an engine-provided function under the next id. The source is trusted
    /// (built-ins are validated by the `variants` tests of the crate that ships them);
    /// only a duplicate name is refused (`material.fn_duplicate`).
    pub fn register_builtin(
        &mut self,
        name: &str,
        wgsl: &str,
    ) -> Result<MaterialFnId, ShaderError> {
        self.push(name, wgsl, true)
    }

    /// Registers a user function (WGSL from JS) under the next id after checking the S11
    /// user rules, each refused with `material.fn_invalid` and a reason naming the rule:
    /// the source is at most 65536 bytes; it uses none of `@group`, `@binding`,
    /// `@vertex`, `@fragment`, `@compute`, `var<storage`, `var<uniform`, `var<workgroup`,
    /// `enable`, `override` (token scan, comments ignored); it parses; it defines no
    /// module-scope identifier starting `raa_`; it defines `material_main` with the exact
    /// signature `fn material_main(in: MaterialInput) -> MaterialOutput`; and the Forward
    /// variant (MSAA 4) composed with `composer` validates (naga message and line:col
    /// kept). A registered name is refused with `material.fn_duplicate`; a slot without
    /// provider in `composer` with `shader.slot_missing`. A refused function leaves the
    /// registry unchanged.
    pub fn register_user(
        &mut self,
        name: &str,
        wgsl: &str,
        composer: &Composer,
    ) -> Result<MaterialFnId, ShaderError> {
        if self.id_of(name).is_some() {
            return Err(duplicate(name));
        }
        check_user_source(wgsl).map_err(|reason| invalid(name, reason))?;
        let id = self.push(name, wgsl, false)?;
        let key = VariantKey {
            material_fn: id,
            pass: PassKind::Forward,
            msaa: 4,
            flags: 0,
        };
        let checked = composer.compose(self, &key).and_then(|module| {
            Composer::validate(&module)
                .map_err(|e| invalid(name, composed_reason(&module, wgsl, &e)))
        });
        if let Err(e) = checked {
            self.entries.pop();
            return Err(e);
        }
        Ok(id)
    }

    pub(crate) fn push(
        &mut self,
        name: &str,
        wgsl: &str,
        builtin: bool,
    ) -> Result<MaterialFnId, ShaderError> {
        if self.id_of(name).is_some() {
            return Err(duplicate(name));
        }
        let id = u32::try_from(self.entries.len()).map_err(|_| {
            ShaderError::new(
                ShaderError::FN_INVALID,
                format!("cannot register `{name}`: the registry is full"),
            )
        })?;
        self.entries.push(MaterialFnEntry {
            name: name.to_string(),
            wgsl: wgsl.to_string(),
            has_vertex: defines_fn(wgsl, "material_vertex"),
            builtin,
        });
        Ok(MaterialFnId(id))
    }

    /// The function registered under `id`.
    pub fn get(&self, id: MaterialFnId) -> Option<&MaterialFnEntry> {
        self.entries.get(id.0 as usize)
    }

    /// Id of the function registered as `name`.
    pub fn id_of(&self, name: &str) -> Option<MaterialFnId> {
        self.entries
            .iter()
            .position(|e| e.name == name)
            .map(|i| MaterialFnId(i as u32))
    }

    /// Registered names in id order.
    pub fn names(&self) -> Vec<String> {
        self.entries.iter().map(|e| e.name.clone()).collect()
    }

    /// Number of registered functions (at least 1: `raa.standard`).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Always false: `raa.standard` is registered by [`MaterialFnRegistry::new`].
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Largest user material function source, in bytes (S11).
const USER_FN_MAX_BYTES: usize = 65536;

/// Attributes a user function may not use (no bindings, no entry points).
const FORBIDDEN_ATTRIBUTES: [&str; 5] = ["group", "binding", "vertex", "fragment", "compute"];
/// Address spaces a user function may not declare variables in.
const FORBIDDEN_ADDRESS_SPACES: [&str; 3] = ["storage", "uniform", "workgroup"];
/// Directives and declarations a user function may not use.
const FORBIDDEN_KEYWORDS: [&str; 2] = ["enable", "override"];

/// The one signature `material_main` must have.
const MATERIAL_MAIN_SIGNATURE: &str = "fn material_main(in: MaterialInput) -> MaterialOutput";

fn duplicate(name: &str) -> ShaderError {
    ShaderError::new(
        ShaderError::FN_DUPLICATE,
        format!("material function `{name}` is already registered"),
    )
}

fn invalid(name: &str, reason: String) -> ShaderError {
    ShaderError::new(
        ShaderError::FN_INVALID,
        format!("material function `{name}` refused: {reason}"),
    )
}

/// The S11 source rules a user function must pass before composition, in order: size,
/// forbidden attributes and keywords, parse, `raa_` prefix, `material_main` signature.
/// `Err` holds the reason, starting with the rule name.
fn check_user_source(src: &str) -> Result<(), String> {
    if src.len() > USER_FN_MAX_BYTES {
        return Err(format!(
            "size limit: the source is {} bytes, more than {USER_FN_MAX_BYTES}",
            src.len()
        ));
    }
    check_tokens(src)?;

    // naga has no AST API and a user function refers to the prelude structs and LIB
    // helpers, so the user source is parsed after PRELUDE and LIB (no slots, no pass) and
    // only the items whose span falls inside the user source are its definitions.
    let context = format!("{PRELUDE_WGSL}\n{LIB_WGSL}\n");
    let offset = context.len();
    let full = context + src;
    let module = match naga::front::wgsl::parse_str(&full) {
        Ok(module) => module,
        Err(e) => return Err(parse_reason(src, offset, &full, &e)),
    };
    let user_span = |span: naga::Span| {
        span.to_range()
            .filter(|r| r.start >= offset)
            .map(|r| r.start - offset)
    };

    let mut defined: Vec<(usize, &str, &str)> = Vec::new();
    for (h, f) in module.functions.iter() {
        if let (Some(at), Some(name)) = (user_span(module.functions.get_span(h)), &f.name) {
            defined.push((at, "fn", name));
        }
    }
    for (h, t) in module.types.iter() {
        if let (Some(at), Some(name)) = (user_span(module.types.get_span(h)), &t.name) {
            defined.push((at, "struct", name));
        }
    }
    for (h, c) in module.constants.iter() {
        if let (Some(at), Some(name)) = (user_span(module.constants.get_span(h)), &c.name) {
            defined.push((at, "const", name));
        }
    }
    for (h, v) in module.global_variables.iter() {
        if let (Some(at), Some(name)) = (user_span(module.global_variables.get_span(h)), &v.name) {
            defined.push((at, "var", name));
        }
    }
    defined.sort();
    if let Some((at, kind, ident)) = defined.iter().find(|(_, _, n)| n.starts_with("raa_")) {
        let (line, col) = line_col(src, *at);
        return Err(format!(
            "reserved prefix: {kind} `{ident}` at {line}:{col} starts with `raa_` (reserved for the engine)"
        ));
    }

    let main = module.functions.iter().find(|(h, f)| {
        f.name.as_deref() == Some("material_main")
            && user_span(module.functions.get_span(*h)).is_some()
    });
    let Some((_, main)) = main else {
        return Err(format!(
            "material_main signature: the source does not define `{MATERIAL_MAIN_SIGNATURE}`"
        ));
    };
    let type_is = |ty: naga::Handle<naga::Type>, name: &str| {
        module
            .types
            .get_handle(ty)
            .ok()
            .and_then(|t| t.name.as_deref())
            == Some(name)
    };
    let exact = match (main.arguments.as_slice(), &main.result) {
        ([arg], Some(result)) => {
            arg.binding.is_none()
                && type_is(arg.ty, "MaterialInput")
                && result.binding.is_none()
                && type_is(result.ty, "MaterialOutput")
        }
        _ => false,
    };
    if !exact {
        return Err(format!(
            "material_main signature: `material_main` must be `{MATERIAL_MAIN_SIGNATURE}`"
        ));
    }
    Ok(())
}

/// Token scan for the forbidden attributes, address spaces and keywords (comments do not
/// count). The reason gives the first offending token's line:col.
fn check_tokens(src: &str) -> Result<(), String> {
    let tokens = tokens(src);
    let at = |t: &str| line_col(src, t.as_ptr() as usize - src.as_ptr() as usize);
    for (i, &t) in tokens.iter().enumerate() {
        let next = tokens.get(i + 1).copied();
        if t == "@" {
            if let Some(attr) = next.filter(|n| FORBIDDEN_ATTRIBUTES.contains(n)) {
                let (line, col) = at(t);
                return Err(format!(
                    "forbidden attribute: `@{attr}` at {line}:{col} (user functions declare no bindings or entry points)"
                ));
            }
        } else if t == "var" && next == Some("<") {
            if let Some(space) = tokens
                .get(i + 2)
                .copied()
                .filter(|s| FORBIDDEN_ADDRESS_SPACES.contains(s))
            {
                let (line, col) = at(t);
                return Err(format!(
                    "forbidden address space: `var<{space}` at {line}:{col} (only function and private variables)"
                ));
            }
        } else if FORBIDDEN_KEYWORDS.contains(&t) {
            let (line, col) = at(t);
            return Err(format!("forbidden keyword: `{t}` at {line}:{col}"));
        }
    }
    Ok(())
}

/// Reason of a parse failure of `src` placed at `offset` in `full`. A redefinition of a
/// `raa_` name (a LIB helper or prelude global) is the reserved-prefix rule; any other
/// error keeps naga's message, located in the user source when its span lies there.
fn parse_reason(src: &str, offset: usize, full: &str, e: &naga::front::wgsl::ParseError) -> String {
    let span = e.labels().next().and_then(|(span, _)| span.to_range());
    let user_at = span.as_ref().and_then(|r| r.start.checked_sub(offset));
    if let (Some(r), Some(at)) = (&span, user_at) {
        let ident = full.get(r.clone()).unwrap_or("");
        if e.message().starts_with("redefinition") && ident.starts_with("raa_") {
            let (line, col) = line_col(src, at);
            return format!(
                "reserved prefix: `{ident}` at {line}:{col} starts with `raa_` (redefines an engine item)"
            );
        }
    }
    let mut message = e.message().to_string();
    for (_, label) in e.labels() {
        if !label.is_empty() && label != message {
            message.push_str(&format!("; {label}"));
        }
    }
    for note in e.notes() {
        message.push_str(&format!("; {note}"));
    }
    match user_at {
        Some(at) => {
            let (line, col) = line_col(src, at);
            format!("parse: the source does not parse: {line}:{col}: {message}")
        }
        None => format!("parse: the source does not parse: {message}"),
    }
}

/// Reason of a validation failure of the composed Forward variant: the composer's message
/// (line:col in the composed module) followed by the line:col in the user source when the
/// error lies in the material function's section.
fn composed_reason(module: &str, src: &str, e: &ShaderError) -> String {
    let mut reason = format!(
        "composed Forward variant (msaa 4) does not validate: {}: {}",
        e.code, e.message
    );
    let header = "\n// raa section: material\n";
    let first_line = module
        .find(header)
        .map(|i| module[..i + header.len()].matches('\n').count() + 1);
    let located = e.message.split_once(": ").and_then(|(loc, _)| {
        let (line, col) = loc.split_once(':')?;
        Some((line.parse::<usize>().ok()?, col.parse::<usize>().ok()?))
    });
    if let (Some(first), Some((line, col))) = (first_line, located) {
        let user_lines = src.lines().count().max(1);
        if line >= first && line < first + user_lines {
            reason.push_str(&format!(" (user source {}:{col})", line - first + 1));
        }
    }
    reason
}

/// 1-based line and column (in characters) of byte `at` of `src`.
fn line_col(src: &str, at: usize) -> (usize, usize) {
    let prefix = src.get(..at).unwrap_or(src);
    let line = prefix.matches('\n').count() + 1;
    let line_start = prefix.rfind('\n').map_or(0, |i| i + 1);
    (line, prefix[line_start..].chars().count() + 1)
}

/// True when `src` declares a function `name` (`fn` followed by the identifier), ignoring
/// comments.
pub(crate) fn defines_fn(src: &str, name: &str) -> bool {
    let tokens = tokens(src);
    tokens
        .iter()
        .zip(tokens.iter().skip(1))
        .any(|(a, b)| *a == "fn" && *b == name)
}

/// WGSL tokens of `src`: identifiers (and numbers) as whole words, every other
/// non-blank character on its own; line and (nested) block comments skipped.
pub(crate) fn tokens(src: &str) -> Vec<&str> {
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    let mut out = Vec::new();
    let mut chars = src.char_indices().peekable();
    while let Some((start, c)) = chars.next() {
        if c.is_whitespace() {
            continue;
        }
        let next = chars.peek().map(|&(_, n)| n);
        if c == '/' && next == Some('/') {
            for (_, n) in chars.by_ref() {
                if n == '\n' {
                    break;
                }
            }
        } else if c == '/' && next == Some('*') {
            chars.next();
            let mut depth = 1u32;
            while depth > 0 {
                match chars.next() {
                    Some((_, '*')) if chars.peek().map(|&(_, n)| n) == Some('/') => {
                        chars.next();
                        depth -= 1;
                    }
                    Some((_, '/')) if chars.peek().map(|&(_, n)| n) == Some('*') => {
                        chars.next();
                        depth += 1;
                    }
                    Some(_) => {}
                    None => break,
                }
            }
        } else if is_word(c) {
            let mut end = start + c.len_utf8();
            while let Some(&(i, n)) = chars.peek() {
                if !is_word(n) {
                    break;
                }
                end = i + n.len_utf8();
                chars.next();
            }
            out.push(&src[start..end]);
        } else {
            out.push(&src[start..start + c.len_utf8()]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const VERTEX_FN: &str = "fn material_main(in: MaterialInput) -> MaterialOutput {
    return raa_default_output(in);
}
fn material_vertex(in: VertexMaterialInput) -> VertexMaterialOutput {
    return VertexMaterialOutput(in.local_position, in.normal);
}
";

    #[test]
    fn new_registers_standard_as_id_zero() {
        let reg = MaterialFnRegistry::new();
        assert_eq!(reg.len(), 1);
        assert!(!reg.is_empty());
        assert_eq!(reg.id_of("raa.standard"), Some(MaterialFnId::STANDARD));
        let standard = reg.get(MaterialFnId::STANDARD).expect("standard entry");
        assert_eq!(standard.name, "raa.standard");
        assert!(standard.builtin);
        assert!(!standard.has_vertex);
        assert!(standard.wgsl.contains("raa_default_output(in)"));
    }

    #[test]
    fn builtin_ids_are_dense() {
        let mut reg = MaterialFnRegistry::new();
        let a = reg.register_builtin("raa.a", STANDARD_WGSL).expect("a");
        let b = reg.register_builtin("raa.b", STANDARD_WGSL).expect("b");
        let c = reg.register_builtin("raa.c", STANDARD_WGSL).expect("c");
        assert_eq!(
            [a, b, c],
            [MaterialFnId(1), MaterialFnId(2), MaterialFnId(3)]
        );
        assert_eq!(reg.len(), 4);
        assert_eq!(reg.names(), ["raa.standard", "raa.a", "raa.b", "raa.c"]);
        for (i, name) in reg.names().iter().enumerate() {
            let id = MaterialFnId(i as u32);
            assert_eq!(reg.id_of(name), Some(id));
            assert_eq!(reg.get(id).map(|e| e.name.as_str()), Some(name.as_str()));
        }
        assert_eq!(reg.get(MaterialFnId(4)), None);
        assert_eq!(reg.id_of("raa.missing"), None);
    }

    #[test]
    fn duplicate_name_is_refused() {
        let mut reg = MaterialFnRegistry::new();
        reg.register_builtin("raa.a", STANDARD_WGSL).expect("a");
        for name in ["raa.a", "raa.standard"] {
            let e = reg.register_builtin(name, STANDARD_WGSL).unwrap_err();
            assert_eq!(e.code(), "material.fn_duplicate");
            assert!(e.message.contains(name), "{e}");
        }
        assert_eq!(reg.len(), 2);
        assert_eq!(
            reg.register_builtin("raa.b", STANDARD_WGSL),
            Ok(MaterialFnId(2))
        );
    }

    #[test]
    fn has_vertex_follows_material_vertex_definition() {
        let mut reg = MaterialFnRegistry::new();
        let id = reg
            .register_builtin("raa.wobble", VERTEX_FN)
            .expect("wobble");
        assert!(reg.get(id).expect("entry").has_vertex);

        let commented = format!(
            "{STANDARD_WGSL}// fn material_vertex(in: VertexMaterialInput)\n\
             /* fn material_vertex /* nested */ fn material_vertex */\n\
             fn material_vertex_helper() {{}}\n\
             fn f() {{ let material_vertex = 1; }}\n"
        );
        let id = reg.register_builtin("raa.flat", &commented).expect("flat");
        assert!(!reg.get(id).expect("entry").has_vertex);
    }

    #[test]
    fn tokens_split_words_and_punctuation() {
        let t = tokens("fn  f(a: vec3<f32>) -> f32 { return a.x*2.0; } // tail");
        assert_eq!(
            t,
            [
                "fn", "f", "(", "a", ":", "vec3", "<", "f32", ">", ")", "-", ">", "f32", "{",
                "return", "a", ".", "x", "*", "2", ".", "0", ";", "}"
            ]
        );
        assert_eq!(tokens("a /* unterminated"), ["a"]);
    }
}
