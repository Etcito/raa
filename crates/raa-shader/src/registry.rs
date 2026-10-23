//! Registry of material functions (S11): built-in reference functions and user WGSL.

use crate::error::ShaderError;
use crate::ids::MaterialFnId;

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

    pub(crate) fn push(
        &mut self,
        name: &str,
        wgsl: &str,
        builtin: bool,
    ) -> Result<MaterialFnId, ShaderError> {
        if self.id_of(name).is_some() {
            return Err(ShaderError::new(
                ShaderError::FN_DUPLICATE,
                format!("material function `{name}` is already registered"),
            ));
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
