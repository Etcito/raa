//! The asset loader trait and the format registry.

use crate::{LoadContext, LoadError, LoadedAsset};

/// One asset format.
pub trait AssetLoader {
    /// Stable format name, for example `raa.prims`.
    fn format(&self) -> &'static str;
    /// Cheap check whether `bytes` look like this format.
    fn sniff(&self, bytes: &[u8]) -> bool;
    /// Parses and builds the asset.
    fn load(&self, bytes: &[u8], ctx: &LoadContext<'_>) -> Result<LoadedAsset, LoadError>;
}

/// Loaders tried in registration order.
#[derive(Default)]
pub struct LoaderRegistry {
    loaders: Vec<Box<dyn AssetLoader>>,
}

impl LoaderRegistry {
    pub fn new() -> LoaderRegistry {
        LoaderRegistry {
            loaders: Vec::new(),
        }
    }

    /// Adds a loader; a format already present gives `loader.duplicate_format`.
    pub fn register(&mut self, l: Box<dyn AssetLoader>) -> Result<(), LoadError> {
        let format = l.format();
        if self.loaders.iter().any(|x| x.format() == format) {
            return Err(LoadError::new(
                "loader.duplicate_format",
                "",
                format!("a loader for format '{format}' is already registered"),
            ));
        }
        self.loaders.push(l);
        Ok(())
    }

    /// Loads with the first loader whose sniff accepts `bytes`, else `loader.unknown_format`.
    pub fn load(&self, bytes: &[u8], ctx: &LoadContext<'_>) -> Result<LoadedAsset, LoadError> {
        match self.loaders.iter().find(|l| l.sniff(bytes)) {
            Some(l) => l.load(bytes, ctx),
            None => Err(LoadError::new(
                "loader.unknown_format",
                "",
                format!(
                    "no registered loader recognises the bytes (registered formats: [{}])",
                    self.formats().join(", ")
                ),
            )),
        }
    }

    /// Registered formats in registration order.
    pub fn formats(&self) -> Vec<&'static str> {
        self.loaders.iter().map(|l| l.format()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NoBooleans;
    use raa_geom::{GeometryRegistry, Quality};

    struct Dummy {
        format: &'static str,
        prefix: &'static [u8],
    }

    impl AssetLoader for Dummy {
        fn format(&self) -> &'static str {
            self.format
        }
        fn sniff(&self, bytes: &[u8]) -> bool {
            bytes.starts_with(self.prefix)
        }
        fn load(&self, _bytes: &[u8], _ctx: &LoadContext<'_>) -> Result<LoadedAsset, LoadError> {
            Ok(LoadedAsset {
                format: self.format.to_string(),
                id: String::new(),
                name: String::new(),
                meshes: Vec::new(),
                materials: Vec::new(),
                anchors: Vec::new(),
                colliders: Vec::new(),
                triangles: 0,
                budget: 0,
            })
        }
    }

    fn dummy(format: &'static str, prefix: &'static [u8]) -> Box<dyn AssetLoader> {
        Box::new(Dummy { format, prefix })
    }

    fn with_ctx(f: impl FnOnce(&LoadContext<'_>)) {
        let geometry = GeometryRegistry::new();
        let ctx = LoadContext {
            geometry: &geometry,
            booleans: &NoBooleans,
            quality: Quality::LOD0,
            strict_kinds: true,
        };
        f(&ctx);
    }

    #[test]
    fn duplicate_format_is_refused() {
        let mut r = LoaderRegistry::new();
        r.register(dummy("a", b"A")).unwrap();
        let e = r.register(dummy("a", b"B")).unwrap_err();
        assert_eq!(e.code, "loader.duplicate_format");
        assert!(!e.message.is_empty());
        assert_eq!(r.formats(), vec!["a"]);
    }

    #[test]
    fn unknown_format_lists_registered_formats() {
        let mut r = LoaderRegistry::new();
        r.register(dummy("first", b"A")).unwrap();
        r.register(dummy("second", b"B")).unwrap();
        with_ctx(|ctx| {
            let e = r.load(b"zzz", ctx).unwrap_err();
            assert_eq!(e.code, "loader.unknown_format");
            assert_eq!(e.path, "");
            assert!(e.message.contains("first, second"), "{}", e.message);
        });
    }

    #[test]
    fn first_sniffing_loader_wins() {
        let mut r = LoaderRegistry::new();
        r.register(dummy("one", b"X")).unwrap();
        r.register(dummy("two", b"")).unwrap();
        r.register(dummy("three", b"X")).unwrap();
        with_ctx(|ctx| {
            assert_eq!(r.load(b"Xyz", ctx).unwrap().format, "one");
            assert_eq!(r.load(b"yz", ctx).unwrap().format, "two");
        });
    }

    #[test]
    fn formats_in_registration_order() {
        let mut r = LoaderRegistry::new();
        for f in ["c", "a", "b"] {
            r.register(dummy(f, b"")).unwrap();
        }
        assert_eq!(r.formats(), vec!["c", "a", "b"]);
    }
}
