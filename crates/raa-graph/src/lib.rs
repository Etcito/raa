//! Render graph: the `raa.render-graph` document format, the node catalog, validation,
//! planning and the executor of the lit pipeline (S50-S53).

pub mod catalog;
pub mod error;
pub mod format;
pub mod plan;
pub mod validate;

#[rustfmt::skip] // the file lands with the executor cards
#[cfg(raa_graph_exec)]
pub mod exec;

pub mod registry {
    include!(concat!(env!("OUT_DIR"), "/registry.rs"));
}

pub use catalog::Catalog;
pub use error::Issue;
pub use format::GraphDoc;
pub use plan::GraphPlan;
pub use plan::plan_from_json;

/// The frozen pipelines discovered by the build, in registry order.
pub fn frozen_names() -> Vec<&'static str> {
    registry::frozen_all()
        .into_iter()
        .map(|(name, _, _)| name)
        .collect()
}

/// The plan of the frozen pipeline with that name, `None` when absent.
pub fn frozen_plan(name: &str) -> Option<GraphPlan> {
    registry::frozen_all()
        .into_iter()
        .find(|(candidate, _, _)| *candidate == name)
        .map(|(_, plan, _)| plan())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_is_empty_without_extension_directories() {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let dirs = [
            "src/nodes",
            "src/exec_ext",
            "src/frozen",
            "graphs",
            "presets",
        ];
        if dirs.iter().any(|dir| manifest.join(dir).exists()) {
            return;
        }
        assert!(registry::STARTERS.is_empty());
        assert!(registry::PRESETS.is_empty());
        assert!(registry::families_all().is_empty());
        assert!(frozen_names().is_empty());
        assert!(frozen_plan("simple_forward").is_none());
    }
}
