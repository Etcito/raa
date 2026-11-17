//! Render graph issues (S52): severity, validation stage, the issue record and its order.

use raa_loader::json::JsonValue;

use crate::format::GraphDoc;

/// How an issue affects the graph: an error refuses it, a warning only reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

/// The validation stage an issue belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Parse,
    Structure,
    Budget,
    Plan,
    Apply,
}

impl Stage {
    /// The order stages are reported in: parse 0, structure 1, budget 2, plan 3, apply 4.
    pub fn rank(&self) -> u8 {
        match self {
            Stage::Parse => 0,
            Stage::Structure => 1,
            Stage::Budget => 2,
            Stage::Plan => 3,
            Stage::Apply => 4,
        }
    }
}

/// One refusal or warning of a render graph document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Issue {
    pub code: &'static str,
    pub severity: Severity,
    pub stage: Stage,
    pub node: Option<String>,
    pub port: Option<String>,
    pub param: Option<String>,
    pub link: Option<u32>,
    pub message: String,
    pub fix: String,
}

impl Issue {
    /// The issue as `{ code, severity, nodeId?, port?, param?, link?, message, fix }`.
    pub fn to_json(&self) -> JsonValue {
        let mut entries = Vec::with_capacity(8);
        entries.push(("code".to_string(), JsonValue::String(self.code.to_string())));
        let severity = match self.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
        };
        entries.push((
            "severity".to_string(),
            JsonValue::String(severity.to_string()),
        ));
        if let Some(node) = &self.node {
            entries.push(("nodeId".to_string(), JsonValue::String(node.clone())));
        }
        if let Some(port) = &self.port {
            entries.push(("port".to_string(), JsonValue::String(port.clone())));
        }
        if let Some(param) = &self.param {
            entries.push(("param".to_string(), JsonValue::String(param.clone())));
        }
        if let Some(link) = self.link {
            entries.push(("link".to_string(), JsonValue::Number(f64::from(link))));
        }
        entries.push((
            "message".to_string(),
            JsonValue::String(self.message.clone()),
        ));
        entries.push(("fix".to_string(), JsonValue::String(self.fix.clone())));
        JsonValue::Object(entries)
    }
}

/// The sort class and index of an issue's node within the document.
fn node_rank(node: Option<&str>, doc: Option<&GraphDoc>) -> (u8, usize) {
    let Some(id) = node else {
        return (0, 0);
    };
    match doc.and_then(|doc| doc.nodes.iter().position(|known| known.id == id)) {
        Some(index) => (1, index),
        None => (2, usize::MAX),
    }
}

/// The sort class and index of an issue's link.
fn link_rank(link: Option<u32>) -> (u8, u32) {
    match link {
        Some(index) => (1, index),
        None => (0, 0),
    }
}

/// Sorts issues stably by stage rank, node document index (issues without a node first,
/// a node absent from `doc` last), link index (none first) and `code` byte-wise.
///
/// Handoff: `sort_issues` is the entry point of this file. Card 'raa-graph validate 1/2:
/// nodes, params, links and canonical_doc' calls it on the issues `validate` collected
/// with `Some(doc)`, passing the `Issue` list and the document whose node order ranks
/// them. Card 'raa-graph format: parse_graph and parse-stage refusals' calls the same
/// `sort_issues` with `None` on its parse-stage issues.
pub fn sort_issues(issues: &mut [Issue], doc: Option<&GraphDoc>) {
    issues.sort_by(|a, b| {
        a.stage
            .rank()
            .cmp(&b.stage.rank())
            .then_with(|| node_rank(a.node.as_deref(), doc).cmp(&node_rank(b.node.as_deref(), doc)))
            .then_with(|| link_rank(a.link).cmp(&link_rank(b.link)))
            .then_with(|| a.code.cmp(b.code))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{GraphDoc, NodeDoc};

    fn issue(
        code: &'static str,
        severity: Severity,
        stage: Stage,
        node: Option<&str>,
        link: Option<u32>,
    ) -> Issue {
        Issue {
            code,
            severity,
            stage,
            node: node.map(str::to_string),
            port: None,
            param: None,
            link,
            message: format!("message {code}"),
            fix: format!("fix {code}"),
        }
    }

    fn doc() -> GraphDoc {
        let node = |id: &str| NodeDoc {
            id: id.to_string(),
            type_id: "pass.forward".to_string(),
            version: None,
            params: Vec::new(),
            compute: None,
        };
        GraphDoc {
            name: "sort".to_string(),
            nodes: vec![node("a"), node("b"), node("c")],
            links: Vec::new(),
            editor: None,
        }
    }

    #[test]
    fn to_json_key_order_and_omission() {
        let mut full = issue(
            "graph.param_range",
            Severity::Warning,
            Stage::Structure,
            Some("forward"),
            Some(3),
        );
        full.port = Some("color".to_string());
        full.param = Some("radius".to_string());
        let JsonValue::Object(entries) = full.to_json() else {
            panic!("issue json is not an object");
        };
        let keys: Vec<&str> = entries.iter().map(|(key, _)| key.as_str()).collect();
        assert_eq!(
            keys,
            [
                "code", "severity", "nodeId", "port", "param", "link", "message", "fix"
            ]
        );
        assert_eq!(entries[1].1, JsonValue::String("warning".to_string()));
        assert_eq!(entries[1].0, "severity");
        assert_eq!(entries[5].1, JsonValue::Number(3.0));

        let bare = issue("graph.json", Severity::Error, Stage::Parse, None, None);
        let JsonValue::Object(entries) = bare.to_json() else {
            panic!("issue json is not an object");
        };
        let keys: Vec<&str> = entries.iter().map(|(key, _)| key.as_str()).collect();
        assert_eq!(keys, ["code", "severity", "message", "fix"]);
        assert_eq!(entries[1].1, JsonValue::String("error".to_string()));
    }

    #[test]
    fn sort_by_stage_node_link_and_code() {
        let doc = doc();
        let mut issues = vec![
            issue("p.plan", Severity::Error, Stage::Plan, Some("c"), None),
            issue("b.stage", Severity::Error, Stage::Parse, Some("b"), None),
            issue("s.none_2", Severity::Error, Stage::Structure, None, Some(2)),
            issue(
                "s.none_1",
                Severity::Warning,
                Stage::Structure,
                None,
                Some(1),
            ),
            issue(
                "s.a_link_1",
                Severity::Error,
                Stage::Structure,
                Some("a"),
                Some(1),
            ),
            issue(
                "s.gone",
                Severity::Error,
                Stage::Structure,
                Some("gone"),
                None,
            ),
            issue(
                "s.a_link_0",
                Severity::Error,
                Stage::Structure,
                Some("a"),
                Some(0),
            ),
            issue(
                "s.a_no_link",
                Severity::Error,
                Stage::Structure,
                Some("a"),
                None,
            ),
            issue("b.budget", Severity::Error, Stage::Budget, Some("b"), None),
        ];
        sort_issues(&mut issues, Some(&doc));
        let order: Vec<&str> = issues.iter().map(|issue| issue.code).collect();
        assert_eq!(
            order,
            [
                "b.stage",
                "s.none_1",
                "s.none_2",
                "s.a_no_link",
                "s.a_link_0",
                "s.a_link_1",
                "s.gone",
                "b.budget",
                "p.plan",
            ]
        );

        let mut issues = vec![
            issue("x.node", Severity::Error, Stage::Structure, Some("b"), None),
            issue("x.none", Severity::Error, Stage::Structure, None, None),
        ];
        sort_issues(&mut issues, None);
        let order: Vec<&str> = issues.iter().map(|issue| issue.code).collect();
        assert_eq!(order, ["x.none", "x.node"]);
    }
}
