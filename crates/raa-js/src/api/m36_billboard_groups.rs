//! m36 whole-tree billboard groups (M6): the shared Rust registry behind
//! `setBillboardGroup` and `billboardGroupState`, its install into `FeatureMap` and the
//! Engine registration/query adapters.
//!
//! Per-view preparation is the second share of `m6-group-registry-api`, owned by
//! "Prepare billboard groups for each camera"; until it is attached a group keeps LOD0
//! visible and reports `ready == false`. This file supplies validated state registration
//! and does not render.
//!
//! The registry part below compiles on the host and is tested by
//! `tests/billboard_groups.rs`.

use std::collections::{BTreeMap, BTreeSet};

use raa_core::RaaError;

/// Stable code of every refusal from the group registry, the same code
/// `crate::billboard_selection::BILLBOARD_INVALID` carries (this file is compiled
/// standalone by `tests/billboard_groups.rs`, which has no `crate::billboard_selection`).
pub const BILLBOARD_INVALID: &str = "billboard.invalid";

fn invalid(message: impl Into<String>) -> RaaError {
    RaaError::FrameGraph {
        code: BILLBOARD_INVALID,
        message: message.into(),
    }
}

/// The requested level of a group: automatic switching, forced LOD0 or forced proxy.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BillboardLevel {
    Auto,
    Full,
    Proxy,
}

/// The validated settings of one billboard group.
#[derive(Clone, Debug, PartialEq)]
pub struct BillboardGroupConfig {
    /// JS scene-node handle shared by the group's instances.
    pub root: u32,
    /// The complete, unique LOD0 instance handles; never contains `proxy`.
    pub full: Vec<u32>,
    /// The proxy instance handle.
    pub proxy: u32,
    /// Bake-store id; this registry does not resolve it.
    pub bake: u32,
    /// Positive projected-size threshold in pixels.
    pub switch_pixels: f32,
    /// Whole-number switch bias in -3..=3.
    pub bias: i8,
    /// Requested level.
    pub level: BillboardLevel,
}

/// The last prepared status of a group. A fresh group is level 0, direction 0,
/// not ready and has no error.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BillboardGroupStatus {
    pub level: u8,
    pub direction: u8,
    pub ready: bool,
    pub error: Option<String>,
}

/// The hysteresis record of one viewport (0..=3).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BillboardViewState {
    pub level: u8,
    pub direction: u8,
}

/// One registered group: its settings, last status, per-view hysteresis records and the
/// caller visibility saved before any group override.
#[derive(Clone, Debug, PartialEq)]
pub struct BillboardGroupRecord {
    pub config: BillboardGroupConfig,
    pub status: BillboardGroupStatus,
    pub views: BTreeMap<u8, BillboardViewState>,
    pub original_visibility: BTreeMap<u32, bool>,
}

/// The private registry of whole-tree billboard groups, keyed by caller id.
#[derive(Debug, Default)]
pub struct BillboardGroupRegistry {
    groups: BTreeMap<String, BillboardGroupRecord>,
}

/// Validates one group against the live handles and the instances `taken` by the other
/// groups. Nothing is mutated; every refusal carries [`BILLBOARD_INVALID`].
pub(crate) fn validate_group(
    config: &BillboardGroupConfig,
    live_roots: &BTreeSet<u32>,
    live_instances: &BTreeMap<u32, bool>,
    taken: &BTreeSet<u32>,
) -> Result<(), RaaError> {
    if !live_roots.contains(&config.root) {
        return Err(invalid(format!(
            "root {} is not a live scene node",
            config.root
        )));
    }
    if config.full.is_empty() {
        return Err(invalid("full must hold at least one LOD0 instance"));
    }
    let mut seen = BTreeSet::new();
    for &handle in &config.full {
        if !live_instances.contains_key(&handle) {
            return Err(invalid(format!("full instance {handle} is not live")));
        }
        if !seen.insert(handle) {
            return Err(invalid(format!("full instance {handle} is listed twice")));
        }
        if handle == config.proxy {
            return Err(invalid(format!("full instance {handle} is also the proxy")));
        }
        if taken.contains(&handle) {
            return Err(invalid(format!(
                "instance {handle} belongs to another group"
            )));
        }
    }
    if !live_instances.contains_key(&config.proxy) {
        return Err(invalid(format!(
            "proxy instance {} is not live",
            config.proxy
        )));
    }
    if taken.contains(&config.proxy) {
        return Err(invalid(format!(
            "proxy instance {} belongs to another group",
            config.proxy
        )));
    }
    if !config.switch_pixels.is_finite() || config.switch_pixels <= 0.0 {
        return Err(invalid("switch_pixels must be finite and positive"));
    }
    if !(-3..=3).contains(&config.bias) {
        return Err(invalid("bias must be in -3..=3"));
    }
    Ok(())
}

fn validate_id(id: &str) -> Result<(), RaaError> {
    if id.is_empty() {
        Err(invalid("a group id must not be empty"))
    } else {
        Ok(())
    }
}

impl BillboardGroupRegistry {
    pub fn new() -> Self {
        Self {
            groups: BTreeMap::new(),
        }
    }

    /// Handoff: "Prepare billboard groups for each camera" takes over from
    /// [`BillboardGroupRegistry::set`], the symbol this comment is anchored on. It reads
    /// the saved visibility records in [`BillboardGroupRecord::original_visibility`] (one
    /// `handle -> caller visibility` entry per full and proxy instance, captured before
    /// any group override), the unresolved-bake semantics
    /// ([`BillboardGroupConfig::bake`] is a bake-store id this registry does not resolve,
    /// so a group stays `ready == false` until preparation validates it) and the
    /// LIFECYCLE sentence: "install creates BillboardGroupRegistry in FeatureMap; Engine
    /// destruction drops it. set captures caller visibility before overriding it;
    /// removal/replacement restores prior visibility before any new group preparation."
    /// The earlier handoff for the same downstream work is
    /// `crates/raa-js/src/billboard_selection.rs:61 select_billboard`.
    ///
    /// Today this validates `id`, the live root, the nonempty unique `full` list, the
    /// proxy handle, disjoint membership against every other group, `switch_pixels` and
    /// `bias` before any mutation; then it stores the record with an empty per-view map and
    /// a not-ready status and returns the prior handles and their saved visibility in
    /// ascending handle order for the Engine adapter to restore.
    pub fn set(
        &mut self,
        id: &str,
        config: Option<BillboardGroupConfig>,
        live_roots: &BTreeSet<u32>,
        live_instances: &BTreeMap<u32, bool>,
    ) -> Result<Vec<(u32, bool)>, RaaError> {
        validate_id(id)?;
        let Some(config) = config else {
            return Ok(self.remove(id));
        };
        let mut taken = BTreeSet::new();
        for (other_id, record) in &self.groups {
            if other_id == id {
                continue;
            }
            taken.extend(record.config.full.iter().copied());
            taken.insert(record.config.proxy);
        }
        validate_group(&config, live_roots, live_instances, &taken)?;
        let restored = self.remove(id);
        let original_visibility = config
            .full
            .iter()
            .copied()
            .chain(std::iter::once(config.proxy))
            .map(|handle| (handle, live_instances[&handle]))
            .collect();
        self.groups.insert(
            id.to_owned(),
            BillboardGroupRecord {
                config,
                status: BillboardGroupStatus::default(),
                views: BTreeMap::new(),
                original_visibility,
            },
        );
        Ok(restored)
    }

    pub fn get(&self, id: &str) -> Option<&BillboardGroupRecord> {
        self.groups.get(id)
    }

    fn remove(&mut self, id: &str) -> Vec<(u32, bool)> {
        self.groups.remove(id).map_or_else(Vec::new, |record| {
            record.original_visibility.into_iter().collect()
        })
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) use web::{frame, install};

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use crate::engine::{Engine, InstallCtx};
    use crate::state::EngineState;
    use js_sys::{Array, Object, Reflect};
    use raa_core::GpuContext;
    use raa_web::FrameInput;
    use wasm_bindgen::{JsCast, prelude::*};

    pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
        ctx.features.insert(BillboardGroupRegistry::new());
        Ok(())
    }

    /// No preparation provider is attached yet, so every group keeps LOD0 visible and
    /// `billboardGroupState` reports `ready == false`.
    pub(crate) fn frame(
        _st: &mut EngineState,
        _gpu: &GpuContext,
        _input: &FrameInput,
    ) -> Result<(), RaaError> {
        Ok(())
    }

    fn required(obj: &JsValue, key: &str) -> Result<JsValue, RaaError> {
        if !obj.is_object() {
            return Err(invalid("a billboard group description must be an object"));
        }
        let v = Reflect::get(obj, &JsValue::from_str(key))
            .map_err(|_| invalid(format!("cannot read '{key}'")))?;
        if v.is_undefined() || v.is_null() {
            return Err(invalid(format!("'{key}' is required")));
        }
        Ok(v)
    }

    fn number(obj: &JsValue, key: &str) -> Result<f64, RaaError> {
        required(obj, key)?
            .as_f64()
            .ok_or_else(|| invalid(format!("'{key}' must be a number")))
    }

    fn handle_number(v: f64, name: &str) -> Result<u32, RaaError> {
        if v.fract() == 0.0 && (0.0..=f64::from(u32::MAX)).contains(&v) {
            Ok(v as u32)
        } else {
            Err(invalid(format!(
                "'{name}' must be a handle number, got {v}"
            )))
        }
    }

    fn handle(obj: &JsValue, key: &str) -> Result<u32, RaaError> {
        handle_number(number(obj, key)?, key)
    }

    fn full_list(obj: &JsValue) -> Result<Vec<u32>, RaaError> {
        let v = required(obj, "full")?;
        if !Array::is_array(&v) {
            return Err(invalid("'full' must be an array of instance handles"));
        }
        let a: &Array = v.unchecked_ref();
        let mut out = Vec::with_capacity(a.length() as usize);
        for i in 0..a.length() {
            let item = a
                .get(i)
                .as_f64()
                .ok_or_else(|| invalid(format!("'full[{i}]' must be a number")))?;
            out.push(handle_number(item, "full[]")?);
        }
        Ok(out)
    }

    fn level(obj: &JsValue) -> Result<BillboardLevel, RaaError> {
        let v = required(obj, "level")?;
        if let Some(s) = v.as_string() {
            if s == "auto" {
                return Ok(BillboardLevel::Auto);
            }
        } else if let Some(n) = v.as_f64() {
            if n == 0.0 {
                return Ok(BillboardLevel::Full);
            }
            if n == 1.0 {
                return Ok(BillboardLevel::Proxy);
            }
        }
        Err(invalid("'level' must be 'auto', 0 or 1"))
    }

    fn parse_desc(desc: &JsValue) -> Result<Option<BillboardGroupConfig>, RaaError> {
        if desc.is_null() || desc.is_undefined() {
            return Ok(None);
        }
        let bias = number(desc, "bias")?;
        if bias.fract() != 0.0 || !(-3.0..=3.0).contains(&bias) {
            return Err(invalid(format!(
                "'bias' must be a whole number from -3 to 3, got {bias}"
            )));
        }
        Ok(Some(BillboardGroupConfig {
            root: handle(desc, "root")?,
            full: full_list(desc)?,
            proxy: handle(desc, "proxy")?,
            bake: handle(desc, "bake")?,
            switch_pixels: number(desc, "switchPixels")? as f32,
            bias: bias as i8,
            level: level(desc)?,
        }))
    }

    fn live_roots(st: &EngineState) -> BTreeSet<u32> {
        st.nodes.iter().map(|(handle, _)| handle).collect()
    }

    fn live_instances(st: &EngineState) -> BTreeMap<u32, bool> {
        st.instances
            .iter()
            .filter(|(_, group)| !group.stale)
            .map(|(handle, group)| (handle, group.visible))
            .collect()
    }

    /// Restores the caller visibility the registry saved for the prior group. An instance
    /// the caller has removed since has nothing to restore.
    fn restore(st: &mut EngineState, prior: Vec<(u32, bool)>) {
        for (handle, visible) in prior {
            if let Ok(group) = st.instances.get_mut(handle) {
                let _ = group.set_visible(&mut st.scene, visible);
            }
        }
    }

    fn apply(st: &mut EngineState, id: &str, desc: &JsValue) -> Result<(), RaaError> {
        let config = parse_desc(desc)?;
        let roots = live_roots(st);
        let instances = live_instances(st);
        let prior = st
            .features
            .require_mut::<BillboardGroupRegistry>("billboard groups")?
            .set(id, config, &roots, &instances)?;
        restore(st, prior);
        Ok(())
    }

    fn set_field(obj: &Object, key: &str, value: JsValue) {
        let _ = Reflect::set(obj, &JsValue::from_str(key), &value);
    }

    fn state_value(record: &BillboardGroupRecord) -> JsValue {
        let obj = Object::new();
        set_field(
            &obj,
            "level",
            JsValue::from_f64(f64::from(record.status.level)),
        );
        set_field(
            &obj,
            "direction",
            JsValue::from_f64(f64::from(record.status.direction)),
        );
        set_field(&obj, "ready", JsValue::from_bool(record.status.ready));
        set_field(
            &obj,
            "error",
            match &record.status.error {
                Some(error) => JsValue::from_str(error),
                None => JsValue::NULL,
            },
        );
        obj.into()
    }

    #[wasm_bindgen(js_class = Engine)]
    impl Engine {
        /// Registers or replaces the whole-tree billboard group `id`, or removes it with a
        /// null description. A refusal (`billboard.invalid`) changes nothing.
        #[wasm_bindgen(js_name = setBillboardGroup)]
        pub fn set_billboard_group(&mut self, id: &str, desc: JsValue) -> Result<(), JsValue> {
            self.with_state(|_, st| apply(st, id, &desc))
        }

        /// The last prepared status of `id`, or null when no such group is registered.
        #[wasm_bindgen(js_name = billboardGroupState)]
        pub fn billboard_group_state(&self, id: &str) -> Result<JsValue, JsValue> {
            self.with_state(|_, st| {
                Ok(st
                    .features
                    .get::<BillboardGroupRegistry>()
                    .and_then(|registry| registry.get(id))
                    .map_or(JsValue::NULL, state_value))
            })
        }
    }
}
