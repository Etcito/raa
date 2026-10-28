//! Tick-based transition control, independent of pose sampling.
use crate::{Compiled, CondOp, Condition, Crossfade, GraphState, ParamValue, Params, WILDCARD};

fn compare<T: PartialOrd>(a: T, b: T, op: CondOp) -> bool {
    match op {
        CondOp::Eq => a == b,
        CondOp::Ne => a != b,
        CondOp::Gt => a > b,
        CondOp::Lt => a < b,
        CondOp::Ge => a >= b,
        CondOp::Le => a <= b,
        CondOp::Trigger => false,
    }
}
fn matches(condition: &Condition, params: &Params, pending: &[bool]) -> bool {
    let Some(index) = params.index_of(&condition.param) else { return false; };
    match params.value(index) {
        Some(ParamValue::Trigger(_)) => condition.op == CondOp::Trigger && pending[index],
        Some(ParamValue::Float(v)) => compare(v, condition.value.rt, condition.op),
        Some(ParamValue::Int(v)) => compare(v, condition.value.rt as i32, condition.op),
        Some(ParamValue::Bool(v)) => compare(v, condition.value.rt != 0.0, condition.op),
        None => false,
    }
}
impl Compiled {
    /// Advance transition control exactly one tick per call. Inputs belong to this
    /// graph; trigger inputs are pulses (resubmitting a true input fires it again).
    /// Pending pulses survive failed conditions and exit gates. Only triggers in
    /// the winning transition are consumed. Machines run in document order, so
    /// the first machine to fire consumes a shared trigger before later machines.
    ///
    /// Each referenced machine runs once, using its first layer as its cursor;
    /// all layers referencing it receive the same cursor. No transition chains.
    /// Gates use the pre-call local clock, inclusively. A non-transitioning clock
    /// advances by one; a destination starts at clock/phase zero. Clocks saturate.
    /// Motion sampling is responsible for advancing phases.
    ///
    /// Existing fades advance before selection. A new fade starts at elapsed zero,
    /// reaches weight one after `blend_ticks` further calls, then is removed.
    /// Zero ticks switches immediately. Interrupting discards the previous fade
    /// and starts from the current destination at its current phase, even if the
    /// old fade had not completed. The source phase is frozen during the fade.
    pub fn advance_transitions(&self, state: &mut GraphState, params: &Params) {
        state.apply(params);
        for (machine_index, machine) in self.graph.machines.iter().enumerate() {
            let Some(index) = state.layers.iter().position(|l| l.machine == machine_index)
                else { continue; };
            let mut cursor = state.layers[index].clone();
            if let Some(fade) = &mut cursor.fade {
                fade.elapsed = fade.elapsed.saturating_add(1);
                if fade.elapsed >= fade.ticks { cursor.fade = None; }
            }
            let current = &machine.states[cursor.state].name;
            let winner = machine.transitions.iter().enumerate().filter(|(_, t)| {
                (t.from == *current || t.from == WILDCARD)
                    && t.exit_tick.is_none_or(|exit| cursor.clock >= exit)
                    && t.when.iter().all(|c| matches(c, params, &state.pending))
            }).min_by_key(|(order, t)| (t.from == WILDCARD, t.priority, *order));
            if let Some((_, transition)) = winner {
                for condition in &transition.when {
                    if condition.op == CondOp::Trigger {
                        state.pending[params.index_of(&condition.param).unwrap()] = false;
                    }
                }
                cursor.fade = (transition.blend_ticks != 0).then_some(Crossfade {
                    from: cursor.state, from_phase: cursor.phase,
                    elapsed: 0, ticks: transition.blend_ticks,
                });
                cursor.state = machine.states.iter().position(|s| s.name == transition.to).unwrap();
                cursor.clock = 0;
                cursor.phase = 0.0;
            } else {
                cursor.clock = cursor.clock.saturating_add(1);
            }
            for layer in state.layers.iter_mut().filter(|l| l.machine == machine_index) {
                *layer = cursor.clone();
            }
        }
        state.tick = state.tick.saturating_add(1);
    }
}
