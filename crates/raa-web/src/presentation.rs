//! Selection policy shared by the browser runtime and CPU-only tests.

/// Engine timeline identity for a single canvas.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct PresentTarget {
    pub revision: u64,
    pub tick: u64,
    pub phase: f64,
}

impl PresentTarget {
    pub fn includes(self, source: Self) -> bool {
        self.revision == source.revision && source <= self
    }
}

/// Identity of the cutoff snapshot from which a candidate was recorded.
#[derive(Clone, Debug, PartialEq)]
pub struct PredictionKey {
    pub target: PresentTarget,
    pub cutoff_ms: f64,
    pub snapshot_identity: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Selection {
    Real,
    Prediction,
    Hold,
}

pub(crate) fn select(
    target: PresentTarget,
    real: Option<(PresentTarget, u64)>,
    request: Option<&PredictionKey>,
    candidate: Option<(&PredictionKey, u64)>,
) -> Selection {
    let Some((real_target, real_id)) = real.filter(|(source, _)| target.includes(*source)) else {
        return Selection::Hold;
    };
    if real_target != target
        && let (Some(request), Some((key, source_id))) = (request, candidate)
        && request.target == target
        && key == request
        && source_id == real_id
    {
        return Selection::Prediction;
    }
    Selection::Real
}

#[cfg(test)]
mod tests {
    use super::*;

    const OLD: PresentTarget = PresentTarget {
        revision: 1,
        tick: 40,
        phase: 1.0,
    };
    const TARGET: PresentTarget = PresentTarget { tick: 41, ..OLD };

    #[test]
    fn missed_real_deadline_then_late_real_completion() {
        let key = PredictionKey {
            target: TARGET,
            cutoff_ms: 16.0,
            snapshot_identity: "input".into(),
        };
        let candidate = Some((&key, 9));
        assert_eq!(
            select(TARGET, Some((OLD, 9)), Some(&key), None),
            Selection::Real
        );
        assert_eq!(
            select(TARGET, Some((OLD, 9)), Some(&key), candidate),
            Selection::Prediction
        );
        assert_eq!(
            select(TARGET, Some((TARGET, 10)), Some(&key), candidate),
            Selection::Real
        );
    }

    #[test]
    fn rejects_changed_cutoff_source_rollback_and_future_real() {
        let key = PredictionKey {
            target: TARGET,
            cutoff_ms: 16.0,
            snapshot_identity: "input".into(),
        };
        let flash = PredictionKey {
            cutoff_ms: 17.0,
            snapshot_identity: "flash".into(),
            ..key.clone()
        };
        let candidate = Some((&key, 9));
        assert_eq!(
            select(TARGET, Some((OLD, 9)), Some(&flash), candidate),
            Selection::Real
        );
        assert_eq!(
            select(TARGET, Some((OLD, 10)), Some(&key), candidate),
            Selection::Real
        );
        assert_eq!(
            select(TARGET, Some((OLD, 9)), None, candidate),
            Selection::Real
        );
        let rollback = PresentTarget {
            revision: 2,
            ..TARGET
        };
        assert_eq!(
            select(rollback, Some((OLD, 9)), Some(&key), candidate),
            Selection::Hold
        );
        assert_eq!(select(OLD, Some((TARGET, 10)), None, None), Selection::Hold);
    }
}
