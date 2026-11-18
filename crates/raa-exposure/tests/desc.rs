use raa_core::RaaError;
use raa_exposure::AutoExposureDesc;

/// The default desc with one field set by `edit`.
fn edited(edit: impl FnOnce(&mut AutoExposureDesc)) -> AutoExposureDesc {
    let mut desc = AutoExposureDesc::default();
    edit(&mut desc);
    desc
}

fn refusal(desc: AutoExposureDesc) -> (&'static str, String) {
    match desc.validate() {
        Err(RaaError::FrameGraph { code, message }) => (code, message),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn defaults_equal_the_s46_table() {
    let d = AutoExposureDesc::default();
    assert!(!d.enabled);
    assert_eq!(d.compensation_ev, 0.263);
    assert_eq!(d.low_percent, 0.2);
    assert_eq!(d.high_percent, 0.7);
    assert_eq!(d.speed_up, 2.4);
    assert_eq!(d.speed_down, 1.1);
    assert_eq!(d.min_ev, -6.0);
    assert_eq!(d.max_ev, 6.0);
    assert!(d.validate().is_ok());
}

#[test]
fn each_rule_refuses_naming_its_js_key() {
    let cases = [
        ("compensationEv", edited(|d| d.compensation_ev = f32::NAN)),
        ("compensationEv", edited(|d| d.compensation_ev = 9.0)),
        ("lowPercent", edited(|d| d.low_percent = -0.1)),
        ("highPercent", edited(|d| d.high_percent = 1.5)),
        ("speedUp", edited(|d| d.speed_up = 0.0)),
        ("speedDown", edited(|d| d.speed_down = -1.0)),
        ("maxEv", edited(|d| d.max_ev = f32::NAN)),
        ("minEv", edited(|d| (d.min_ev, d.max_ev) = (3.0, 2.0))),
    ];
    for (key, desc) in cases {
        let (code, message) = refusal(desc);
        assert_eq!(code, "post.invalid");
        assert!(
            message.starts_with(&format!("`{key}`")),
            "{key}: unexpected message {message:?}"
        );
    }
}

#[test]
fn low_percent_must_be_below_high_percent() {
    let desc = edited(|d| (d.low_percent, d.high_percent) = (0.7, 0.7));
    let (code, message) = refusal(desc);
    assert_eq!(code, "post.invalid");
    assert_eq!(
        message,
        "`lowPercent` must be below `highPercent`, got 0.7 >= 0.7"
    );
}

#[test]
fn an_edited_desc_within_the_table_passes() {
    let desc = edited(|d| {
        d.enabled = true;
        d.compensation_ev = -8.0;
        (d.low_percent, d.high_percent) = (0.0, 1.0);
        (d.speed_up, d.speed_down) = (0.5, 10.0);
        (d.min_ev, d.max_ev) = (2.0, 2.0);
    });
    assert!(desc.validate().is_ok());
}
