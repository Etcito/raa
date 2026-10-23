use raa_bloom::mip_sizes;

fn levels(w: u32, h: u32) -> Vec<(u32, u32)> {
    let (a, n) = mip_sizes(w, h);
    a[..n].to_vec()
}

#[test]
fn hd_720p_has_six_levels() {
    let l = levels(1280, 720);
    assert_eq!(l.len(), 6);
    assert_eq!(l[0], (640, 360));
    assert_eq!(l[5], (20, 11));
}

#[test]
fn uhd_is_capped_at_eight() {
    assert_eq!(levels(3840, 2160).len(), 8);
}

#[test]
fn tiny_targets_have_none() {
    assert_eq!(levels(15, 15).len(), 0);
    assert_eq!(levels(1, 1).len(), 0);
}
