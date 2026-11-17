//! Pins the clamped contact-shadow load of the `csm` Shadow slot (`src/wgsl/csm.wgsl`,
//! `raa_slot_shadow`). With `shadow.contact` off the lighting binds the 1x1 dummy contact texture
//! (value 1); an unclamped `textureLoad` at the pixel coordinate is out of bounds for every pixel
//! but (0, 0), robust backends answer 0 and the sun term vanishes. Device-free: text only.

const CSM_WGSL: &str = include_str!("../src/wgsl/csm.wgsl");

/// The body of `raa_slot_shadow`, from its signature to the end of the file.
fn slot_shadow() -> &'static str {
    let start = CSM_WGSL
        .find("fn raa_slot_shadow(")
        .expect("csm.wgsl defines raa_slot_shadow");
    &CSM_WGSL[start..]
}

/// Whitespace removed, so the checks do not depend on formatting.
fn compact(text: &str) -> String {
    text.chars().filter(|c| !c.is_whitespace()).collect()
}

#[test]
fn contact_load_is_clamped_to_the_texture_dimensions_minus_one() {
    let body = compact(slot_shadow());
    let size = "letcontact_size=vec2<i32>(textureDimensions(raa_contact_tex));";
    let clamp = "letcontact_px=clamp(vec2<i32>(frag.xy),vec2<i32>(0),contact_size-vec2<i32>(1));";
    let load = "letcontact=textureLoad(raa_contact_tex,contact_px,0).r;";
    let size_at = body
        .find(size)
        .expect("contact_size reads textureDimensions(raa_contact_tex)");
    let clamp_at = body
        .find(clamp)
        .expect("contact_px is clamped to contact_size - 1");
    let load_at = body.find(load).expect("the contact load uses contact_px");
    assert!(size_at < clamp_at && clamp_at < load_at);
}

#[test]
fn no_unclamped_contact_load_remains() {
    let all = compact(CSM_WGSL);
    assert!(
        !all.contains("textureLoad(raa_contact_tex,vec2<i32>(frag.xy)"),
        "an unclamped textureLoad(raa_contact_tex, vec2<i32>(frag.xy), ..) remains in csm.wgsl"
    );
    // Every load of the contact texture goes through the clamped coordinate.
    let loads = all.matches("textureLoad(raa_contact_tex,").count();
    let clamped = all
        .matches("textureLoad(raa_contact_tex,contact_px,")
        .count();
    assert_eq!(loads, clamped);
    assert!(loads >= 1);
}
