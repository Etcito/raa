//! Pure Oklab transforms for offline colour tooling and unit tests.
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter().zip(b).map(|(x, y)| x * y).sum()
}
pub fn to_oklab(rgb: [f64; 3]) -> [f64; 3] {
    let lms = [
        dot(rgb, [0.4122214708, 0.5363325363, 0.0514459929]),
        dot(rgb, [0.2119034982, 0.6806995451, 0.1073969566]),
        dot(rgb, [0.0883024619, 0.2817188376, 0.6299787005]),
    ]
    .map(f64::cbrt);
    [
        dot(lms, [0.2104542553, 0.7936177850, -0.0040720468]),
        dot(lms, [1.9779984951, -2.4285922050, 0.4505937099]),
        dot(lms, [0.0259040371, 0.7827717662, -0.8086757660]),
    ]
}
pub fn from_oklab(lab: [f64; 3]) -> [f64; 3] {
    let [l, a, b] = lab;
    let q = [
        l + 0.3963377774 * a + 0.2158037573 * b,
        l - 0.1055613458 * a - 0.0638541728 * b,
        l - 0.0894841775 * a - 1.2914855480 * b,
    ]
    .map(|x| x * x * x);
    [
        dot(q, [4.0767416621, -3.3077115913, 0.2309699292]),
        dot(q, [-1.2684380046, 2.6097574011, -0.3413193965]),
        dot(q, [-0.0041960863, -0.7034186147, 1.7076147010]),
    ]
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn oklab_round_trip() {
        for r in [0.0, 0.01, 0.18, 0.5, 1.0] {
            for g in [0.0, 0.2, 1.0] {
                for b in [0.0, 0.6, 1.0] {
                    let rgb = [r, g, b];
                    let got = from_oklab(to_oklab(rgb));
                    for i in 0..3 {
                        assert!((got[i] - rgb[i]).abs() < 3e-7, "{rgb:?}: {got:?}");
                    }
                }
            }
        }
    }
}
